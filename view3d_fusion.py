"""Live 3D viewer, fusion edition: all cameras build ONE room, and one box per person.

    python view3d_fusion.py [--cameras 0] [--stride 8]
    python view3d_fusion.py --cameras 0,1   # two cameras, one fused world

What's new over view3d_ultra.py (which just overlays per-camera views):

- One merged live room cloud (world/points) — every camera reprojects its
  current depth into world space and the fusion thread concatenates them into a
  SINGLE point cloud (not one per camera): where the views overlap it densifies,
  and each camera fills the others' blind spots. People are masked out (they get
  their own cloud), so this is the live room only, in true color.
- Persistent room map (world/map) — the same masked depth is also integrated
  over time into one shared TSDF voxel grid (fusion.py). Hundreds of noisy
  monocular-depth frames average into a single stable surface; free-space
  carving erases residue; it appears over the first ~10-20 s and keeps refining.
  The refined, denoised counterpart to the instant world/points cloud.
- One fused person cloud (world/people/points) — the depth pixels inside each
  detection's bbox are reprojected to world space in true color; both cameras'
  views of the same person are merged into one cloud, so you see the person's
  actual shape in space, not just a box.
- Tracked people (SORT in 3D) — per-camera detections are transformed to world
  space, clustered across cameras (two detections from the same camera never
  merge), then fed to one constant-velocity Kalman filter per person with
  Hungarian assignment (fusion.py). A person seen by both cameras is ONE box at
  world/people with a persistent #ID, a stable color, a speed label, and a
  velocity arrow when moving; through short detection dropouts the box coasts
  on its predicted path (dimmed, "predicted"). Per-camera raw boxes are off by
  default (--per-cam-boxes to debug association).

Architecture: one worker thread per camera (capture -> depth -> detection),
each publishing world-space room points, person clouds + detections, and masked
depth to a single fusion thread that owns the merged cloud, the map, and the
tracked people. --profile prints per-stage timings; --input-size / --capture
trade depth detail for FPS.

Each camera needs calibration/camera_<idx>.json from calibrate.py; multi-camera
also needs "extrinsics" from extrinsics.py (that is what makes world space
shared). The map covers an 8x8x3 m box around the world origin (the
checkerboard) at 4 cm voxels by default — --bounds / --voxel to change.
Ctrl+C in the terminal to stop.
"""
import argparse
import json
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metric_depth'))

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.nn.functional as TF
from ultralytics import YOLO

from depth_anything_v2.dpt import DepthAnythingV2  # metric variant
from fusion import TSDFGrid, fuse_people, PersonTracker

DEVICE = os.environ.get('DEVICE') or (  # auto: NVIDIA > Apple GPU > CPU
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu')
INPUT_SIZE = 384
MAX_DEPTH = 20.0
DEPTH_SCALE = 0.71  # tape-measure correction, see test.py — per-camera override via
                    # a "depth_scale" key in calibration/camera_<idx>.json
BOX_COLORS = [(80, 200, 255), (255, 170, 80), (170, 255, 120), (255, 120, 200)]  # per camera
TRACK_COLORS = [(255, 90, 90), (90, 200, 255), (255, 210, 80), (150, 255, 130),
                (230, 120, 255), (255, 150, 60), (120, 160, 255), (255, 240, 180)]  # per track ID
DISPLAY_SCALE = 0.5  # camera images shown in the viewer at this scale (geometry unaffected)
PERSON_MERGE_M = 0.75  # detections from different cameras closer than this = same person
PERSON_MERGE_RANGE_FRAC = 0.08  # + this fraction of distance-from-camera, widening the
                                # budget for far-away people (proportional depth-scale error)
DET_MAX_AGE = 0.5  # s — a camera's detections older than this drop out of fusion
CLOUD_MAX_AGE = 2.0  # s — a camera's room points older than this stop feeding the merged cloud
MASK_PAD = 0.15  # person bbox padding (fraction) when masking depth for the map

parser = argparse.ArgumentParser()
parser.add_argument('--cameras', default='0', help='comma-separated camera indices, e.g. 0,1')
parser.add_argument('--stride', type=int, default=8, help='live point cloud subsampling (pixels)')
parser.add_argument('--voxel', type=float, default=0.04, help='room map voxel size (meters)')
parser.add_argument('--bounds', default=None,
                    help='room map box "x0,x1,y0,y1,z0,z1" in world meters (default 8x8x3 around origin)')
parser.add_argument('--map-interval', type=float, default=0.4,
                    help='seconds between map integrations per camera')
parser.add_argument('--map-log-interval', type=float, default=2.0,
                    help='seconds between map updates sent to the viewer')
parser.add_argument('--no-map', action='store_true', help='disable the persistent room map')
parser.add_argument('--per-cam-boxes', action='store_true',
                    help='also show per-camera person boxes (debug association)')
parser.add_argument('--save', default=None, help='record to .rrd file instead of opening the viewer')
parser.add_argument('--max-frames', type=int, default=0, help='stop after N frames per camera (0 = run forever)')
parser.add_argument('--input-size', type=int, default=384,
                    help='depth model input size (multiple of 14; lower = faster, coarser depth)')
parser.add_argument('--capture', default='1920x1080',
                    help='camera capture WxH; lower = faster, sparser (intrinsics rescale automatically)')
parser.add_argument('--person-stride', type=int, default=3,
                    help='pixel subsampling for the per-person point clouds (lower = denser)')
parser.add_argument('--profile', action='store_true',
                    help='print per-stage timings (depth/detect/rest ms) in the status line')
args = parser.parse_args()
cam_indices = [int(x) for x in args.cameras.split(',')]
INPUT_SIZE = args.input_size
cap_w, cap_h = (int(v) for v in args.capture.lower().split('x'))

# --- depth model (shared: its forward pass is stateless, safe across threads) ---
if DEVICE == 'cuda':
    torch.set_float32_matmul_precision('high')  # TF32 matmuls, ~20% faster forward
    torch.backends.cudnn.benchmark = True
depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384],
                              max_depth=MAX_DEPTH)
depth_model.load_state_dict(torch.load('checkpoints/depth_anything_v2_metric_hypersim_vits.pth',
                                       map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()
print(f'device: {DEVICE}')

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def infer_depth(frame_bgr, model_size_hw):
    """model.infer_image with the resize/normalize moved onto the GPU — the numpy
    preprocess costs ~19 ms per 1080p frame, this path ~2 ms. Runs on any GPU
    (CUDA or Apple MPS); CPU falls back to the stock numpy infer_image."""
    if DEVICE == 'cpu':
        return depth_model.infer_image(frame_bgr, input_size=INPUT_SIZE)
    down_mode = 'bicubic' if DEVICE == 'cuda' else 'bilinear'  # MPS has no bicubic kernel
    x = torch.from_numpy(frame_bgr).to(DEVICE)
    x = x.flip(-1).permute(2, 0, 1)[None].float().div_(255)  # BGR uint8 -> RGB 0..1
    x = TF.interpolate(x, size=model_size_hw, mode=down_mode, align_corners=False)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    with torch.no_grad():
        d = depth_model(x)
    d = TF.interpolate(d[:, None], size=frame_bgr.shape[:2], mode='bilinear', align_corners=True)
    return d[0, 0].cpu().numpy()


class Grabber(threading.Thread):
    """Reads frames as fast as the camera delivers them, undistorts off the main
    thread, and keeps only the newest frame — the pipeline never blocks on I/O."""

    def __init__(self, cap, undistort_maps):
        super().__init__(daemon=True)
        self.cap, self.maps = cap, undistort_maps
        self.lock = threading.Lock()
        self.frame = None
        self.alive = True

    def run(self):
        while self.alive:
            ret, frame = self.cap.read()
            if not ret:
                self.alive = False
                break
            if self.maps is not None:
                frame = cv2.remap(frame, *self.maps, cv2.INTER_LINEAR)
            with self.lock:
                self.frame = frame

    def take(self):
        """Newest unseen frame, or None if the camera hasn't produced one yet."""
        with self.lock:
            frame, self.frame = self.frame, None
            return frame


class Worker(threading.Thread):
    """Per-camera pipeline: newest frame -> depth -> detection -> live point cloud,
    publishing world-space detections + person-masked depth for the fusion thread.
    One per camera; while this one's inference is on the GPU (GIL released), the
    other workers run their CPU stages."""

    def __init__(self, cam, yolo, color, stop):
        super().__init__(daemon=True)
        self.cam, self.yolo, self.color, self.stop = cam, yolo, color, stop
        self.fps, self.frames, self.people = 0.0, 0, 0
        self.t_depth = self.t_det = self.t_rest = 0.0  # EMA per-stage ms (--profile)

    def run(self):
        c = self.cam
        base = f'world/cam{c["idx"]}'
        t_prev = None
        while not self.stop.is_set():
            if args.max_frames and self.frames >= args.max_frames:
                break
            frame = c['grabber'].take()
            if frame is None:
                if not c['grabber'].alive:
                    break
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            depth = infer_depth(frame, c['model_size_hw']) * c['depth_scale']  # HxW meters
            t1 = time.perf_counter()
            det = self.yolo(frame, classes=[0], device=DEVICE, verbose=False)[0]  # class 0 = person
            t2 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time()

            # person boxes: median depth over the torso region -> camera-frame center + height
            people = []  # (x1, y1, x2, y2, center_cam, z_p, height_m)
            for box in det.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                bw, bh = x2 - x1, y2 - y1
                torso = depth[y1 + bh // 4: y2 - bh // 4, x1 + bw // 4: x2 - bw // 4]
                if torso.size == 0:
                    continue
                z_p = float(np.median(torso))
                u_c, v_c = (x1 + x2) / 2, (y1 + y2) / 2
                center = np.array([(u_c - c['cx']) / c['fx'] * z_p,
                                   (v_c - c['cy']) / c['fy'] * z_p, z_p])
                people.append((x1, y1, x2, y2, center, z_p, bh / c['fy'] * z_p))
            self.people = len(people)

            # Each detection carries its own world-space point cloud (the depth pixels
            # inside its bbox, subsampled, in true color). The fusion thread merges the
            # two cameras' clouds of the same person into one. A shared person_mask keeps
            # people out of the room clouds (they get their own cloud instead).
            H, W = depth.shape
            person_mask = np.zeros((H, W), bool)
            ps = args.person_stride
            dets = []
            for x1, y1, x2, y2, center_cam, z_p, height in people:
                px, py = int((x2 - x1) * MASK_PAD), int((y2 - y1) * MASK_PAD)
                person_mask[max(y1 - py, 0):y2 + py, max(x1 - px, 0):x2 + px] = True
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
                if x2 <= x1 or y2 <= y1:
                    continue
                sub_z = depth[y1:y2:ps, x1:x2:ps]
                vv, uu = np.mgrid[y1:y2:ps, x1:x2:ps]
                keep = sub_z > 0.05
                zc = sub_z[keep]
                cam_pts = np.stack([(uu[keep] - c['cx']) / c['fx'] * zc,
                                    (vv[keep] - c['cy']) / c['fy'] * zc, zc], axis=-1)
                dets.append(dict(center=c['R_w'] @ center_cam + c['t_w'], height=height, range=z_p,
                                 points=(cam_pts @ c['R_w'].T + c['t_w']).astype(np.float32),
                                 colors=rgb[y1:y2:ps, x1:x2:ps][keep]))

            # masked depth for the persistent TSDF map (0 = "no data", so people never enter it)
            dmap = depth
            if people and not args.no_map:
                dmap = depth.copy()
                dmap[person_mask] = 0.0
            with c['lock']:
                c['dets'] = (now, dets)
                c['map_frame'] = (now, dmap, rgb)

            # rerun's current time is thread-local: per-camera frame sequence plus a
            # shared wall clock so entities from different cameras can be correlated
            rr.set_time('frame', sequence=self.frames)
            rr.set_time('time', timestamp=now)

            # this camera's contribution to the ONE merged room cloud: people masked out,
            # invalid/zero-depth pixels dropped (so nothing piles up at the camera origin),
            # reprojected into world space in true color. The fusion thread concatenates
            # every camera's latest into a single world/points cloud — overlap densifies
            # and each camera fills the others' blind spots.
            s = args.stride
            z = depth[::s, ::s]
            keep = (z > 0.05) & ~person_mask[::s, ::s]
            zc = z[keep]
            cam_pts = np.stack([c['xn'][keep] * zc, c['yn'][keep] * zc, zc], axis=-1)
            with c['lock']:
                c['room'] = (now, (cam_pts @ c['R_w'].T + c['t_w']).astype(np.float32),
                             rgb[::s, ::s][keep])

            if args.per_cam_boxes:  # debug: this camera's own (unfused) boxes
                centers, half_sizes, labels = [], [], []
                for x1, y1, x2, y2, center, z_p, _h in people:
                    centers.append(center.tolist())
                    half_sizes.append([(x2 - x1) / c['fx'] * z_p / 2,
                                       (y2 - y1) / c['fy'] * z_p / 2, 0.25])
                    labels.append(f'cam{c["idx"]} {z_p:.2f}m')
                rr.log(f'{base}/people', rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                                                    labels=labels, colors=[self.color]))

            # camera image, downscaled + jpeg-compressed so the stream stays light
            small = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            rr.log(f'{base}/image',
                   rr.EncodedImage(contents=cv2.imencode('.jpg', small)[1].tobytes(),
                                   media_type='image/jpeg'))

            t3 = time.perf_counter()
            self.t_depth = 0.9 * self.t_depth + 0.1 * (t1 - t0) * 1e3
            self.t_det = 0.9 * self.t_det + 0.1 * (t2 - t1) * 1e3
            self.t_rest = 0.9 * self.t_rest + 0.1 * (t3 - t2) * 1e3
            if t_prev is not None:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            self.frames += 1


class FusionThread(threading.Thread):
    """Owns the shared room. Merges every camera's person detections into
    world-space clusters, runs the 3D SORT tracker over them (world/people:
    one box + persistent ID + velocity per person), and integrates the masked
    depth into the persistent TSDF map (world/map)."""

    def __init__(self, cams, grid, stop):
        super().__init__(daemon=True)
        self.cams, self.grid, self.stop = cams, grid, stop
        self.tracker = PersonTracker(gate=PERSON_MERGE_M * 1.6)
        self.n_people, self.n_map, self.n_cloud, self.ticks = 0, 0, 0, 0

    def run(self):
        last_int = {c['idx']: 0.0 for c in self.cams}
        room_t = {c['idx']: -1.0 for c in self.cams}
        last_log = 0.0
        while not self.stop.is_set():
            now = time.time()
            rr.set_time('frame', sequence=self.ticks)
            rr.set_time('time', timestamp=now)

            # --- people: cluster fresh detections across cameras, then track ---
            dets = []
            for c in self.cams:
                with c['lock']:
                    t_d, ds = c['dets']
                if now - t_d < DET_MAX_AGE:
                    dets += [(c['idx'], d) for d in ds]
            meas, ppts, pcols = [], [], []
            for cl in fuse_people(dets, merge_dist=PERSON_MERGE_M, range_frac=PERSON_MERGE_RANGE_FRAC):
                center = np.mean([d['center'] for _, d in cl], axis=0)
                meas.append(dict(center=center, height=max(d['height'] for _, d in cl), n_cams=len(cl)))
                # each camera reconstructs the person at a slightly different spot (per-camera
                # monocular depth-scale error), so raw concatenation shows two offset copies.
                # Shift each view so its centroid lands on the fused center -> they overlap.
                for _, d in cl:
                    if len(d['points']):
                        ppts.append(d['points'] + (center - d['center']).astype(np.float32))
                        pcols.append(d['colors'])
            tracks = self.tracker.step(meas, now)

            # one fused, true-color point cloud of everyone — their actual shape in space
            rr.log('world/people/points',
                   rr.Points3D(np.concatenate(ppts) if ppts else np.zeros((0, 3), np.float32),
                               colors=np.concatenate(pcols) if pcols else None, radii=0.015))

            centers, half_sizes, labels, colors = [], [], [], []
            arr_o, arr_v, arr_c = [], [], []
            for tr in tracks:
                col = TRACK_COLORS[tr['id'] % len(TRACK_COLORS)]
                h = max(tr['height'], 0.5)
                centers.append(tr['center'].tolist())
                # axis-aligned person box: world Z is up with extrinsics, camera -Y without
                half_sizes.append([0.3, 0.3, h / 2] if have_extrinsics else [0.3, h / 2, 0.3])
                speed = float(np.linalg.norm(tr['vel']))
                if tr['matched']:
                    labels.append(f'#{tr["id"]} {speed:.1f} m/s ({tr["n_cams"]} cam)')
                    colors.append(col)
                else:  # coasting on the Kalman prediction through a detection dropout
                    labels.append(f'#{tr["id"]} (predicted)')
                    colors.append(tuple(v // 2 for v in col))
                if speed > 0.3:  # velocity arrow: where they're headed in the next ~0.7 s
                    arr_o.append(tr['center'].tolist())
                    arr_v.append((tr['vel'] * 0.7).tolist())
                    arr_c.append(col)
            rr.log('world/people', rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                                              labels=labels, colors=colors))
            rr.log('world/people/velocity', rr.Arrows3D(origins=arr_o, vectors=arr_v, colors=arr_c))
            self.n_people = len(tracks)

            # --- room map: integrate each camera's newest masked depth, throttled ---
            if self.grid is not None:
                for c in self.cams:
                    if now - last_int[c['idx']] < args.map_interval:
                        continue
                    with c['lock']:
                        mf, c['map_frame'] = c['map_frame'], None  # each frame integrates once
                    if mf is None:
                        continue
                    _, dmap, rgb = mf
                    self.grid.integrate(dmap, rgb, c['fx'], c['fy'], c['cx'], c['cy'],
                                        c['R_w'], c['t_w'])
                    last_int[c['idx']] = now
                if now - last_log > args.map_log_interval:
                    pts, cols = self.grid.extract()
                    rr.log('world/map', rr.Points3D(pts, colors=cols, radii=self.grid.voxel / 2))
                    self.n_map = len(pts)
                    last_log = now

            # --- merged live room cloud: all cameras' latest world-space room points as
            #     ONE cloud, re-logged only when a camera delivers a fresh frame ---
            rpts, rcols, fresh = [], [], False
            for c in self.cams:
                with c['lock']:
                    r = c['room']
                if r is None or now - r[0] > CLOUD_MAX_AGE:
                    continue
                rpts.append(r[1])
                rcols.append(r[2])
                if r[0] != room_t[c['idx']]:
                    room_t[c['idx']], fresh = r[0], True
            if fresh:
                rr.log('world/points',
                       rr.Points3D(np.concatenate(rpts), colors=np.concatenate(rcols), radii=0.01))
                self.n_cloud = sum(len(p) for p in rpts)

            self.ticks += 1
            time.sleep(0.05)


# --- cameras: capture + intrinsics + extrinsics per index ---
cams = []
for idx in cam_indices:
    cap = cv2.VideoCapture(idx)
    # request the configured resolution (--capture); intrinsics rescale to whatever the
    # camera actually delivers, so this trades detail for speed without breaking geometry
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
    ret, frame = cap.read()
    assert ret, f'no frame from camera {idx}'
    H, W = frame.shape[:2]

    calib_path = f'calibration/camera_{idx}.json'
    calib = None
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calib = json.load(f)
        print(f'cam{idx}: intrinsics from {calib_path} (rms {calib["rms_reprojection_error_px"]:.3f} px)')
    else:
        print(f'cam{idx}: WARNING: {calib_path} not found — run calibrate.py. Guessing 70 deg hFOV.')

    if calib:
        sx, sy = W / calib['resolution'][0], H / calib['resolution'][1]  # rescale if resolution differs
        fx, fy, cx, cy = calib['fx'] * sx, calib['fy'] * sy, calib['cx'] * sx, calib['cy'] * sy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        undistort_maps = cv2.initUndistortRectifyMap(K, np.array(calib['dist']), None, K, (W, H), cv2.CV_32FC1)
    else:
        fx = fy = W / (2 * np.tan(np.radians(70 / 2)))
        cx, cy = W / 2, H / 2
        undistort_maps = None

    extrinsics = (calib or {}).get('extrinsics')
    if extrinsics:
        t = extrinsics['t']
        print(f'cam{idx}: extrinsics loaded — at world ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) m')
    elif len(cam_indices) > 1:
        print(f'cam{idx}: WARNING: no extrinsics — run extrinsics.py, else this camera sits at the world origin')

    # model input size: aspect-preserving, both dims >= INPUT_SIZE, multiples of 14
    scale = INPUT_SIZE / min(H, W)
    model_size_hw = (math.ceil(H * scale / 14) * 14, math.ceil(W * scale / 14) * 14)

    us, vs = np.meshgrid(np.arange(0, W, args.stride), np.arange(0, H, args.stride))
    grabber = Grabber(cap, undistort_maps)
    grabber.start()
    cams.append(dict(
        idx=idx, cap=cap, grabber=grabber, W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy,
        extrinsics=extrinsics, model_size_hw=model_size_hw,
        depth_scale=(calib or {}).get('depth_scale', DEPTH_SCALE),
        xn=(us - cx) / fx, yn=(vs - cy) / fy,  # normalized ray directions at stride
        R_w=np.array(extrinsics['R'], np.float32) if extrinsics else np.eye(3, dtype=np.float32),
        t_w=np.array(extrinsics['t'], np.float32) if extrinsics else np.zeros(3, np.float32),
        lock=threading.Lock(), dets=(0.0, []), map_frame=None, room=None,
    ))

# --- rerun ---
rr.init('camera-map', spawn=args.save is None)
if args.save:
    rr.save(args.save)
# explicit layout: 3D scene + one live camera view per camera
rr.send_blueprint(rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='world', name='3D scene'),
        rrb.Vertical(*[rrb.Spatial2DView(origin=f'world/cam{c["idx"]}/image', name=f'camera {c["idx"]}')
                       for c in cams]),
        column_shares=[2, 1],
    ),
))
# with extrinsics the world is the board frame (Z up); without, the camera frame (RDF)
have_extrinsics = any(c['extrinsics'] for c in cams)
rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP if have_extrinsics else rr.ViewCoordinates.RDF,
       static=True)
for c in cams:
    base = f'world/cam{c["idx"]}'
    if c['extrinsics']:
        rr.log(base, rr.Transform3D(translation=c['extrinsics']['t'],
                                    mat3x3=c['extrinsics']['R']), static=True)
    d = DISPLAY_SCALE  # pinhole matches the downscaled display image; same FOV either way
    rr.log(f'{base}/image',
           rr.Pinhole(resolution=[c['W'] * d, c['H'] * d], focal_length=[c['fx'] * d, c['fy'] * d],
                      principal_point=[c['cx'] * d, c['cy'] * d], image_plane_distance=0.3,
                      camera_xyz=rr.ViewCoordinates.RDF), static=True)

# --- the shared room map ---
grid = None
if not args.no_map:
    if args.bounds:
        bounds = [float(x) for x in args.bounds.split(',')]
    else:  # board world: 8x8 m around the origin, floor to 2.7 m up.
        # camera world (no extrinsics, RDF): x right, y down, z forward into the room
        bounds = [-4, 4, -4, 4, -0.3, 2.7] if have_extrinsics else [-4, 4, -1.8, 1.2, 0.1, 6.5]
    grid = TSDFGrid(bounds, voxel=args.voxel)
    print(f'map: {grid.pts.shape[0] / 1e6:.1f}M voxels @ {args.voxel * 100:.0f} cm, bounds {bounds}')

# --- threads: one pipeline per camera + one fusion thread for the shared room ---
stop = threading.Event()
# YOLO predictors are not thread-safe -> one instance per worker (created here,
# sequentially, so the first-run auto-download can't race)
workers = [Worker(c, YOLO('yolo11n.pt'), BOX_COLORS[i % len(BOX_COLORS)], stop)
           for i, c in enumerate(cams)]
fusion = FusionThread(cams, grid, stop)
for w in workers:
    w.start()
fusion.start()

try:
    while any(w.is_alive() for w in workers):
        time.sleep(2.0)
        status = ' | '.join(f'cam{w.cam["idx"]}: {w.fps:.1f} FPS, {w.people} det(s)' for w in workers)
        if args.profile:
            status += '  ' + ' '.join(
                f'[c{w.cam["idx"]} depth {w.t_depth:.0f} / det {w.t_det:.0f} / rest {w.t_rest:.0f} ms]'
                for w in workers)
        print(f'{status} || fused: {fusion.n_people} person(s), '
              f'cloud {fusion.n_cloud / 1000:.0f}k + map {fusion.n_map / 1000:.0f}k pts')
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    for w in workers:
        w.join(timeout=2.0)
    fusion.join(timeout=2.0)
    for c in cams:
        c['grabber'].alive = False
    for c in cams:
        c['grabber'].join(timeout=1.0)
        c['cap'].release()
