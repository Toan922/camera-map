"""Live 3D viewer, threaded edition: one full pipeline worker per camera.

    python view3d_ultra.py [--cameras 0] [--stride 8]
    python view3d_ultra.py --cameras 0,1    # multi-camera: one shared world

Same scene as view3d.py, but every camera runs capture -> depth -> detection ->
logging in its own worker thread, so one camera's GPU inference overlaps another's
CPU work and cameras are fully decoupled — each runs at its own rate. Prefer this
on a machine with a discrete GPU; view3d.py is the simpler sequential fallback.

Each camera needs calibration/camera_<idx>.json from calibrate.py (falls back to a
guessed 70-degree FOV with a warning). For multi-camera, each json also needs an
"extrinsics" entry from extrinsics.py — that is what places every camera in the same
world frame (the checkerboard's frame, Z up) so their point clouds merge into one
scene. Without extrinsics a camera sits at the world origin.

Frames are logged on a per-camera 'frame' sequence and a shared wall-clock 'time'
timeline (use the latter to correlate cameras; it's what fusion will match on).

Opens the rerun viewer: merged point cloud, one frustum + live image per camera,
and a 3D box per detected person with distance label. Ctrl+C in the terminal to stop.
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

DEVICE = os.environ.get('DEVICE') or (  # auto: NVIDIA > Apple GPU > CPU
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu')
INPUT_SIZE = 384
MAX_DEPTH = 20.0
DEPTH_SCALE = 0.71  # tape-measure correction, see test.py — per-camera override via
                    # a "depth_scale" key in calibration/camera_<idx>.json
FADE_COLOR = np.array([168.0, 85.0, 247.0])  # point cloud fades toward this (purple)
FADE_STRENGTH = 0.85  # how tinted the farthest point gets (1.0 = solid purple, 0 = off)
BOX_COLORS = [(80, 200, 255), (255, 170, 80), (170, 255, 120), (255, 120, 200)]  # per camera
DISPLAY_SCALE = 0.5  # camera images shown in the viewer at this scale (geometry unaffected)

parser = argparse.ArgumentParser()
parser.add_argument('--cameras', default='0', help='comma-separated camera indices, e.g. 0,1')
parser.add_argument('--stride', type=int, default=8, help='point cloud subsampling (pixels)')
parser.add_argument('--save', default=None, help='record to .rrd file instead of opening the viewer')
parser.add_argument('--max-frames', type=int, default=0, help='stop after N frames per camera (0 = run forever)')
args = parser.parse_args()
cam_indices = [int(x) for x in args.cameras.split(',')]

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
    preprocess costs ~19 ms per 1080p frame, this path ~2 ms. CUDA only; other
    devices use the stock infer_image."""
    if DEVICE != 'cuda':
        return depth_model.infer_image(frame_bgr, input_size=INPUT_SIZE)
    x = torch.from_numpy(frame_bgr).to(DEVICE)
    x = x.flip(-1).permute(2, 0, 1)[None].float().div_(255)  # BGR uint8 -> RGB 0..1
    x = TF.interpolate(x, size=model_size_hw, mode='bicubic', align_corners=False)
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
    """Full per-camera pipeline: newest frame -> depth -> detection -> rerun logs.
    One per camera; while this one's inference is on the GPU (GIL released), the
    other workers run their CPU stages."""

    def __init__(self, cam, yolo, color, stop):
        super().__init__(daemon=True)
        self.cam, self.yolo, self.color, self.stop = cam, yolo, color, stop
        self.fps, self.frames, self.people = 0.0, 0, 0

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

            depth = infer_depth(frame, c['model_size_hw']) * c['depth_scale']  # HxW meters
            det = self.yolo(frame, classes=[0], device=DEVICE, verbose=False)[0]  # class 0 = person
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # rerun's current time is thread-local: per-camera frame sequence plus a
            # shared wall clock so entities from different cameras can be correlated
            rr.set_time('frame', sequence=self.frames)
            rr.set_time('time', timestamp=time.time())

            # point cloud in camera coords — the Transform3D on world/cam<idx> places it in the world
            s = args.stride
            z = depth[::s, ::s]
            pts = np.stack([c['xn'] * z, c['yn'] * z, z], axis=-1).reshape(-1, 3)
            # distance fade: nearest point true color -> farthest most purple, rescaled
            # per frame (percentiles, not min/max, so a few outlier pixels don't flicker it)
            z_lo, z_hi = np.percentile(z, [2, 98])
            fade = np.clip((z - z_lo) / max(z_hi - z_lo, 1e-6), 0, 1)[..., None] * FADE_STRENGTH
            colors = (rgb[::s, ::s] * (1 - fade) + FADE_COLOR * fade).astype(np.uint8)
            rr.log(f'{base}/points', rr.Points3D(pts, colors=colors.reshape(-1, 3), radii=0.01))

            # person boxes: median depth over the torso region of each bbox -> 3D box
            centers, half_sizes, labels = [], [], []
            for box in det.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                bw, bh = x2 - x1, y2 - y1
                torso = depth[y1 + bh // 4: y2 - bh // 4, x1 + bw // 4: x2 - bw // 4]
                if torso.size == 0:
                    continue
                z_p = float(np.median(torso))
                u_c, v_c = (x1 + x2) / 2, (y1 + y2) / 2
                centers.append([(u_c - c['cx']) / c['fx'] * z_p, (v_c - c['cy']) / c['fy'] * z_p, z_p])
                half_sizes.append([bw / c['fx'] * z_p / 2, bh / c['fy'] * z_p / 2, 0.25])
                labels.append(f'cam{c["idx"]} {z_p:.2f}m')
            rr.log(f'{base}/people', rr.Boxes3D(centers=centers, half_sizes=half_sizes, labels=labels,
                                                colors=[self.color]))
            self.people = len(centers)

            # camera image, downscaled + jpeg-compressed so the stream stays light
            small = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            rr.log(f'{base}/image',
                   rr.EncodedImage(contents=cv2.imencode('.jpg', small)[1].tobytes(),
                                   media_type='image/jpeg'))

            now = time.time()
            if t_prev is not None:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            self.frames += 1


# --- cameras: capture + intrinsics + extrinsics per index ---
cams = []
for idx in cam_indices:
    cap = cv2.VideoCapture(idx)
    # Windows opens webcams at 640x480 by default; request full res to match calibration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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

# --- workers: one full pipeline per camera ---
stop = threading.Event()
# YOLO predictors are not thread-safe -> one instance per worker (created here,
# sequentially, so the first-run auto-download can't race)
workers = [Worker(c, YOLO('yolo11n.pt'), BOX_COLORS[i % len(BOX_COLORS)], stop)
           for i, c in enumerate(cams)]
for w in workers:
    w.start()

try:
    while any(w.is_alive() for w in workers):
        time.sleep(2.0)
        print(' | '.join(f'cam{w.cam["idx"]}: {w.fps:.1f} FPS, {w.people} person(s)' for w in workers))
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    for w in workers:
        w.join(timeout=2.0)
    for c in cams:
        c['grabber'].alive = False
    for c in cams:
        c['grabber'].join(timeout=1.0)
        c['cap'].release()
