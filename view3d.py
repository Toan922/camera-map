"""Live 3D viewer: webcam -> metric depth -> point cloud + person boxes, in rerun.

    python view3d.py [--camera 0] [--calib calibration/camera_0.json] [--stride 8]

Needs calibration/camera_<idx>.json from calibrate.py. Without it, falls back to a
guessed 70-degree FOV (geometry will be roughly right, distances less trustworthy).

Opens the rerun viewer: point cloud of the scene in camera coordinates, the camera
frustum + live image, and a 3D box per detected person with distance label.
Ctrl+C in the terminal to stop.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metric_depth'))

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from ultralytics import YOLO

from depth_anything_v2.dpt import DepthAnythingV2  # metric variant

DEVICE = os.environ.get('DEVICE') or (  # auto: NVIDIA > Apple GPU > CPU
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu')
INPUT_SIZE = 384
MAX_DEPTH = 20.0
DEPTH_SCALE = 0.71  # tape-measure correction, see test.py (set 1.0 to disable)
FADE_COLOR = np.array([168.0, 85.0, 247.0])  # point cloud fades toward this (purple)
FADE_STRENGTH = 0.85  # how tinted the farthest point gets (1.0 = solid purple, 0 = off)

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--calib', default=None, help='intrinsics json (default calibration/camera_<idx>.json)')
parser.add_argument('--stride', type=int, default=8, help='point cloud subsampling (pixels)')
parser.add_argument('--save', default=None, help='record to .rrd file instead of opening the viewer')
parser.add_argument('--max-frames', type=int, default=0, help='stop after N frames (0 = run forever)')
args = parser.parse_args()

# --- intrinsics ---
calib_path = args.calib or f'calibration/camera_{args.camera}.json'
calib = None
if os.path.exists(calib_path):
    with open(calib_path) as f:
        calib = json.load(f)
    print(f'loaded intrinsics from {calib_path} (rms {calib["rms_reprojection_error_px"]:.3f} px)')
else:
    print(f'WARNING: {calib_path} not found — run calibrate.py. Falling back to guessed 70 deg hFOV.')

# --- models ---
depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384],
                              max_depth=MAX_DEPTH)
depth_model.load_state_dict(torch.load('checkpoints/depth_anything_v2_metric_hypersim_vits.pth',
                                       map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()
print(f'device: {DEVICE}')
yolo = YOLO('yolo11n.pt')  # auto-downloads on first run

# --- camera ---
cap = cv2.VideoCapture(args.camera)
ret, frame = cap.read()
assert ret, 'no webcam frame'
H, W = frame.shape[:2]

if calib:
    sx, sy = W / calib['resolution'][0], H / calib['resolution'][1]  # rescale if resolution differs
    fx, fy, cx, cy = calib['fx'] * sx, calib['fy'] * sy, calib['cx'] * sx, calib['cy'] * sy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array(calib['dist'])
    undistort_maps = cv2.initUndistortRectifyMap(K, dist, None, K, (W, H), cv2.CV_32FC1)
else:
    fx = fy = W / (2 * np.tan(np.radians(70 / 2)))
    cx, cy = W / 2, H / 2
    undistort_maps = None

# precomputed pixel grid for back-projection (at point cloud stride)
s = args.stride
us, vs = np.meshgrid(np.arange(0, W, s), np.arange(0, H, s))
xn = (us - cx) / fx  # normalized ray directions
yn = (vs - cy) / fy

# --- rerun ---
rr.init('camera-map', spawn=args.save is None)
if args.save:
    rr.save(args.save)
# explicit layout: 3D scene + live camera view (also overrides any stale saved layout)
rr.send_blueprint(rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='world', name='3D scene'),
        rrb.Spatial2DView(origin='world/camera/image', name='camera'),
        column_shares=[2, 1],
    ),
))
rr.log('world', rr.ViewCoordinates.RDF, static=True)  # X right, Y down, Z forward (camera frame)
rr.log('world/camera/image', rr.Pinhole(resolution=[W, H], focal_length=[fx, fy],
                                        principal_point=[cx, cy], image_plane_distance=0.3), static=True)

frame_i = 0
fps = 0.0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if undistort_maps is not None:
            frame = cv2.remap(frame, *undistort_maps, cv2.INTER_LINEAR)

        t0 = time.time()
        depth = depth_model.infer_image(frame, input_size=INPUT_SIZE) * DEPTH_SCALE  # HxW meters
        detections = yolo(frame, classes=[0], device=DEVICE, verbose=False)[0]  # class 0 = person

        rr.set_time('frame', sequence=frame_i)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # point cloud: back-project subsampled depth through the pinhole model
        z = depth[::s, ::s]
        pts = np.stack([xn * z, yn * z, z], axis=-1).reshape(-1, 3)
        # distance fade: nearest point true color -> farthest most purple, rescaled
        # per frame (percentiles, not min/max, so a few outlier pixels don't flicker it)
        z_lo, z_hi = np.percentile(z, [2, 98])
        fade = np.clip((z - z_lo) / max(z_hi - z_lo, 1e-6), 0, 1)[..., None] * FADE_STRENGTH
        colors = (rgb[::s, ::s] * (1 - fade) + FADE_COLOR * fade).astype(np.uint8)
        rr.log('world/points', rr.Points3D(pts, colors=colors.reshape(-1, 3), radii=0.01))

        # person boxes: median depth over the torso region of each bbox -> 3D box
        centers, half_sizes, labels = [], [], []
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            bw, bh = x2 - x1, y2 - y1
            torso = depth[y1 + bh // 4: y2 - bh // 4, x1 + bw // 4: x2 - bw // 4]
            if torso.size == 0:
                continue
            z_p = float(np.median(torso))
            u_c, v_c = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append([(u_c - cx) / fx * z_p, (v_c - cy) / fy * z_p, z_p])
            half_sizes.append([bw / fx * z_p / 2, bh / fy * z_p / 2, 0.25])
            labels.append(f'person {z_p:.2f}m')
        rr.log('world/people', rr.Boxes3D(centers=centers, half_sizes=half_sizes, labels=labels,
                                          colors=[(80, 200, 255)]))

        # camera image (jpeg-compressed so the stream stays light)
        rr.log('world/camera/image',
               rr.EncodedImage(contents=cv2.imencode('.jpg', frame)[1].tobytes(),
                               media_type='image/jpeg'))

        fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t0, 1e-6))
        frame_i += 1
        if frame_i % 30 == 0:
            print(f'{fps:.1f} FPS | {len(centers)} person(s)')
        if args.max_frames and frame_i >= args.max_frames:
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
