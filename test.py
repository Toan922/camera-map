"""Metric depth testbed — webcam + Depth Anything V2 (Hypersim indoor, meters).

Controls:
  click      move the depth probe to that pixel
  r          toggle color scale: fixed 0-5m  <->  autoscale per frame
  q / ESC    quit
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metric_depth'))

import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2  # metric variant (sigmoid * max_depth)

DEVICE = 'mps'          # MacBook GPU (Metal)
INPUT_SIZE = 384        # model input; smaller = faster, 518 = finer detail
MAX_DEPTH = 20.0        # Hypersim indoor model range
DEPTH_SCALE = 0.71      # tape-measure correction: model over-reads ~1.4-1.6x on this camera
                        # (fit 2026-07: actual 1/2/3m read 1.58/3.0/4.0m). Set 1.0 to disable.
VIS_RANGE = 5.0         # fixed color scale in meters (indoor-friendly contrast)
DISPLAY_W = 1600        # total width of the side-by-side window

model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384],
                        max_depth=MAX_DEPTH)
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_metric_hypersim_vits.pth',
                                 map_location='cpu'))
model = model.to(DEVICE).eval()

cap = cv2.VideoCapture(0)
probe = None            # (x, y) in full-frame coords; None = frame center
fixed_scale = True
fps = 0.0

def on_mouse(event, x, y, flags, param):
    """Map a click in the display window back to full-frame coords."""
    global probe
    if event == cv2.EVENT_LBUTTONDOWN:
        half_w = DISPLAY_W // 2
        scale = param['frame_w'] / half_w
        probe = (int((x % half_w) * scale), int(y * scale))

cv2.namedWindow('Webcam | Metric Depth')
mouse_param = {'frame_w': 1920}
cv2.setMouseCallback('Webcam | Metric Depth', on_mouse, mouse_param)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    mouse_param['frame_w'] = w

    t0 = time.time()
    depth = model.infer_image(frame, input_size=INPUT_SIZE) * DEPTH_SCALE  # HxW meters
    fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t0, 1e-6))

    # Colorize: fixed scale keeps colors stable frame-to-frame (better for metric sanity)
    if fixed_scale:
        depth_vis = np.clip(depth / VIS_RANGE, 0, 1) * 255.0
    else:
        depth_vis = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6) * 255.0
    depth_colored = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_INFERNO)

    # Probe readout (median of a small patch — single pixels are noisy)
    px, py = probe if probe else (w // 2, h // 2)
    patch = depth[max(py - 5, 0):py + 6, max(px - 5, 0):px + 6]
    d = float(np.median(patch))
    for img in (frame, depth_colored):
        cv2.drawMarker(img, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(img, f'{d:.2f} m', (px + 20, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(img, f'{d:.2f} m', (px + 20, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    scale_label = f'scale: 0-{VIS_RANGE:.0f}m fixed' if fixed_scale else 'scale: auto'
    hud = f'{fps:.1f} FPS | {scale_label} | click=probe  r=scale  q=quit'
    cv2.putText(frame, hud, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(frame, hud, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    combined = np.hstack([frame, depth_colored])
    disp_h = int(combined.shape[0] * DISPLAY_W / combined.shape[1])
    cv2.imshow('Webcam | Metric Depth', cv2.resize(combined, (DISPLAY_W, disp_h)))

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('r'):
        fixed_scale = not fixed_scale

cap.release()
cv2.destroyAllWindows()
