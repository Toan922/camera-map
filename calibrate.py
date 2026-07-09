"""Camera intrinsics calibration from a printed checkerboard.

Print calibration/checkerboard_9x6_20mm.png at 100% scale (verify: one square = 20mm),
tape it to something flat, then:

    python calibrate.py [--camera 0] [--square-size 0.020] [--out calibration/camera_0.json]

Show the board to the camera and press SPACE each time corners are detected (green).
Vary it: near/far, left/right/top/bottom of frame, tilted toward/away, rotated.
15-20 captures is good. Press C to calibrate + save, Q to quit.
"""
import argparse
import json
import time

import cv2
import numpy as np

CHESSBOARD = (9, 6)  # inner corners

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--square-size', type=float, default=0.020, help='square edge in meters (measure your print!)')
parser.add_argument('--out', default=None, help='output json (default calibration/camera_<idx>.json)')
args = parser.parse_args()
out_path = args.out or f'calibration/camera_{args.camera}.json'

# 3D corner positions on the board plane (z=0), scaled to real size
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * args.square_size

cap = cv2.VideoCapture(args.camera)
obj_points, img_points = [], []
frame_size = None
flash_until = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_size = (frame.shape[1], frame.shape[0])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect on a half-res image for speed, refine on full res at capture time
    small = cv2.resize(gray, None, fx=0.5, fy=0.5)
    found, corners_small = cv2.findChessboardCorners(
        small, CHESSBOARD, None,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)

    if found:
        cv2.drawChessboardCorners(frame, CHESSBOARD, corners_small * 2, found)

    n = len(img_points)
    status = f'{n} captures | SPACE=capture  C=calibrate+save  Q=quit'
    color = (0, 255, 0) if found else (0, 0, 255)
    cv2.putText(frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if time.time() < flash_until:
        cv2.rectangle(frame, (0, 0), frame_size, (255, 255, 255), 30)

    cv2.imshow('Calibrate', cv2.resize(frame, None, fx=0.6, fy=0.6))
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and found:
        # refine corner positions to sub-pixel accuracy on the full-res image
        corners = cv2.cornerSubPix(gray, corners_small * 2, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)
        flash_until = time.time() + 0.15
        print(f'captured {len(img_points)}')
    elif key == ord('c') and len(img_points) >= 8:
        print(f'calibrating on {len(img_points)} views...')
        rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, frame_size, None, None)
        data = {
            'camera_index': args.camera,
            'resolution': list(frame_size),
            'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2],
            'dist': dist.ravel().tolist(),
            'rms_reprojection_error_px': rms,
            'square_size_m': args.square_size,
            'num_views': len(img_points),
        }
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        hfov = 2 * np.degrees(np.arctan(frame_size[0] / (2 * K[0, 0])))
        print(f'RMS reprojection error: {rms:.3f} px  (under ~0.5 is good, over 1.0 = recapture)')
        print(f'fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}  hFOV={hfov:.1f} deg')
        print(f'saved -> {out_path}')
        break
    elif key == ord('c'):
        print(f'need at least 8 captures (have {len(img_points)})')
    elif key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
