"""Camera extrinsics: where each camera sits in the shared world frame (Phase 3).

Lay the printed checkerboard flat (floor / table) where the camera can see it — the
board defines the world origin. With the board and camera both completely static:

    python extrinsics.py [--camera 0] [--square-size 0.020]

SPACE captures a sample when corners are detected (green) — each sample is solved
individually and must fit to < 2 px. Take ~10, then C averages the poses and saves
into calibration/camera_<idx>.json under "extrinsics". Q quits.

Repeat for each camera WITHOUT moving the board: every camera measured against the
same board placement lands in the same world frame, which is what lets view3d.py
merge their point clouds into one scene.

The checkerboard can be detected 180-deg rotated; orientation is anchored to the
printed pattern (a flip inverts the square colors), so all cameras — even facing
each other across the board — agree on one world.

World frame: origin at the board's first inner corner, X along the long side,
Y along the short side, Z pointing up out of the printed face.

Requires intrinsics first (calibrate.py).
"""
import argparse
import json
import time

import cv2
import numpy as np

CHESSBOARD = (9, 6)  # inner corners (cols, rows), must match calibrate.py
COLS = CHESSBOARD[0]  # corner index k = row * COLS + col
F = np.diag([1.0, -1.0, -1.0])  # board frame -> world frame (Z up out of the face)


def canonicalize(corners, gray):
    """Resolve the 180-deg detection ambiguity from the printed pattern: a flipped
    detection lands these probes on opposite-colored squares, so 'darker square at
    the origin corner' is a physical convention independent of camera viewpoint."""
    def patch_mean(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return float(gray[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].mean())
    pts = corners.reshape(-1, 2)  # corner array shape differs across OpenCV versions
    a = patch_mean((pts[0] + pts[COLS + 1]) / 2)      # square inside origin corner
    b = patch_mean((pts[-1] + pts[-(COLS + 2)]) / 2)  # square inside far corner
    return corners if a < b else np.ascontiguousarray(corners[::-1])


def camera_pose(rvec, tvec):
    """solvePnP gives board->camera; invert for camera->board, flip into world."""
    R_cb, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))
    R = F @ R_cb.T
    t = (-F @ R_cb.T @ np.asarray(tvec).reshape(3, 1)).ravel()
    return R, t


parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--square-size', type=float, default=0.020, help='square edge in meters (measure your print!)')
args = parser.parse_args()
calib_path = f'calibration/camera_{args.camera}.json'

try:
    with open(calib_path) as f:
        calib = json.load(f)
except FileNotFoundError:
    raise SystemExit(f'{calib_path} not found — run calibrate.py for this camera first')

# 3D corner positions on the board plane (z=0), scaled to real size
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * args.square_size

cap = cv2.VideoCapture(args.camera)
# Windows opens webcams at 640x480 by default; request full res to match calibration
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = cap.read()
assert ret, 'no webcam frame'
H, W = frame.shape[:2]

sx, sy = W / calib['resolution'][0], H / calib['resolution'][1]  # rescale if resolution differs
K = np.array([[calib['fx'] * sx, 0, calib['cx'] * sx],
              [0, calib['fy'] * sy, calib['cy'] * sy],
              [0, 0, 1]])
dist = np.array(calib['dist'])

samples = []  # (rvec, tvec, rms_px) per capture — solved individually, averaged at the end
flash_until = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(gray, None, fx=0.5, fy=0.5)
    found, corners_small = cv2.findChessboardCorners(
        small, CHESSBOARD, None,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)

    if found:
        cv2.drawChessboardCorners(frame, CHESSBOARD, corners_small * 2, found)
        # live pose preview: axes drawn on the board (X red, Y green, Z blue)
        ok, rvec, tvec = cv2.solvePnP(objp, corners_small * 2, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 3 * args.square_size)

    status = f'{len(samples)} samples | SPACE=capture  C=solve+save  Q=quit'
    color = (0, 255, 0) if found else (0, 0, 255)
    cv2.putText(frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if time.time() < flash_until:
        cv2.rectangle(frame, (0, 0), (W, H), (255, 255, 255), 30)

    cv2.imshow('Extrinsics', cv2.resize(frame, None, fx=0.6, fy=0.6))
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and found:
        corners = cv2.cornerSubPix(gray, corners_small * 2, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners = canonicalize(corners, gray)
        rms = np.inf
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(objp, corners, K, dist, rvec, tvec)
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            err = proj.reshape(-1, 2) - corners.reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        if not ok or rms > 2.0:
            print(f'sample rejected (RMS {rms:.2f} px) — board not flat, or intrinsics off?')
        else:
            samples.append((rvec.ravel(), tvec.ravel(), rms))
            flash_until = time.time() + 0.15
            print(f'captured {len(samples)} (RMS {rms:.2f} px)')
    elif key == ord('c') and len(samples) >= 3:
        # each sample independently claims a camera position — their spread is a
        # direct measurement of stability (motion or a slipped flip shows up huge)
        positions = np.stack([camera_pose(rv, tv)[1] for rv, tv, _ in samples])
        spread = float(np.linalg.norm(positions - positions.mean(axis=0), axis=1).max())
        if spread > 0.02:
            print(f'camera position spread {spread * 1000:.0f} mm across samples — '
                  f'something moved. Samples cleared, recapture.')
            samples.clear()
            continue

        # samples agree to <2 cm, so a plain average of the pose vectors is safe
        rvec = np.mean([rv for rv, _, _ in samples], axis=0)
        tvec = np.mean([tv for _, tv, _ in samples], axis=0)
        rms = float(np.mean([r for _, _, r in samples]))
        R, t = camera_pose(rvec, tvec)

        calib['extrinsics'] = {
            'R': R.tolist(),  # camera->world rotation
            't': t.tolist(),  # camera position in world (meters)
            'rms_reprojection_error_px': rms,
            'position_spread_m': spread,
            'num_views': len(samples),
            'square_size_m': args.square_size,
        }
        with open(calib_path, 'w') as f:
            json.dump(calib, f, indent=2)
        print(f'mean per-sample RMS: {rms:.3f} px | position spread: {spread * 1000:.1f} mm')
        print(f'camera at world ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m — '
              f'{t[2]:.2f} m above the board plane, {np.linalg.norm(t):.2f} m from the origin')
        print(f'saved -> {calib_path}')
        break
    elif key == ord('c'):
        print(f'need at least 3 samples (have {len(samples)})')
    elif key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
