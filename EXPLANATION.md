# How this all works

Goal: track people in a shared 3D space built from multiple webcams. Depth estimation
is one ingredient — the rest is camera geometry, detection, and (later) tracking.

## The pieces, and who does what

| Piece | What it is | What it does | What it does NOT do |
|---|---|---|---|
| **Depth Anything V2** (metric) | Neural net | Turns one RGB frame into a 2D depth map — meters per pixel | No people, no 3D, no point clouds |
| **YOLO** (yolo11n) | A second, unrelated neural net | Finds people as 2D pixel rectangles in the frame | Knows nothing about depth |
| **Back-projection** | Plain math, no AI | Converts depth map → 3D point cloud, and 2D boxes → 3D boxes | — |
| **Camera calibration** | One-time measurement | Measures your camera's real geometry (focal length, center, distortion) so the math above is correct | — |
| **rerun** | Viewer only | Draws points/boxes/images in an interactive 3D window with a timeline | Computes nothing. Zero AI, zero math |

## Data flow per frame (`view3d.py`)

```
webcam frame (OpenCV) ──┬──▶ Depth Anything V2 ──▶ depth map: meters per pixel (2D!)
                        │
                        └──▶ YOLO ──▶ person boxes: 2D pixel rectangles
                                             │
depth map ──▶ back-projection ──▶ point cloud│
              X=(u-cx)·Z/fx                  │
              Y=(v-cy)·Z/fy                  │
                                             │
depth map + 2D box ──▶ same math ──▶ 3D person box (median depth of torso region)
                                             │
point cloud + boxes + camera image ──▶ rerun (just draws it)
```

Key idea: **everything 3D is math, not AI.** Each pixel is "pushed back" along its
camera ray by its predicted depth. That push uses the camera intrinsics
(`fx, fy, cx, cy`) — which is why calibration exists.

## Why the *metric* model matters

The stock Depth Anything V2 outputs **relative** depth: arbitrary scale and shift,
different every frame. You cannot build a stable 3D space from that.

The metric variant (`metric_depth/`) is the same network with a different last layer
(`sigmoid × max_depth` instead of ReLU) **fine-tuned on Hypersim**, a synthetic indoor
dataset with ground truth in meters. The architecture change is ~3 lines; the
knowledge lives in the fine-tuned weights. Caveat: its metric priors assume the
training cameras' field of view, so expect a consistent scale bias on a webcam —
measure it with the tape-measure test and correct it in software.

## Why calibration matters

Back-projection needs to know how pixels map to rays in the real world. That is 4
numbers + distortion, unique to each physical camera:

- `fx, fy` — focal length in pixels (how "zoomed" the camera is)
- `cx, cy` — where the optical axis actually hits the sensor (never exactly the center)
- distortion coefficients — lens bending, straight lines curving near the edges

`calibrate.py` measures these by looking at a printed checkerboard of known size from
many angles (OpenCV solves for the numbers that best explain all views). Without it,
`view3d.py` guesses a 70° FOV — shapes look roughly right, distances less trustworthy.

## The full roadmap

1. **Phase 0 — metric depth** ✅ `test.py` (probe a pixel, read meters)
2. **Phase 1 — one camera → 3D** ✅ `calibrate.py` + `view3d.py` (point cloud, camera pose, person boxes in rerun)
3. **Phase 2 — humans in 3D** ✅ partially: YOLO boxes lifted to 3D (keypoints/skeletons later)
4. **Phase 3 — multiple cameras, one world** — extrinsics: where each camera sits in a
   shared world frame (AprilTag/board visible to all cameras defines the origin), time
   sync, then all point clouds merge into one scene
5. **Phase 4 — fusion + tracking** — match the same person across cameras by 3D
   proximity, Kalman filter per person, Hungarian assignment per frame (SORT in 3D)
6. **Phase 5 — realtime engineering** — one shared depth model round-robining ~4
   cameras, capture thread per camera keeping only the latest frame, MJPEG/720p to
   survive USB bandwidth

## Current performance (M1 Pro, MPS)

- Depth alone @ input 384: ~14 FPS
- Depth + YOLO + point cloud + rerun logging: ~9.5 FPS (single camera)
