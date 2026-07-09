# Setup — Metric Depth + 3D Camera Pipeline (macOS)

Real-time metric depth (in **meters**) and a live 3D scene — point cloud, camera pose,
person boxes — from the MacBook webcam. Runs on Apple Silicon GPU via MPS.

**Read [`EXPLANATION.md`](EXPLANATION.md) first** for what each piece does and how the
data flows.

## Requirements

- macOS on Apple Silicon (uses the `mps` torch backend)
- Python 3.13+ (older 3.10+ likely fine)
- A webcam (built-in works)

## Install

```bash
# 1. Clone and enter the repo
git clone https://github.com/Toan922/camera-map.git
cd camera-map

# 2. Create a virtualenv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Download the metric depth checkpoint (~99 MB)
curl -L --create-dirs -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
```

(The YOLO person-detection model, ~5 MB, auto-downloads on first run of `view3d.py`.)

First run of any script: macOS asks for camera permission for your terminal.

## The scripts

### 1. `test.py` — depth testbed (2D)

```bash
python test.py
```

Side-by-side webcam + depth map (~14 FPS). Crosshair probe reads meters.

| Key / action | Effect |
|---|---|
| **click** | Move the depth probe (reads median depth of an 11×11 patch) |
| **r** | Toggle color scale: fixed 0–5 m ↔ per-frame autoscale |
| **q / ESC** | Quit |

**Sanity check:** place an object at a measured 1 m / 2 m / 3 m, click it, compare.
A consistent *linear* bias is expected (webcam FOV differs from the model's training
cameras) and gets corrected via calibration.

### 2. `calibrate.py` — camera intrinsics (one-time per camera)

Print `calibration/checkerboard_9x6_20mm.png` at **100% scale** (no "fit to page"),
verify one square = 20 mm with a ruler, tape it to something rigid. Then:

```bash
python calibrate.py                    # --camera 1, --square-size 0.020 if needed
```

Show the board to the camera; corners turn green when detected → **SPACE** to capture.
Take 15–20 shots varying distance, position in frame, and especially **tilt**.
Press **C** to solve and save `calibration/camera_0.json`.

Quality gate: RMS reprojection error **< 0.5 px** is good; > 1 px → recapture.

### 3. `view3d.py` — live 3D scene (rerun viewer)

```bash
python view3d.py                       # Ctrl+C to stop
```

Opens the [rerun](https://rerun.io) viewer: live point cloud of the room, camera
frustum with the video feed, and a 3D box per detected person labeled with distance.

- Uses `calibration/camera_0.json` automatically (falls back to a guessed 70° FOV
  with a warning — works, but calibrate for real geometry)
- `--stride 8` — point cloud density (lower = denser = slower)
- `--save out.rrd --max-frames 300` — record headless, open later with `rerun out.rrd`

## Suggested first-run order

1. `python view3d.py` — see the room in 3D immediately (guessed FOV)
2. Print the checkerboard → `python calibrate.py`
3. `python view3d.py` again — geometry straightens out with real intrinsics
4. `python test.py` — tape-measure the depth accuracy at known distances

## Tuning

Constants at the top of `test.py` / `view3d.py`:

- `INPUT_SIZE` — 384 (fast) → 518 (finer detail, ~half the FPS)
- `MAX_DEPTH` — 20 m for the Hypersim indoor model; 80 for a VKITTI (outdoor) checkpoint
- `DEVICE` — `mps`; change to `cuda` / `cpu` off-Mac
- `VIS_RANGE` (test.py) — fixed color scale range in meters

## Notes

- Scripts import the **metric** model variant from `metric_depth/depth_anything_v2/`
  (sigmoid head × `max_depth`, outputs meters) — not the root package, which is the
  relative-depth model (arbitrary scale).
- Larger checkpoints (Base/Large, indoor + outdoor) are listed in
  [`metric_depth/README.md`](metric_depth/README.md) — same code, better accuracy, slower.
