# Setup — Metric Depth Webcam Testbed (macOS)

Real-time metric depth estimation (in **meters**) from the MacBook webcam, using
Depth Anything V2 fine-tuned on Hypersim (indoor). Runs on Apple Silicon GPU via MPS.

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

## Run

```bash
python test.py
```

First run: macOS will ask for camera permission for your terminal.

### Controls

| Key / action | Effect |
|---|---|
| **click** | Move the depth probe to that pixel (reads meters at that point) |
| **r** | Toggle color scale: fixed 0–5 m ↔ per-frame autoscale |
| **q / ESC** | Quit |

### What you should see

Side-by-side webcam feed and depth map (~14 FPS on M-series at input size 384).
The crosshair probe shows median depth of an 11×11 patch in meters.

**Sanity check:** place an object at a measured distance (1 m / 2 m / 3 m), click it,
and compare the readout. Expect a consistent scale bias (the model's metric priors
assume a different camera FOV than a webcam) — a *linear* error is fine and gets
corrected during camera calibration.

## Tuning

Constants at the top of `test.py`:

- `INPUT_SIZE` — 384 (fast) → 518 (finer detail, ~half the FPS)
- `VIS_RANGE` — fixed color scale range in meters (default 5 m, indoor-friendly)
- `MAX_DEPTH` — 20 m for the Hypersim indoor model; use 80 with a VKITTI (outdoor) checkpoint
- `DEVICE` — `mps`; change to `cuda` / `cpu` off-Mac

## Notes

- `test.py` imports the **metric** model variant from `metric_depth/depth_anything_v2/`
  (sigmoid head × `max_depth`, outputs meters) — not the root package, which is the
  relative-depth model (arbitrary scale).
- Larger checkpoints (Base/Large, indoor + outdoor) are listed in
  [`metric_depth/README.md`](metric_depth/README.md) — same code, better accuracy, slower.
