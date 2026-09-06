# camera-map

Turn a couple of ordinary webcams into one persistent, metric 3D reconstruction of a
room — Tesla-Autopilot-style occupancy, but built from monocular depth estimation
instead of a learned bird's-eye network — with every person in it tracked as a single
fused entity that keeps its identity as it moves.

<!-- TODO: assets/hero.png — rerun viewer, fused room + tracked people with IDs -->
![Fused room with tracked people](assets/hero.png)

## What it does

- Estimates **metric depth** (real meters, not relative) from a single RGB webcam
  frame, using Depth Anything V2 fine-tuned for metric output.
- Detects people with YOLO and lifts their 2D boxes into 3D using the depth map and
  the camera's calibrated intrinsics.
- Calibrates each camera's intrinsics (checkerboard) and extrinsics (same
  checkerboard, un-moved) so multiple cameras agree on one shared world frame.
- Fuses every camera's depth over time into one persistent TSDF voxel map of the
  static room — noise in any single frame averages out, and cameras fill each
  other's blind spots.
- Clusters and tracks people across all cameras in 3D: one box per human, a
  Kalman-filter velocity estimate, and a persistent ID that survives brief
  detection dropouts.

Full explanation of how the pieces fit together: [`EXPLANATION.md`](EXPLANATION.md).
Full setup and per-script usage: [`SETUP.md`](SETUP.md).

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Metric depth on a single frame (`test.py`) | done |
| 1 | One camera → 3D point cloud + person boxes (`calibrate.py`, `view3d.py`) | done |
| 2 | Humans lifted into 3D | done (boxes; skeletons later) |
| 3 | Multiple cameras, one shared world (`extrinsics.py`) | done |
| 4 | Fused room map + cross-camera person tracking (`view3d_fusion.py`, `fusion.py`) | done |
| 5 | Realtime engineering — shared model round-robin, capture threads, bandwidth | in progress |

See the [roadmap in `EXPLANATION.md`](EXPLANATION.md#the-full-roadmap) for details,
including what's known-rough (e.g. per-camera depth-scale calibration isn't done yet).

## Screenshots

<!-- TODO: drop images into assets/ with these filenames and they'll render here -->

| | |
|---|---|
| ![Single camera point cloud](assets/screenshot-single-camera.png) | ![Two cameras merged into one world](assets/screenshot-multi-camera.png) |
| Single camera — depth point cloud + person box | Two cameras, one shared world frame |
| ![Fused room map](assets/screenshot-fused-room.png) | ![Tracked people across cameras](assets/screenshot-tracking.png) |
| Persistent TSDF room map after ~20s | Tracked people, persistent IDs + velocity arrows |

## Requirements

- Python 3.10+ (3.13 tested)
- A CUDA GPU (Windows/Linux) or Apple Silicon (MPS) — CPU works but is slow
- One or more webcams
- ~100 MB for the metric depth checkpoint, ~5 MB for YOLO (auto-downloads on first run)

## Quickstart

```bash
git clone <this-repo-url>
cd camera-map

# macOS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Windows (conda)
conda create -n torch_env python=3.13 && conda activate torch_env
pip install -r requirements_windows.txt

# Download the metric depth checkpoint (~99 MB)
curl -L --create-dirs -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"

python view3d_fusion.py --cameras 0,1
```

Calibration (intrinsics + extrinsics) has to happen once per camera before multi-camera
output is trustworthy — see the [suggested first-run order in `SETUP.md`](SETUP.md#suggested-first-run-order).

## The scripts

| Script | Purpose |
|---|---|
| `test.py` | 2D depth testbed — probe a pixel, read meters |
| `calibrate.py` | Per-camera intrinsics (checkerboard) |
| `extrinsics.py` | Per-camera pose in the shared world (same checkerboard, un-moved) |
| `view3d.py` | Live 3D scene, single or multi-camera, sequential |
| `view3d_ultra.py` | Same scene, threaded per-camera for higher FPS |
| `view3d_fusion.py` | **Main script** — fused room map + cross-camera person tracking |
| `fusion.py` | TSDF fusion + SORT-style 3D tracking (importable; also self-tests standalone) |
| `app.py` | Gradio demo for the underlying depth model (not part of the tracking pipeline) |

## How it works (short version)

```
webcam frame → Depth Anything V2 → metric depth map (meters/pixel)
            └→ YOLO → person boxes (2D)

depth map + camera intrinsics → back-projection → 3D point cloud / 3D person boxes

per-camera depth (people masked out) → TSDF fusion → persistent room voxel map
per-camera person boxes → world-space clustering → Kalman tracking → one ID per person
```

Everything that turns 2D into 3D is plain geometry (back-projection through calibrated
intrinsics) — the two neural nets only ever produce a depth map and a person box each.
Diagrams and the reasoning behind each design choice: [`EXPLANATION.md`](EXPLANATION.md).

## Known limitations

- Monocular depth isn't a fixed scale — two cameras can disagree by 10–30 cm on the
  same wall. TSDF averaging softens this; per-camera `depth_scale` calibration is the
  real fix and isn't done for every camera yet.
- No re-identification yet — someone who leaves the room and comes back gets a new ID.
- Phase 5 (one shared depth model round-robining several cameras) isn't built — each
  camera currently runs its own inference.

## Acknowledgements

Built on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
(metric variant, fine-tuned on Hypersim) for depth estimation,
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for person detection,
and [rerun](https://rerun.io) for 3D visualization. See [`DA-2K.md`](DA-2K.md) for the
original depth benchmark and the
[Depth Anything V2 paper](https://arxiv.org/abs/2406.09414) for the depth model itself.

## License

Apache 2.0 (see [`LICENSE`](LICENSE)), inherited from the upstream Depth Anything V2
repository this project builds on. The Depth-Anything-V2-Base/Large/Giant checkpoints
are CC-BY-NC-4.0; the Small checkpoint used here is Apache-2.0.
