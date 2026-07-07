"""
depth_worker.py
---------------
Background depth-inference worker using Depth Anything 3 (DA3Mono-Large).

Loads the model once on first use (downloaded from HuggingFace ~1.3 GB).
Each camera gets its own DepthWorker instance, which runs inference on
a background thread and exposes the latest colorized depth map thread-safely.

Model:  depth-anything/DA3MONO-LARGE  (relative monocular depth)
Device: CUDA (RTX 3070) if available, else CPU

Inference API (from depth_anything_3.api):
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
    predictions = model([pil_image])          # list of PIL Images
    depth = predictions.depth[0]              # H×W float32 numpy array
"""

from __future__ import annotations

import threading
import time
import queue
from typing import Optional

import numpy as np

# ── Status constants ─────────────────────────────────────────────────────────
STATUS_IDLE        = "idle"
STATUS_LOADING     = "loading"
STATUS_READY       = "ready"
STATUS_ERROR       = "error"
STATUS_UNAVAILABLE = "unavailable"   # DA3 package not installed

# ── Colormap ─────────────────────────────────────────────────────────────────
# Turbo-inspired colormap: blue (far) → green → yellow → red (near)
_COLORMAP_LUT: Optional[np.ndarray] = None


def _build_lut() -> np.ndarray:
    """Build a 256×3 uint8 LUT mapping [0,255] → RGB colour (0=cold/far, 255=warm/near)."""
    t = np.linspace(0.0, 1.0, 256)

    # R channel: low at cold end, ramps up through warm
    r = np.clip(
        np.where(t < 0.5,
                 0.2 + t * 2.0 * (1.0 - 0.2),
                 np.ones_like(t)),
        0.0, 1.0)

    # G channel: peaks in the mid range (greenish teal → yellow)
    g = np.clip(
        np.where(t < 0.4,
                 t * 2.5,
                 np.where(t < 0.6,
                          1.0,
                          1.0 - (t - 0.6) * 2.5)),
        0.0, 1.0)

    # B channel: strong at cold/far end, fades to near zero at warm end
    b = np.clip(
        np.where(t < 0.3,
                 1.0,
                 1.0 - (t - 0.3) / 0.7),
        0.0, 1.0)

    lut = np.stack([r, g, b], axis=1)
    return (lut * 255).astype(np.uint8)


def _colorize_depth(depth_np: np.ndarray) -> np.ndarray:
    """
    Convert a float32 depth map (H×W) to a uint8 RGB image (H×W×3).
    Closer = warm (red), farther = cool (blue).
    """
    global _COLORMAP_LUT
    if _COLORMAP_LUT is None:
        _COLORMAP_LUT = _build_lut()

    d = depth_np.astype(np.float32)
    dmin, dmax = d.min(), d.max()
    if dmax > dmin:
        d = (d - dmin) / (dmax - dmin)   # [0, 1]
    else:
        d = np.zeros_like(d)

    # Invert: DA3 outputs larger values for *farther* objects.
    # We want near = warm (high LUT index), so we invert.
    d = 1.0 - d
    idx = (d * 255).clip(0, 255).astype(np.uint8)
    return _COLORMAP_LUT[idx]   # H×W×3 RGB


# ── Shared model loader (singleton) ─────────────────────────────────────────

_model_lock    = threading.Lock()
_model         = None
_model_status  = STATUS_IDLE
_model_error   = ""
_model_callbacks: list = []


def _load_model_thread():
    """Load DA3Mono-Large in a background thread (called once globally)."""
    global _model, _model_status, _model_error

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # DA3's image processor outputs float32 tensors, so we must keep model weights in float32
        # (autocast handles the mixed precision internally in forward())
        dtype = torch.float32

        print(f"[DepthWorker] Loading DA3Mono-Large on {device} (torch.float32)…")

        try:
            # Primary: use the installed depth_anything_3 package
            from depth_anything_3.api import DepthAnything3   # noqa: PLC0415
            model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
            model = model.to(device).to(dtype).eval()
            model_type = "da3"
        except (ImportError, Exception) as primary_err:
            print(f"[DepthWorker] DA3 primary load failed ({primary_err}), "
                  "trying HF depth-estimation pipeline…")
            # Fallback: HuggingFace transformers pipeline (Depth-Anything-V2-Large)
            from transformers import pipeline as hf_pipeline   # noqa: PLC0415
            hf_device = 0 if torch.cuda.is_available() else -1
            model = hf_pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf",
                device=hf_device,
            )
            model_type = "hf_pipeline"

        with _model_lock:
            _model = (model, model_type)
            _model_status = STATUS_READY
        print(f"[DepthWorker] Model ready (type={model_type}).")

    except Exception as exc:
        with _model_lock:
            _model_status = STATUS_ERROR
            _model_error  = str(exc)
        print(f"[DepthWorker] Model load failed: {exc}")

    _notify_callbacks()


def _notify_callbacks():
    with _model_lock:
        cbs = list(_model_callbacks)
    for cb in cbs:
        try:
            cb(_model_status)
        except Exception:
            pass


def get_model_status() -> str:
    with _model_lock:
        return _model_status


def get_model_error() -> str:
    with _model_lock:
        return _model_error


def register_status_callback(cb):
    """Register a function(status: str) to be called when model status changes."""
    with _model_lock:
        _model_callbacks.append(cb)


def ensure_model_loading():
    """Trigger model load if not already started. Safe to call multiple times."""
    global _model_status
    with _model_lock:
        if _model_status != STATUS_IDLE:
            return
        _model_status = STATUS_LOADING

    t = threading.Thread(target=_load_model_thread, daemon=True, name="DA3-Loader")
    t.start()


# ── DepthWorker ──────────────────────────────────────────────────────────────

class DepthWorker:
    """
    Per-camera depth inference worker.

    Usage:
        worker = DepthWorker(camera_index=0)
        worker.submit(bgr_frame)        # non-blocking
        depth_rgb = worker.get_depth()  # returns latest result or None
        worker.stop()
    """

    def __init__(self, camera_index: int, target_w: int = 320, target_h: int = 240):
        self.camera_index = camera_index
        self.target_w = target_w
        self.target_h = target_h

        self._in_queue:  queue.Queue = queue.Queue(maxsize=1)
        self._out_lock   = threading.Lock()
        self._latest_depth: Optional[np.ndarray] = None   # H×W×3 RGB uint8
        self._running    = True

        self._thread = threading.Thread(
            target=self._infer_loop,
            daemon=True,
            name=f"DA3-Cam{camera_index}",
        )
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(self, bgr_frame: np.ndarray) -> None:
        """Submit a BGR frame for depth inference. Drops the frame if busy."""
        try:
            self._in_queue.put_nowait(bgr_frame.copy())
        except queue.Full:
            pass  # previous frame still being processed; skip this one

    def get_depth(self) -> Optional[np.ndarray]:
        """Return the latest colorized depth map (H×W×3 RGB) or None."""
        with self._out_lock:
            return self._latest_depth.copy() if self._latest_depth is not None else None

    def stop(self) -> None:
        self._running = False
        try:
            self._in_queue.put_nowait(None)   # unblock the thread
        except queue.Full:
            pass
        self._thread.join(timeout=3)

    # ── Internal inference loop ───────────────────────────────────────────────

    def _infer_loop(self):
        while self._running:
            try:
                frame = self._in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                break

            with _model_lock:
                model_bundle = _model
                status       = _model_status

            if model_bundle is None or status != STATUS_READY:
                continue

            try:
                model, model_type = model_bundle
                depth_rgb = self._run_inference(model, model_type, frame)
                with self._out_lock:
                    self._latest_depth = depth_rgb
            except Exception as exc:
                print(f"[DepthWorker cam{self.camera_index}] Inference error: {exc}")

    def _run_inference(self, model, model_type: str,
                       bgr_frame: np.ndarray) -> np.ndarray:
        """Run one inference pass; returns H×W×3 uint8 RGB depth image."""
        import torch
        from PIL import Image as PilImage

        # BGR → RGB → PIL
        rgb = bgr_frame[:, :, ::-1]
        pil_img = PilImage.fromarray(rgb.astype(np.uint8))

        if model_type == "da3":
            # ── DA3Mono-Large native API ────────────────────────────────
            # model.inference([pil_img]) → Predictions object
            # predictions.depth[0] → H×W float32 numpy array
            with torch.inference_mode():
                predictions = model.inference([pil_img])
            depth_np = predictions.depth[0]           # H×W float32

        elif model_type == "hf_pipeline":
            # ── HuggingFace transformers pipeline fallback ───────────────
            result   = model(pil_img)
            depth_pil = result["depth"]               # PIL grayscale Image
            depth_np  = np.array(depth_pil, dtype=np.float32)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Resize depth to display dimensions
        import cv2
        depth_resized = cv2.resize(
            depth_np.astype(np.float32),
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Apply colourmap: near=warm, far=cool
        colored = _colorize_depth(depth_resized)   # H×W×3 uint8 RGB
        return colored
