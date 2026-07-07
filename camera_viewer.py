"""
camera_viewer.py
----------------
Opens a popup window showing all detected webcam feeds simultaneously,
arranged in a responsive grid layout.

Each camera card shows two panels side-by-side:
  • Left  – live RGB feed  (always on)
  • Right – colorized depth map from Depth Anything 3 (DA3Mono-Large)
             or "Loading model…" / "Depth unavailable" if the model
             is still initialising or failed to load.

Requirements:
    pip install opencv-python pillow numpy
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install transformers einops huggingface_hub safetensors
"""

import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import math
import threading
import time

import depth_worker as dw


# ── Config ────────────────────────────────────────────────────────────────────
MAX_CAMERAS_TO_SCAN = 10   # how many indices to probe for cameras
PANEL_W             = 320  # width of EACH panel (RGB or depth) in px
PANEL_H             = 240  # height per panel
FPS_INTERVAL_MS     = 33   # ~30 fps Tkinter refresh
DEPTH_SUBMIT_EVERY  = 2    # submit a frame for depth inference every N RGB frames
WINDOW_TITLE        = "Camera Map – Live Feeds + Depth"
BG_COLOR            = "#0f1117"
CARD_COLOR          = "#1e2130"
LABEL_COLOR         = "#94a3b8"
ACCENT_COLOR        = "#6366f1"
DEPTH_ACCENT        = "#f59e0b"
# ──────────────────────────────────────────────────────────────────────────────

TOTAL_W = PANEL_W * 2   # card canvas spans both panels


def _try_open(index: int) -> cv2.VideoCapture | None:
    """
    Try backends in order and return the first VideoCapture that
    successfully reads a frame.  Avoids forcing MJPEG/resolution hints
    that cause MSMF streaming failures on many webcams.
    """
    # CAP_DSHOW first (most stable on Windows for multiple webcams),
    # then MSMF and ANY
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # Request a moderate resolution — let the driver pick the format
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PANEL_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PANEL_H)

        # Warm-up: give the sensor time to start streaming
        ok = False
        for _ in range(5):
            ret, _ = cap.read()
            if ret:
                ok = True
                break
            time.sleep(0.1)

        if ok:
            print(f"  Camera {index}: opened with backend {backend}")
            return cap
        cap.release()
    return None


def detect_cameras(max_index: int = MAX_CAMERAS_TO_SCAN) -> list[tuple[int, cv2.VideoCapture]]:
    """
    Scan indices 0..max_index and return a list of (index, open_capture)
    pairs for every camera that produces frames.
    Cameras are intentionally kept open to avoid Windows re-open races.
    """
    found = []
    for i in range(max_index):
        print(f"Probing camera {i}…")
        cap = _try_open(i)
        if cap is not None:
            found.append((i, cap))
            # Brief pause so the OS USB stack can stabilise before the next open
            time.sleep(0.3)
    return found


# ── CameraFeed ────────────────────────────────────────────────────────────────

class CameraFeed:
    """Receives an already-open VideoCapture and streams frames in a thread."""

    def __init__(self, index: int, cap: cv2.VideoCapture):
        self.index = index
        self.cap   = cap
        self.frame = None          # latest RGB numpy frame (H×W×3)
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        consecutive_failures = 0
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                consecutive_failures = 0
                frame = cv2.resize(frame, (PANEL_W, PANEL_H))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self.frame = frame
            else:
                consecutive_failures += 1
                time.sleep(0.05)
                # After 30 consecutive failures (~1.5s), try to reopen the camera
                if consecutive_failures >= 30:
                    print(f"  Camera {self.index}: too many failures, reopening…")
                    self.cap.release()
                    time.sleep(0.5)
                    new_cap = _try_open(self.index)
                    if new_cap is not None:
                        self.cap = new_cap
                        consecutive_failures = 0
                        print(f"  Camera {self.index}: reopened successfully")
                    else:
                        consecutive_failures = 0  # reset to avoid spin-loop

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def get_bgr_frame(self):
        """Return latest frame as BGR numpy array (for depth inference)."""
        with self._lock:
            if self.frame is None:
                return None
            return self.frame[:, :, ::-1].copy()   # RGB→BGR

    @property
    def has_signal(self) -> bool:
        return self.frame is not None

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


# ── Placeholder image helpers ─────────────────────────────────────────────────

def _make_placeholder(text: str, w: int = PANEL_W, h: int = PANEL_H,
                      bg: str = "#0a0c12", fg: str = "#475569") -> Image.Image:
    """Return a PIL Image with centred text."""
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    # Use default font (no external fonts needed)
    draw.text((w // 2, h // 2), text, fill=fg, anchor="mm")
    return img


def _make_depth_bar(w: int = PANEL_W, h: int = 6) -> Image.Image:
    """Return a small gradient bar near→far for the depth legend."""
    import numpy as np
    gradient = np.linspace(0, 1, w, dtype=np.float32)
    # Apply the same colormap as depth_worker
    lut = dw._build_lut()
    idx = ((1.0 - gradient) * 255).clip(0, 255).astype(np.uint8)
    bar_row = lut[idx]   # w×3
    bar = np.tile(bar_row[np.newaxis, :, :], (h, 1, 1))   # h×w×3
    return Image.fromarray(bar.astype(np.uint8))


# ── CameraViewerApp ───────────────────────────────────────────────────────────

class CameraViewerApp(tk.Tk):
    def __init__(self, camera_pairs: list[tuple[int, cv2.VideoCapture]]):
        super().__init__()
        self.title(WINDOW_TITLE)
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)

        self.feeds:         list[CameraFeed]                       = []
        self.depth_workers: list[dw.DepthWorker]                  = []
        self.canvases:      list[tk.Canvas]                        = []
        self.status_dots:   list[tk.Label]                         = []
        self.depth_labels:  list[tk.Label]                         = []
        self.photo_refs:    list[ImageTk.PhotoImage | None]        = []
        self._frame_counts: list[int]                              = []

        # Depth status label in the header
        self._model_status_var = tk.StringVar(value="⏳ Loading depth model…")

        self._build_ui(camera_pairs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Register for model status changes
        dw.register_status_callback(self._on_model_status_change)

        # Kick off model loading immediately
        dw.ensure_model_loading()

        self._update_frames()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self, camera_pairs: list[tuple[int, cv2.VideoCapture]]):
        n = len(camera_pairs)

        # ── Header ──
        header = tk.Frame(self, bg=BG_COLOR, pady=12)
        header.pack(fill="x", padx=20)

        tk.Label(
            header,
            text="📷  Camera Map",
            font=("Segoe UI", 18, "bold"),
            bg=BG_COLOR, fg="white"
        ).pack(side="left")

        tk.Label(
            header,
            text=f"{n} camera{'s' if n != 1 else ''} detected",
            font=("Segoe UI", 10),
            bg=BG_COLOR, fg=LABEL_COLOR
        ).pack(side="left", padx=14)

        # Depth model status (right side of header)
        tk.Label(
            header,
            textvariable=self._model_status_var,
            font=("Segoe UI", 9),
            bg=BG_COLOR, fg=DEPTH_ACCENT
        ).pack(side="right")

        tk.Frame(self, bg=ACCENT_COLOR, height=2).pack(fill="x", padx=20, pady=(0, 12))

        # ── Depth legend bar ──
        try:
            bar_img   = _make_depth_bar(PANEL_W * 2)
            bar_photo = ImageTk.PhotoImage(image=bar_img)
            legend_frame = tk.Frame(self, bg=BG_COLOR)
            legend_frame.pack(padx=24, fill="x")
            tk.Label(legend_frame, text="Near", font=("Segoe UI", 8),
                     bg=BG_COLOR, fg="#f97316").pack(side="left")
            lbl = tk.Label(legend_frame, image=bar_photo, bg=BG_COLOR)
            lbl.image = bar_photo   # keep alive
            lbl.pack(side="left", expand=True, fill="x", padx=6)
            tk.Label(legend_frame, text="Far", font=("Segoe UI", 8),
                     bg=BG_COLOR, fg="#3b82f6").pack(side="right")
        except Exception:
            pass   # if numpy not ready yet, skip the legend

        # ── Camera grid ──
        cols       = max(1, math.ceil(math.sqrt(n)))
        grid_frame = tk.Frame(self, bg=BG_COLOR)
        grid_frame.pack(padx=16, pady=(8, 16))

        for i, (idx, cap) in enumerate(camera_pairs):
            row, col = divmod(i, cols)

            # ── Card ──────────────────────────────────────────────────────
            cell = tk.Frame(
                grid_frame,
                bg=CARD_COLOR,
                highlightthickness=2,
                highlightbackground="#2d3148",
            )
            cell.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

            # ── Card header row ──
            cam_header = tk.Frame(cell, bg=CARD_COLOR, pady=6)
            cam_header.pack(fill="x", padx=10)

            tk.Label(
                cam_header,
                text=f"  Camera {idx}",
                font=("Segoe UI", 9, "bold"),
                bg=CARD_COLOR, fg=LABEL_COLOR
            ).pack(side="left")

            dot = tk.Label(
                cam_header, text="●",
                font=("Segoe UI", 8),
                bg=CARD_COLOR, fg="#f59e0b"   # amber while connecting
            )
            dot.pack(side="right")
            self.status_dots.append(dot)

            # ── Panel labels row (RGB | DEPTH) ──
            labels_row = tk.Frame(cell, bg=CARD_COLOR)
            labels_row.pack(fill="x", padx=2)

            tk.Label(
                labels_row, text="RGB",
                font=("Segoe UI", 8, "bold"),
                bg=CARD_COLOR, fg="#22c55e", width=PANEL_W // 8
            ).pack(side="left", expand=True)

            depth_lbl = tk.Label(
                labels_row, text="DEPTH",
                font=("Segoe UI", 8, "bold"),
                bg=CARD_COLOR, fg=DEPTH_ACCENT, width=PANEL_W // 8
            )
            depth_lbl.pack(side="right", expand=True)
            self.depth_labels.append(depth_lbl)

            # ── Wide canvas spanning both panels ──
            canvas = tk.Canvas(
                cell,
                width=TOTAL_W, height=PANEL_H,
                bg="#0a0c12",
                highlightthickness=0,
            )
            canvas.pack(padx=2, pady=(0, 2))

            # Initial placeholder
            canvas.create_text(
                PANEL_W // 2, PANEL_H // 2,
                text="Connecting…",
                fill=LABEL_COLOR,
                font=("Segoe UI", 10),
                tags="placeholder_rgb",
            )
            canvas.create_text(
                PANEL_W + PANEL_W // 2, PANEL_H // 2,
                text="Loading model…",
                fill="#4b5563",
                font=("Segoe UI", 10),
                tags="placeholder_depth",
            )

            # Thin divider line between panels
            canvas.create_line(
                PANEL_W, 0, PANEL_W, PANEL_H,
                fill="#2d3148", width=1,
                tags="divider"
            )

            feed   = CameraFeed(idx, cap)
            worker = dw.DepthWorker(idx, target_w=PANEL_W, target_h=PANEL_H)

            self.feeds.append(feed)
            self.depth_workers.append(worker)
            self.canvases.append(canvas)
            self.photo_refs.append(None)
            self._frame_counts.append(0)

        # Equalise column widths
        for c in range(cols):
            grid_frame.columnconfigure(c, weight=1, uniform="col")

        # No cameras fallback
        if not camera_pairs:
            tk.Label(
                grid_frame,
                text="No cameras detected.\nConnect a webcam and restart.",
                font=("Segoe UI", 13),
                bg=BG_COLOR, fg=LABEL_COLOR,
                pady=40
            ).pack()

    # ── Model status callback ─────────────────────────────────────────────────

    def _on_model_status_change(self, status: str):
        """Called from the model-loader thread; schedule UI update on main thread."""
        self.after(0, self._apply_model_status, status)

    def _apply_model_status(self, status: str):
        labels = {
            dw.STATUS_LOADING:     "⏳ Loading depth model…",
            dw.STATUS_READY:       "🟢 Depth: Active (DA3Mono-Large)",
            dw.STATUS_ERROR:       f"⚠️ Depth error: {dw.get_model_error()[:50]}",
            dw.STATUS_UNAVAILABLE: "❌ Depth unavailable (install torch + transformers)",
        }
        text = labels.get(status, f"Depth: {status}")
        self._model_status_var.set(text)

        if status == dw.STATUS_READY:
            for lbl in self.depth_labels:
                lbl.configure(fg="#22c55e")
        elif status == dw.STATUS_ERROR:
            for lbl in self.depth_labels:
                lbl.configure(fg="#ef4444")

    # ── Frame Update Loop ─────────────────────────────────────────────────────

    def _update_frames(self):
        model_ready = dw.get_model_status() == dw.STATUS_READY

        for i, feed in enumerate(self.feeds):
            canvas = self.canvases[i]
            frame  = feed.get_frame()   # RGB numpy H×W×3

            # ── Left panel: RGB ──────────────────────────────────────────
            if frame is not None:
                self.status_dots[i].configure(fg="#22c55e")
                rgb_img   = Image.fromarray(frame)

                # ── Right panel: DEPTH ───────────────────────────────────
                worker = self.depth_workers[i]

                # Submit a frame for inference periodically (not every frame)
                count = self._frame_counts[i]
                if model_ready and count % DEPTH_SUBMIT_EVERY == 0:
                    bgr = feed.get_bgr_frame()
                    if bgr is not None:
                        worker.submit(bgr)
                self._frame_counts[i] = count + 1

                depth_arr = worker.get_depth()   # H×W×3 RGB or None
                if depth_arr is not None:
                    depth_img = Image.fromarray(depth_arr)
                else:
                    # Show a placeholder with status text
                    status = dw.get_model_status()
                    msg = {
                        dw.STATUS_LOADING: "Loading model…",
                        dw.STATUS_ERROR:   "Depth error",
                        dw.STATUS_IDLE:    "Initialising…",
                    }.get(status, "No depth yet")
                    depth_img = _make_placeholder(msg, PANEL_W, PANEL_H)

                # ── Combine into one wide image and blit ─────────────────
                combined = Image.new("RGB", (TOTAL_W, PANEL_H))
                combined.paste(rgb_img,   (0, 0))
                combined.paste(depth_img, (PANEL_W, 0))

                photo = ImageTk.PhotoImage(image=combined)
                canvas.delete("all")
                canvas.create_image(0, 0, anchor="nw", image=photo)
                # Redraw divider line on top
                canvas.create_line(
                    PANEL_W, 0, PANEL_W, PANEL_H,
                    fill="#2d3148", width=1,
                )
                self.photo_refs[i] = photo
            else:
                # Still connecting — keep amber dot
                self.status_dots[i].configure(fg="#f59e0b")

        self.after(FPS_INTERVAL_MS, self._update_frames)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self):
        for worker in self.depth_workers:
            worker.stop()
        for feed in self.feeds:
            feed.stop()
        self.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    print("Scanning for cameras…")
    camera_pairs = detect_cameras()

    if not camera_pairs:
        print("No cameras found. Launching window with placeholder.")
    else:
        indices = [idx for idx, _ in camera_pairs]
        print(f"Found {len(camera_pairs)} camera(s): indices {indices}")

    app = CameraViewerApp(camera_pairs)
    app.mainloop()


if __name__ == "__main__":
    main()
