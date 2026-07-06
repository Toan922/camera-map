"""
camera_viewer.py
----------------
Opens a popup window showing all detected webcam feeds simultaneously,
arranged in a responsive grid layout.

Requirements:
    pip install opencv-python pillow
"""

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import math
import threading
import time


# ── Config ────────────────────────────────────────────────────────────────────
MAX_CAMERAS_TO_SCAN = 10   # how many indices to probe for cameras
FRAME_WIDTH         = 320  # display width per feed (px)
FRAME_HEIGHT        = 240  # display height per feed (px)
FPS_INTERVAL_MS     = 33   # ~30 fps refresh rate
WINDOW_TITLE        = "Camera Map – Live Feeds"
BG_COLOR            = "#0f1117"
LABEL_COLOR         = "#94a3b8"
ACCENT_COLOR        = "#6366f1"
# ──────────────────────────────────────────────────────────────────────────────


def _try_open(index: int) -> cv2.VideoCapture | None:
    """
    Try every backend in order and return the first VideoCapture that
    successfully reads a frame.  The capture is left OPEN so the caller
    can hand it directly to CameraFeed without re-opening.

    Forces MJPEG compression + target resolution before any reads to
    keep USB bandwidth low enough for 3 simultaneous cameras.
    """
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # ── Reduce USB bandwidth: MJPEG instead of raw YUV ──────────────
        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # ────────────────────────────────────────────────────────────────

        # Warm up: drain frames so the sensor stabilises after mode change
        for _ in range(15):
            cap.read()
        ret, _ = cap.read()
        if ret:
            print(f"  Camera {index}: opened with backend {backend}")
            return cap          # returned OPEN — caller must eventually release
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
        self.frame = None
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
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self.frame = frame
            else:
                consecutive_failures += 1
                # Brief pause before retrying to avoid busy-spin on failure
                time.sleep(0.05)

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    @property
    def has_signal(self) -> bool:
        return self.frame is not None

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


# ── CameraViewerApp ───────────────────────────────────────────────────────────

class CameraViewerApp(tk.Tk):
    def __init__(self, camera_pairs: list[tuple[int, cv2.VideoCapture]]):
        super().__init__()
        self.title(WINDOW_TITLE)
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)

        self.feeds:      list[CameraFeed]             = []
        self.canvases:   list[tk.Canvas]              = []
        self.status_dots: list[tk.Label]              = []
        self.photo_refs: list[ImageTk.PhotoImage | None] = []

        self._build_ui(camera_pairs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
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

        tk.Frame(self, bg=ACCENT_COLOR, height=2).pack(fill="x", padx=20, pady=(0, 12))

        # ── Camera grid ──
        cols      = max(1, math.ceil(math.sqrt(n)))
        grid_frame = tk.Frame(self, bg=BG_COLOR)
        grid_frame.pack(padx=16, pady=(0, 16))

        for i, (idx, cap) in enumerate(camera_pairs):
            row, col = divmod(i, cols)

            # Card
            cell = tk.Frame(
                grid_frame,
                bg="#1e2130",
                highlightthickness=2,
                highlightbackground="#2d3148",
            )
            cell.grid(row=row, column=col, padx=8, pady=8,
                      sticky="nsew")

            # Card header row
            cam_header = tk.Frame(cell, bg="#1e2130", pady=6)
            cam_header.pack(fill="x", padx=10)

            tk.Label(
                cam_header,
                text=f"  Camera {idx}",
                font=("Segoe UI", 9, "bold"),
                bg="#1e2130", fg=LABEL_COLOR
            ).pack(side="left")

            dot = tk.Label(
                cam_header, text="●",
                font=("Segoe UI", 8),
                bg="#1e2130", fg="#f59e0b"   # amber while connecting
            )
            dot.pack(side="right")
            self.status_dots.append(dot)

            # Fixed-size canvas — always 320×240 regardless of signal state
            canvas = tk.Canvas(
                cell,
                width=FRAME_WIDTH, height=FRAME_HEIGHT,
                bg="#0a0c12",
                highlightthickness=0
            )
            canvas.pack(padx=2, pady=(0, 2))
            canvas.create_text(
                FRAME_WIDTH // 2, FRAME_HEIGHT // 2,
                text="Connecting…",
                fill=LABEL_COLOR,
                font=("Segoe UI", 10),
            )

            # Hand the already-open capture straight to CameraFeed
            feed = CameraFeed(idx, cap)
            self.feeds.append(feed)
            self.canvases.append(canvas)
            self.photo_refs.append(None)

        # Equalise column widths so all cells are the same size
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

    # ── Frame Update Loop ─────────────────────────────────────────────────────

    def _update_frames(self):
        for i, feed in enumerate(self.feeds):
            canvas = self.canvases[i]
            frame  = feed.get_frame()

            if frame is not None:
                img   = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=img)
                canvas.delete("all")
                canvas.create_image(0, 0, anchor="nw", image=photo)
                self.photo_refs[i] = photo          # keep reference alive
                self.status_dots[i].configure(fg="#22c55e")  # green = live
            else:
                # Still connecting — keep amber dot, leave "Connecting…" text
                self.status_dots[i].configure(fg="#f59e0b")

        self.after(FPS_INTERVAL_MS, self._update_frames)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self):
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
