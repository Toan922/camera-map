# Camera Map

A personal project built with friends to view all webcam feeds simultaneously in a single popup window.

## Requirements

- Python 3.10+
- Webcam(s) connected via USB

## Setup

**Option 1 – Double-click (Windows)**

Run `setup.bat`. It will check for Python, install pip if needed, and install all dependencies automatically.

**Option 2 – Manual**

```bash
python -m pip install -r requirements.txt
```

## Running

```bash
python camera_viewer.py
```

The app will scan for all connected webcams and display their live feeds in a tiled grid.

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture and frame processing |
| `Pillow` | Image conversion for the Tkinter display |
| `numpy` | Array handling (auto-installed with opencv) |
| `tkinter` | GUI window (built into Python — no install needed) |
