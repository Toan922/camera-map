"""Generate a printable checkerboard for camera calibration.

10x7 squares -> 9x6 inner corners (what calibrate.py looks for), 20mm squares.
Saved with 300 DPI metadata so 'print at 100%' comes out true to size.
ALWAYS verify with a ruler after printing: one square must be exactly 20mm.
"""
import numpy as np
from PIL import Image

DPI = 300
SQUARE_MM = 20
COLS, ROWS = 10, 7          # squares (inner corners = 9x6)
MARGIN_MM = 12

px = lambda mm: int(round(mm / 25.4 * DPI))
sq = px(SQUARE_MM)
margin = px(MARGIN_MM)

board = np.kron((np.indices((ROWS, COLS)).sum(axis=0) % 2) * 255, np.ones((sq, sq))).astype(np.uint8)
canvas = np.full((board.shape[0] + 2 * margin, board.shape[1] + 2 * margin), 255, np.uint8)
canvas[margin:margin + board.shape[0], margin:margin + board.shape[1]] = board

img = Image.fromarray(canvas)
img.save('calibration/checkerboard_9x6_20mm.png', dpi=(DPI, DPI))
print(f"saved calibration/checkerboard_9x6_20mm.png "
      f"({canvas.shape[1]}x{canvas.shape[0]}px @ {DPI}dpi = "
      f"{canvas.shape[1]/DPI*25.4:.0f}x{canvas.shape[0]/DPI*25.4:.0f}mm)")
print("Print at 100% scale (no 'fit to page'), then verify one square = 20mm with a ruler.")
