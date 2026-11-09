"""
Ship Hull Drawing Pad
---------------------
A small Tkinter application for drawing a ship's outer boundary.

Controls:
  - Drag the mouse to draw the hull outline.
  - "Undo stroke" removes the last stroke.
  - "Clear" clears the canvas.
  - "Export hull" closes/fills the outline and saves:
       * ship_hull_mask.png  (256x256 binary mask: hull=white, background=black)
       * ship_hull_polygon.json (normalized polygon coordinates in [0, 1])
       * ship_hull_preview.png (larger preview)

Requirements:
  - Python 3
  - Pillow: pip install pillow

The export assumes you draw one continuous, non-self-intersecting outer outline.
"""

import json
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Pillow is required. Install it with: pip install pillow")


CANVAS_W = 800
CANVAS_H = 560
OUTPUT_SIZE = 256
PADDING = 12

INDEX = 0

class HullDrawingPad:
    def __init__(self, root):
        self.root = root
        self.root.title("Ship Hull Drawing Pad")
        self.root.resizable(False, False)

        self.strokes = []
        self.current_stroke = []
        self.last_xy = None

        header = tk.Frame(root, padx=12, pady=10)
        header.pack(fill="x")

        tk.Label(
            header,
            text="Draw the ship's outer boundary",
            font=("Arial", 15, "bold")
        ).pack(anchor="w")

        tk.Label(
            header,
            text=("Click and drag to draw one continuous outline. "
                  "Export will connect the final point to the first point "
                  "and fill the hull."),
            font=("Arial", 10)
        ).pack(anchor="w", pady=(3, 0))

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_W,
            height=CANVAS_H,
            bg="white",
            cursor="cross",
            highlightthickness=1,
            highlightbackground="#999999"
        )
        self.canvas.pack(padx=12, pady=4)

        controls = tk.Frame(root, padx=12, pady=10)
        controls.pack(fill="x")

        tk.Button(
            controls, text="Undo stroke", command=self.undo,
            width=14
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            controls, text="Clear", command=self.clear,
            width=10
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            controls, text="Export hull", command=self.export,
            width=14, bg="#d9ead3"
        ).pack(side="right")

        self.status = tk.StringVar(value="Ready — draw a closed hull outline.")
        tk.Label(
            root, textvariable=self.status, anchor="w",
            padx=12, pady=5, fg="#444444"
        ).pack(fill="x")

        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

    def start_stroke(self, event):
        self.current_stroke = [(event.x, event.y)]
        self.last_xy = (event.x, event.y)
        self.canvas.create_oval(
            event.x - 2, event.y - 2, event.x + 2, event.y + 2,
            fill="#174a7e", outline="#174a7e"
        )

    def draw(self, event):
        if self.last_xy is None:
            return

        x, y = event.x, event.y
        lx, ly = self.last_xy

        # Keep points within the drawing area.
        x = max(0, min(CANVAS_W, x))
        y = max(0, min(CANVAS_H, y))

        self.canvas.create_line(
            lx, ly, x, y,
            fill="#174a7e",
            width=3,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
            smooth=True
        )

        self.current_stroke.append((x, y))
        self.last_xy = (x, y)

    def end_stroke(self, event):
        if self.current_stroke:
            if len(self.current_stroke) == 1:
                self.current_stroke.append(self.current_stroke[0])
            self.strokes.append(self.current_stroke[:])
        self.current_stroke = []
        self.last_xy = None
        self.status.set(f"{len(self.strokes)} stroke(s) on canvas.")

    def redraw(self):
        self.canvas.delete("all")
        for stroke in self.strokes:
            if len(stroke) == 1:
                x, y = stroke[0]
                self.canvas.create_oval(
                    x - 2, y - 2, x + 2, y + 2,
                    fill="#174a7e", outline="#174a7e"
                )
            else:
                flattened = [coordinate for point in stroke for coordinate in point]
                self.canvas.create_line(
                    *flattened,
                    fill="#174a7e",
                    width=3,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    smooth=True
                )

    def undo(self):
        if self.strokes:
            self.strokes.pop()
            self.redraw()
            self.status.set(f"{len(self.strokes)} stroke(s) on canvas.")
        else:
            self.status.set("Nothing to undo.")

    def clear(self):
        self.strokes.clear()
        self.current_stroke = []
        self.last_xy = None
        self.canvas.delete("all")
        self.status.set("Canvas cleared.")

    def collect_outline(self):
        # Combine strokes in drawing order. For a clean hull, use one continuous stroke.
        points = []
        for stroke in self.strokes:
            for point in stroke:
                if not points or point != points[-1]:
                    points.append(point)

        # Remove consecutive duplicate points.
        cleaned = []
        for point in points:
            if not cleaned or point != cleaned[-1]:
                cleaned.append(point)

        if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()

        return cleaned

    def export(self):
        points = self.collect_outline()

        if len(points) < 3:
            messagebox.showwarning(
                "Not enough points",
                "Draw a hull outline with at least three points first."
            )
            return

        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return

        folder = Path(folder)

        # Fit the drawn outline into a square output while preserving aspect ratio.
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max(max_x - min_x, 1)
        height = max(max_y - min_y, 1)
        scale = (OUTPUT_SIZE - 2 * PADDING) / max(width, height)

        drawn_w = width * scale
        drawn_h = height * scale
        offset_x = (OUTPUT_SIZE - drawn_w) / 2
        offset_y = (OUTPUT_SIZE - drawn_h) / 2

        scaled_points = [
            (
                offset_x + (x - min_x) * scale,
                offset_y + (y - min_y) * scale
            )
            for x, y in points
        ]

        # Binary mask: white (255) inside hull, black (0) outside.
        mask = Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(scaled_points, fill=255)

        # Save a larger preview with the same fitted geometry.
        preview_scale = 3
        preview = mask.resize(
            (OUTPUT_SIZE * preview_scale, OUTPUT_SIZE * preview_scale),
            Image.Resampling.NEAREST
        )

        # Coordinates normalized to [0, 1], with y increasing downward.
        polygon_normalized = [
            [round(x / OUTPUT_SIZE, 6), round(y / OUTPUT_SIZE, 6)]
            for x, y in scaled_points
        ]

        global INDEX        
        mask_path = folder / f"ship_hull_mask_{INDEX}.png"
        polygon_path = folder / f"ship_hull_polygon_{INDEX}.json"
        # preview_path = folder / "ship_hull_preview.png"

        mask.save(mask_path)
        # preview.save(preview_path)

        payload = {
            "image_size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "coordinate_system": "normalized [0,1], origin at top-left",
            "polygon": polygon_normalized,
            "note": "Single outer hull polygon; polygon is implicitly closed."
        }
        polygon_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8"
        )

        self.status.set(f"Exported hull files to: {folder}")
        messagebox.showinfo(
            "Export complete",
            "Saved:\n"
            f"{mask_path}\n"
            f"{polygon_path}\n"
        )

        INDEX += 1


if __name__ == "__main__":
    root = tk.Tk()
    app = HullDrawingPad(root)
    root.mainloop()
