import argparse
import json
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.spatial.transform import Rotation


def load_json(path):
    with open(path) as f:
        return json.load(f)


def rot6d_to_euler_degrees(rot6d):
    """Reconstruct rotation matrix from 6D repr, return XYZ Euler angles in degrees.

    The 6D representation stores the first two columns of a rotation matrix.
    The third column is recovered via their cross product.
    A Gram-Schmidt step ensures orthonormality before the cross product.
    """
    a1 = np.array(rot6d[:3], dtype=float)
    a2 = np.array(rot6d[3:6], dtype=float)
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 - np.dot(a2, a1) * a1
    a2 = a2 / np.linalg.norm(a2)
    a3 = np.cross(a1, a2)
    R = np.stack([a1, a2, a3], axis=1)  # columns → 3×3 rotation matrix
    return Rotation.from_matrix(R).as_euler("xyz", degrees=True)


def plot_rotation_6d(data, ax, use_degrees):
    ax.clear()
    frames = data["frames"]
    timestamps = [f["timestamp"] for f in frames]
    source = data.get("video_info", {}).get("source", "")

    if use_degrees:
        eulers = np.array(
            [rot6d_to_euler_degrees(f["rotation_6d"]) for f in frames]
        )
        labels = ["X", "Y", "Z"]
        colors = ["tab:red", "tab:green", "tab:blue"]
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(timestamps, eulers[:, i], label=label, color=color)
        ax.set_title(f"Euler Angles XYZ (degrees)  —  {source}", fontsize=11)
        ax.set_ylabel("Degrees")
    else:
        components = [
            [f["rotation_6d"][i] for f in frames] for i in range(6)
        ]
        labels = ["r0", "r1", "r2", "r3", "r4", "r5"]
        colors = [
            "tab:red", "tab:orange", "tab:green",
            "tab:blue", "tab:purple", "tab:brown",
        ]
        for comp, label, color in zip(components, labels, colors):
            ax.plot(timestamps, comp, label=label, color=color)
        ax.set_title(f"Rotation 6D  —  {source}", fontsize=11)
        ax.set_ylabel("Rotation 6D")

    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)


class App(tk.Tk):
    def __init__(self, use_degrees=False):
        super().__init__()
        self.title("Motion JSON — Rotation 6D Viewer")
        self.geometry("900x560")
        self._data = None
        self._build_ui(use_degrees)

    def _build_ui(self, use_degrees):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(toolbar, text="Open JSON…", command=self._open_file).pack(
            side=tk.LEFT
        )
        self.path_var = tk.StringVar(value="No file loaded")
        ttk.Label(toolbar, textvariable=self.path_var, foreground="gray").pack(
            side=tk.LEFT, padx=10
        )

        self.degrees_var = tk.BooleanVar(value=use_degrees)
        ttk.Checkbutton(
            toolbar,
            text="Euler angles (degrees)",
            variable=self.degrees_var,
            command=self._replot,
        ).pack(side=tk.LEFT, padx=8)

        self.fig, self.ax = plt.subplots(figsize=(9, 4.5))
        self.fig.tight_layout(pad=2.5)

        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

        nav = NavigationToolbar2Tk(canvas, self)
        nav.update()

    def _replot(self):
        if self._data is None:
            return
        plot_rotation_6d(self._data, self.ax, self.degrees_var.get())
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select motion.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = load_json(path)
        except Exception as exc:
            tk.messagebox.showerror("Error", f"Could not load file:\n{exc}")
            return

        self._data = data
        self.path_var.set(path)
        self._replot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot rotation_6d from motion.json"
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Start in Euler-angles mode (degrees) instead of raw 6D",
    )
    args = parser.parse_args()
    app = App(use_degrees=args.degrees)
    app.mainloop()
