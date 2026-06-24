import json
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_translation(data, ax):
    ax.clear()
    frames = data["frames"]
    timestamps = [f["timestamp"] for f in frames]
    tx = [f["translation"][0] for f in frames]
    ty = [f["translation"][1] for f in frames]
    tz = [f["translation"][2] for f in frames]

    ax.plot(timestamps, tx, label="X", color="tab:red")
    ax.plot(timestamps, ty, label="Y", color="tab:green")
    ax.plot(timestamps, tz, label="Z", color="tab:blue")

    source = data.get("video_info", {}).get("source", "")
    ax.set_title(f"Translation  —  {source}", fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Translation")
    ax.legend()
    ax.grid(True, alpha=0.3)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Motion JSON — Translation Viewer")
        self.geometry("900x560")
        self._build_ui()

    def _build_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(toolbar, text="Open JSON…", command=self._open_file).pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value="No file loaded")
        ttk.Label(toolbar, textvariable=self.path_var, foreground="gray").pack(
            side=tk.LEFT, padx=10
        )

        self.fig, self.ax = plt.subplots(figsize=(9, 4.5))
        self.fig.tight_layout(pad=2.5)

        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

        nav = NavigationToolbar2Tk(canvas, self)
        nav.update()

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

        self.path_var.set(path)
        plot_translation(data, self.ax)
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()
