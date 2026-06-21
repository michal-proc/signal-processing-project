"""SAR 1D Viewer.

A standalone Tkinter desktop application for exploring Sentinel-3 / CryoSat-like
L1A SAR altimetry data stored in NetCDF files. It mirrors the processing done in
``main.ipynb`` and validates the derived elevation against the Open-Elevation API.
The trajectory is shown on an interactive slippy map (tkintermapview).
"""

from __future__ import annotations

import queue
import ssl
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import netCDF4
import numpy as np
import requests
import tkintermapview
import urllib3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

COLORS = {
    "background": "#0a0a0a",
    "foreground": "#fafafa",
    "card": "#1c1c1c",
    "popover": "#1c1c1c",
    "primary": "#26d6a0",
    "primary_fg": "#04261a",
    "secondary": "#2a2a2e",
    "secondary_fg": "#fafafa",
    "muted": "#262626",
    "muted_fg": "#a1a1a1",
    "accent": "#26d6a0",
    "destructive": "#f1664d",
    "border": "#2b2b2b",
    "input": "#333333",
    "ring": "#3f8f78",
    "chart_2": "#26d6a0",
}

HEIGHT_CMAP = "RdYlGn_r"
_CMAP = matplotlib.colormaps[HEIGHT_CMAP]

DARK_TILE_SERVER = "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"

SPEED_OF_LIGHT = 299_792_458.0
BANDWIDTH = 350e6
GATE_RESOLUTION = SPEED_OF_LIGHT / (2 * BANDWIDTH)
N_SAMPLES = 128
SAMPLE_RATE = 50e6
REFERENCE_GATE = 64

VAR_I = "i_meas_ku_l1a_echo_sar_ku"
VAR_Q = "q_meas_ku_l1a_echo_sar_ku"
VAR_LAT = "lat_l1a_echo_sar_ku"
VAR_LON = "lon_l1a_echo_sar_ku"
VAR_RANGE = "range_ku_l1a_echo_sar_ku"
VAR_ALT = "alt_l1a_echo_sar_ku"
VAR_SURF = "surf_type_l1a_echo_sar_ku"

OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

SURFACE_TYPES = {
    0: "open ocean / sea",
    1: "enclosed sea / lake",
    2: "continental ice",
    3: "land",
}

GATE_TOOLTIP = (
    "Range gate (0–127): the sample bin of the range-compressed waveform where the "
    "ground return is detected via cross-correlation with a step reference. The offset "
    "from the reference gate (64) gives the fine range:  R_fine = (gate − 64) × c/(2·B)."
)

FREQUENCIES = np.fft.fftshift(np.fft.fftfreq(N_SAMPLES, d=1 / SAMPLE_RATE))


def enable_dark_titlebar(window: tk.Misc) -> None:
    """Force a dark Windows title bar for the given window (no-op elsewhere)."""
    try:
        import ctypes

        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1)
        for attribute in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
            )
    except Exception:
        pass


def elevation_color(value: float, vmin: float, vmax: float) -> str:
    """Map an elevation to a hex colour on the green→red scale."""
    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = min(max(t, 0.0), 1.0)
    r, g, b, _ = _CMAP(t)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _read_complex(dataset: netCDF4.Dataset, idx: int) -> np.ndarray:
    """Reconstruct the complex signal for a single burst ``idx``."""
    i = np.asarray(dataset.variables[VAR_I][idx, :, :], dtype=float)
    q = np.asarray(dataset.variables[VAR_Q][idx, :, :], dtype=float)
    return i + 1j * q


def compute_point(dataset: netCDF4.Dataset, idx: int) -> dict:
    """Run the full processing chain for one burst and return diagnostics."""
    complex_signal = _read_complex(dataset, idx)

    range_compressed = np.fft.fft(complex_signal, axis=-1)
    range_compressed_shifted = np.fft.fftshift(range_compressed, axes=-1)

    power_pulses = np.abs(range_compressed_shifted) ** 2
    averaged_waveform = np.mean(power_pulses, axis=0)

    reference_step = np.zeros(N_SAMPLES)
    reference_step[REFERENCE_GATE:] = np.max(averaged_waveform)
    cross_corr = np.correlate(averaged_waveform, reference_step, mode="same")
    detected_gate = int(np.argmax(cross_corr))

    r_coarse = float(dataset.variables[VAR_RANGE][idx])
    r_fine = (detected_gate - REFERENCE_GATE) * GATE_RESOLUTION
    r_total = r_coarse + r_fine
    h_satellite = float(dataset.variables[VAR_ALT][idx])
    surface_elevation = h_satellite - r_total

    try:
        surf_type = int(dataset.variables[VAR_SURF][idx])
    except (KeyError, ValueError, TypeError):
        surf_type = None

    return {
        "elevation": surface_elevation,
        "detected_gate": detected_gate,
        "averaged_waveform": averaged_waveform,
        "complex_first": complex_signal[0],
        "spectrum_first": np.abs(range_compressed_shifted[0]),
        "r_coarse": r_coarse,
        "r_fine": r_fine,
        "r_total": r_total,
        "h_satellite": h_satellite,
        "surf_type": surf_type,
    }


def fetch_elevations(lats: np.ndarray, lons: np.ndarray, timeout: float = 90.0) -> np.ndarray:
    """Batch-query the Open-Elevation API for reference ground heights."""
    locations = "|".join(f"{la:.6f},{lo:.6f}" for la, lo in zip(lats, lons))
    response = requests.get(
        OPEN_ELEVATION_URL,
        params={"locations": locations},
        timeout=timeout,
        verify=False,
    )
    response.raise_for_status()
    payload = response.json()
    return np.array([p["elevation"] for p in payload["results"]], dtype=float)


def fetch_single_elevation(lat: float, lon: float, timeout: float = 30.0) -> float:
    """Query a single point for its reference ground height."""
    import json
    import urllib.parse
    import urllib.request

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    query = urllib.parse.urlencode({"locations": f"{lat:.6f},{lon:.6f}"})
    url = f"{OPEN_ELEVATION_URL}?{query}"
    with urllib.request.urlopen(url, timeout=timeout, context=context) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return float(payload["results"][0]["elevation"])


@dataclass
class SarData:
    dataset: netCDF4.Dataset
    indices: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    elevations: np.ndarray
    validation_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    validation_lats: np.ndarray = field(default_factory=lambda: np.array([]))
    validation_lons: np.ndarray = field(default_factory=lambda: np.array([]))
    validation_computed: np.ndarray = field(default_factory=lambda: np.array([]))
    validation_reference: np.ndarray | None = None
    validation_error: str | None = None
    correction: float = 0.0


class Tooltip:
    """A minimal hover tooltip for a Tk widget."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event) -> None:
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.tip,
            text=self.text,
            background=COLORS["popover"],
            foreground=COLORS["foreground"],
            wraplength=300,
            justify="left",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            font=("Segoe UI", 9),
        ).pack()

    def _hide(self, _event) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class BoundsDialog(tk.Toplevel):
    """Modal dialog for the bounding box and number of validation points."""

    def __init__(self, parent: tk.Misc, available: dict):
        super().__init__(parent)
        self.result: dict | None = None
        self.available = available

        self.title("Area parameters")
        self.configure(bg=COLORS["popover"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._build()
        self.bind("<Return>", lambda _e: self._on_ok())
        self.bind("<Escape>", lambda _e: self._on_cancel())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.update_idletasks()
        enable_dark_titlebar(self)
        self._center(parent)

    def _build(self) -> None:
        wrap = ttk.Frame(self, style="Card.TFrame", padding=20)
        wrap.pack(fill="both", expand=True)

        ttk.Label(wrap, text="Geographic range", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        ttk.Label(
            wrap,
            text="Choose the bounding box and the number of validation points.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        a = self.available
        info = ttk.Frame(wrap, style="Info.TFrame", padding=(12, 10))
        info.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        info.columnconfigure(1, weight=1)
        ttk.Label(info, text="AVAILABLE RANGE IN FILE", style="InfoTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        ttk.Label(info, text="Latitude", style="InfoKey.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 16)
        )
        ttk.Label(
            info, text=f"{a['lat_min']:.3f}°  …  {a['lat_max']:.3f}°", style="InfoVal.TLabel"
        ).grid(row=1, column=1, sticky="e")
        ttk.Label(info, text="Longitude", style="InfoKey.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 16), pady=(4, 0)
        )
        ttk.Label(
            info, text=f"{a['lon_min']:.3f}°  …  {a['lon_max']:.3f}°", style="InfoVal.TLabel"
        ).grid(row=2, column=1, sticky="e", pady=(4, 0))

        self.vars = {
            "lat_min": tk.StringVar(value=f"{a['lat_min']:.3f}"),
            "lat_max": tk.StringVar(value=f"{a['lat_max']:.3f}"),
            "lon_min": tk.StringVar(value=f"{a['lon_min']:.3f}"),
            "lon_max": tk.StringVar(value=f"{a['lon_max']:.3f}"),
            "n_val": tk.StringVar(value="30"),
            "correction": tk.StringVar(value="-40"),
        }

        fields = [
            ("Latitude min", "lat_min"),
            ("Latitude max", "lat_max"),
            ("Longitude min", "lon_min"),
            ("Longitude max", "lon_max"),
            ("Validation points", "n_val"),
            ("Elevation correction [m]", "correction"),
        ]
        for r, (label, key) in enumerate(fields, start=3):
            ttk.Label(wrap, text=label, style="Card.TLabel").grid(
                row=r, column=0, sticky="w", pady=4, padx=(0, 12)
            )
            entry = ttk.Entry(wrap, textvariable=self.vars[key], width=12, style="Dark.TEntry")
            entry.grid(row=r, column=1, sticky="ew", pady=4)

        btns = ttk.Frame(wrap, style="Card.TFrame")
        btns.grid(row=len(fields) + 3, column=0, columnspan=2, sticky="e", pady=(18, 0))
        ttk.Button(btns, text="Cancel", style="Ghost.TButton", command=self._on_cancel).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btns, text="Compute", style="Accent.TButton", command=self._on_ok).pack(
            side="left"
        )

    def _center(self, parent: tk.Misc) -> None:
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

    def _on_ok(self) -> None:
        try:
            lat_min = float(self.vars["lat_min"].get())
            lat_max = float(self.vars["lat_max"].get())
            lon_min = float(self.vars["lon_min"].get())
            lon_max = float(self.vars["lon_max"].get())
            n_val = int(float(self.vars["n_val"].get()))
            correction = float(self.vars["correction"].get())
        except ValueError:
            messagebox.showerror("Error", "All fields must be numbers.", parent=self)
            return

        if lat_min >= lat_max or lon_min >= lon_max:
            messagebox.showerror(
                "Error", "Min values must be smaller than max values.", parent=self
            )
            return
        if not (1 <= n_val <= 50):
            messagebox.showerror(
                "Error", "The number of validation points must be between 1 and 50.", parent=self
            )
            return

        self.result = {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "n_val": n_val,
            "correction": correction,
        }
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()


class SarViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.data: SarData | None = None
        self.sel_marker = None
        self.queue: queue.Queue = queue.Queue()
        self.progress_win: tk.Toplevel | None = None
        self.vmin = 0.0
        self.vmax = 1.0

        root.title("SAR 1D Viewer")
        root.geometry("1500x900")
        root.minsize(1150, 720)
        root.configure(bg=COLORS["background"])

        self._setup_style()
        self._build_layout()
        enable_dark_titlebar(root)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame", background=COLORS["background"])
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("Sidebar.TFrame", background=COLORS["card"])

        style.configure(
            "TLabel", background=COLORS["background"], foreground=COLORS["foreground"]
        )
        style.configure(
            "Card.TLabel", background=COLORS["card"], foreground=COLORS["foreground"]
        )
        style.configure(
            "Muted.TLabel", background=COLORS["card"], foreground=COLORS["muted_fg"]
        )
        style.configure(
            "Coord.TLabel",
            background=COLORS["card"],
            foreground=COLORS["muted_fg"],
            font=("Consolas", 10),
        )
        style.configure("Info.TFrame", background=COLORS["muted"])
        style.configure(
            "InfoTitle.TLabel",
            background=COLORS["muted"],
            foreground=COLORS["muted_fg"],
            font=("Segoe UI", 8),
        )
        style.configure(
            "InfoKey.TLabel",
            background=COLORS["muted"],
            foreground=COLORS["foreground"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "InfoVal.TLabel",
            background=COLORS["muted"],
            foreground=COLORS["primary"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Heading.TLabel",
            background=COLORS["card"],
            foreground=COLORS["primary"],
            font=("Segoe UI Semibold", 13),
        )
        style.configure(
            "StatKey.TLabel",
            background=COLORS["card"],
            foreground=COLORS["muted_fg"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "StatVal.TLabel",
            background=COLORS["card"],
            foreground=COLORS["foreground"],
            font=("Consolas", 10),
        )

        style.configure(
            "Accent.TButton",
            background=COLORS["primary"],
            foreground=COLORS["primary_fg"],
            font=("Segoe UI Semibold", 10),
            borderwidth=0,
            focusthickness=0,
            padding=(16, 9),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#19a87c"), ("pressed", "#14805e")],
        )
        style.configure(
            "Ghost.TButton",
            background=COLORS["secondary"],
            foreground=COLORS["foreground"],
            borderwidth=0,
            padding=(14, 8),
        )
        style.map("Ghost.TButton", background=[("active", COLORS["muted"])])

        style.configure(
            "Dark.TEntry",
            fieldbackground=COLORS["input"],
            foreground=COLORS["foreground"],
            insertcolor=COLORS["foreground"],
            borderwidth=0,
            padding=6,
        )
        style.configure(
            "Omni.Horizontal.TProgressbar",
            background=COLORS["primary"],
            troughcolor=COLORS["muted"],
            borderwidth=0,
        )

    def _style_axes(self, ax) -> None:
        ax.set_facecolor(COLORS["card"])
        ax.tick_params(colors=COLORS["muted_fg"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        ax.title.set_color(COLORS["foreground"])
        ax.xaxis.label.set_color(COLORS["muted_fg"])
        ax.yaxis.label.set_color(COLORS["muted_fg"])

    def _placeholder(self, ax, message: str) -> None:
        ax.text(
            0.5,
            0.5,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=COLORS["muted_fg"],
            fontsize=8,
            wrap=True,
        )

    def _build_layout(self) -> None:
        bottom = ttk.Frame(self.root, style="Card.TFrame", padding=(20, 16))
        bottom.pack(side="bottom", fill="x")
        ttk.Button(
            bottom, text="Load", style="Accent.TButton", command=self.on_load
        ).pack(side="left", padx=(0, 16))
        self.file_label = ttk.Label(bottom, text="No file loaded", style="Muted.TLabel")
        self.file_label.pack(side="left")
        self.status_label = ttk.Label(bottom, text="", style="Muted.TLabel")
        self.status_label.pack(side="right")

        main = ttk.Frame(self.root, style="TFrame", padding=(12, 10))
        main.pack(side="top", fill="both", expand=True)
        main.columnconfigure(0, weight=3, uniform="cols")
        main.columnconfigure(1, weight=2, uniform="cols")
        main.rowconfigure(0, weight=1)

        self._build_map(main)
        self._build_right(main)

    def _build_map(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=8)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(card, text="Trajectory map (terrain elevation)", style="Heading.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        self.map_widget = tkintermapview.TkinterMapView(card, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)
        try:
            self.map_widget.set_tile_server(DARK_TILE_SERVER, max_zoom=20)
        except Exception:
            pass
        self.map_widget.set_position(52.0, 19.0)
        self.map_widget.set_zoom(6)
        self.map_widget.add_left_click_map_command(self.on_map_left_click)
        self.map_widget.canvas.bind("<Motion>", self.on_map_motion)

        footer = ttk.Frame(card, style="Card.TFrame")
        footer.pack(fill="x", pady=(6, 0))
        ttk.Label(footer, text="Elevation", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self.legend_min = ttk.Label(footer, text="low", style="Muted.TLabel")
        self.legend_min.pack(side="left", padx=(0, 4))
        self.legend_canvas = tk.Canvas(
            footer, width=150, height=12, highlightthickness=0, bg=COLORS["card"]
        )
        for i in range(150):
            self.legend_canvas.create_line(i, 0, i, 12, fill=elevation_color(i / 149, 0, 1))
        self.legend_canvas.pack(side="left")
        self.legend_max = ttk.Label(footer, text="high", style="Muted.TLabel")
        self.legend_max.pack(side="left", padx=(4, 0))

        self.coord_label = ttk.Label(footer, text="lat —,  lon —", style="Coord.TLabel")
        self.coord_label.pack(side="right")

    def _build_right(self, parent: ttk.Frame) -> None:
        right = ttk.Frame(parent, style="TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_general_stats(right)
        self._build_stats(right)
        self._build_plots(right)

    def _build_general_stats(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Sidebar.TFrame", padding=12)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(card, text="General statistics (validation)", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        card.columnconfigure(1, weight=1)

        self.gen_vars: dict[str, tk.StringVar] = {}
        rows = [
            ("Validation points", "n_val"),
            ("Mean difference (SAR − API)", "mean"),
            ("Standard deviation", "std"),
            ("RMSE", "rmse"),
            ("Max abs difference", "maxabs"),
        ]
        for r, (label, key) in enumerate(rows, start=1):
            self.gen_vars[key] = tk.StringVar(value="—")
            ttk.Label(card, text=label, style="StatKey.TLabel").grid(
                row=r, column=0, sticky="w", padx=(0, 10), pady=2
            )
            ttk.Label(card, textvariable=self.gen_vars[key], style="StatVal.TLabel").grid(
                row=r, column=1, sticky="e", pady=2
            )

    def _build_stats(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Sidebar.TFrame", padding=12)
        card.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(card, text="Point statistics", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
        )

        self.stat_vars: dict[str, tk.StringVar] = {}
        self.stat_key_labels: dict[str, ttk.Label] = {}
        rows = [
            ("Index", "index", "Lat [°]", "lat"),
            ("Lon [°]", "lon", "Surface", "surf"),
            ("SAR elev [m]", "sar_h", "API elev [m]", "api_h"),
            ("Difference [m]", "diff", "Gate", "gate"),
            ("Sat alt [m]", "h_sat", "Range [m]", "range"),
        ]
        for r, (k1, v1, k2, v2) in enumerate(rows, start=1):
            self._stat_cell(card, r, 0, k1, v1)
            self._stat_cell(card, r, 2, k2, v2)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(3, weight=1)

        Tooltip(self.stat_key_labels["gate"], GATE_TOOLTIP)

    def _stat_cell(self, parent, row, col, label, key) -> None:
        self.stat_vars[key] = tk.StringVar(value="—")
        key_label = ttk.Label(parent, text=label, style="StatKey.TLabel")
        key_label.grid(row=row, column=col, sticky="w", padx=(0, 6), pady=3)
        self.stat_key_labels[key] = key_label
        ttk.Label(parent, textvariable=self.stat_vars[key], style="StatVal.TLabel").grid(
            row=row, column=col + 1, sticky="w", padx=(0, 14), pady=3
        )

    def _build_plots(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=8)
        card.grid(row=2, column=0, sticky="nsew")

        self.plot_fig = Figure(figsize=(5, 8), dpi=100, facecolor=COLORS["card"])
        self.plot_fig.subplots_adjust(hspace=0.6, top=0.96, bottom=0.07, left=0.16, right=0.95)
        self.ax_complex = self.plot_fig.add_subplot(311)
        self.ax_spectrum = self.plot_fig.add_subplot(312)
        self.ax_profile = self.plot_fig.add_subplot(313)

        self.ax_complex.set_title("Complex signal on the complex plane", fontsize=9)
        self.ax_complex.set_xlabel("Re")
        self.ax_complex.set_ylabel("Im")
        self.ax_complex.set_xlim(-1.0, 1.0)
        self.ax_complex.set_ylim(-1.0, 1.0)
        self._placeholder(self.ax_complex, "Click a point to show the I/Q signal")

        self.ax_spectrum.set_title("Frequency spectrum of the signal", fontsize=9)
        self.ax_spectrum.set_xlabel("Frequency [Hz]")
        self.ax_spectrum.set_ylabel("Amplitude")
        self.ax_spectrum.set_xlim(FREQUENCIES.min(), FREQUENCIES.max())
        self.ax_spectrum.set_ylim(0.0, 1.0)
        self._placeholder(self.ax_spectrum, "Click a point to show the spectrum and detected gate")

        self.ax_profile.set_title("Ground elevation profile for measurements", fontsize=9)
        self.ax_profile.set_xlabel("Measurement index")
        self.ax_profile.set_ylabel("Elevation [m]")
        self.ax_profile.set_xlim(0, 1)
        self.ax_profile.set_ylim(0, 600)
        self._placeholder(self.ax_profile, "Load a file to compute the elevation profile")

        for ax in (self.ax_complex, self.ax_spectrum, self.ax_profile):
            self._style_axes(ax)

        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=card)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    def on_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a NetCDF file",
            filetypes=[("NetCDF", "*.nc"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            dataset = netCDF4.Dataset(path)
        except OSError as exc:
            messagebox.showerror("Error", f"Cannot open the file:\n{exc}")
            return

        lats = np.asarray(dataset.variables[VAR_LAT][:], dtype=float)
        lons = np.asarray(dataset.variables[VAR_LON][:], dtype=float)
        valid = (
            np.isfinite(lats)
            & np.isfinite(lons)
            & (np.abs(lats) <= 90)
            & (np.abs(lons) <= 180)
        )
        if not np.any(valid):
            messagebox.showerror("Error", "The file has no valid lat/lon coordinates.")
            dataset.close()
            return

        available = {
            "lat_min": float(np.min(lats[valid])),
            "lat_max": float(np.max(lats[valid])),
            "lon_min": float(np.min(lons[valid])),
            "lon_max": float(np.max(lons[valid])),
        }

        dialog = BoundsDialog(self.root, available)
        self.root.wait_window(dialog)
        if dialog.result is None:
            dataset.close()
            return

        params = dialog.result
        mask = (
            (lats >= params["lat_min"])
            & (lats <= params["lat_max"])
            & (lons >= params["lon_min"])
            & (lons <= params["lon_max"])
        )
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            messagebox.showwarning(
                "No data", "No measurements were found inside the given area."
            )
            dataset.close()
            return

        self.file_label.configure(text=path.split("/")[-1].split("\\")[-1])
        self._open_progress(len(indices))
        worker = threading.Thread(
            target=self._compute_worker,
            args=(
                dataset,
                indices,
                lats[indices],
                lons[indices],
                params["n_val"],
                params["correction"],
            ),
            daemon=True,
        )
        worker.start()
        self.root.after(80, self._poll_queue)

    def _compute_worker(self, dataset, indices, plats, plons, n_val, correction) -> None:
        try:
            n = len(indices)
            elevations = np.empty(n)
            for i, idx in enumerate(indices):
                elevations[i] = compute_point(dataset, int(idx))["elevation"] + correction
                if i % 20 == 0:
                    self.queue.put(("progress", i))
            self.queue.put(("progress", n))

            self.queue.put(("status", "Downloading validation points from Open-Elevation…"))
            section = np.unique(np.linspace(0, n - 1, min(n_val, n)).astype(int))
            v_idx = indices[section]
            v_lat = plats[section]
            v_lon = plons[section]
            v_comp = elevations[section]
            reference = None
            error = None
            try:
                reference = fetch_elevations(v_lat, v_lon)
            except Exception as exc:
                error = str(exc)

            data = SarData(
                dataset=dataset,
                indices=indices,
                lats=plats,
                lons=plons,
                elevations=elevations,
                validation_indices=v_idx,
                validation_lats=v_lat,
                validation_lons=v_lon,
                validation_computed=v_comp,
                validation_reference=reference,
                validation_error=error,
                correction=correction,
            )
            self.queue.put(("done", data))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    self._update_progress(payload)
                elif kind == "status":
                    if self.progress_win is not None:
                        self.progress_status.configure(text=payload)
                elif kind == "done":
                    self._close_progress()
                    self._on_data_ready(payload)
                    return
                elif kind == "error":
                    self._close_progress()
                    messagebox.showerror("Computation error", payload)
                    return
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _open_progress(self, total: int) -> None:
        win = tk.Toplevel(self.root)
        win.title("Processing")
        win.configure(bg=COLORS["popover"])
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        win.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ttk.Frame(win, style="Card.TFrame", padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Computing elevation profile", style="Heading.TLabel").pack(
            anchor="w"
        )
        self.progress_status = ttk.Label(
            frame, text="Processing measurements…", style="Muted.TLabel"
        )
        self.progress_status.pack(anchor="w", pady=(4, 12))
        self.progress_bar = ttk.Progressbar(
            frame,
            style="Omni.Horizontal.TProgressbar",
            length=340,
            maximum=total,
            mode="determinate",
        )
        self.progress_bar.pack(fill="x")
        self.progress_win = win

        self.root.update_idletasks()
        enable_dark_titlebar(win)
        px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
        pw, ph = self.root.winfo_width(), self.root.winfo_height()
        w, h = win.winfo_width(), win.winfo_height()
        win.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

    def _update_progress(self, value: int) -> None:
        if self.progress_win is not None:
            self.progress_bar["value"] = value
            self.progress_status.configure(text=f"Processed {value} measurements…")

    def _close_progress(self) -> None:
        if self.progress_win is not None:
            self.progress_win.grab_release()
            self.progress_win.destroy()
            self.progress_win = None

    def _on_data_ready(self, data: SarData) -> None:
        if self.data is not None and self.data.dataset is not data.dataset:
            try:
                self.data.dataset.close()
            except Exception:
                pass
        self.data = data
        self.sel_marker = None

        self._render_map()
        self._draw_profile()
        self.plot_canvas.draw_idle()
        self._update_general_stats(data)

        parts = [f"Ready • {len(data.indices)} measurements"]
        if data.validation_error:
            parts.append(f"validation unavailable: {data.validation_error[:50]}")
        else:
            parts.append(f"{len(data.validation_indices)} validation points")
        self.status_label.configure(text="  •  ".join(parts))

    def _update_general_stats(self, data: SarData) -> None:
        self.gen_vars["n_val"].set(str(len(data.validation_indices)))
        if data.validation_reference is None:
            for key in ("mean", "std", "rmse", "maxabs"):
                self.gen_vars[key].set("—")
            return
        diff = data.validation_computed - data.validation_reference
        self.gen_vars["mean"].set(f"{np.mean(diff):.2f} m")
        self.gen_vars["std"].set(f"{np.std(diff):.2f} m")
        self.gen_vars["rmse"].set(f"{np.sqrt(np.mean(diff ** 2)):.2f} m")
        self.gen_vars["maxabs"].set(f"{np.max(np.abs(diff)):.2f} m")

    def _render_map(self) -> None:
        data = self.data
        self.map_widget.delete_all_path()
        self.map_widget.delete_all_marker()
        self.sel_marker = None

        lats = data.lats
        lons = data.lons
        elevations = data.elevations
        self.vmin = float(elevations.min())
        self.vmax = float(elevations.max())
        self.legend_min.configure(text=f"{self.vmin:.0f} m")
        self.legend_max.configure(text=f"{self.vmax:.0f} m")

        n = len(lats)
        sample = np.unique(np.linspace(0, n - 1, min(n, 160)).astype(int))
        for a, b in zip(sample[:-1], sample[1:]):
            color = elevation_color(elevations[a], self.vmin, self.vmax)
            self.map_widget.set_path(
                [(float(lats[a]), float(lons[a])), (float(lats[b]), float(lons[b]))],
                color=color,
                width=6,
            )

        try:
            self.map_widget.fit_bounding_box(
                (float(lats.max()), float(lons.min())),
                (float(lats.min()), float(lons.max())),
            )
        except Exception:
            self.map_widget.set_position(float(lats.mean()), float(lons.mean()))

    def _draw_profile(self, selected_pos: int | None = None) -> None:
        data = self.data
        self.ax_profile.clear()
        self._style_axes(self.ax_profile)
        self.ax_profile.plot(
            data.indices,
            data.elevations,
            color=COLORS["chart_2"],
            linewidth=1.0,
            label="SAR 1D",
        )
        if data.validation_reference is not None:
            self.ax_profile.scatter(
                data.validation_indices,
                data.validation_reference,
                color=COLORS["destructive"],
                s=18,
                zorder=5,
                label="API",
            )
        if selected_pos is not None:
            self.ax_profile.scatter(
                [data.indices[selected_pos]],
                [data.elevations[selected_pos]],
                facecolors="none",
                edgecolors=COLORS["foreground"],
                s=80,
                linewidths=1.5,
                zorder=6,
                label="Selected point",
            )
        self.ax_profile.set_title("Ground elevation profile for measurements", fontsize=9)
        self.ax_profile.set_xlabel("Measurement index")
        self.ax_profile.set_ylabel("Elevation [m]")
        legend = self.ax_profile.legend(
            fontsize=7, facecolor=COLORS["card"], edgecolor=COLORS["border"]
        )
        for text in legend.get_texts():
            text.set_color(COLORS["foreground"])

    def on_map_motion(self, event) -> None:
        try:
            lat, lon = self.map_widget.convert_canvas_coords_to_decimal_coords(
                event.x, event.y
            )
        except Exception:
            return
        self.coord_label.configure(text=f"lat {lat:.4f}°,  lon {lon:.4f}°")

    def on_map_left_click(self, coords) -> None:
        if self.data is None:
            return
        lat, lon = coords
        dlat = self.data.lats - lat
        dlon = self.data.lons - lon
        pos = int(np.argmin(dlat * dlat + dlon * dlon))
        self._select_point(pos)

    def _select_point(self, pos: int) -> None:
        data = self.data
        idx = int(data.indices[pos])
        info = compute_point(data.dataset, idx)
        lat = float(data.lats[pos])
        lon = float(data.lons[pos])

        self.map_widget.delete_all_marker()
        self.sel_marker = self.map_widget.set_marker(
            lat,
            lon,
            marker_color_circle=COLORS["primary"],
            marker_color_outside=COLORS["foreground"],
            text_color=COLORS["foreground"],
        )

        self._draw_complex(info)
        self._draw_spectrum(info)
        self._draw_profile(selected_pos=pos)
        self.plot_canvas.draw_idle()

        corrected_elev = info["elevation"] + data.correction
        surf = info["surf_type"]
        surf_text = SURFACE_TYPES.get(surf, "—" if surf is None else str(surf))
        self.stat_vars["index"].set(str(idx))
        self.stat_vars["lat"].set(f"{lat:.5f}")
        self.stat_vars["lon"].set(f"{lon:.5f}")
        self.stat_vars["surf"].set(surf_text)
        self.stat_vars["sar_h"].set(f"{corrected_elev:.1f}")
        self.stat_vars["gate"].set(str(info["detected_gate"]))
        self.stat_vars["h_sat"].set(f"{info['h_satellite']:.1f}")
        self.stat_vars["range"].set(f"{info['r_total']:.1f}")
        self.stat_vars["api_h"].set("loading…")
        self.stat_vars["diff"].set("…")

        threading.Thread(
            target=self._validate_single,
            args=(lat, lon, corrected_elev),
            daemon=True,
        ).start()

    def _validate_single(self, lat: float, lon: float, sar_h: float) -> None:
        try:
            reference = fetch_single_elevation(lat, lon)
            self.root.after(
                0,
                lambda: (
                    self.stat_vars["api_h"].set(f"{reference:.1f}"),
                    self.stat_vars["diff"].set(f"{sar_h - reference:.1f}"),
                ),
            )
        except Exception as exc:
            msg = str(exc)
            self.root.after(
                0,
                lambda: (
                    self.stat_vars["api_h"].set("error"),
                    self.stat_vars["diff"].set("—"),
                    self.status_label.configure(text=f"Point validation: {msg[:50]}"),
                ),
            )

    def _draw_complex(self, info: dict) -> None:
        self.ax_complex.clear()
        self._style_axes(self.ax_complex)
        signal = info["complex_first"]
        self.ax_complex.scatter(signal.real, signal.imag, color=COLORS["chart_2"], s=12)
        self.ax_complex.set_title("Complex signal on the complex plane", fontsize=9)
        self.ax_complex.set_xlabel("Re")
        self.ax_complex.set_ylabel("Im")
        self.ax_complex.axhline(0, color=COLORS["border"], linewidth=0.6)
        self.ax_complex.axvline(0, color=COLORS["border"], linewidth=0.6)

    def _draw_spectrum(self, info: dict) -> None:
        self.ax_spectrum.clear()
        self._style_axes(self.ax_spectrum)
        self.ax_spectrum.plot(
            FREQUENCIES, info["spectrum_first"], color=COLORS["chart_2"], linewidth=1.0
        )
        gate = info["detected_gate"]
        self.ax_spectrum.axvline(
            FREQUENCIES[gate],
            color=COLORS["destructive"],
            linestyle="--",
            linewidth=1.2,
            label=f"detected_gate = {gate}",
        )
        self.ax_spectrum.set_title("Frequency spectrum of the signal", fontsize=9)
        self.ax_spectrum.set_xlabel("Frequency [Hz]")
        self.ax_spectrum.set_ylabel("Amplitude")
        legend = self.ax_spectrum.legend(
            fontsize=7, facecolor=COLORS["card"], edgecolor=COLORS["border"]
        )
        for text in legend.get_texts():
            text.set_color(COLORS["foreground"])


def main() -> None:
    root = tk.Tk()
    SarViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
