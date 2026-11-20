"""Plot Styling"""

import matplotlib as mpl

PLOT_COLORS = {
    "light": {
        # signal
        "signal": "darkslategray",
        "trend": "#a569bd",
        "envelope": "orange",
        "detrend": "black",
        "fourier": "cadetblue",
        # spectra
        "cmap": "YlGnBu_r",
        "ridge": "crimson",
        "coi": "0.6",
        "hist": "cadetblue",
        # readouts
        "period": "cornflowerblue",
        "phase": "#e05353",
        "amplitude": "#111111",
        "power": "cadetblue",
        "powerkde": "#0a0a0aa0"
    },
    "dark": {
        # signal
        "signal": "#7fc4c4",
        "trend": "#d2a0e6",
        "envelope": "#ffb347",
        "detrend": "#e8e8e8",
        "fourier": "#88c8d0",
        # spectra
        "cmap": "magma",
        "ridge": "#1a2a26",
        "coi": "0.85",
        "hist": "powderblue",
        # readouts
        "period": "#7aa8f8",
        "phase": "#ff6e6e",
        "amplitude": "#e8e8e8",
        "power": "#88c8d0",
        "powerkde": "#e6e6e6a0"
    },
}

_dark = "#21252b"  # ashen gray
DARK_THEME_COLORS = {
    "figure.facecolor": _dark,
    "axes.facecolor": _dark,      # Kills the pitch-black void!
    "savefig.facecolor": _dark,
    "axes.edgecolor": "#555555",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#3e3e42",
    "grid.alpha": 0.6,
    "legend.facecolor": _dark,
    "legend.edgecolor": "#555555",
}


# --- Set Light/Dark Styling ---

COLORS = dict(PLOT_COLORS["light"])

def set_dark_mode(enabled: bool) -> None:

    theme_key = "dark" if enabled else "light"
    COLORS.clear()
    COLORS.update(PLOT_COLORS[theme_key])
    if enabled:
        mpl.pyplot.style.use("dark_background")
        mpl.rcParams.update(DARK_THEME_COLORS)
    else:
        mpl.pyplot.style.use("default")
