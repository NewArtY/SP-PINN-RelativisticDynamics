"""Shared publication-quality Matplotlib styling.

A colour-blind-friendly palette is used consistently across every figure:
    Boris    -- orange   (#E69F00)
    RK4      -- blue     (#0072B2)
    SP-PINN  -- green    (#009E73)
    RK8 ref  -- black dashed
"""

import matplotlib as mpl

COLORS = {
    "Boris": "#E69F00",
    "RK4": "#0072B2",
    "SP-PINN": "#009E73",
    "RK8": "#000000",
    "PINN": "#CC79A7",
}
MARKERS = {"Boris": "s", "RK4": "o", "SP-PINN": "^", "RK8": None, "PINN": "D"}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.9,
        "figure.constrained_layout.use": True,
    })
