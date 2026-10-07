"""Shared publication style for all manuscript figures (npj CAS).

Calibrated against npj Climate and Atmospheric Science exemplars
(paper/figure_refs/, CC-BY) and Nature Portfolio artwork specs:
183 mm double-column width, sans-serif 5-8 pt at final size, bold
lowercase panel labels, colorblind-safe palette, RGB, >=300 dpi.

Every figure script imports this module and calls apply() before
plotting. One palette, one typography, one spine style across the
entire figure set — consistency is the main quality lever.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

FIGDIR = Path(__file__).resolve().parents[1] / "paper" / "figures"

# Okabe-Ito colorblind-safe anchors, with semantic roles kept constant
# across ALL figures: the deployed 5-term closure is always HERO blue,
# the uncorrected SLUCM always GREY, the identification-stage MLP always SKY.
C = {
    "grey":   "#8a8a8a",   # uncorrected / de-emphasised
    "hero":   "#0072B2",   # deployed 5-term closure (the deliverable)
    "sky":    "#56B4E9",   # MLP identification stage
    "teal":   "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "verm":   "#D55E00",   # sparing accent for external references/thresholds
    "dark":   "#333333",
    "light":  "#e9e9e9",
}

# Same site -> same colour in every figure it appears in.
SITE_COLOURS = {
    "CA-Sunset":    C["teal"],
    "FI-Kumpula":   C["hero"],
    "US-Baltimore": C["orange"],
}

# Design widths (inches). W_DOUBLE targets the REVIEW manuscript, where
# figures print at \textwidth = 371 pt = 5.13 in (sn-jnl): designing at
# 5.15 in keeps the 6.5-8 pt fonts at true size in the compiled PDF
# (designing at the 183 mm production width shrank all text by ~0.7x).
W_DOUBLE = 5.15
W_SINGLE = 3.5


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.edgecolor": C["dark"],
        "axes.labelcolor": C["dark"],
        "xtick.color": C["dark"],
        "ytick.color": C["dark"],
        "text.color": C["dark"],
        "legend.frameon": False,
        "axes.grid": False,
        "grid.color": "#d5d5d5",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # keep text editable in the PDF (journal requirement)
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


def panel_label(ax, letter: str, dx: float = -0.10, dy: float = 1.04) -> None:
    """Nature convention: bold lowercase letter, no parentheses, outside top-left."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="left", va="bottom")


def ygrid(ax) -> None:
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def save_dual(fig, stem: str) -> tuple[Path, Path]:
    png = FIGDIR / f"{stem}.png"
    pdf = FIGDIR / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    return png, pdf
