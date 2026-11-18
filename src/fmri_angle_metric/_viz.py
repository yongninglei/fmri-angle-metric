"""Shared plotting palette and helpers for the cross-session summary figures.

Categorical hues are assigned in fixed order and never cycled, so a session
keeps its colour no matter how many sessions are plotted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

# ── palette ──────────────────────────────────────────────────────────────────
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

# Diverging blue <-> red with a neutral gray midpoint, for correlations.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#184f95", "#2a78d6", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#a52422"],
)

# Sequential single-hue blue, light -> dark, for magnitudes such as degrees.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq",
    ["#e8f1fd", "#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#184f95"],
)


def _text_on(cmap, norm_value: float) -> str:
    """Pick ink or white for a label sitting on a coloured cell."""
    r, g, b = to_rgb(cmap(norm_value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 0.55 else INK


def style_axes(ax) -> None:
    """Recessive grid and axes; the data carries the emphasis."""
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def titles(ax, title: str, subtitle: str) -> None:
    """Bold title with a muted subtitle below it, clear of the axes."""
    ax.set_title(title, fontsize=11, color=INK, pad=26, loc="left", weight="bold")
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5,
            color=INK_MUTED, va="bottom", ha="left")


def heatmap(ax, df: pd.DataFrame, title: str, subtitle: str, *,
            vmin: float, vmax: float, cmap, fmt: str = "{:.2f}",
            xlabel: str = "session"):
    """Session x session matrix with every cell labelled.

    Labelling each cell means the matrix doubles as its own table view, so a
    reader never has to judge a value by colour alone.
    """
    data = df.values.astype(float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Rows and columns are labelled independently — the matrix is not always
    # square (ROI x session, ROI x pair).
    row_labels = [str(s) for s in df.index]
    col_labels = [str(s) for s in df.columns]
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=9, color=INK_MUTED)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=9, color=INK_MUTED)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)

    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isfinite(data[i, j]):
                continue
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                    fontsize=8, color=_text_on(cmap, float(norm(data[i, j]))))

    titles(ax, title, subtitle)
    return im


def series_legend(fig, labels: list[str], colors: dict[str, str],
                  prefix: str = "") -> None:
    """Figure-level legend, outside the panels so it cannot cover data."""
    import matplotlib.pyplot as plt

    handles = [plt.Line2D([], [], color=colors[k], marker="o", markersize=7,
                          linewidth=2, label=f"{prefix}{k}") for k in labels]
    fig.legend(handles=handles, loc="upper right", frameon=False,
               ncol=len(labels), fontsize=9.5, labelcolor=INK_MUTED,
               bbox_to_anchor=(0.995, 1.0))


def session_legend(fig, sessions: list[str], colors: dict[str, str]) -> None:
    series_legend(fig, sessions, colors, prefix="ses-")
