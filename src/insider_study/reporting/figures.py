"""Static figures for the report (matplotlib, validated reference palette, light mode)."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots validate all-pairs
NEUTRAL = "#8a8984"


def _style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _fig() -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Nudge end-of-line label positions apart so they never overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for a, b in pairwise(order):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def cumulative(net: pd.DataFrame, cols: list[str], bench: str, title: str, path: Path) -> Path:
    fig, ax = _fig()
    wealth = (1 + net[[*cols, bench]].fillna(0.0)).cumprod()
    names = [*cols, bench]
    for i, c in enumerate(cols):
        ax.plot(wealth.index, wealth[c], color=SERIES[i], linewidth=2, label=c)
    ax.plot(wealth.index, wealth[bench], color=NEUTRAL, linewidth=1.5, linestyle="--", label=bench)
    ends = [float(wealth[c].iloc[-1]) for c in names]
    span = float(wealth.max().max() - wealth.min().min())
    for c, y in zip(names, _spread_labels(ends, 0.05 * span), strict=True):
        ax.annotate(
            c,
            (wealth.index[-1], y),
            color=INK if c != bench else INK_2,
            fontsize=8,
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
        )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    _style(ax, title, "Crecimiento de 1 USD (neto de costes)")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def capacity(curves: dict[str, dict[str, Any]], title: str, path: Path) -> Path:
    fig, ax = _fig()
    for i, (name, data) in enumerate(curves.items()):
        aum = [int(a) for a in data["curve"]]
        alpha = [100 * v["alpha_net_annual"] for v in data["curve"].values()]
        ax.plot(
            aum,
            alpha,
            color=SERIES[i % 3] if i < 3 else NEUTRAL,
            linewidth=2,
            marker="o",
            markersize=4,
            label=name,
        )
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    _style(ax, title, "Alpha FF5+UMD neto anual (%)")
    ax.set_xlabel("AUM (USD, escala log)", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def bars(
    values: dict[str, float], errors: dict[str, float], title: str, ylabel: str, path: Path
) -> Path:
    fig, ax = _fig()
    names = list(values)
    vals = [100 * values[n] for n in names]
    errs = [100 * 1.96 * errors.get(n, 0.0) for n in names]
    ax.bar(
        names,
        vals,
        color=SERIES[0],
        width=0.6,
        yerr=errs,
        error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3},
    )
    ax.axhline(0, color=INK_2, linewidth=0.8)
    for i, v in enumerate(vals):
        ax.annotate(
            f"{v:.1f}",
            (i, v),
            xytext=(0, 3 if v >= 0 else -10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _style(ax, title, ylabel)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def decay(yearly: dict[str, dict[str, float]], title: str, path: Path) -> Path:
    fig, ax = _fig()
    for i, (name, series) in enumerate(yearly.items()):
        years = [int(y) for y in series]
        ax.plot(
            years,
            [100 * v for v in series.values()],
            color=SERIES[i],
            linewidth=2,
            marker="o",
            markersize=4,
            label=name,
        )
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    _style(ax, title, "BHAR factorial medio a 6 meses (%)")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
