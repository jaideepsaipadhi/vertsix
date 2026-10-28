#!/usr/bin/env python3
"""P(island exists) against Delta, one curve per lattice size.

    python3 experiments/plot_island_exists.py island_exists.csv out.png

The predicate is the crossing criterion in experiments/island_exists.py: a
constant-parity-corrected-height cluster spanning the central window. Panel
(a) is the probability itself with binomial intervals; panel (b) is the mean
widest extent, a continuous companion that shows HOW the marginal cases fail
rather than only that they do.

What to read off panel (a): whether the curves steepen as N grows, and where
they cross. A transition sharpening towards Delta = -1 is the prediction; a
family of curves that simply shift, or that steepen about some other point,
is not that, and the figure is drawn so either outcome is visible rather than
only the expected one. The crossing point of successive sizes is marked
because that, not the midpoint of any single curve, is the finite-size
estimate of the transition.
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"
ACCENT = "#eb6834"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e8e7e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5d0")
    ax.tick_params(colors=INK2, labelsize=9.5)


def main(src, dst):
    rows = list(csv.DictReader(open(src)))
    sizes = sorted({int(r["n"]) for r in rows})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=195,
                                   facecolor=SURFACE, layout="constrained")
    for ax in (axA, axB):
        style(ax)
        ax.axvline(-1.0, color=ACCENT, linewidth=1.3, linestyle=(0, (4, 3)), zorder=1)

    for k, n in enumerate(sizes):
        sel = sorted((r for r in rows if int(r["n"]) == n),
                     key=lambda r: float(r["delta"]))
        d = np.array([float(r["delta"]) for r in sel])
        p = np.array([float(r["p_exists"]) for r in sel])
        se = np.array([float(r["se"]) for r in sel])
        wd = np.array([float(r["mean_width"]) for r in sel])
        col = RAMP[k % len(RAMP)]
        axA.fill_between(d, np.clip(p - se, 0, 1), np.clip(p + se, 0, 1),
                         color=col, alpha=0.16, linewidth=0, zorder=2)
        axA.plot(d, p, "-o", color=col, linewidth=2, markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
                 label=f"N = {n}")
        axB.plot(d, wd, "-o", color=col, linewidth=2, markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
                 label=f"N = {n}")

    axA.annotate(r"$\Delta = -1$", (-1.0, 1.02), color=ACCENT, fontsize=9.5,
                 ha="center", va="bottom")
    axA.set_ylim(-0.04, 1.08)
    axA.set_xlabel(r"$\Delta$", color=INK2, fontsize=10.5)
    axA.set_ylabel("P(island exists)", color=INK2, fontsize=10.5)
    axA.set_title("(a)  does a constant-height cluster cross?",
                  color=INK, fontsize=11.5, loc="left", pad=9)
    axA.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="center left")

    axB.set_xlabel(r"$\Delta$", color=INK2, fontsize=10.5)
    axB.set_ylabel("mean widest extent / window", color=INK2, fontsize=10.5)
    axB.set_title("(b)  how far the best cluster gets",
                  color=INK, fontsize=11.5, loc="left", pad=9)
    axB.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")

    fig.savefig(dst, facecolor=SURFACE)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
