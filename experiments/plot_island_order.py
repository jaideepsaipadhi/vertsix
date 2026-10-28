#!/usr/bin/env python3
"""The island order parameter and its decay exponent.

    python3 experiments/plot_island_order.py island_order.csv out.png

A(Delta, N) is the area of the largest solid-c component over N^2, with no
window and no span threshold in its definition -- see experiments/
island_order.py for why the crossing probability needed replacing.

Panel (a) is A itself, one curve per size. Panel (b) is the fitted exponent
in A ~ N^-theta, which is what carries the claim: A ~ N^-theta means the core
covers N^(2-theta) faces, so theta = 0 is a macroscopic island and theta = 1
is a FILAMENT, an object of length N and bounded width. Measured, theta runs
from about 1 at Delta >= -1 to 0.16 at Delta = -3. So above the transition
the solid-c set has only filaments, and below it the core thickens towards a
blob.

What the figure does NOT show is a sharp jump at Delta = -1. theta decreases
steadily across the whole range at these sizes, and the three sizes span a
factor of two, so each theta carries an error of order 0.2. The ORDERING is
solid; the location of any threshold is not, and the panel is drawn with
theta = 0 and theta = 1 marked so that a reader can see how far the deepest
point still sits from a genuinely macroscopic island.

One point is not to be trusted: theta at Delta = -0.5 comes out 0.61, BELOW
the 1.15 at Delta = -0.75, which breaks the monotone trend. That is a floor
effect, not structure. At Delta = -0.5 the largest solid component is 13
faces at N = 64 and 14 at N = 48 -- a handful of cells, the smallest object
the erosion can leave behind -- so A cannot decay any further and the fitted
exponent is measuring the floor rather than the physics. The same caution
applies to Delta = -0.75 at 14 faces. Only the points from Delta = -1
downwards, where the core is tens to hundreds of faces, carry a meaningful
exponent.
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
    deltas = sorted({float(r["delta"]) for r in rows})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=195,
                                   facecolor=SURFACE, layout="constrained")
    for ax in (axA, axB):
        style(ax)
        ax.axvline(-1.0, color=ACCENT, linewidth=1.3, linestyle=(0, (4, 3)),
                   zorder=1)

    for k, n in enumerate(sizes):
        sel = sorted((r for r in rows if int(r["n"]) == n),
                     key=lambda r: float(r["delta"]))
        d = np.array([float(r["delta"]) for r in sel])
        a = np.array([float(r["area_frac"]) for r in sel])
        se = np.array([float(r["se"]) for r in sel])
        col = RAMP[k % len(RAMP)]
        axA.fill_between(d, a - se, a + se, color=col, alpha=0.18,
                         linewidth=0, zorder=2)
        axA.plot(d, a, "-o", color=col, linewidth=2, markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
                 label=f"N = {n}")

    axA.set_yscale("log")
    axA.set_xlabel(r"$\Delta$", color=INK2, fontsize=10.5)
    axA.set_ylabel(r"$A$ = largest solid core / $N^2$", color=INK2,
                   fontsize=10.5)
    axA.set_title("(a)  the order parameter", color=INK, fontsize=11.5,
                  loc="left", pad=9)
    axA.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")

    th = []
    for dd in deltas:
        a = [float(next(r for r in rows if float(r["delta"]) == dd
                        and int(r["n"]) == n)["area_frac"]) for n in sizes]
        th.append(-np.polyfit(np.log(sizes), np.log(a), 1)[0])
    th = np.array(th)
    dd = np.array(deltas)

    axB.axhline(0.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    axB.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    axB.plot(dd, th, "-o", color=ACCENT, linewidth=2, markersize=5,
             markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
    axB.annotate("macroscopic island", (dd.min(), 0.0),
                 textcoords="offset points", xytext=(6, 6), ha="left",
                 color=MUTED, fontsize=9)
    axB.annotate("filament (area ~ N)", (dd.min(), 1.0),
                 textcoords="offset points", xytext=(6, 6), ha="left",
                 color=MUTED, fontsize=9)
    axB.set_ylim(-0.25, 1.45)
    axB.set_xlabel(r"$\Delta$", color=INK2, fontsize=10.5)
    axB.set_ylabel(r"exponent $\theta$ in $A \sim N^{-\theta}$", color=INK2,
                   fontsize=10.5)
    axB.set_title(r"(b)  filament above, blob below", color=INK,
                  fontsize=11.5, loc="left", pad=9)

    fig.savefig(dst, facecolor=SURFACE)
    print(f"wrote {dst}")
    for d, t in zip(deltas, th):
        print(f"  Delta={d:+5.2f}  theta = {t:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
