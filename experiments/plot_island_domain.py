#!/usr/bin/env python3
"""Does the island look the same on Gorin-Kenyon domains as on the square?

    python3 experiments/plot_island_domain.py island_domain_sample.csv out.png

Measured at n=48, Delta=-3, the same observable on four domains.

The qualitative picture SURVIVES. On every domain the island height is
concentrated on two or three lattice levels, with all of the mass on
{mode-2, mode, mode+2} -- levels differ by 2 because the parity-corrected
height steps by 2 between adjacent plateaus. So the narrow, bounded
fluctuation the discrete-Gaussian statement describes is not special to DWBC.

The WIDTH is not domain-independent. R_lo spans a factor of five across these
four domains at the SAME Delta and the same n, from 0.232 on the square to
0.047 on the notch. That is far outside the error bars and is not explained by
dispersion, since the histograms are equally narrow everywhere.

NOR BY AREA, which is the first objection to make and which this note used to
leave unanswered. "The same n" does not mean the same number of cells: the
four domains hold 2304, 1536, 1280 and 2048 faces, a range of 1.80x. That
matters because R_lo IS size-dependent -- experiments/plot_shift_size.py
measures, on the CSVs shipped here, +0.0101 +- 0.0039 (2.6 sigma) at
Delta=-1.5 and +0.0092 +- 0.0049 (1.9 sigma) at Delta=-3.0 going from n=16 to
n=32, a 4x change in cells. Scaled to a 1.80x range -- the shift goes as the
log of the cell count, so by log(1.80)/log(4) = 0.42 -- that allows at most
about 0.004, which is 2% of the 0.185 spread seen here.

(This paragraph used to cite "+0.0088 ... at 3.0 sigma", which is neither of
the two measurements and overstates the better of them. The bound it supports
is unchanged -- 0.0101 scaled by 0.42 is still 0.004 -- but the number was
not one plot_shift_size produces, and it is pinned to that script's own
output now.)

The ordering settles it more directly than the bound does. If area drove
R_lo, R_lo would rise with cell count, as it does with n on the square. It
does not: the two LARGEST domains sit at the two extremes --

    square      2304 cells    R_lo 0.232   (highest)
    notch       2048 cells    R_lo 0.047   (lowest)
    staircase   1536 cells    R_lo 0.208
    plus        1280 cells    R_lo 0.083

-- so the spread is not ordered by area in any direction, and the domain
dependence is a property of the shape.

One caveat the figure shows rather than hides: the island is only
well-separated on some of these domains. Panel (b) is the ratio of the largest
c-component to the runner-up. On the square and the notch it is 45-51x, so
"the island" is unambiguous. On the plus it is 2.6x and on the staircase 1.1x
-- there the largest component barely beats the second, and the staircase's
three lobes can each host their own, so which one is measured may change from
sample to sample. The staircase number is reported, not trusted.

The notch matters most for the conclusion: its island IS cleanly separated
(50.6x) and its R_lo is still a factor of five FROM the square's -- 0.047
against 0.232, the two ends of the whole spread -- so the domain dependence
is not an artifact of ambiguous component selection. (This read "five times
the square's", which has the ratio upside down: the notch is the low end, at
one fifth of the square, not five times it. The argument is the same either
way, since what it needs is that the two extremes of the spread are both
cleanly separated.)
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"
CLEAN = 5.0          # separation above which the island is unambiguous


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color="#e8e7e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5d0")
    ax.tick_params(colors=INK2, labelsize=9.5)


def main(src, dst):
    rows = list(csv.DictReader(open(src)))
    name = [r["domain"] for r in rows]
    R = np.array([float(r["R_lo"]) for r in rows])
    se = np.array([float(r["se_lo"]) for r in rows])
    sep = np.array([float(r["separation"]) for r in rows])
    x = np.arange(len(rows))
    col = [BLUE if s >= CLEAN else ORANGE for s in sep]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.7), dpi=195,
                                   facecolor=SURFACE, layout="constrained")
    for ax in (axA, axB):
        style(ax)

    axA.bar(x, R, yerr=se, width=0.6, color=col, edgecolor=SURFACE,
            linewidth=2, capsize=4, error_kw=dict(ecolor=INK2, elinewidth=1.4),
            zorder=3)
    for xi, r, s in zip(x, R, se):
        axA.annotate(f"{r:.3f}", (xi, r + s), textcoords="offset points",
                     xytext=(0, 5), ha="center", color=INK2, fontsize=9.5)
    axA.set_xticks(x); axA.set_xticklabels(name)
    axA.set_ylabel("$P(\\mathrm{mode}-2)\\,/\\,P(\\mathrm{mode})$",
                   color=INK, fontsize=11)
    axA.set_title("(a)  the width is domain-dependent", color=INK,
                  fontsize=12, loc="left", pad=10)
    axA.set_ylim(0, max(R + se) * 1.22)

    axB.bar(x, sep, width=0.6, color=col, edgecolor=SURFACE, linewidth=2,
            zorder=3)
    axB.axhline(CLEAN, color=MUTED, linewidth=1.3, linestyle=(0, (5, 4)),
                zorder=2)
    axB.set_yscale("log")
    axB.set_xticks(x); axB.set_xticklabels(name)
    axB.set_ylabel("largest c-component / runner-up", color=INK, fontsize=11)
    axB.set_title("(b)  is there one island at all?", color=INK, fontsize=12,
                  loc="left", pad=10)
    for xi, s in zip(x, sep):
        axB.annotate(f"{s:.1f}x", (xi, s), textcoords="offset points",
                     xytext=(0, 5), ha="center", color=INK2, fontsize=9.5)
    axB.set_ylim(0.8, max(sep) * 3.2)
    # anchored INSIDE the axes: at the boundary it was silently clipped
    axB.annotate("below this the largest component\nbarely beats the second",
                 (1.5, CLEAN), textcoords="offset points",
                 xytext=(0, 8), ha="center", va="bottom", color=INK2,
                 fontsize=9.0, linespacing=1.4, annotation_clip=False)

    fig.text(0.006, -0.085,
             "Island height on $\\Omega^{X,Y,I}$, $n$=48, $\\Delta=-3$, "
             "$a=b=1$, $c=\\sqrt{2(1-\\Delta)}$.  Stationary chains from "
             "independent exact CFTP starts; errors are chain-to-chain batch "
             "means.  Blue: the island is unambiguous.  Orange: it is not, and "
             "the number is reported rather than relied on.",
             color=MUTED, fontsize=8.2)
    fig.savefig(dst, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {dst}")
    for nm, r, s, sp in zip(name, R, se, sep):
        print(f"    {nm:10s} R_lo {r:.4f}+-{s:.4f}   separation {sp:6.1f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "island_domain_sample.csv",
         sys.argv[2] if len(sys.argv) > 2 else "island_domain.png")
