#!/usr/bin/env python3
"""P(island exists) against Delta, one curve per lattice size.

    python3 experiments/plot_island_exists.py island_exists.csv out.png

The predicate is the crossing criterion in experiments/island_exists.py: a
cluster of SOLID c-faces -- c-faces all of whose eight neighbours are also
c-faces -- spanning the central N/5 window. Panel
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

That last sentence was false for as long as it stood: nothing in this file
computed a crossing. The only vertical line was Delta = -1 -- the PREDICTION
-- so the figure marked the expected answer and not the measured one, which
is precisely the bias the paragraph above claims to guard against. The
crossings are now computed and drawn, in ink rather than the accent colour so
they cannot be confused with the prediction, and they do not support it:

    N=32 vs N=48    crossing at Delta = -1.125
    N=48 vs N=64    crossing at Delta = -1.357

The crossing moves DEEPER as N grows rather than sharpening onto -1, which
agrees with island_order.py's account of this predicate as biased deep. It is
not established, though: the P differences in the cells that BRACKET each
crossing are 0.025 and 0.050 against standard errors of 0.038 and 0.047, so
both sit inside the noise, and the script says so per crossing rather than
leaving the drift to be read off the picture. The bracketing cells are the
right comparison -- at the crossing the difference is zero by construction,
so judging it by the largest difference anywhere on the sweep (which is at
Delta = -1.5, where the curves are far apart) would make any crossing look
well-determined.
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


def curve_crossings(rows, n1, n2):
    """Delta values where the P curves for two sizes cross, by interpolation.

    This file's docstring promised "the crossing point of successive sizes is
    marked" and nothing in it computed one: the only vertical line was the
    Delta = -1 PREDICTION, so the figure marked the expected answer and not
    the measured one -- the exact bias the surrounding paragraph was written
    to guard against.

    Returns a list of (Delta, bracket_diff, bracket_se) per crossing. The
    quantity to judge a crossing by is the P difference in the two cells that
    BRACKET it, not the largest difference anywhere on the curve -- at the
    crossing itself the difference is zero by construction, so a global
    maximum taken from the far end of the sweep makes any crossing look
    well-determined. On the committed sample the crossings are at -1.125
    (N=32 vs 48) and -1.357 (N=48 vs 64), moving DEEPER as N grows rather
    than sharpening onto -1 -- but the bracketing differences are 0.025 to
    0.038 against combined standard errors of the same size, so the drift is
    suggestive and not established.
    """
    def curve(n):
        sel = sorted((r for r in rows if int(r["n"]) == n),
                     key=lambda r: float(r["delta"]))
        return (np.array([float(r["delta"]) for r in sel]),
                np.array([float(r["p_exists"]) for r in sel]),
                np.array([float(r["se"]) for r in sel]))

    d1, p1, s1 = curve(n1)
    d2, p2, s2 = curve(n2)
    if d1.size != d2.size or not np.allclose(d1, d2):
        return [], 0.0, 0.0
    diff = p1 - p2
    se = np.hypot(s1, s2)
    sgn = np.sign(diff)
    out = []
    for k in range(1, diff.size):
        if sgn[k] == 0 or sgn[k - 1] == 0 or sgn[k] == sgn[k - 1]:
            continue
        t = diff[k - 1] / (diff[k - 1] - diff[k])
        out.append((float(d1[k - 1] + t * (d1[k] - d1[k - 1])),
                    float(max(abs(diff[k - 1]), abs(diff[k]))),
                    float(max(se[k - 1], se[k]))))
    return out


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

    axA.annotate(r"$\Delta = -1$ (predicted)", (-1.0, 1.02), color=ACCENT,
                 fontsize=9.5, ha="center", va="bottom")

    # the MEASURED crossings, which the docstring promised and the figure
    # never drew. Marked in ink, not in the accent colour, so they are not
    # mistaken for the prediction.
    marks = []
    for i in range(len(sizes) - 1):
        for x, bd, bse in curve_crossings(rows, sizes[i], sizes[i + 1]):
            marks.append((x, sizes[i], sizes[i + 1], bd, bse))
            axA.axvline(x, color=INK2, linewidth=1.0, linestyle=(0, (1, 2)),
                        zorder=1)
    if marks:
        axA.annotate("crossings of successive $N$",
                     (float(np.mean([m[0] for m in marks])), 0.62),
                     textcoords="offset points", xytext=(4, 0), ha="left",
                     color=INK2, fontsize=8.8)
    axA.set_ylim(-0.04, 1.08)
    axA.set_xlabel(r"$\Delta$", color=INK2, fontsize=10.5)
    axA.set_ylabel("P(island exists)", color=INK2, fontsize=10.5)
    axA.set_title("(a)  does a solid block of c-vertices span the middle?",
                  color=INK, fontsize=11.5, loc="left", pad=9)
    axA.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="center left")

    axB.set_xlabel(r"$\Delta$", color=INK2, fontsize=10.5)
    axB.set_ylabel("mean widest extent / window", color=INK2, fontsize=10.5)
    axB.set_title("(b)  how far the widest solid block gets",
                  color=INK, fontsize=11.5, loc="left", pad=9)
    axB.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")

    fig.savefig(dst, facecolor=SURFACE)
    print(f"wrote {dst}")
    for x, n1, n2, bd, bse in marks:
        verdict = ("inside the noise" if bd <= 1.5 * bse
                   else "larger than the noise")
        print(f"  crossing of N={n1} and N={n2} at Delta = {x:+.3f}"
              f"   (bracketing |P1-P2| {bd:.4f} vs s.e. {bse:.4f}"
              f" -- {verdict})")
    if len(marks) >= 2 and marks[-1][0] < marks[0][0]:
        print("  the crossing moves DEEPER as N grows, not towards -1")


if __name__ == "__main__":
    main(*sys.argv[1:3])
