#!/usr/bin/env python3
"""The deterministic shift as a function of Delta.

    python3 experiments/plot_shift_delta.py shift_delta_chain_sample.csv out.png

Panel (a) is the result: mu and sigma across the antiferroelectric phase, both
drifting steadily with depth and neither settling. They share one axis because
they happen to occupy the same range in magnitude -- a second y-scale would
invent a comparison that is not in the data.

Panel (b) is why no limit is claimed. The marginal quantities 1/sigma^2 and
2mu/sigma^2 are each fit well by a + b ln c (chi2 5.3 and 6.7 on 4 dof), and
their fitted slopes come out equal and opposite to within 0.2 sigma, which
would force mu -> -1/2. But that statement is exactly "R_lo is constant", and
R_lo is 220 chi2 on 5 dof away from constant with a clear minimum near
Delta=-3. The slopes only look equal because each slope's own error, 0.12, is
far larger than the structure their DIFFERENCE contains. Two marginal fits can
both pass and still be wrong together.

SIZE DEPENDENCE -- read experiments/plot_shift_size before quoting numbers
from here. This sweep is at n=16, and repeating it at n=32 shows the SHAPE
survives (the minimum sits at Delta=-3 at both sizes) while the LEVEL does
not: R_lo at n=32 sits above n=16 by a pooled +0.0088 +- 0.0029, 3.0 sigma.
mu tracks closely across the two sizes and sigma a little less so. The
Delta-trend in mu spans about 0.06, an order of magnitude larger than the
size effect, so the trend is safe; the fourth decimal of mu and sigma at one n
is not a universal number.

Panel (c) is the same lesson learned the hard way. A saturating law a + b/|D|
fit to |D| in [2,4] matched those points to 0.002 and then missed |D|=6 and 8
by 3-9 sigma, the error growing with extrapolation distance.
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical pair, light
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"


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
    rows = sorted(csv.DictReader(open(src)), key=lambda r: float(r["delta"]))
    D = np.array([float(r["delta"]) for r in rows])
    aD = np.abs(D)
    Rlo = np.array([float(r["R_lo"]) for r in rows])
    slo = np.array([float(r["se_lo"]) for r in rows])
    Rhi = np.array([float(r["R_hi"]) for r in rows])
    shi = np.array([float(r["se_hi"]) for r in rows])
    mu = np.array([float(r["mu"]) for r in rows])
    sg = np.array([float(r["sigma"]) for r in rows])
    inv, y = -np.log(Rlo * Rhi), np.log(Rhi / Rlo)
    se = np.hypot(slo / Rlo, shi / Rhi)
    # propagate to mu and sigma
    s_sg = sg ** 3 * se / 2.0
    s_mu = np.abs(mu) * np.hypot(se / np.abs(y), se / inv)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=195,
                                        facecolor=SURFACE, layout="constrained")
    for ax in (axA, axB, axC):
        style(ax)

    # ---- (a) mu and sigma
    axA.errorbar(D, mu, yerr=s_mu, fmt="-o", color=BLUE, linewidth=2,
                 markersize=7, capsize=3, elinewidth=1.4,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4,
                 label="$\\mu$  (the shift)")
    axA.errorbar(D, sg, yerr=s_sg, fmt="-s", color=ORANGE, linewidth=2,
                 markersize=7, capsize=3, elinewidth=1.4,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4,
                 label="$\\sigma$")
    axA.axhline(0, color=MUTED, linewidth=1.0)
    axA.set_xlabel("$\\Delta$", color=INK, fontsize=11)
    axA.set_ylabel("discrete-Gaussian parameter", color=INK, fontsize=11)
    axA.set_title("(a)  the shift and the width", color=INK, fontsize=12,
                  loc="left", pad=10)
    leg = axA.legend(frameon=False, fontsize=10, loc="center right")
    for t in leg.get_texts():
        t.set_color(INK2)

    # ---- (b) R_lo, the structure that forbids the extrapolation
    axB.errorbar(D, Rlo, yerr=slo, fmt="-o", color=BLUE, linewidth=2,
                 markersize=7, capsize=3, elinewidth=1.4,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
    i = int(np.argmin(Rlo))
    axB.annotate(f"minimum near $\\Delta\\approx{D[i]:.0f}$",
                 (D[i], Rlo[i]), textcoords="offset points", xytext=(10, 62),
                 ha="left", color=ORANGE, fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.3))
    axB.set_xlabel("$\\Delta$", color=INK, fontsize=11)
    axB.set_ylabel("$P(k=-1)\\,/\\,P(k=0)$", color=INK, fontsize=11)
    axB.set_title("(b)  why no limit is claimed", color=INK, fontsize=12,
                  loc="left", pad=10)
    axB.set_xlim(-8.6, -1.2)
    axB.set_ylim(Rlo.min() - 0.0075, Rlo.max() + 0.0035)
    axB.annotate("constant would force $\\mu\\to-1/2$;\n"
                 "this is $\\chi^2$=220 on 5 d.o.f. from constant",
                 (0.5, 0.03), xycoords="axes fraction", ha="center",
                 va="bottom", color=INK2, fontsize=9.0, linespacing=1.5)

    # ---- (c) the failed extrapolation, on 1/sigma^2
    k = (aD >= 2) & (aD <= 4)
    X = np.vstack([np.ones(k.sum()), 1 / aD[k]]).T
    co, *_ = np.linalg.lstsq(X, inv[k], rcond=None)
    grid = np.linspace(1.4, 8.6, 200)
    axC.plot(-grid, co[0] + co[1] / grid, color=MUTED, linewidth=1.6,
             linestyle=(0, (5, 4)), zorder=2,
             label="$a+b/|\\Delta|$ fit on $2\\leq|\\Delta|\\leq4$")
    axC.errorbar(D[k], inv[k], yerr=se[k], fmt="o", color=BLUE, markersize=7,
                 capsize=3, elinewidth=1.4, markeredgecolor=SURFACE,
                 markeredgewidth=1.4, zorder=4, label="points it was fitted to")
    out = ~k
    axC.errorbar(D[out], inv[out], yerr=se[out], fmt="s", color=ORANGE,
                 markersize=7, capsize=3, elinewidth=1.4,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=5,
                 label="held out")
    for j in np.where(aD >= 6)[0]:
        pred = co[0] + co[1] / aD[j]
        axC.annotate(f"{abs(inv[j]-pred)/se[j]:.0f}$\\sigma$ off",
                     (D[j], inv[j]), textcoords="offset points",
                     xytext=(0, 13), ha="center", va="bottom",
                     color=ORANGE, fontsize=9.5)
    axC.set_xlabel("$\\Delta$", color=INK, fontsize=11)
    axC.set_ylabel("$1/\\sigma^{2}$", color=INK, fontsize=11)
    axC.set_title("(c)  a law that fitted, then failed", color=INK,
                  fontsize=12, loc="left", pad=10)
    leg = axC.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK2)
    axC.set_xlim(-8.9, -1.4)
    axC.set_ylim(inv.min() - 0.55, inv.max() + 0.75)

    fig.text(0.006, -0.055,
             "Island height on DWBC, $n$=16, $a=b=1$, $c=\\sqrt{2(1-\\Delta)}$.  "
             "Stationary chains started from exact CFTP samples; 32 chains "
             "x 10000 observations per point, thinning set from the measured "
             "$\\tau_{\\rm int}$.  Errors are chain-to-chain batch means, "
             "propagated to $\\mu$ and $\\sigma$.  Whole sweep: 25 minutes.",
             color=MUTED, fontsize=8.2)
    fig.savefig(dst, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "shift_delta_chain_sample.csv",
         sys.argv[2] if len(sys.argv) > 2 else "shift_delta.png")
