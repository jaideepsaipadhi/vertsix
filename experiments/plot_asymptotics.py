#!/usr/bin/env python3
"""Figure: the deep-antiferroelectric slowdown is not a power law.

    python3 experiments/plot_asymptotics.py asym.csv asym.png

Panel (a) fits tau ~ n^alpha on n=32..48 ONLY and extends that fit as a dashed
line to n=80. A power law would keep tracking it. Delta=-4 leaves it; the
Delta=-1 control does not.

Panel (b) is the same statement without a fitted reference: the local exponent
between successive sizes. Constant in n means a power law.

The control is the part that makes this an argument rather than an
observation. Coalescence time is heavy-tailed and censoring biases exactly the
slowest runs, so a rising exponent at one Delta could be the measurement
failing rather than the model. Running the identical pipeline at Delta=-1,
where the exponent is known to be flat, tests that: it stays flat (+0.21 +-
0.19, p=0.26) while Delta=-4 rises (+2.06 +- 0.68, p=0.002). No run at either
Delta was censored.
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"    # validated categorical pair, light mode
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"
LOW, HIGH = (32, 48), (56, 80)


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
    by = {}
    for r in csv.DictReader(open(src)):
        by.setdefault(float(r["delta"]), []).append(
            (int(r["n"]), float(r["median_tau"]), float(r["lo"]),
             float(r["hi"]), int(r["censored"])))
    for v in by.values():
        v.sort()
    order = sorted(by, reverse=True)                  # control first
    colour = {order[0]: BLUE, order[-1]: ORANGE}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=200,
                                   facecolor=SURFACE)
    for ax in (axL, axR):
        style(ax)

    for d in order:
        v = by[d]
        n = np.array([x[0] for x in v], float)
        t = np.array([x[1] for x in v], float)
        lo = np.array([x[2] for x in v]); hi = np.array([x[3] for x in v])
        c = colour[d]

        axL.errorbar(n, t, yerr=[t - lo, hi - t], fmt="o", color=c,
                     markersize=7, capsize=3, elinewidth=1.4,
                     markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4,
                     label=f"$\\Delta={d:+.0f}$")
        # fit on the LOW range only, then extend
        k = (n >= LOW[0]) & (n <= LOW[1])
        p = np.polyfit(np.log(n[k]), np.log(t[k]), 1)
        axL.plot(n, np.exp(np.polyval(p, np.log(n))), color=c, linewidth=1.5,
                 linestyle=(0, (5, 4)), alpha=0.85, zorder=3)
        axL.annotate(f"$n^{{{p[0]:.2f}}}$ fitted on $n\\leq{LOW[1]}$",
                     (n[-1], np.exp(np.polyval(p, np.log(n[-1])))),
                     textcoords="offset points", xytext=(6, -3), ha="left",
                     va="top", color=c, fontsize=8.8)

        # local exponent between successive sizes, at the geometric midpoint
        sd = (np.log(hi) - np.log(lo)) / (2 * 1.96)
        xm = np.sqrt(n[:-1] * n[1:])
        al = np.log(t[1:] / t[:-1]) / np.log(n[1:] / n[:-1])
        se = np.hypot(sd[1:], sd[:-1]) / np.log(n[1:] / n[:-1])
        axR.errorbar(xm, al, yerr=se, fmt="-o", color=c, linewidth=2,
                     markersize=7, capsize=3, elinewidth=1.4,
                     markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4,
                     label=f"$\\Delta={d:+.0f}$")

    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xticks([32, 40, 48, 56, 64, 80])
    axL.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # a log axis keeps drawing its own minor labels (3x10^1, 6x10^1, ...)
    # straight through the explicit ticks. Caught by comparing TICK LABEL
    # bounding boxes, which an annotation-only overlap check does not see.
    axL.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axL.set_xlabel("$n$", color=INK, fontsize=11)
    axL.set_ylabel("median coalescence time  $\\tau$  (sweeps)",
                   color=INK, fontsize=11)
    axL.set_title("(a)  a power law fitted low, extended high",
                  color=INK, fontsize=12, loc="left", pad=10)
    axL.set_xlim(29, 122)

    axR.set_xscale("log")
    axR.set_xticks([36, 44, 52, 60, 72])
    axR.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axR.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axR.set_xlabel("$n$  (geometric midpoint of the pair)",
                   color=INK, fontsize=11)
    axR.set_ylabel("local exponent  $\\alpha_{\\rm local}$",
                   color=INK, fontsize=11)
    axR.set_title("(b)  a power law keeps this flat",
                  color=INK, fontsize=12, loc="left", pad=10)
    for ax in (axL, axR):
        leg = ax.legend(frameon=False, fontsize=10, loc="upper left")
        for tx in leg.get_texts():
            tx.set_color(INK2)

    fig.text(0.006, 0.015,
             "Exact CFTP coupling, DWBC, $a=b=1$, $c=\\sqrt{2(1-\\Delta)}$.  "
             "40 independent runs per point, disjoint seed block each; medians "
             "with 95% bootstrap intervals (the law is heavy-tailed).  "
             "No run was censored.  Exponent measured on $n$=32–48 vs $n$=56–80: "
             "$\\Delta=-4$ rises $+2.06\\pm0.68$ ($p$=0.002), the control "
             "$+0.21\\pm0.19$ ($p$=0.26).",
             color=MUTED, fontsize=8.2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(dst, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "asym.csv",
         sys.argv[2] if len(sys.argv) > 2 else "asym.png")
