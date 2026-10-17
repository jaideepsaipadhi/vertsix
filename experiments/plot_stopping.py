#!/usr/bin/env python3
"""The CFTP stopping time T* as a function of Delta.

    python3 experiments/plot_stopping.py stop.csv stop_raw.csv out.png

This is the plot Gorin asked for: "the mixing time (how long it takes until
CFTP stops) as a function of Delta". T* is the backward coalescence time,

    T* = min { T : the composed update map from -T to 0 is constant },

measured exactly by experiments/cftp_stopping.py -- not the power of two the
sampler reports, and not the forward coalescence time that the earlier
experiment (experiments/run_mixing.py) measured as a stand-in.

Panel (a) is the answer. T* is NON-MONOTONE in Delta at every size measured:
it falls from the ferroelectric side to a minimum at Delta = -0.5 to -1 --
at or just past the free-fermion-adjacent disordered regime -- and climbs
again into the antiferroelectric phase. The minimum does not sit at the
uniform point Delta = -1/2 in any clean way; at n=12 and n=32 the two
candidates are within each other's intervals, so the location is pinned only
to that interval and is not claimed more precisely.

Panel (b) is what makes (a) a statement about the model rather than about one
lattice size. Fitting T* ~ n^alpha across n = 12..32 at each Delta, the
exponent sits on a plateau at about 2.3 for Delta >= -1.5 -- flat to within
its intervals across the whole ferroelectric and disordered range, including
across the minimum in (a) -- and then RISES into the antiferroelectric phase,
reaching 2.86 [2.63, 2.98] at Delta = -4, which does not overlap the plateau.
A curve in (a) alone only says which Delta is slower at one n; a rising
exponent says the slowdown compounds with size, which is what the
antiferroelectric slowdown has to mean to survive the limit.

Note what panel (b) does NOT say. The minimum in (a) is not visible in the
exponent at all: the cost difference between Delta = +0.5 and Delta = -1 is a
constant factor, not a different power. Only the antiferroelectric side
changes the exponent.

Panel (c) is the audit of the earlier substitution. T* and the forward
coalescence time agree to within 13% at every (n, Delta) measured, ratios in
[0.88, 1.13] with no trend in Delta. So the earlier figure was not measuring
the wrong shape -- but that is now a measurement, not an argument from the
doubling schedule, which was the wrong argument: the schedule quantises the
REPORTED number, not the quantity.

A caution on (b). The fit spans n = 12..32, under a factor of three, and the
exponents carry bootstrap intervals of roughly +-0.1. The ORDERING across
Delta is well outside that; the third digit of any single exponent is not.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sequential ramp: lattice size is an ordered quantity, so it gets one hue
# light->dark rather than categorical hues. Same ramp as plot_mixing.
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


def bootstrap_alpha(by_n, reps=2000, seed=0):
    """Exponent in T* ~ n^alpha, resampling seeds within each size.

    The fit is on log(median T*) against log n, so each bootstrap replicate
    redraws the seeds at every size and refits. Using the medians alone with
    no spread would give a point estimate and no honest interval.
    """
    rng = np.random.default_rng(seed)
    ns = np.array(sorted(by_n))
    x = np.log(ns)
    out = []
    for _ in range(reps):
        y = []
        for n in ns:
            s = by_n[n]
            y.append(np.log(np.median(rng.choice(s, len(s)))))
        out.append(np.polyfit(x, np.array(y), 1)[0])
    y0 = np.log([np.median(by_n[n]) for n in ns])
    return float(np.polyfit(x, y0, 1)[0]), np.percentile(out, [2.5, 97.5])


def main(src, raw_src, dst):
    rows = list(csv.DictReader(open(src)))
    sizes = sorted({int(r["n"]) for r in rows})
    deltas = sorted({float(r["delta"]) for r in rows})

    raw = defaultdict(list)                       # (n, delta) -> [T*]
    for r in csv.DictReader(open(raw_src)):
        if r["Tstar"]:
            raw[(int(r["n"]), float(r["delta"]))].append(int(r["Tstar"]))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=195,
                                        facecolor=SURFACE, layout="constrained")
    for ax in (axA, axB, axC):
        style(ax)

    # ---- (a) T* against Delta, one curve per size
    for k, n in enumerate(sizes):
        sel = sorted((r for r in rows if int(r["n"]) == n),
                     key=lambda r: float(r["delta"]))
        d = np.array([float(r["delta"]) for r in sel])
        m = np.array([float(r["median_Tstar"]) for r in sel])
        lo = np.array([float(r["lo"]) for r in sel])
        hi = np.array([float(r["hi"]) for r in sel])
        col = RAMP[k % len(RAMP)]
        axA.fill_between(d, lo, hi, color=col, alpha=0.16, linewidth=0, zorder=2)
        axA.plot(d, m, "-o", color=col, linewidth=2, markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
                 label=f"n = {n}")
        axA.annotate(f"n = {n}", (d[-1], m[-1]), textcoords="offset points",
                     xytext=(-6, 9), ha="right", color=col, fontsize=9.5,
                     fontweight="semibold")
    axA.set_yscale("log")
    axA.set_xlabel(r"$\Delta$", color=INK2, fontsize=10.5)
    axA.set_ylabel(r"stopping time $T^{*}$ (sweeps)", color=INK2, fontsize=10.5)
    axA.set_title("(a)  CFTP stops latest deep on both sides",
                  color=INK, fontsize=11.5, loc="left", pad=9)
    axA.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")

    # ---- (b) exponent in T* ~ n^alpha
    al, lo_a, hi_a = [], [], []
    for d in deltas:
        by_n = {n: raw[(n, d)] for n in sizes if raw[(n, d)]}
        a, (l, h) = bootstrap_alpha(by_n)
        al.append(a); lo_a.append(l); hi_a.append(h)
    al, lo_a, hi_a = np.array(al), np.array(lo_a), np.array(hi_a)
    dd = np.array(deltas)
    axB.fill_between(dd, lo_a, hi_a, color=ACCENT, alpha=0.18, linewidth=0, zorder=2)
    axB.plot(dd, al, "-o", color=ACCENT, linewidth=2, markersize=5,
             markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
    plateau = float(np.mean(al[dd >= -1.5]))
    axB.axhline(plateau, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    axB.annotate(rf"plateau, $\Delta \geq -1.5$:  $\alpha = {plateau:.2f}$",
                 (0.97, 0.96), xycoords="axes fraction", ha="right",
                 va="top", color=MUTED, fontsize=9)
    axB.set_xlabel(r"$\Delta$", color=INK2, fontsize=10.5)
    axB.set_ylabel(r"exponent $\alpha$ in $T^{*}\sim n^{\alpha}$",
                   color=INK2, fontsize=10.5)
    axB.set_title(r"(b)  the slowdown compounds with $n$",
                  color=INK, fontsize=11.5, loc="left", pad=9)

    # ---- (c) T* against the forward time it stood in for
    for k, n in enumerate(sizes):
        sel = sorted((r for r in rows if int(r["n"]) == n),
                     key=lambda r: float(r["delta"]))
        d = np.array([float(r["delta"]) for r in sel])
        rat = np.array([float(r["median_Tstar"]) / float(r["median_forward"])
                        for r in sel])
        axC.plot(d, rat, "-o", color=RAMP[k % len(RAMP)], linewidth=2,
                 markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.4,
                 zorder=3, label=f"n = {n}")
    axC.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    axC.set_ylim(0.75, 1.25)
    axC.set_xlabel(r"$\Delta$", color=INK2, fontsize=10.5)
    axC.set_ylabel(r"$T^{*}$ / forward coalescence time",
                   color=INK2, fontsize=10.5)
    axC.set_title("(c)  the earlier stand-in, audited",
                  color=INK, fontsize=11.5, loc="left", pad=9)
    axC.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left", ncol=2)

    fig.savefig(dst, facecolor=SURFACE)
    print(f"wrote {dst}")
    for d, a, l, h in zip(deltas, al, lo_a, hi_a):
        print(f"  Delta={d:+5.2f}  alpha = {a:.2f}  [{l:.2f}, {h:.2f}]")


if __name__ == "__main__":
    main(*sys.argv[1:4])
