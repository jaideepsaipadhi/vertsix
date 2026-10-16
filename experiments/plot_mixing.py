#!/usr/bin/env python3
"""Figure for experiment 1: coalescence time as a function of Delta.

    python3 experiments/plot_mixing.py mixing_grid.json mixing.png

Reads the JSON written by the grid run and produces a two-panel figure:

  (a) median forward coalescence time against Delta, one curve per lattice size
  (b) the fitted scaling exponent alpha in tau ~ n^alpha, against Delta,
      with bootstrap confidence intervals

Panel (b) is the one that carries the claim. A single curve in (a) only says
which Delta is slower at one size; the exponent says whether the slowdown
survives the large-n limit, which is what "negative Delta is slower" has to
mean to be a statement about the model rather than about n=40.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ordinal ramp, blue, light mode -- validated (monotone L, adjacent dL >= 0.06,
# light end 2.06:1 on the surface, single hue). Lattice size is an ordered
# quantity, so it gets a sequential ramp rather than categorical hues.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
ACCENT = "#eb6834"


def bootstrap_alpha(samples, ns, reps=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.log(ns)
    out = []
    for _ in range(reps):
        med = [np.median(rng.choice(samples[n], len(samples[n]))) for n in ns]
        out.append(np.polyfit(x, np.log(med), 1)[0])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main(src, dst):
    blob = json.load(open(src))
    ns = blob["ns"]
    seeds = blob["seeds"]
    raw = blob["data"]
    deltas = sorted((float(d) for d in raw), reverse=True)

    samples = {d: {n: np.array(raw[str(d)][str(n)], float) for n in ns}
               for d in deltas}
    med = {d: np.array([np.median(samples[d][n]) for n in ns]) for d in deltas}
    alpha = {d: float(np.polyfit(np.log(ns), np.log(med[d]), 1)[0]) for d in deltas}
    ci = {d: bootstrap_alpha(samples[d], ns) for d in deltas}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=200,
                                   facecolor=SURFACE)
    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color="#e8e7e3", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d6d5d0")
        ax.tick_params(colors=INK2, labelsize=9.5)

    # ---- (a) tau vs Delta, one curve per n
    for k, n in enumerate(ns):
        y = [med[d][k] for d in deltas]
        axL.plot(deltas, y, "-o", color=RAMP[k], linewidth=2, markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        # direct label OUTSIDE the data, past the largest Delta. deltas is
        # sorted descending, so deltas[0] is the rightmost point.
        axL.annotate(f"n={n}", (deltas[0], y[0]), textcoords="offset points",
                     xytext=(8, 0), ha="left", va="center",
                     color=RAMP[k], fontsize=9.5, fontweight="bold",
                     annotation_clip=False)
    axL.set_yscale("log")
    axL.set_xlabel("$\\Delta$", color=INK, fontsize=11)
    axL.set_ylabel("median coalescence time  $\\tau$  (sweeps)",
                   color=INK, fontsize=11)
    axL.set_title("(a)  coalescence time against anisotropy",
                  color=INK, fontsize=12, loc="left", pad=10)
    axL.set_xlim(min(deltas) - 0.3, max(deltas) + 0.35)

    # the minimum is the surprise -- mark it once, on the largest n
    kbig = len(ns) - 1
    ybig = np.array([med[d][kbig] for d in deltas])
    imin = int(np.argmin(ybig))
    axL.annotate(f"minimum near $\\Delta\\approx{deltas[imin]:.1f}$",
                 (deltas[imin], ybig[imin]),
                 textcoords="offset points", xytext=(0, 34), ha="center",
                 color=ACCENT, fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, linewidth=1.3))

    # ---- (b) exponent vs Delta
    a = np.array([alpha[d] for d in deltas])
    lo = np.array([ci[d][0] for d in deltas])
    hi = np.array([ci[d][1] for d in deltas])
    axR.fill_between(deltas, lo, hi, color=RAMP[2], alpha=0.18, zorder=2,
                     linewidth=0)
    axR.plot(deltas, a, "-o", color=RAMP[3], linewidth=2, markersize=6,
             markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)

    flat = float(np.median(a[np.array(deltas) >= -2.0]))
    axR.axhline(flat, color=MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    axR.annotate(f"$\\alpha \\approx {flat:.2f}$ for $\\Delta \\gtrsim -2.5$",
                 (max(deltas), flat), textcoords="offset points",
                 xytext=(-4, -26), ha="right", color=INK2, fontsize=9.5)
    axR.set_xlabel("$\\Delta$", color=INK, fontsize=11)
    axR.set_ylabel("fitted exponent  $\\alpha$  in  $\\tau \\sim n^{\\alpha}$",
                   color=INK, fontsize=11)
    axR.set_title("(b)  does the slowdown survive large $n$?",
                  color=INK, fontsize=12, loc="left", pad=10)
    axR.set_xlim(min(deltas) - 0.3, max(deltas) + 0.3)
    # deltas is sorted descending, so the deep antiferroelectric end -- where
    # the exponent climbs -- is deltas[-1], not deltas[0].
    axR.annotate("exponent rises here:\nthe slowdown is asymptotic,\n"
                 "not a finite-size effect",
                 (deltas[-1], a[-1]), textcoords="offset points",
                 xytext=(26, -14), va="top", ha="left",
                 color=ACCENT, fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, linewidth=1.3))

    fig.text(0.008, 0.015,
             f"Exact CFTP samples, DWBC, weights a=b=1 and c=$\\sqrt{{2(1-\\Delta)}}$ "
             f"(the monotone region, which is contained in $\\Delta\\leq 1/2$).  "
             f"{seeds} independent runs per point; n = {', '.join(map(str, ns))}.  "
             f"Band in (b) is a 95% bootstrap interval.",
             color=MUTED, fontsize=8.2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(dst, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {dst}")

    print("\n  Delta   alpha   95% CI")
    for d in deltas:
        print(f"  {d:+5.2f}   {alpha[d]:5.2f}   [{ci[d][0]:.2f}, {ci[d][1]:.2f}]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mixing_grid.json",
         sys.argv[2] if len(sys.argv) > 2 else "mixing.png")
