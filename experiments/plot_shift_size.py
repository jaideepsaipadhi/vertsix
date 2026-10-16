#!/usr/bin/env python3
"""Is the Delta-dependence of the island height a finite-size artifact?

    python3 experiments/plot_shift_size.py n16.csv n32.csv out.png

The Delta sweep in experiments/run_shift_delta_chain was run at n=16, chosen
because it is the cheapest place to measure. That choice rested on an
n-independence check made at ONE value of Delta, so the shape it found -- a
minimum in R_lo near Delta=-3 -- could have been a property of n=16 rather
than of the model. This repeats the sweep at n=32 and compares.

Two separate questions, and they get different answers:

* The SHAPE survives. R_lo falls from Delta=-1.5 to -3 and rises after, at
  both sizes.
* The LEVEL does not. n=32 sits systematically above n=16 by about 0.010 at
  the shallow points. The earlier test that reported R_lo "flat in n"
  (p=0.27, over n=16..48 from independent sampling) simply did not have the
  precision to see a shift that small -- the chain estimator does.

So the Delta-trend in mu and sigma is safe (it spans about 0.06, an order of
magnitude larger than the size effect), but the fourth decimal places quoted
for mu and sigma at a single n are not universal numbers.

Deep points at n=32 are expensive and correspondingly imprecise -- the k=+1
cell is rare there and tau_int is large -- so the rise after the minimum is
well resolved at n=16 and only suggestive at n=32. The error bars say so.
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


def load(path):
    rows = sorted(csv.DictReader(open(path)), key=lambda r: float(r["delta"]))
    g = lambda k: np.array([float(r[k]) for r in rows])
    d = g("delta")
    Rlo, slo, Rhi, shi = g("R_lo"), g("se_lo"), g("R_hi"), g("se_hi")
    inv, y = -np.log(Rlo * Rhi), np.log(Rhi / Rlo)
    se = np.hypot(slo / Rlo, shi / Rhi)
    mu, sg = g("mu"), g("sigma")
    return dict(d=d, Rlo=Rlo, slo=slo, mu=mu, sg=sg,
                s_sg=sg ** 3 * se / 2.0,
                s_mu=np.abs(mu) * np.hypot(se / np.abs(y), se / inv),
                n=int(rows[0]["n"]))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e8e7e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5d0")
    ax.tick_params(colors=INK2, labelsize=9.5)


def main(f16, f32, dst):
    A, B = load(f16), load(f32)
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=195,
                                        facecolor=SURFACE, layout="constrained")
    for ax in (axL, axM, axR):
        style(ax)

    for S, col, mk in ((A, BLUE, "o"), (B, ORANGE, "s")):
        axL.errorbar(S["d"], S["Rlo"], yerr=S["slo"], fmt="-" + mk, color=col,
                     linewidth=2, markersize=7, capsize=3, elinewidth=1.4,
                     markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4,
                     label=f"$n$={S['n']}")
        axM.errorbar(S["d"], S["mu"], yerr=S["s_mu"], fmt="-" + mk, color=col,
                     linewidth=2, markersize=7, capsize=3, elinewidth=1.4,
                     markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4,
                     label=f"$n$={S['n']}")
        axR.errorbar(S["d"], S["sg"], yerr=S["s_sg"], fmt="-" + mk, color=col,
                     linewidth=2, markersize=7, capsize=3, elinewidth=1.4,
                     markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4,
                     label=f"$n$={S['n']}")

    axL.set_ylabel("$P(k=-1)\\,/\\,P(k=0)$", color=INK, fontsize=11)
    axL.set_title("(a)  the shape survives, the level does not",
                  color=INK, fontsize=12, loc="left", pad=10)
    axM.set_ylabel("$\\mu$", color=INK, fontsize=11)
    axM.set_title("(b)  the shift", color=INK, fontsize=12, loc="left", pad=10)
    axR.set_ylabel("$\\sigma$", color=INK, fontsize=11)
    axR.set_title("(c)  the width", color=INK, fontsize=12, loc="left", pad=10)
    for ax in (axL, axM, axR):
        ax.set_xlabel("$\\Delta$", color=INK, fontsize=11)
        leg = ax.legend(frameon=False, fontsize=10, loc="best")
        for t in leg.get_texts():
            t.set_color(INK2)

    fig.text(0.006, -0.085,
             "Island height on DWBC, $a=b=1$, $c=\\sqrt{2(1-\\Delta)}$.  "
             "Stationary chains from independent exact CFTP starts; errors are "
             "chain-to-chain batch means, which stay valid however slowly a "
             "single chain mixes.  The deep $n$=32 points are imprecise "
             "because the $k=+1$ cell is rare and $\\tau_{\\rm int}$ is large "
             "there.",
             color=MUTED, fontsize=8.2)
    fig.savefig(dst, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {dst}")

    print("\n  R_lo, n=16 vs n=32 at shared Delta:")
    for d in sorted(set(A["d"]) & set(B["d"]), reverse=True):
        i = int(np.where(A["d"] == d)[0][0]); j = int(np.where(B["d"] == d)[0][0])
        diff = B["Rlo"][j] - A["Rlo"][i]
        s = float(np.hypot(A["slo"][i], B["slo"][j]))
        print(f"    Delta={d:+5.1f}  n16 {A['Rlo'][i]:.4f}+-{A['slo'][i]:.4f}   "
              f"n32 {B['Rlo'][j]:.4f}+-{B['slo'][j]:.4f}   "
              f"diff {diff:+.4f}+-{s:.4f} ({abs(diff)/s:.1f} sigma)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
