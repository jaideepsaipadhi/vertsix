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

What the figure does NOT show is a sharp jump at Delta = -1, and it does not
show a resolved exponent either. The three sizes span a factor of two, so
propagating the CSV's own per-point standard errors gives a standard error on
theta of 0.12 to 0.20 and a 95% interval of about +-0.35:

    Delta    theta    s.e.   95% interval
    -0.50    0.61     0.18   [+0.25, +0.96]
    -0.75    1.15     0.17   [+0.80, +1.49]
    -1.00    0.96     0.19   [+0.56, +1.31]
    -1.25    0.47     0.20   [+0.06, +0.86]
    -1.50    0.61     0.16   [+0.29, +0.91]
    -2.00    0.39     0.16   [+0.06, +0.71]
    -3.00    0.16     0.12   [-0.09, +0.40]

Panel (b) now draws that band. It used to plot theta as a bare line while
this paragraph claimed an error "of order 0.2" -- the se column was read for
panel (a) and never used for the exponent, so seven values were drawn as if
resolved. (plot_mixing.py has always bootstrapped its exponent and drawn the
interval; this was the sibling that did not.)

With the band in view, the earlier claim that "the ORDERING is solid" does
not hold. There are TWO inversions in the committed sample, not one:

  - Delta = -0.5 at theta 0.61 against -0.75 at 1.15. This one was already
    documented, and it is a floor effect: the largest solid component is 13
    faces at N = 64 and 14 at N = 48, the smallest object the erosion can
    leave behind, so A cannot decay further and the fit measures the floor.

  - Delta = -1.25 at theta 0.47 against -1.50 at 0.61. This one was NOT
    documented, and it is not a floor effect -- those cores are 14 to 53
    faces. It is simply noise: the gap is 0.14 against a combined standard
    error of about 0.26.

So what the data supports is the CONTRAST between the ends -- theta is
consistent with 1 at Delta = -0.75 and -1.0 and with 0 at Delta = -3,
excluding the other value at both -- and not a monotone sequence, and not any
individual exponent. The script prints the inversions it finds rather than
leaving a reader to spot them.

That caution reaches further than this paragraph used to admit. It said the
floor applied to Delta = -0.5 and -0.75, and that "only the points from
Delta = -1 downwards, where the core is tens to hundreds of faces, carry a
meaningful exponent". Measured core sizes, in faces, at the three sizes the
fit uses (80 samples at N = 32, 48; 40 at N = 64):

    Delta      N=32    N=48    N=64
    -0.75       7.9    12.7    14.9     <- dismissed as floor
    -1.00       9.5    16.4    20.0     <- claimed meaningful
    -1.25      14.3    21.1       -
    -1.50      19.2    31.5    52.9
    -2.00         -    50.7    77.0
    -3.00      57.4   118.0   199.8

Delta = -1 is 20 faces at N = 64 against 14.9 at Delta = -0.75 -- the same
order as the point being rejected, and nowhere near "tens to hundreds". By
this document's own criterion the meaningful range starts around
Delta = -1.5, not -1. Quote the exponent only from there down, and prefer
to state the cutoff by CORE SIZE rather than by Delta, since the floor is a
property of how many faces survive the erosion.

One further caution at Delta = -1.5: mean 52.9 against median 32.5 at
N = 64, so that point is skewed by occasional large cores and its mean-based
A is noisier than the others.
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


def cell(rows, delta, n, field):
    """One CSV cell, with a readable error when the grid is ragged.

    The fit below needs every (Delta, N) cell present. A bare next() raised
    StopIteration from inside a comprehension, which is an unhelpful way to
    learn that a run was partial -- and partial runs are the normal case here,
    since the deep points cost the most.
    """
    for r in rows:
        if float(r["delta"]) == delta and int(r["n"]) == n:
            return float(r[field])
    raise SystemExit(
        f"{__file__}: no row for Delta={delta}, N={n}. The exponent is fitted "
        f"across all sizes, so every Delta needs every N; either finish the "
        f"run or drop that Delta from the CSV.")


def theta_with_interval(rows, delta, sizes, reps=20000, seed=0):
    """theta in A ~ N^-theta, with the interval the CSV's own errors imply.

    Panel (b) used to plot theta as a bare line while this file's docstring
    stated an error "of order 0.2" -- the se column was read for panel (a)
    and never used for the exponent, so seven values ranging 0.16 to 1.15
    were drawn as if resolved. plot_mixing.py bootstraps its exponent and
    draws a band; this did not.

    Propagated from the per-point standard errors, the standard error on
    theta is 0.12 to 0.20 across the sweep, so a 95% interval is about +-0.35
    -- wide enough that no single theta is pinned and that the two ordering
    inversions in the committed sample are both inside the noise.
    """
    if len(sizes) < 3:
        raise SystemExit(
            f"{__file__}: {len(sizes)} lattice size(s) in the CSV. theta is "
            f"the slope of log A against log N, so one size fits a line to a "
            f"single point and returns a number with a tiny interval and no "
            f"meaning (a one-size CSV gave 0.65 +- 0.02 before this guard); "
            f"two sizes fit exactly and leave no residual at all. Use at "
            f"least three.")
    rng = np.random.default_rng(seed)
    a = np.array([cell(rows, delta, n, "area_frac") for n in sizes])
    se = np.array([cell(rows, delta, n, "se") for n in sizes])
    x = np.log(sizes)
    theta = -np.polyfit(x, np.log(a), 1)[0]
    draws = []
    for _ in range(reps):
        ap = a + se * rng.standard_normal(a.size)
        if (ap <= 0).any():
            continue
        draws.append(-np.polyfit(x, np.log(ap), 1)[0])
    draws = np.array(draws)
    return (float(theta), float(draws.std()),
            float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


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

    th, lo, hi, sd = [], [], [], []
    for d0 in deltas:
        t, s0, l0, h0 = theta_with_interval(rows, d0, sizes)
        th.append(t); sd.append(s0); lo.append(l0); hi.append(h0)
    th = np.array(th); lo = np.array(lo); hi = np.array(hi); sd = np.array(sd)
    dd = np.array(deltas)

    axB.axhline(0.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    axB.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    axB.fill_between(dd, lo, hi, color=ACCENT, alpha=0.16, linewidth=0,
                     zorder=2)
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
    for d, t, s0, l0, h0 in zip(deltas, th, sd, lo, hi):
        print(f"  Delta={d:+5.2f}  theta = {t:.2f} +- {s0:.2f}"
              f"   95% [{l0:+.2f}, {h0:+.2f}]")
    # deltas is ascending (-3 first), and theta should RISE with Delta
    # (filament at the shallow end, blob at the deep end). A pair where it
    # falls instead is an inversion; quote it with its interval.
    worse = [(deltas[i + 1], deltas[i]) for i in range(len(deltas) - 1)
             if th[i + 1] < th[i]]
    if worse:
        print("  theta inverts (rises as Delta falls) at: "
              + ", ".join(f"{a:+.2f}->{b:+.2f}" for a, b in worse)
              + "\n  Check each against the intervals above before reading "
                "structure into it.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
