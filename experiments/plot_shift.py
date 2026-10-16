#!/usr/bin/env python3
"""Figure for experiment 2: what the island height actually pins down.

    python3 experiments/plot_shift.py shift_delta.csv shift_n.csv shift.png

Panel (a) is the point of the figure. Almost all the mass sits on two lattice
points, so a two-parameter discrete Gaussian fitted to this histogram has zero
degrees of freedom -- it agrees by construction, and a goodness-of-fit p-value
computed from it would mean nothing however many samples were taken.

Panels (b) and (c) show the quantity that survives that: the near-neighbour
ratio R = P(k=-1)/P(k=0), which needs only the two populated cells. They share
a y-axis on purpose, so "flat in Delta" and "flat in n" are read at the same
visual scale rather than each being stretched to fill its own panel.

Delta and n get separate panels because they are separate independent
variables. An earlier draft put both in one panel, which stacked four points
at Delta=-3 on top of the Delta curve and read as a measurement disagreeing
with itself.
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"   # validated categorical pair, light mode
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        c = {float(k): int(v)
             for k, v in (x.split(":") for x in r["counts"].split(";"))}
        mode = max(c, key=c.get)
        lo, hi = c.get(mode - 1, 0), c.get(mode + 1, 0)
        rows.append(dict(n=int(r["n"]), delta=float(r["delta"]), counts=c,
                         mode=mode, lo=lo, hi=hi, m=c[mode], r_lo=lo / c[mode],
                         # Poisson on the numerator dominates; the mode cell
                         # carries ~1600 counts and adds <1% in quadrature
                         se=np.sqrt(max(lo, 1)) / c[mode]))
    return rows


def constant_fit(v, s):
    """Weighted mean and its chi-square against a constant."""
    from scipy.stats import chi2
    w = 1 / s ** 2
    mu = float((w * v).sum() / w.sum())
    X = float((((v - mu) / s) ** 2).sum())
    dof = len(v) - 1
    return mu, X, dof, float(chi2.sf(X, dof)), float(1 / np.sqrt(w.sum()))


def trend(x, v, s):
    """Weighted linear regression slope and its two-sided p.

    Reported alongside the constant fit because the two disagree, and the
    disagreement is about statistical power, not about the data. The omnibus
    chi-square spends one degree of freedom per point testing every direction
    of departure at once and returns p=0.09 here; the slope tests the single
    direction actually at issue -- does R drift with Delta -- and returns
    p=0.008 on the same numbers.

    The cut point is not chosen from the data. An earlier version split the
    points at Delta=-2 after seeing where the drop looked like it was, which
    makes the resulting p meaningless; a slope has no cut point to choose.
    """
    from scipy.stats import norm
    x = np.asarray(x, float)
    w = 1 / np.asarray(s, float) ** 2
    X = np.vstack([np.ones_like(x), x]).T
    C = np.linalg.inv(X.T @ np.diag(w) @ X)
    b = C @ X.T @ np.diag(w) @ np.asarray(v, float)
    se = float(np.sqrt(np.diag(C)[1]))
    z = float(b[1]) / se
    return float(b[1]), se, z, float(2 * norm.sf(abs(z)))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e8e7e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5d0")
    ax.tick_params(colors=INK2, labelsize=9.5)


def ratio_panel(ax, xs, v, s, colour, xlabel, title, band, tr=None):
    mu, X, dof, p, sem = band
    ax.axhspan(mu - sem, mu + sem, color=MUTED, alpha=0.16, zorder=1,
               linewidth=0)
    ax.axhline(mu, color=MUTED, linewidth=1.3, linestyle=(0, (5, 4)), zorder=2)
    ax.errorbar(xs, v, yerr=s, fmt="o", color=colour, markersize=7,
                capsize=3, elinewidth=1.4, linestyle="none",
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
    ax.set_xlabel(xlabel, color=INK, fontsize=11)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    txt = (f"constant fit  $R={mu:.3f}$,  "
           f"$\\chi^2$={X:.1f}/{dof} d.o.f.,  $p$={p:.2f}")
    if tr is not None:
        b, se, z, pt = tr
        txt += (f"\nweighted slope  ${b:+.4f}\\pm{se:.4f}$,  "
                f"$z={z:+.2f}$,  $p={pt:.3f}$")
    ax.annotate(txt, (0.5, 0.035), xycoords="axes fraction", ha="center",
                va="bottom", color=INK2, fontsize=9.0, linespacing=1.5)


def main(f_delta, f_n, dst):
    D = sorted(load(f_delta), key=lambda r: r["delta"])
    N = sorted(load(f_n), key=lambda r: r["n"])

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(15.2, 4.9), dpi=200, facecolor=SURFACE,
        gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.15]))
    for ax in (axA, axB, axC):
        style(ax)

    # ---- (a) the histogram, log scale so the near-empty cell is visible
    ref = next((r for r in N if r["n"] == 32), N[-1])
    ks = sorted(set(ref["counts"]) | {ref["mode"] - 1, ref["mode"] + 1})
    vals = [ref["counts"].get(k, 0) for k in ks]
    tot = sum(ref["counts"].values())
    axA.bar(ks, [max(v, 0.6) for v in vals], width=0.62, color=BLUE,
            edgecolor=SURFACE, linewidth=2, zorder=3)
    axA.set_yscale("log")
    axA.set_ylim(0.5, tot * 26)
    axA.set_xticks(ks)
    for k, v in zip(ks, vals):
        axA.annotate(f"{v}", (k, max(v, 0.6)), textcoords="offset points",
                     xytext=(0, 5), ha="center", color=INK2, fontsize=9.5)
    axA.set_xlabel("centred island height  $k$", color=INK, fontsize=11)
    axA.set_ylabel(f"samples (of {tot})", color=INK, fontsize=11)
    axA.set_title(f"(a)  $n$={ref['n']},  $\\Delta$={ref['delta']:+.1f}",
                  color=INK, fontsize=12, loc="left", pad=10)
    frac = 100 * (ref["m"] + ref["lo"]) / tot
    # sits in the headroom above the tallest bar, not over it
    # left-aligned over the SHORT bar. Centred at the top it collided with
    # the "1635" count label -- which looked fine by eye and was caught only
    # by comparing the rendered text bounding boxes.
    axA.annotate(f"two cells hold {frac:.2f}%\nof the mass: a 2-parameter\n"
                 "discrete Gaussian fitted\nhere has 0 d.o.f.",
                 (0.03, 0.98), xycoords="axes fraction", ha="left",
                 va="top", color=ORANGE, fontsize=9.3, linespacing=1.35)

    # ---- (b), (c) the ratio that IS measured, on a shared y-axis
    dv = np.array([r["r_lo"] for r in D]); ds = np.array([r["se"] for r in D])
    nv = np.array([r["r_lo"] for r in N]); ns = np.array([r["se"] for r in N])
    dx = [r["delta"] for r in D]
    nx = [r["n"] for r in N]
    ratio_panel(axB, dx, dv, ds, BLUE, "$\\Delta$",
                f"(b)  against $\\Delta$   ($n$={D[0]['n']})",
                constant_fit(dv, ds), trend(dx, dv, ds))
    ratio_panel(axC, nx, nv, ns, ORANGE, "$n$",
                f"(c)  against $n$   ($\\Delta$={N[0]['delta']:+.1f})",
                constant_fit(nv, ns), trend(nx, nv, ns))
    axB.set_ylabel("$P(k=-1)\\,/\\,P(k=0)$", color=INK, fontsize=11)
    lo = min((dv - ds).min(), (nv - ns).min())
    hi = max((dv + ds).max(), (nv + ns).max())
    pad = 0.42 * (hi - lo)
    for ax in (axB, axC):          # shared scale: flat must look flat in both
        ax.set_ylim(lo - pad, hi + 0.10 * (hi - lo))
    axC.set_xticks([r["n"] for r in N])

    fig.text(0.006, 0.015,
             "Exact CFTP samples, DWBC, $a=b=1$, $c=\\sqrt{2(1-\\Delta)}$.  "
             "2000 samples per point, every config on a disjoint seed block.  "
             "Error bars are Poisson on the $k=-1$ count; dashed line and band "
             "are the weighted mean $\\pm$ its standard error.  "
             "(b) and (c) share a $y$-axis.  The $k=+1$ cell holds ~0.15% of "
             "the mass, so it alone separates $\\mu$ from $\\sigma$ and is not "
             "measured here.",
             color=MUTED, fontsize=8.2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(dst, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {dst}")
    for tag, rows in (("Delta", D), ("n", N)):
        print(f"\n  sweep in {tag}")
        for r in rows:
            print(f"    n={r['n']:3d} Delta={r['delta']:+5.1f}  "
                  f"R={r['r_lo']:.4f} +-{r['se']:.4f}   k=+1 count {r['hi']}")


if __name__ == "__main__":
    main(*(sys.argv[1:4] if len(sys.argv) > 3
           else ("shift_delta.csv", "shift_n.csv", "shift.png")))
