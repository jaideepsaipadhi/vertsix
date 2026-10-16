#!/usr/bin/env python3
"""Exact samples on the Gorin-Kenyon domains (arXiv:2408.14446, Section 2.1).

    python3 experiments/plot_domains.py domains.png

Each panel is one exact CFTP sample of the height function on a domain built
from a convex k x l array I, at two values of Delta. Cells outside the domain
are left blank, so the shape of the region is the shape of the coloured area.

The height field gets a single-hue sequential ramp, validated light-to-dark
against the page surface, because height is an ordered quantity and a
categorical or rainbow scale would invent structure in it that is not there.

Two things the first version of this figure got wrong, both visible only once
it was rendered:

* It plotted the RAW height, which alternates between L and L+1 on the
  checkerboard by construction. Every panel came out covered in a fine
  checker pattern that reads as antiferroelectric order and is nothing of the
  sort -- it is parity. What is plotted here is H - ((i+j) mod 2), the same
  correction experiments/island.py applies.

* Each panel autoscaled its own colour range, so the same colour meant a
  different height in different panels. The scale is now shared down each
  column, where the domain and hence the height range is the same, and a
  colour bar states it.
"""
from __future__ import annotations

import sys, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sixvertex.domains import cftp_sample, corner_mask, domain_mask
from experiments.mixing import weights_for_delta

RAMP = ["#86b6ef", "#5f9ae0", "#3b7ccc", "#2259a0", "#123f74"]  # validated
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"
CMAP = LinearSegmentedColormap.from_list("h", RAMP)

DOMAINS = [
    ("staircase", [[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
    ("plus",      [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    ("notch",     [[1, 1, 1], [1, 1, 1], [0, 1, 1]]),
]
DELTAS = [0.0, -3.0]


def main(dst, N=60, seed=7):
    cuts = [0, N // 3, 2 * N // 3, N]
    fig, axes = plt.subplots(len(DELTAS), len(DOMAINS),
                             figsize=(4.1 * len(DOMAINS), 4.3 * len(DELTAS)),
                             dpi=190, facecolor=SURFACE,
                             layout="constrained")
    axes = np.atleast_2d(axes)
    fields = {}
    titles = {}
    for c, (name, I) in enumerate(DOMAINS):
        mask = domain_mask(cuts, cuts, I)
        cm = corner_mask(mask)
        titles[c] = (name, I)
        for r, d in enumerate(DELTAS):
            ax = axes[r, c]
            H, info = cftp_sample(mask, weights_for_delta(d), master_seed=seed)
            ii, jj = np.meshgrid(np.arange(H.shape[0]), np.arange(H.shape[1]),
                                 indexing="ij")
            G = H - ((ii + jj) & 1)            # remove the parity checkerboard
            F = np.where(cm, G, np.nan).astype(float)
            fields.setdefault(c, []).append((r, F, info))
    # shared colour range down each column: same domain, same height range
    for c, items in fields.items():
        vals = np.concatenate([F[np.isfinite(F)].ravel() for _, F, _ in items])
        vmin, vmax = float(vals.min()), float(vals.max())
        name, I = titles[c]
        for r, F, info in items:
            ax = axes[r, c]
            im = ax.imshow(F, cmap=CMAP, interpolation="nearest",
                           origin="upper", vmin=vmin, vmax=vmax)
            ax.set_facecolor(SURFACE)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"{name}    $I$ = {I}".replace("], [", "] ["),
                             color=INK, fontsize=10.5, pad=9)
            if c == 0:
                ax.set_ylabel(f"$\\Delta = {DELTAS[r]:+.0f}$", color=INK,
                              fontsize=12)
            ax.annotate(f"$T$={info['T']}", (0.99, 0.01),
                        xycoords="axes fraction", ha="right", va="bottom",
                        color=MUTED, fontsize=8.5)
        cb = fig.colorbar(im, ax=list(axes[:, c]), fraction=0.046, pad=0.012, shrink=0.82)
        cb.set_label("corrected height", color=INK2, fontsize=8.5)
        cb.ax.tick_params(colors=INK2, labelsize=8)
        cb.outline.set_visible(False)

    fig.suptitle("Exact samples on Gorin-Kenyon domains", color=INK,
                 fontsize=13.5, x=0.008, ha="left")
    fig.text(0.008, 0.004,
             f"Height function on $\\Omega^{{X,Y,I}}$, $N$={N}, cuts at "
             f"{cuts[1]} and {cuts[2]}.  $a=b=1$, $c=\\sqrt{{2(1-\\Delta)}}$.  "
             "Exact CFTP, one sample per panel; $T$ is the coupling window "
             "reached.  Blank cells are outside the domain.  Plotted is "
             "$H-((i+j)\\,\\mathrm{mod}\\,2)$, with the colour range shared down each "
             "column.",
             color=MUTED, fontsize=8.4)
    fig.savefig(dst, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "domains.png")
