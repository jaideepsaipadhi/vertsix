#!/usr/bin/env python3
"""A True/False definition of "the island exists", and P(island) against Delta.

    python3 experiments/island_exists.py --n 32 48 64 --reps 120 --out isl.csv

COST, and the large-N run. Exact sampling at Delta = -3 costs about 0.12s at
n=32, 2.4s at n=64 and 28s at n=96 -- roughly n^6, since T* itself grows like
n^2.5 and each sweep is n^2. N=200 is therefore around 40 minutes PER SAMPLE
and is a batch job, not an interactive one. The run that would settle the
sharpening:

    python3 experiments/island_exists.py --n 96 128 160 200 \
        --deltas -0.5 -0.75 -1.0 -1.25 -1.5 -2.0 --reps 200 \
        --out island_exists_large.csv

Budget it as a few thousand core-hours and split it by --n across machines;
the rows are independent, and the CSVs concatenate.

Gorin's question: give a combinatorial predicate that answers yes or no on a
single configuration, then study P(island exists) as a function of Delta,
expecting a sharp change at Delta = -1 as N grows.

WHY THE OBVIOUS DEFINITIONS DO NOT WORK. Measured on exact samples, the
largest 8-connected component of c-faces is MACROSCOPIC on both sides of
Delta = -1: area fraction 0.23 at Delta = -0.5 and 0.35 at Delta = -3, each
with no decay in n over n = 16..48 (log-log slopes -0.09 and +0.03). It is not
a percolation artifact either -- the c-face density at Delta = -0.5 is 0.31,
below the 8-connected site threshold of about 0.407, so an uncorrelated set
that sparse would give only logarithmic clusters. A predicate built on the
SIZE of the c-region therefore says "yes" everywhere and detects nothing.

WHAT ACTUALLY DIFFERS is whether the height is CONSTANT on that region. Inside
a real island the raw height alternates between L and L+1 in a checkerboard,
so the right variable is the parity-corrected height

    G(i, j) = H(i, j) - ((i + j) mod 2),

which is genuinely constant there. At Delta = -0.5 the central region is
c-rich but G still fluctuates across it.

THE PREDICATE. Fix a level L and call a FACE level-L when all four of its
corners have G = L. Take the level-L faces with 8-connectivity. The island
exists iff some component, for some L, CROSSES the central window -- spans it
from the left edge to the right. Crossing is the natural threshold-free yes/no
here: it needs no tuned area cutoff, and crossing probabilities are the
observables that actually develop sharp transitions in N. The central window
of side N/2 keeps the frozen corners of DWBC out, since those would otherwise
supply their own enormous constant-G regions.

Faces, not lattice points. Running this on the POINT set {G = L} gives
P = 1 at every Delta measured, because an 8-connected set of points can be a
one-cell-wide filament and a filament always crosses something. Requiring a
whole face to be level forces the crossing object to have width, which is what
"the height is constant on a region" was supposed to mean. This is the third
definition tried here and the first that separates the two phases; the two
discarded ones are recorded above and in experiments/island.py so the same
ground is not covered again.

A note on defects. An earlier attempt required strict flatness of whole faces
and measured the largest such component; it decayed with n at Delta = -3 as
well as at Delta = -0.5, because a single interior defect splits the region
and the detector was reporting fragmentation rather than absence. Crossing is
far less brittle: a defect has to disconnect the component ALL the way across
the window to change the answer.
"""
from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sixvertex.cftp_exact import cftp_sample
from experiments.mixing import weights_for_delta


def corrected_height(H, n):
    """G = H - ((i+j) mod 2), constant on an island."""
    H = np.asarray(H, dtype=np.int64)
    ii, jj = np.indices(H.shape)
    return H - ((ii + jj) & 1)


def island_exists(H, n, frac=0.5):
    """True iff a level FACE cluster crosses the central window left-to-right.

    Returns (exists, level, width_fraction) where width_fraction is the widest
    horizontal extent achieved by any level cluster, as a fraction of the
    window -- a continuous companion to the yes/no answer, which shows HOW a
    marginal case fails rather than only that it does.
    """
    G = corrected_height(H, n)
    m = G.shape[0]
    lo = int(round(m * (1 - frac) / 2))
    hi = int(round(m * (1 + frac) / 2))
    W = G[lo:hi, lo:hi]
    if W.shape[0] < 2:
        return False, None, 0.0

    # A face is level iff all four of its corners agree.
    level_face = ((W[:-1, :-1] == W[:-1, 1:]) & (W[:-1, :-1] == W[1:, :-1])
                  & (W[:-1, :-1] == W[1:, 1:]))
    lev = W[:-1, :-1]
    w = level_face.shape[1]

    best_frac, best_level, found = 0.0, None, False
    structure = np.ones((3, 3))
    for L in np.unique(lev[level_face]):
        lab, nl = ndimage.label(level_face & (lev == L), structure=structure)
        if nl == 0:
            continue
        cols = np.indices(lab.shape)[1]
        for c in range(1, nl + 1):
            sel = cols[lab == c]
            ext = float(sel.max() - sel.min() + 1) / w
            if ext > best_frac:
                best_frac, best_level = ext, int(L)
            if ext >= 1.0:
                found = True
    return found, best_level, best_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--frac", type=float, default=0.5,
                    help="side of the central window, as a fraction of N")
    ap.add_argument("--out", default="island_exists.csv")
    args = ap.parse_args()

    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "delta", "reps", "p_exists", "se", "mean_width"])
        for n in args.n:
            for d in args.deltas:
                w = weights_for_delta(d)
                hits, widths = 0, []
                for s in range(args.reps):
                    H, _ = cftp_sample(n, w, master_seed=31_000 + 97 * s)
                    ok, _lvl, ext = island_exists(H, n, args.frac)
                    hits += int(ok)
                    widths.append(ext)
                p = hits / args.reps
                se = (p * (1 - p) / args.reps) ** 0.5
                wr.writerow([n, d, args.reps, f"{p:.4f}", f"{se:.4f}",
                             f"{np.mean(widths):.4f}"])
                fh.flush()
                print(f"  n={n:4d} Delta={d:+5.2f}  P(island)={p:.3f} +- {se:.3f}"
                      f"   mean widest extent {np.mean(widths):.3f}"
                      f"   [{time.time()-t0:5.0f}s]", flush=True)
    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
