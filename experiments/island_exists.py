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

WHAT THE PICTURE ACTUALLY SHOWS. Gorin's N=200, Delta=-3 simulation shows the
island as a near-SOLID block of c-vertices in the middle, surrounded by a
speckled c region, both inside the arctic curve. The radial profile confirms
it: at n=96, Delta=-3 the c-density is 0.89-0.93 for r/R <= 0.17 and 0.58 at
the edge, whereas at Delta=-0.5 the centre reaches only 0.57-0.65 against an
edge value of 0.44. So the island is where the c region is SOLID, not merely
where it is large -- and the solid core is only about 0.25N across.

THE PREDICATE. Call a c-face SOLID when all eight of its neighbours are also
c-faces (one step of binary erosion). The island exists iff the solid c-faces
contain a connected component spanning the central window, taken with
8-connectivity. The window is N/5 by default, because the core is about a
quarter of N: demanding a span of N/2, as an earlier version did, asks the
island to be twice the size it ever reaches, and that alone held P below 0.4
at Delta=-3 where the island is unmistakable.

Measured at n=64 with 40 samples per point, this gives P = 0.00 at
Delta = -0.5, -0.75 and -1.0, then 0.03, 0.07, 0.30, 0.68 at -1.25, -1.5,
-2.0 and -3.0: zero above the transition and rising below it, with the onset
at Delta = -1.

DEFINITIONS THAT FAIL, recorded so the ground is not covered twice:

  - SIZE of the c region. The largest 8-connected component of c-faces is
    macroscopic on both sides of Delta = -1 -- area fraction 0.23 at
    Delta = -0.5 and 0.35 at Delta = -3, neither decaying in n over
    n = 16..48. Not a percolation artifact either: the c-density at
    Delta = -0.5 is 0.31, below the 8-connected site threshold of about
    0.407. A size-based predicate says yes everywhere.

  - CROSSING ON LATTICE POINTS of a level set of the height. Trivially true,
    P = 1 at every Delta, because an 8-connected set of points can be one
    cell wide and a filament always crosses.

  - CROSSING BY LEVEL FACES, where a face is level when its four corners
    share the parity-corrected height G = H - ((i+j) mod 2). This separates
    the phases but much less cleanly than solidity: at n=64 it still fires
    with probability 0.20 at Delta = -0.5, where there is no island. Keeping
    the height flat is a weaker signal than keeping the c region solid.

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


def solid_c_faces(H, n):
    """c-faces all of whose eight neighbours are c-faces."""
    from experiments.island import c_face_mask
    cm = c_face_mask(np.asarray(H, dtype=np.int64), n)
    return ndimage.binary_erosion(cm, structure=np.ones((3, 3)),
                                  border_value=0)


def island_exists(H, n, frac=0.2):
    """True iff a solid-c cluster spans the central window left-to-right.

    Returns (exists, None, width_fraction) where width_fraction is the widest
    horizontal extent achieved by any solid-c cluster, as a fraction of the
    window -- a continuous companion to the yes/no answer, which shows HOW a
    marginal case fails rather than only that it does. The middle slot is kept
    for call compatibility; solidity carries no level label.
    """
    solid = solid_c_faces(H, n)
    m = solid.shape[0]
    lo = int(round(m * (1 - frac) / 2))
    hi = int(round(m * (1 + frac) / 2))
    W = solid[lo:hi, lo:hi]
    if W.size == 0 or W.shape[1] == 0:
        return False, None, 0.0
    w = W.shape[1]

    lab, nl = ndimage.label(W, structure=np.ones((3, 3)))
    best = 0.0
    for c in range(1, nl + 1):
        cols = np.nonzero((lab == c).any(axis=0))[0]
        if cols.size:
            ext = float(cols.max() - cols.min() + 1) / w
            best = max(best, ext)
    return best >= 1.0, None, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--frac", type=float, default=0.2,
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
