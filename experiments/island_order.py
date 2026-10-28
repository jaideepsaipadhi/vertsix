#!/usr/bin/env python3
"""The island's order parameter: the solid-c core's area fraction.

    python3 experiments/island_order.py --n 32 48 64 --reps 60 --out ord.csv

WHY THIS AND NOT THE CROSSING PROBABILITY. experiments/island_exists.py asks
whether a solid-c cluster spans the central N/5 window. That predicate is a
clean yes/no and it is zero above Delta = -1, but its threshold is BIASED
DEEP, and necessarily so: near the transition the island occupies a vanishing
fraction of the lattice, so it fails a span test against any fixed fraction
of N while being genuinely present. Measured crossing probabilities put the
crossover at Delta = -1.5 to -2 at N <= 64 for exactly this reason.

The unbiased quantity is the size of the solid core itself,

    A(Delta, N) = (area of the largest solid-c component) / N^2,

where a c-face is solid when all eight of its neighbours are c-faces. A is an
order parameter in the usual sense: it should tend to a positive constant
where the island exists and to zero where it does not, with no window and no
span threshold anywhere in its definition. The transition is then located by
where A stops decaying in N, which is a statement about the sequence A(., N)
rather than about a cutoff chosen by hand.

Read the output as a family of curves in N. A(Delta, N) flat or growing in N
means a macroscopic island; A decaying like a power of N means none, and the
fitted exponent says how fast. The crossing probability remains the yes/no
answer to Gorin's question; this is how to find where that answer changes.
"""
from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sixvertex.cftp_exact import cftp_sample
from experiments.mixing import weights_for_delta
from experiments.island_exists import solid_c_faces


def core_area_fraction(H, n):
    """Area of the largest solid-c component, as a fraction of n^2."""
    solid = solid_c_faces(H, n)
    lab, nl = ndimage.label(solid, structure=np.ones((3, 3)))
    if nl == 0:
        return 0.0
    sizes = ndimage.sum(solid, lab, range(1, nl + 1))
    return float(sizes.max()) / (n * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[-0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0])
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--out", default="island_order.csv")
    args = ap.parse_args()

    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "delta", "reps", "area_frac", "se"])
        for n in args.n:
            for d in args.deltas:
                w = weights_for_delta(d)
                a = np.array([core_area_fraction(
                    cftp_sample(n, w, master_seed=51_000 + 97 * s)[0], n)
                    for s in range(args.reps)])
                se = a.std(ddof=1) / np.sqrt(len(a))
                wr.writerow([n, d, args.reps, f"{a.mean():.5f}", f"{se:.5f}"])
                fh.flush()
                print(f"  n={n:4d} Delta={d:+5.2f}  A = {a.mean():.4f}"
                      f" +- {se:.4f}   [{time.time()-t0:5.0f}s]", flush=True)
    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
