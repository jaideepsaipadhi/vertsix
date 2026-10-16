#!/usr/bin/env python3
"""Experiment 2: the height of the c-vertex island, and whether it is a
discrete Gaussian.

    python3 experiments/validate.py          # run this FIRST
    python3 experiments/run_island.py        # then this

Writes island.csv with one row per exact sample, and prints a discrete-Gaussian
fit per (n, Delta).

Sample count matters more than n here, and that is not obvious. The measured
variance of the centred height k is about 0.13-0.22 across Delta from -1.2 to
-5, and it does NOT grow with n (0.14-0.18 for n = 24..64) -- which is itself
the bounded variance a discrete Gaussian predicts. With sd ~ 0.42, the mass
sits on two or three lattice points and the tails carry the actual shape
information: a fit at P(k) ~ 1e-2 needs ~1e4 samples before that cell is
populated at all. Running at larger n instead buys nothing and costs n^4.

So: moderate n, many samples.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sixvertex.cftp_exact as ce
from sixvertex.cftp_exact import cftp_sample
from experiments.island import discrete_gaussian_fit, island
from experiments.mixing import weights_for_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[32, 40])
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[-1.5, -2.0, -3.0, -5.0])
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--min-separation", type=float, default=2.0,
                    help="flag samples where the island does not dominate")
    ap.add_argument("--out", default="island.csv")
    args = ap.parse_args()

    if not ce._HAS_NUMBA:
        print("WARNING: numba is not installed, so this will be ~70x slower.")
        print("         pip install numba")
        print()

    total = len(args.n) * len(args.deltas) * args.samples
    print(f"{total} exact samples -> {args.out}")
    t0 = time.time()
    done = 0

    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "delta", "seed", "k", "level", "size", "runner_up",
                     "separation", "purity", "area_fraction", "ambiguous"])
        for n in args.n:
            for d in args.deltas:
                w = weights_for_delta(d)
                ks, amb = [], 0
                for s in range(args.samples):
                    H, _ = cftp_sample(n, w, master_seed=s)
                    r = island(H, n)
                    a = int(r["separation"] < args.min_separation)
                    amb += a
                    wr.writerow([n, d, s, r["k"], r["level"], r["size"],
                                 r["runner_up"], f"{r['separation']:.3f}",
                                 f"{r['purity']:.4f}",
                                 f"{r['area_fraction']:.4f}", a])
                    if r["k"] is not None:
                        ks.append(r["k"])
                    done += 1
                fh.flush()

                cnt = Counter(ks)
                fit = discrete_gaussian_fit(ks)
                el = time.time() - t0
                print(f"\n  n={n}, Delta={d:+.2f}   ({done}/{total}, {el/60:.1f} min)")
                print(f"    mean k {np.mean(ks):+.4f}   var {np.var(ks):.4f}   "
                      f"ambiguous {amb}/{args.samples}")
                print(f"    counts {dict(sorted(cnt.items()))}")
                print(f"    discrete Gaussian fit: mu={fit['mu']:+.4f} "
                      f"sigma={fit['sigma']:.4f}  "
                      f"chi2={fit['chi2']:.2f} p={fit['pvalue']:.3f} "
                      f"({fit['n_cells_used']} cells)")
                if fit["n_cells_used"] < 3:
                    print("    NOTE: fewer than 3 cells have enough mass to test a")
                    print("          shape. This is not evidence for or against the")
                    print("          conjecture -- it needs more samples.")

    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
