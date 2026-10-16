#!/usr/bin/env python3
"""Experiment 1: coalescence time as a function of Delta.

    python3 experiments/validate.py          # run this FIRST
    python3 experiments/run_mixing.py        # then this

Writes mixing.csv with one row per (n, Delta, seed).

What is measured is the **forward coupling time**: the two extremal
configurations are started at time 0, driven by one shared random stream, and
the first sweep at which they agree is recorded.

It is deliberately not the number CFTP prints. That number is the first window
of a doubling schedule, so it is quantised to powers of two -- measured at
n=16 it sat at 256 for Delta = +0.5, 0, -1, -2 and -3 alike, a staircase that
hides the entire dependence being asked about. The forward time is the
unquantised quantity the schedule is bracketing.

Coalescence time is heavy-tailed, so the summary to trust is the median and
the quantiles, not the mean. Runs that do not coalesce within --max-sweeps are
recorded as censored rather than as the cap; treating them as the cap would
bias the average downwards exactly where the chain is slowest, which is the
regime the experiment is about.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sixvertex.cftp_exact as ce
from experiments.mixing import forward_coalescence_time, weights_for_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[16, 24, 32, 40],
                    help="lattice sizes")
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0,
                             -1.25, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0],
                    help="Delta values (must be <= 0.5)")
    ap.add_argument("--seeds", type=int, default=200,
                    help="independent runs per (n, Delta)")
    ap.add_argument("--max-sweeps", type=int, default=1 << 20)
    ap.add_argument("--out", default="mixing.csv")
    args = ap.parse_args()

    if not ce._HAS_NUMBA:
        print("WARNING: numba is not installed, so this will be ~70x slower.")
        print("         pip install numba")
        print()

    total = len(args.n) * len(args.deltas) * args.seeds
    print(f"{total} runs -> {args.out}")
    t0 = time.time()
    done = 0

    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "delta", "c", "seed", "tau", "censored"])
        for n in args.n:
            for d in args.deltas:
                w = weights_for_delta(d)
                taus = []
                for s in range(args.seeds):
                    tau = forward_coalescence_time(n, w, seed=s,
                                                   max_sweeps=args.max_sweeps)
                    wr.writerow([n, d, w["c1"], s,
                                 tau if tau is not None else "",
                                 0 if tau is not None else 1])
                    if tau is not None:
                        taus.append(tau)
                    done += 1
                fh.flush()
                cens = args.seeds - len(taus)
                med = np.median(taus) if taus else float("nan")
                q = np.percentile(taus, [10, 90]) if taus else (float("nan"),) * 2
                el = time.time() - t0
                print(f"  n={n:4d} Delta={d:+6.2f}  median tau {med:9.0f}  "
                      f"[p10 {q[0]:.0f}, p90 {q[1]:.0f}]  "
                      f"censored {cens:3d}/{args.seeds}   "
                      f"[{done}/{total}, {el/60:.1f} min]")

    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")
    print("Plot median tau against Delta, one line per n. Censored rows are")
    print("lower bounds, not values -- if any appear, say so on the plot.")


if __name__ == "__main__":
    main()
