#!/usr/bin/env python3
"""How does the deterministic shift depend on Delta?

    python3 experiments/run_shift_delta_chain.py --out shift_delta_chain.csv

Gorin's statement is that the island height is a discrete Gaussian "after a
deterministic shift", and leaves the shift unspecified. experiments/
run_shift_chain pins it at one Delta; this maps it across the
antiferroelectric phase. It is only affordable because of the stationary-chain
estimator -- the same measurement by independent CFTP sampling is tens of
hours per Delta value.

Two things make this harder than repeating one measurement:

* tau_int grows with depth. Measured at n=16: about 27 sweeps at Delta=-3,
  67 at -4, 98 at -6, 193 at -8. Thinning is set from that per Delta, so the
  island computation stays a small share of the cost at every point and the
  deep points are not silently under-sampled.

* The k=+1 cell gets RARER with depth (its count fell 7,6,7,3,1,1,0 across
  Delta=-1.2..-5 in the independent campaign), and that cell is the only thing
  separating mu from sigma. So the deep end needs more sweeps for the same
  precision, not fewer. The per-point relative precision is reported so an
  under-determined point is visible rather than quietly folded into a curve.

Errors are chain-to-chain batch means throughout.
"""
from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sixvertex.cftp_exact as ce
from experiments.mixing import weights_for_delta
from experiments.run_shift_chain import chain_observations, summarise

# thinning set from measured tau_int(k) at n=16
THIN = {-1.5: 32, -2.0: 32, -2.5: 32, -3.0: 32, -4.0: 64, -5.0: 80,
        -6.0: 96, -8.0: 192}


def thin_for(d):
    keys = sorted(THIN)
    return THIN[min(keys, key=lambda k: abs(k - d))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[-1.5, -2.0, -3.0, -4.0, -6.0, -8.0])
    ap.add_argument("--chains", type=int, default=32)
    ap.add_argument("--meas", type=int, default=10000)
    ap.add_argument("--out", default="shift_delta_chain.csv")
    args = ap.parse_args()

    if not ce._HAS_NUMBA:
        print("WARNING: numba missing, this will be far slower.\n")

    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "delta", "thin", "chains", "meas", "R_lo", "se_lo",
                     "R_hi", "se_hi", "hits", "mu", "sigma", "seconds"])
        for d in args.deltas:
            thin = thin_for(d)
            w = weights_for_delta(d)
            per = []
            # disjoint seed block per Delta
            base = 50_000 + int(abs(d) * 100) * 100
            for c in range(args.chains):
                ks = chain_observations(args.n, w, args.meas, thin,
                                        seed=base + c, start_seed=base + 500 + c)
                per.append((int((ks == -1).sum()), int((ks == 0).sum()),
                            int((ks == 1).sum())))
            s = summarise(per)
            el = time.time() - t0
            wr.writerow([args.n, d, thin, args.chains, args.meas,
                         f"{s['R_lo']:.5f}", f"{s['se_lo']:.5f}",
                         f"{s['R_hi']:.6f}", f"{s['se_hi']:.6f}", s["hits"],
                         f"{s['mu']:.4f}", f"{s['sigma']:.4f}", f"{el:.0f}"])
            fh.flush()
            rel = 100 * s["se_hi"] / s["R_hi"] if s["R_hi"] > 0 else float("nan")
            print(f"  Delta={d:+5.1f} thin={thin:3d}  R_lo={s['R_lo']:.4f}"
                  f"+-{s['se_lo']:.4f}  R_hi={s['R_hi']:.5f}+-{s['se_hi']:.5f}"
                  f" ({s['hits']:5d} hits, {rel:3.0f}%)   "
                  f"mu={s['mu']:+.4f} sigma={s['sigma']:.4f}   [{el:.0f}s]",
                  flush=True)
    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
