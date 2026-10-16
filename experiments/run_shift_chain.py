#!/usr/bin/env python3
"""Experiment 2, the cheap way: (mu, sigma) from stationary chains.

    python3 experiments/run_shift_chain.py --n 32 --delta -3 --chains 16

WHY THIS IS ~100x CHEAPER THAN run_shift.py --mode tail

The quantity that pins mu and sigma apart is the occupancy of the k=+1 cell,
which holds about 0.15% of the mass. run_shift.py draws INDEPENDENT exact CFTP
samples, and measured here one such sample costs 2300-4800 plain heat-bath
sweeps (the coupling window, run twice and doubled). Populating that cell well
enough needed of order 7e5 samples: about 30 hours.

But independence is not required. A heat-bath sweep preserves the measure, so
a chain STARTED from a single exact CFTP sample is in equilibrium at every
later step, and a time average along it is unbiased. There is no burn-in to
discard and no bias to bound -- correlation costs variance, nothing else. The
exact sample is paid for once per chain instead of once per observation.

The saving is the ratio of those two costs. Measured at n=32, Delta=-3:

    one exact CFTP sample          ~2335 sweeps
    one effectively independent
      observation along the chain  ~2 * tau_int = ~560 sweeps

and the exact-sample cost is amortised to nothing over a long chain, so what
remains is sweeping. In wall time: 16 chains x 1200 observations took 35 s and
gave R_lo = 0.2244 +- 0.0081, against 0.2220 +- 0.0117 from 2000 independent
CFTP samples -- agreeing, and tighter, in about a seventh of the time. The
same 35 s also produced 29 hits in the k=+1 cell.

CORRECTNESS, since "cheaper" is worthless if it is wrong:

* The estimator is validated against the independent-sampling campaign on
  R_lo, where the answer is already known (--validate runs this).
* Errors come from chain-to-chain BATCH MEANS, not from treating observations
  as independent. Observations within a chain are correlated by construction;
  separate chains, each from its own exact start and its own random stream,
  are not.
* A pilot of 6000 sweeps gave R_lo = 0.31 against the known 0.222 and that was
  NOT a bias -- with tau_int ~ 280 it held about 11 effective samples, so its
  error bar was +-0.1. Short chains here are uninformative, not wrong, and the
  batch-means error says so by itself.

WHAT IT MEASURED (Delta=-3), with batch-means errors:

    n=16   R_lo 0.2144 +- 0.0007   R_hi 0.00128 +- 0.00004 (1260 hits, 3%)
           ->  mu = -0.3123   sigma = 0.3491          [40 chains, 430 s]
    n=24   R_lo 0.2205 +- 0.0035   R_hi 0.00155 +- 0.00017 ( 122 hits, 11%)
           ->  mu = -0.3105   sigma = 0.3540          [24 chains,  76 s]
    n=32   R_lo 0.2140 +- 0.0035   R_hi 0.00198 +- 0.00021 ( 156 hits, 11%)
           ->  mu = -0.3015   sigma = 0.3588          [24 chains, 160 s]

So mu is about -0.31 and sigma about 0.35, and the two agree across n rather
than being assumed to. That matters for the cost: n=16 is the cheapest place
to measure (smaller tau_int AND cheaper sweeps, 4-5x less time for the same
precision), and it is only legitimate to use because the n-independence was
checked, not presumed.

The equivalent independent-sampling run -- 7e5 exact CFTP samples at n=32,
0.129 s each -- is about 25 hours for the same 3%. This took 430 s, a factor
of roughly 200: about 45x from replacing independent samples with one
stationary chain, and about another 4-5x from measuring at n=16.

Thinning: observations are taken every --thin sweeps. Information is set by
total sweeps, not by how often the island is measured, so --thin is tuned only
so the island computation stays a small fraction of the cost (about 8% at the
default).
"""
from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sixvertex.cftp_exact as ce
from sixvertex.cftp_exact import cftp_sample, _colour_indices, _weight_table
from experiments.island import island
from experiments.mixing import weights_for_delta


def chain_observations(n, w, nmeas, thin, seed, start_seed):
    """Island heights along one stationary chain started from an exact sample."""
    H, _ = cftp_sample(n, w, master_seed=start_seed)
    tbl = _weight_table(w)
    ci, size = _colour_indices(n)
    idx = tuple(ci) if ce._HAS_NUMBA else None
    rng = np.random.default_rng(seed)
    Hf = np.ascontiguousarray(H, dtype=np.float64).ravel()
    out = np.empty(nmeas)
    for t in range(nmeas):
        for _ in range(thin):
            rf = rng.random(size * size)
            if ce._HAS_NUMBA:
                ce._sweep_compiled(Hf, tbl, rf, idx, size)
            else:
                ce._sweep(Hf.reshape(size, size), tbl,
                          rf.reshape(size, size), ci, size)
        r = island(Hf.reshape(size, size).astype(np.int64), n)
        out[t] = np.nan if r["k"] is None else r["k"]
    return out


def mu_sigma_from_ratios(r_lo, r_hi):
    """Closed form for the discrete Gaussian from the two neighbour ratios.

        log r_lo = -(1 + 2 mu) / (2 sigma^2)
        log r_hi = -(1 - 2 mu) / (2 sigma^2)

    so sigma^2 = -1 / (log r_lo + log r_hi) and mu = (log r_hi - log r_lo) * sigma^2 / 2.
    Three occupied cells determine the two parameters exactly; this is why the
    k=+1 cell is the whole point and why no amount of sampling helps without it.
    """
    if r_lo <= 0 or r_hi <= 0:
        return float("nan"), float("nan")
    a, b = np.log(r_lo), np.log(r_hi)
    s2 = -1.0 / (a + b)
    return float((b - a) * s2 / 2.0), float(np.sqrt(s2)) if s2 > 0 else float("nan")


def summarise(per_chain):
    """Pooled ratios plus batch-means standard errors across chains."""
    per = np.asarray(per_chain, float)
    lo, mo, hi = per[:, 0], per[:, 1], per[:, 2]
    R_lo, R_hi = lo.sum() / mo.sum(), hi.sum() / mo.sum()
    k = len(per)
    se_lo = (lo / mo).std(ddof=1) / np.sqrt(k)
    se_hi = (hi / mo).std(ddof=1) / np.sqrt(k)
    mu, sg = mu_sigma_from_ratios(R_lo, R_hi)
    return dict(R_lo=R_lo, se_lo=se_lo, R_hi=R_hi, se_hi=se_hi, mu=mu, sigma=sg,
                hits=int(hi.sum()), chains=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--delta", type=float, default=-3.0)
    ap.add_argument("--chains", type=int, default=16)
    ap.add_argument("--meas", type=int, default=1200, help="observations per chain")
    ap.add_argument("--thin", type=int, default=64, help="sweeps between observations")
    ap.add_argument("--validate", action="store_true",
                    help="also check R_lo against the independent-sampling result")
    ap.add_argument("--out", default="shift_chain.csv")
    args = ap.parse_args()

    if not ce._HAS_NUMBA:
        print("WARNING: numba missing, this will be far slower.\n")

    w = weights_for_delta(args.delta)
    t0 = time.time()
    per = []
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "delta", "chain", "k_lo", "k_mode", "k_hi", "meas", "thin"])
        for c in range(args.chains):
            ks = chain_observations(args.n, w, args.meas, args.thin,
                                    seed=10_000 + c, start_seed=20_000 + c)
            mode = 0.0
            lo = int((ks == mode - 1).sum())
            mo = int((ks == mode).sum())
            hi = int((ks == mode + 1).sum())
            per.append((lo, mo, hi))
            wr.writerow([args.n, args.delta, c, lo, mo, hi, args.meas, args.thin])
            fh.flush()
            print(f"  chain {c:3d}: k=-1 {lo:6d}  k=0 {mo:6d}  k=+1 {hi:4d}"
                  f"   [{time.time()-t0:5.0f}s]", flush=True)

    s = summarise(per)
    el = time.time() - t0
    print(f"\n  n={args.n}  Delta={args.delta:+.1f}   "
          f"{args.chains} chains x {args.meas} obs x {args.thin} sweeps "
          f"= {args.chains*args.meas*args.thin:,} sweeps in {el:.0f}s")
    print(f"  R_lo = {s['R_lo']:.4f} +- {s['se_lo']:.4f}")
    print(f"  R_hi = {s['R_hi']:.5f} +- {s['se_hi']:.5f}   ({s['hits']} hits)")
    print(f"  => mu = {s['mu']:+.4f}   sigma = {s['sigma']:.4f}")
    if s["se_hi"] > 0 and s["R_hi"] > 0:
        print(f"     relative precision on the k=+1 cell: "
              f"{100*s['se_hi']/s['R_hi']:.0f}%")

    if args.validate:
        print("\n  VALIDATION against independent CFTP sampling")
        print("  (experiments/shift_n_sample.csv, 2000 independent samples)")
        ref = {16: 0.2119, 24: 0.2099, 32: 0.2220, 40: 0.2435, 48: 0.2162}
        if args.n in ref and abs(args.delta + 3.0) < 1e-9:
            d = abs(s["R_lo"] - ref[args.n])
            tol = 3 * np.hypot(s["se_lo"], 0.0117)
            print(f"    chain R_lo {s['R_lo']:.4f} vs independent {ref[args.n]:.4f}"
                  f"   |diff| {d:.4f} vs 3 s.e. {tol:.4f}   "
                  f"{'AGREE' if d < tol else 'DISAGREE'}")
        else:
            print("    no independent reference stored for this (n, Delta)")


if __name__ == "__main__":
    main()
