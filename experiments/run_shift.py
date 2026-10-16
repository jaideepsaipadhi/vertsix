#!/usr/bin/env python3
"""Experiment 2, reframed: the DETERMINISTIC SHIFT, not the Gaussian shape.

    python3 experiments/run_shift.py --mode n      # n-sweep at fixed Delta
    python3 experiments/run_shift.py --mode delta  # Delta-sweep at fixed n

Why this and not a histogram goodness-of-fit.

Scouting 400 exact samples at n=32 gave, for the centred island height k:

    Delta=-1.2   k=-1: 80   k=0: 319   k=+1: 1
    Delta=-1.5   k=-1: 81   k=0: 318   k=+1: 1

Three occupied cells. A discrete Gaussian has two parameters, so fitting it to
three cells leaves ZERO degrees of freedom -- the fit is saturated and matches
by construction. That is not a weak test, it is not a test at all, and no
sample size fixes it: with sigma ~ 0.37 the mass on |k| >= 2 is ~1e-6, so the
fourth cell never populates. Running 1e4 samples, as the earlier plan had it,
would have bought a chi-square p-value that means nothing.

What the three cells DO determine, exactly and without any shape assumption,
is mu and sigma. From the counts above:

    log(80/319)  = -(1 + 2mu) / (2 sigma^2)
    log( 1/319)  = -(1 - 2mu) / (2 sigma^2)
    =>  sigma = 0.374,  mu = -0.307

So the measurable content of Gorin's conjecture here is the pair (mu, sigma):
whether sigma stays bounded (it does -- that IS the discrete-Gaussian
signature) and what the deterministic shift mu actually is. mu = -0.307 is
neither 0 nor a half-integer, which is the interesting part and is exactly the
"deterministic shift" his statement leaves unspecified.

Two further things the scout exposed:

* Delta=-1.2 and -1.5 returned 80/319/1 and 81/318/1 -- matching to within one
  count, which is the same seeds producing the same islands, not two
  independent agreements. Every config below therefore gets a DISJOINT seed
  block. (Re-run that way, mu came out -0.298 and -0.304: the agreement was
  real, but the scout had no right to claim it.)

* Errors come from a bootstrap over samples, not from the curvature of a fit
  to three points.

WHAT IS AND IS NOT IDENTIFIED -- read this before quoting mu or sigma.

At 2000 samples the occupied cells are, for Delta=-3:

    n=16   k=-1: 349   k=0: 1647   k=+1: 4
    n=24   k=-1: 347   k=0: 1653   k=+1: 0
    n=32   k=-1: 363   k=0: 1635   k=+1: 2

The k=+1 cell holds about 0.15% of the mass, so its count is a Poisson draw
with mean ~3. n=24 is not a different regime -- it drew a zero.

That single cell is the ONLY thing separating mu from sigma: with two cells a
two-parameter fit is underdetermined and the MLE runs along a ridge, which is
why the n=24 bootstrap interval below comes out absurdly narrow. Those tight
intervals are an artifact of the degeneracy, not precision. The script now
flags it.

The quantity that IS measured, to a few percent and without any shape
assumption, is the near-neighbour ratio

    R = P(k=-1) / P(k=0)   ~ 0.212,  flat in n and in Delta

together with the observation that the distribution is strongly ONE-SIDED:
P(+1)/P(0) ~ 0.0015, some 140x smaller. The island sits one level low about a
fifth of the time and essentially never one level high.

Pinning (mu, sigma) separately therefore needs the +1 cell measured, and
measuring it to 10% needs ~7e5 samples -- about 30 hours at n=32. That is a
handover job (--mode tail), not an in-session one.
"""
from __future__ import annotations

import argparse, csv, os, sys, time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sixvertex.cftp_exact as ce
from sixvertex.cftp_exact import cftp_sample
from experiments.island import island
from experiments.mixing import weights_for_delta


def mu_sigma(ks):
    """Exact MLE for a discrete Gaussian on the integers, from raw samples.

    Solved numerically rather than from the closed form above, so it stays
    correct if a fourth cell ever does populate.
    """
    from scipy.optimize import minimize
    ks = np.asarray(ks, float)
    support = np.arange(ks.min() - 5, ks.max() + 6)

    def nll(p):
        mu, ls = p
        s2 = np.exp(2 * ls)
        logZ = np.log(np.exp(-(support - mu) ** 2 / (2 * s2)).sum())
        return float(np.sum((ks - mu) ** 2 / (2 * s2) + logZ))

    r = minimize(nll, [ks.mean(), np.log(max(ks.std(), 0.2))],
                 method="Nelder-Mead")
    return float(r.x[0]), float(np.exp(r.x[1]))


def boot(ks, reps=400, seed=0):
    rng = np.random.default_rng(seed)
    ks = np.asarray(ks, float)
    mus, sds = [], []
    for _ in range(reps):
        try:
            m, s = mu_sigma(rng.choice(ks, len(ks)))
            mus.append(m); sds.append(s)
        except Exception:
            pass
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(mus), q(sds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["n", "delta", "tail"], required=True)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "tail":
        # the overnight job: one config, enough samples to populate k=+1
        configs = [(32, -3.0)]
        out = args.out or "shift_tail.csv"
    elif args.mode == "n":
        configs = [(n, -3.0) for n in (16, 24, 32, 40, 48, 56)]
        out = args.out or "shift_n.csv"
    else:
        configs = [(32, d) for d in (-1.2, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0)]
        out = args.out or "shift_delta.csv"

    if not ce._HAS_NUMBA:
        print("WARNING: numba missing, ~70x slower.\n")

    print(f"{len(configs)} configs x {args.samples} exact samples -> {out}")
    t0 = time.time()
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "delta", "samples", "mu", "mu_lo", "mu_hi",
                     "sigma", "sigma_lo", "sigma_hi", "ratio_lo_over_mode",
                     "ratio_hi_over_mode", "var_k", "cells", "identified",
                     "counts", "ambiguous"])
        # disjoint seed block per config -- the whole point
        for idx, (n, d) in enumerate(configs):
            base = 1_000_000 + idx * 10_000_000
            w = weights_for_delta(d)
            ks, amb = [], 0
            for s in range(args.samples):
                H, _ = cftp_sample(n, w, master_seed=base + s)
                r = island(H, n)
                if r["separation"] < 2.0:
                    amb += 1
                if r["k"] is not None:
                    ks.append(r["k"])
            c = Counter(ks)
            mu, sd = mu_sigma(ks)
            (mlo, mhi), (slo, shi) = boot(ks, seed=idx)
            # the identifiable summary: neighbour ratios about the mode
            mode = max(c, key=c.get)
            n0 = c[mode]
            r_lo = c.get(mode - 1, 0) / n0
            r_hi = c.get(mode + 1, 0) / n0
            # (mu, sigma) need three populated cells; with two the MLE is a
            # ridge and the bootstrap interval is meaningless, not narrow.
            ident = len([v for v in c.values() if v >= 3]) >= 3

            wr.writerow([n, d, len(ks), f"{mu:.4f}", f"{mlo:.4f}", f"{mhi:.4f}",
                         f"{sd:.4f}", f"{slo:.4f}", f"{shi:.4f}",
                         f"{r_lo:.5f}", f"{r_hi:.5f}",
                         f"{np.var(ks):.4f}", len(c), int(ident),
                         ";".join(f"{k:g}:{v}" for k, v in sorted(c.items())),
                         amb])
            fh.flush()
            flag = "" if ident else "   <-- (mu,sigma) NOT IDENTIFIED: <3 populated cells, CIs meaningless"
            print(f"  n={n:3d} D={d:+.1f}  R_lo={r_lo:.4f} R_hi={r_hi:.5f}  "
                  f"mu={mu:+.3f} sigma={sd:.3f}  cells={len(c)}  "
                  f"amb={amb}  [{(time.time()-t0)/60:.1f} min]{flag}", flush=True)
    print(f"\nwrote {out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
