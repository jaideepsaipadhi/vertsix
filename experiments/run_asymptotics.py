#!/usr/bin/env python3
"""Experiment 1, follow-up: is the deep-antiferroelectric slowdown a power law?

    python3 experiments/run_asymptotics.py --out asym.csv

The Delta sweep left one question open. Fitting tau ~ n^alpha over n=16..48
gave alpha rising from ~2.2 near Delta=-1 to 2.98 at Delta=-4, but a rising
exponent over a short range is equally consistent with a power law of larger
exponent and with the onset of something superpolynomial. The two are only
separated further out in n.

Comparing the two models on the n=16..48 grid does not settle it: a power law
beats a pure exponential at every Delta measured, but at Delta=-4 its residual
sum jumps about fivefold relative to Delta=-3.5 while the exponential's does
not. That is upward curvature on the log-log plot -- the signature of a fit
running out of road, not of a clean exponent.

So this script measures the LOCAL exponent

    alpha_local(n1, n2) = log(tau(n2)/tau(n1)) / log(n2/n1)

at successive pairs, out to n=80, at Delta=-4 and -- as a control -- at
Delta=-1, where the global fit was flat at 2.21. A power law has a local
exponent that is constant in n; anything superpolynomial has one that climbs.
The control matters: if alpha_local climbs at BOTH Delta values then the cause
is the measurement (censoring, the heavy tail, the seed set), not the model.

Coalescence time is heavy-tailed, so every point is a median over --seeds
independent runs and the uncertainty is bootstrapped, not taken from a
normal approximation.
"""
from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sixvertex.cftp_exact as ce
from experiments.mixing import forward_coalescence_time, weights_for_delta


def boot_median(x, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [np.median(rng.choice(x, len(x))) for _ in range(reps)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=float, nargs="+", default=[-4.0, -1.0])
    ap.add_argument("--n", type=int, nargs="+", default=[32, 40, 48, 56, 64, 80])
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--max-sweeps", type=int, default=1 << 22)
    ap.add_argument("--out", default="asym.csv")
    args = ap.parse_args()

    if not ce._HAS_NUMBA:
        print("WARNING: numba missing, this will be ~70x slower.\n")

    t0 = time.time()
    res = {}
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["delta", "n", "seeds", "median_tau", "lo", "hi",
                     "censored", "alpha_local", "prev_n"])
        for d in args.deltas:
            w = weights_for_delta(d)
            prev = None
            for n in args.n:
                taus, cens = [], 0
                # Disjoint seed block per (Delta, n). It used to be
                # `... + n`, which made consecutive sizes share 32 of their
                # 40 seeds (n=32 used base+32..71, n=40 used base+40..79) --
                # and _rand_field draws prod(shape) values from a stream keyed
                # only on (seed, k), so a shared seed hands n=32 and n=40 the
                # same random prefix at every sweep. Measured at Delta=-1,
                # n=16 vs 20, 60 seeds: Spearman rho 0.14 (p=0.29) on shared
                # seeds against -0.12 (p=0.37) on disjoint ones, so the
                # overlap did not in fact correlate the coalescence times and
                # asym_sample.csv is not in question. The blocks are made
                # disjoint anyway, so the comment is true of the code.
                base = 3_000_000 + int(abs(d) * 1000) * 1000 + 1000 * n
                for s in range(args.seeds):
                    t = forward_coalescence_time(n, w, seed=base + s,
                                                 max_sweeps=args.max_sweeps)
                    if t is None:
                        cens += 1
                    else:
                        taus.append(t)
                med = float(np.median(taus))
                lo, hi = boot_median(taus, seed=n)
                al = ""
                if prev is not None:
                    al = f"{np.log(med / prev[1]) / np.log(n / prev[0]):.3f}"
                wr.writerow([d, n, len(taus), f"{med:.0f}", f"{lo:.0f}",
                             f"{hi:.0f}", cens, al, prev[0] if prev else ""])
                fh.flush()
                print(f"  D={d:+.1f} n={n:3d}  median tau {med:9.0f} "
                      f"[{lo:.0f}, {hi:.0f}]  censored {cens:2d}  "
                      f"alpha_local {al or '   -':>6}  "
                      f"[{(time.time()-t0)/60:.1f} min]", flush=True)
                prev = (n, med)
                res.setdefault(d, []).append((n, med))

    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")
    print("\nA power law has alpha_local constant in n. If it climbs at")
    print("Delta=-4 but not at the control, the deep AF slowdown is not a")
    print("power law over this range.")


if __name__ == "__main__":
    main()
