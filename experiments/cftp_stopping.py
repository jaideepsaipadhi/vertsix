#!/usr/bin/env python3
"""The CFTP stopping time itself: how long until CFTP actually stops.

    python3 experiments/cftp_stopping.py --out cftp_stopping.csv

Gorin asked for the mixing time "(how long it takes until CFTP stops)" as a
function of Delta. experiments/run_mixing.py measures the FORWARD coalescence
time instead -- two chains started at time 0 under one shared stream -- and
justifies the substitution by noting that the T the sampler reports is
quantised to powers of two. That is true of the REPORTED number but not of
the quantity: the doubling schedule is a coarse search for the backward
coalescence time

    T* = min { T : the composed update map from -T to 0 is constant },

and T* itself is an ordinary integer. This measures T* exactly, so the plot
is the one that was asked for.

Why a binary search is legitimate here. CFTP indexes its randomness by
ABSOLUTE time -- the field used at step k is _rand_field(master_seed, k),
independent of T -- so extending the window reuses the same fields and
coalescence is monotone in T: if the chains meet from -T they meet from any
earlier start. The doubling run therefore brackets T* between the last power
of two that failed and the first that succeeded, and bisection inside that
bracket finds it. (A test asserts the monotonicity rather than trusting it.)

That argument needs the coupling to be monotone, and the absolute-time
indexing alone is not enough: what is tracked is whether the two EXTREMAL
chains agree, which certifies a constant composed map only while they cannot
cross. Outside the monotone region they can, and the predicate measurably
loses monotonicity in T, so cftp_stopping_time refuses there.

T* is NOT the forward coalescence time and the two answer different
questions: T* is what the sampler pays, the forward time is a property of the
coupling. Both are reported here so the difference is visible rather than
assumed away.
"""
from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sixvertex.cftp_exact as ce
from sixvertex.cftp_exact import (_colour_indices, _rand_field, _weight_table,
                                  extremal_height, in_monotone_region)
from experiments.mixing import forward_coalescence_time, weights_for_delta
from experiments.seeds import seed_block


def _coalesces_from(n, w, seed, T, tbl, colour_idx, size):
    """Do the two extremal chains agree when started at -T?"""
    lo = extremal_height(n, "lo")
    hi = extremal_height(n, "hi")
    idx = tuple(colour_idx) if ce._HAS_NUMBA else None
    for k in range(T, 0, -1):
        rnd = _rand_field(seed, k, lo.shape)
        if ce._HAS_NUMBA:
            ce._sweep_compiled(lo.ravel(), tbl, rnd.ravel(), idx, size)
            ce._sweep_compiled(hi.ravel(), tbl, rnd.ravel(), idx, size)
        else:
            lo = ce._sweep(lo, tbl, rnd, colour_idx, size)
            hi = ce._sweep(hi, tbl, rnd, colour_idx, size)
    return bool(np.array_equal(lo, hi))


def cftp_stopping_time(n, w, seed, max_T=1 << 20):
    """The exact T*, by doubling then bisection. Returns None if censored.

    Refused outside the monotone region, for the same reason
    forward_coalescence_time is: there the two extremal chains can cross, so
    their agreeing no longer certifies that the composed map is constant, and
    -- measured, not assumed -- agreement stops being monotone in T. At
    a1=a2=b1=b2=1, c1=c2=0.9, n=8, 15 of 40 seeds coalesce from -T and then
    fail to coalesce from -(T+1). Bisection on a non-monotone predicate
    returns whichever side of the bracket it happens to land on, so this used
    to hand back a confident wrong number (446 at c=0.5, n=8, seed 0) rather
    than refusing.
    """
    if not in_monotone_region(w):
        raise ValueError(
            "the CFTP stopping time is only defined where the coupling is "
            "monotone (c1c2 >= a1a2 and c1c2 >= b1b2); outside it the chains "
            "can cross, coalescence is not monotone in T, and the bisection "
            "that finds T* is invalid")
    tbl = _weight_table(w)
    colour_idx, size = _colour_indices(n)

    T = 1
    while T <= max_T:
        if _coalesces_from(n, w, seed, T, tbl, colour_idx, size):
            break
        T *= 2
    else:
        return None
    lo_fail, hi_ok = T // 2, T          # lo_fail is known to fail (or is 0)
    while hi_ok - lo_fail > 1:
        mid = (lo_fail + hi_ok) // 2
        if _coalesces_from(n, w, seed, mid, tbl, colour_idx, size):
            hi_ok = mid
        else:
            lo_fail = mid
    return hi_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[12, 16, 24, 32])
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0])
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default="cftp_stopping.csv")
    ap.add_argument("--raw", default=None,
                    help="also write one row per (n, Delta, seed); the "
                         "exponent fit needs these, a median has no spread")
    args = ap.parse_args()

    if not ce._HAS_NUMBA:
        print("WARNING: numba missing, this will be far slower.\n")

    t0 = time.time()
    raw_fh = raw_wr = None
    if args.raw:
        raw_fh = open(args.raw, "w", newline="")
        raw_wr = csv.writer(raw_fh)
        raw_wr.writerow(["n", "delta", "seed", "Tstar", "forward"])
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "delta", "seeds", "median_Tstar", "lo", "hi",
                     "median_forward", "censored"])
        for n in args.n:
            for d in args.deltas:
                w = weights_for_delta(d)
                star, fwd, cens = [], [], 0
                # base = 7_000_000 + n*1000 + s omitted Delta, so every
                # Delta at fixed n drew the same seeds -- and it collided
                # with run_asymptotics at n=16, |Delta|=4, where both land on
                # 7_016_000 and both call forward_coalescence_time, so the
                # two campaigns published the same numbers
                # ([155, 165, 307, 336, 294]). See experiments/seeds.
                for base in seed_block("cftp_stopping", n, d, args.seeds):
                    t = cftp_stopping_time(n, w, base)
                    if t is None:
                        cens += 1
                    else:
                        star.append(t)
                    f = forward_coalescence_time(n, w, seed=base)
                    if f is not None:
                        fwd.append(f)
                    if raw_wr is not None:
                        raw_wr.writerow([n, d, base,
                                         "" if t is None else t,
                                         "" if f is None else f])
                rng = np.random.default_rng(0)
                bs = [np.median(rng.choice(star, len(star))) for _ in range(800)]
                wr.writerow([n, d, len(star), f"{np.median(star):.1f}",
                             f"{np.percentile(bs, 2.5):.1f}",
                             f"{np.percentile(bs, 97.5):.1f}",
                             f"{np.median(fwd):.1f}", cens])
                fh.flush()
                if raw_fh is not None:
                    raw_fh.flush()
                print(f"  n={n:3d} Delta={d:+5.2f}  T* median {np.median(star):8.1f}"
                      f"  [{np.percentile(bs,2.5):.0f}, {np.percentile(bs,97.5):.0f}]"
                      f"   forward {np.median(fwd):8.1f}"
                      f"   ratio {np.median(star)/max(1,np.median(fwd)):5.2f}"
                      f"   [{time.time()-t0:5.0f}s]", flush=True)
    if raw_fh is not None:
        raw_fh.close()
    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
