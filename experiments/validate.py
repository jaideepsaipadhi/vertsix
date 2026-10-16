#!/usr/bin/env python3
"""Known-answer tests for both measurement campaigns.

Gorin: "for either of 1 or 2, you need to do some testing of your code before
showing the final results: it is very easy to do some mistakes here, and since
we do not know the outcome in advance, we will have less ways to detect that
the picture is wrong."

That is the whole point of this file. Neither campaign has a predictable
answer, so there is no wrong-looking plot to catch an error. Everything below
is something whose answer IS known independently of the measurement, so it can
fail loudly.

    python3 experiments/validate.py

Run it before trusting anything the two run_* scripts produce.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sixvertex.cftp_exact as ce
from sixvertex.cftp_exact import cftp_sample, extremal_height
from experiments.mixing import (delta_of, forward_coalescence_time,
                                weights_for_delta)
from experiments.island import island

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- experiment 1
def validate_mixing():
    print("\nEXPERIMENT 1 -- coalescence time vs Delta")

    # (a) the sweep must stay inside the monotone region and hit the right Delta
    ok = True
    for d in (0.5, 0.0, -1.0, -3.0, -5.0):
        w = weights_for_delta(d)
        if abs(delta_of(w) - d) > 1e-9 or not ce.in_monotone_region(w):
            ok = False
    check("the Delta family is inside the monotone region and hits its target", ok)

    # (b) the SAMPLER must be exact at each Delta, or the timings are timings of
    #     the wrong measure. Brute force at n=4.
    from tests.test_sampler import (classify_vertices,
                                    enumerate_height_functions)
    n = 4
    cfgs = enumerate_height_functions(n)
    keys = {c.astype(np.int64).tobytes(): i for i, c in enumerate(cfgs)}
    worst = 0.0
    N = 3000
    for d in (0.5, 0.0, -1.5, -3.0):
        w = weights_for_delta(d)
        wts = []
        for H in cfgs:
            cv = classify_vertices(H, n)
            x = 1.0
            for k, v in cv.items():
                x *= w[k] ** v
            wts.append(x)
        exact = np.array(wts) / sum(wts)
        cnt = np.zeros(len(cfgs))
        for s in range(N):
            H, _ = cftp_sample(n, w, master_seed=770000 + s)
            cnt[keys[np.asarray(H, dtype=np.int64).tobytes()]] += 1
        worst = max(worst, float(np.max(np.abs(cnt / cnt.sum() - exact))))
    check("CFTP reproduces the exact law at every Delta on the sweep",
          worst < 3 / np.sqrt(N), f"max deviation {worst:.4f} vs 3/sqrt(N)={3/np.sqrt(N):.4f}")

    # (c) a bound that holds for any correct implementation: one sweep moves a
    #     site by at most 1 in each chain, so the initial gap closes by <= 2/sweep
    ok, detail = True, []
    for n_ in (8, 12, 16, 24):
        gap = int((extremal_height(n_, "hi") - extremal_height(n_, "lo")).max())
        w = weights_for_delta(0.0)
        taus = [forward_coalescence_time(n_, w, seed=s) for s in range(6)]
        if min(taus) < gap / 2:
            ok = False
        detail.append(f"n={n_}: min tau {min(taus)} >= {gap/2:.0f}")
    check("coalescence time respects the gap/2 lower bound", ok, "; ".join(detail))

    # (d) the two sweep implementations must agree exactly on the same randomness
    if ce._HAS_NUMBA:
        w = weights_for_delta(-3.0)
        pairs = []
        for s in range(5):
            ce._HAS_NUMBA = True
            a = forward_coalescence_time(12, w, seed=s)
            ce._HAS_NUMBA = False
            b = forward_coalescence_time(12, w, seed=s)
            pairs.append((a, b))
        ce._HAS_NUMBA = True
        check("compiled and numpy sweeps give identical coalescence times",
              all(a == b for a, b in pairs), str(pairs))
    else:
        print("  SKIP  compiled/numpy agreement (numba not installed)")

    # (e) monotonicity must never break, or 'coalescence' is meaningless
    ok = True
    for d in (0.0, -3.0):
        try:
            forward_coalescence_time(12, weights_for_delta(d), seed=1,
                                     check_monotone=True)
        except AssertionError:
            ok = False
    check("the coupling stays monotone throughout", ok)

    # (f) disjoint seed ranges must give the same distribution
    w = weights_for_delta(-2.0)
    A = [forward_coalescence_time(12, w, seed=s) for s in range(150)]
    B = [forward_coalescence_time(12, w, seed=s) for s in range(50000, 50150)]
    try:
        from scipy.stats import ks_2samp
        p = float(ks_2samp(A, B).pvalue)
        check("coalescence times are independent of the seed stream", p > 0.01,
              f"KS p={p:.3f}")
    except ImportError:
        check("coalescence times are independent of the seed stream",
              abs(np.median(A) - np.median(B)) < 0.35 * np.median(A))


# ---------------------------------------------------------------- experiment 2
def validate_island():
    print("\nEXPERIMENT 2 -- island height")

    n, N = 32, 40
    w = weights_for_delta(-3.0)
    rows = [island(cftp_sample(n, w, master_seed=880000 + s)[0], n)
            for s in range(N)]

    # (a) unambiguous: Gorin's "large component, not a much smaller one"
    seps = [r["separation"] for r in rows]
    check("the island dominates the next component",
          float(np.median(seps)) > 5,
          f"median separation {np.median(seps):.1f}x, "
          f"{np.mean([s >= 2 for s in seps]):.0%} of samples above 2x")

    # (b) cross-validate the level against an INDEPENDENT definition: the
    #     largest constant plateau of the parity-corrected height, found
    #     without reference to vertex types at all. If the two disagree, one of
    #     them is measuring the wrong thing.
    #
    #     Purity is deliberately NOT thresholded here. The 8-connected
    #     component extends a little past the flat plateau, so its modal level
    #     covers only about half its corners (48-57% measured) by geometry,
    #     not by error -- a purity cut would just be a number chosen to pass.
    from scipy import ndimage
    from experiments.island import corrected_height

    def plateau_level(H, n_):
        G = corrected_height(np.asarray(H, dtype=np.int64), n_)
        best = (None, 0)
        for v in np.unique(G):
            lab, k = ndimage.label(G == v)
            if k == 0:
                continue
            m = int(ndimage.sum(G == v, lab, range(1, k + 1)).max())
            if m > best[1]:
                best = (int(v), m)
        return best[0]

    agree = 0
    for s in range(N):
        H, _ = cftp_sample(n, w, master_seed=880000 + s)
        r = island(H, n)
        if r["level"] is not None and r["level"] == plateau_level(H, n):
            agree += 1
    check("the island level agrees with the independent plateau definition",
          agree / N >= 0.85,
          f"{agree}/{N} agree; mean purity {np.mean([r['purity'] for r in rows]):.1%} "
          f"(~50% is the expected geometry, not an error)")

    # (c) k must land on the integers, or the lattice constant is wrong
    ks = [r["k"] for r in rows if r["k"] is not None]
    check("the centred height k is integer valued",
          all(abs(k - round(k)) < 1e-9 for k in ks),
          f"values {sorted(set(ks))}")

    # (d) a discrete Gaussian has variance bounded in n -- if ours grows, the
    #     observable is wrong
    vs = []
    for n_ in (24, 32, 48):
        kk = [island(cftp_sample(n_, w, master_seed=990000 + s)[0], n_)["k"]
              for s in range(30)]
        vs.append(float(np.var([x for x in kk if x is not None])))
    check("variance of k does not grow with n", max(vs) < 3 * min(vs) + 0.05,
          f"var at n=24,32,48: {[round(v,3) for v in vs]}")

    # (e) Gorin: "This island should be there only for Delta < -1." Reported,
    #     not asserted -- it is a prediction the measurement should be able to
    #     confirm or contradict, so failing it loudly here would beg the
    #     question. The number to watch is the area fraction.
    print("      Gorin's prediction -- island only for Delta < -1 -- area fraction:")
    for d in (0.5, 0.0, -0.5, -1.0, -2.0, -4.0):
        ww = weights_for_delta(d)
        fr = [island(cftp_sample(32, ww, master_seed=1010000 + s)[0], 32)["area_fraction"]
              for s in range(12)]
        sp = [island(cftp_sample(32, ww, master_seed=1010000 + s)[0], 32)["separation"]
              for s in range(12)]
        print(f"        Delta={d:+5.1f}  area {np.mean(fr):6.1%}   separation {np.median(sp):6.1f}x")


if __name__ == "__main__":
    t0 = time.time()
    print("Known-answer tests. Neither campaign has a predictable outcome, so")
    print("these check things whose answers are known independently.")
    print(f"(numba active: {ce._HAS_NUMBA})")
    validate_mixing()
    validate_island()
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}"
          f"   ({time.time() - t0:.0f}s)")
    sys.exit(1 if FAILURES else 0)
