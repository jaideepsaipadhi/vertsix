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

    # (c) k must land on a lattice of spacing 1, or the lattice constant is
    #     wrong. It is NOT integer valued in general: k = (G - n/2)/2 with G
    #     moving in steps of 2, so the offset is set by n mod 4 -- integer at
    #     0 mod 4, half-integer at 2 mod 4, quarter-integer at odd n. Checking
    #     integrality at n=32 alone would pass while asserting something false,
    #     so the check is the spacing, run at all three residues.
    #     The offset itself is left unpinned: it takes four values, one per
    #     residue, and fixing them from a dozen samples each would be fitting
    #     the sample rather than checking the lattice. Spacing 1 is the claim.
    ok, detail = True, []
    for n_ in (32, 34, 33, 35):
        kk = [island(cftp_sample(n_, w, master_seed=880000 + s)[0], n_)["k"]
              for s in range(16)]
        kk = [x for x in kk if x is not None]
        base = min(kk)
        if not all(abs((k - base) - round(k - base)) < 1e-9 for k in kk):
            ok = False
        detail.append(f"n={n_} (n%4={n_ % 4}): {sorted(set(kk))}")
    check("the centred height k sits on a spacing-1 lattice at every n mod 4",
          ok, "; ".join(detail))

    # (d) a discrete Gaussian has variance bounded in n -- if ours grows, the
    #     observable is wrong.
    #
    # This check used to read `max(vs) < 3 * min(vs) + 0.05` over n = 24, 32,
    # 48 at 30 samples, under the title "variance of k does not grow with n".
    # It did not test that. Over a factor-two range in n, max/min < 3 tolerates
    # var ~ n^alpha for any alpha < log2(3) = 1.58, and with the +0.05 slack it
    # passes outright under:
    #
    #     bounded        vs = [0.150, 0.150, 0.150]   ratio 1.00   passes
    #     log n growth   vs = [0.150, 0.164, 0.183]   ratio 1.22   passes
    #     sqrt n         vs = [0.150, 0.173, 0.212]   ratio 1.41   passes
    #     LINEAR in n    vs = [0.150, 0.200, 0.300]   ratio 2.00   passes
    #     n^1.6          vs = [0.150, 0.238, 0.455]   ratio 3.03   passes
    #
    # A check in this file that cannot fail under linear growth is the exact
    # thing the module docstring says this file exists to prevent.
    #
    # Replaced by a log-log slope with a threshold taken from the estimator's
    # MEASURED spread, not from an estimate of it. Six independent repeats of
    # exactly the measurement below (n = 16, 24, 32 at 120 samples) give
    # slopes
    #
    #     -0.010  +0.084  +0.180  +0.418  -0.485  +0.164
    #     mean +0.058,  sd 0.302,  max |slope| 0.485
    #
    # so the slope estimator's sd here is 0.30. That number is the whole
    # story, and it is worth being blunt about what it means: over a factor-
    # two range in n, with var(k) this noisy, this check CANNOT distinguish
    # bounded variance from logarithmic growth (effective exponent 0.29) or
    # even from sqrt-n growth (0.5) -- both sit inside one sigma. My first
    # attempt at this threshold was 0.5, which is 1.65 sigma and would fail
    # roughly one run in ten on correct code; the repeat at -0.485 above is
    # that near-miss.
    #
    # The threshold is 0.9, three measured sigma. What that genuinely
    # excludes is growth as fast as linear, which is what a broken observable
    # would look like -- and that is all it is claimed to do. The old form
    # excluded nothing up to n^1.58.
    #
    # Boundedness itself is established elsewhere, with the lever arm this
    # script cannot afford: n = 16, 24, 32, 48, 64 at 200 samples gives
    # var(k) = 0.163, 0.134, 0.200, 0.134, 0.200, a slope of +0.098 over a
    # factor of FOUR in n, where log growth would give 0.29. Recorded in
    # experiments/island.py with the replicate study behind it. Eight
    # independent 200-sample blocks at n = 32 (0.1444, 0.1630, 0.1600,
    # 0.1816, 0.1375, 0.1716, 0.1570, 0.1476; mean 0.1578, sd 0.0146) also
    # show the bootstrap intervals quoted there are conservative rather than
    # narrow.
    ns = (16, 24, 32)
    vs = []
    for n_ in ns:
        kk = [island(cftp_sample(n_, w, master_seed=990000 + 5000 * n_ + s)[0],
                     n_)["k"] for s in range(120)]
        vs.append(float(np.var([x for x in kk if x is not None])))
    alpha = float(np.polyfit(np.log(ns), np.log(vs), 1)[0])
    check("var(k) shows no growth in n as fast as linear "
          "(3 measured sigma; too noisy here to exclude log n -- see note)",
          abs(alpha) < 0.9,
          f"var at n={','.join(map(str, ns))}: {[round(v, 3) for v in vs]}; "
          f"log-log slope {alpha:+.3f}, sigma 0.30 "
          f"(bounded 0, log n 0.29, sqrt n 0.5, linear 1)")

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
