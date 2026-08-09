"""Regression tests for the six-vertex sampler.

Run with:  python3 -m pytest tests/test_sampler.py -v
       or:  python3 tests/test_sampler.py       (plain, no pytest needed)

These exist because this project has had two subtle correctness bugs that
"looked fine" for months:

  1. The live MCMC acceptance ratio ignored that a single flip changes up to
     four surrounding faces, not just a c1/c2 pair.
  2. CFTP used p_up = c1/(c1+c2), which only targets the correct measure at
     the uniform point -- it was 64% off at c1=c2=sqrt(8).

Both survived because the two samplers were checked *against each other*
while sharing the same flawed assumption. So every correctness test here
compares against brute-force enumeration, which is independent of both.
"""
import math
import os
import sys
from itertools import product

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sixvertex.exact import ExactSampler, exact_sample  # noqa: E402


# --------------------------------------------------------------------------
# Brute-force reference implementation (independent of the samplers)
# --------------------------------------------------------------------------

def enumerate_height_functions(n):
    H = np.zeros((n + 1, n + 1), dtype=int)
    for i in range(n + 1):
        for j in range(n + 1):
            if i == 0:
                H[i, j] = j
            elif i == n:
                H[i, j] = n - j
            elif j == 0:
                H[i, j] = i
            elif j == n:
                H[i, j] = n - i
    results = []
    interior = [(i, j) for i in range(1, n) for j in range(1, n)]

    def backtrack(idx):
        if idx == len(interior):
            for i in range(n + 1):
                for j in range(n):
                    if abs(int(H[i, j + 1]) - int(H[i, j])) != 1:
                        return
            for i in range(n):
                for j in range(n + 1):
                    if abs(int(H[i + 1, j]) - int(H[i, j])) != 1:
                        return
            results.append(H.copy())
            return
        i, j = interior[idx]
        up, left = H[i - 1, j], H[i, j - 1]
        for v in {up - 1, up + 1, left - 1, left + 1}:
            if abs(v - up) == 1 and abs(v - left) == 1:
                H[i, j] = v
                backtrack(idx + 1)
        H[i, j] = 0

    backtrack(0)
    return results


def classify_vertices(H, n):
    counts = {"a1": 0, "a2": 0, "b1": 0, "b2": 0, "c1": 0, "c2": 0}
    mapping = {(0, 0, 0, 0): "a1", (1, 1, 1, 1): "a2",
               (1, 1, 0, 0): "b1", (0, 0, 1, 1): "b2",
               (0, 1, 1, 0): "c1", (1, 0, 0, 1): "c2"}
    for i in range(n):
        for j in range(n):
            t = 1 if H[i, j + 1] - H[i, j] == 1 else 0
            b = 1 if H[i + 1, j + 1] - H[i + 1, j] == 1 else 0
            l = 1 if H[i + 1, j] - H[i, j] == 1 else 0
            r = 1 if H[i + 1, j + 1] - H[i, j + 1] == 1 else 0
            counts[mapping[(l, t, b, r)]] += 1
    return counts


def exact_distribution(n, w):
    cfgs = enumerate_height_functions(n)
    wts = []
    for H in cfgs:
        c = classify_vertices(H, n)
        x = 1.0
        for k, v in c.items():
            x *= w[k] ** v
        wts.append(x)
    return cfgs, np.array(wts) / sum(wts), sum(wts)


UNIFORM = {"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0, "c1": 1.0, "c2": 1.0}
DELTA_M3 = {"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0,
            "c1": math.sqrt(8), "c2": math.sqrt(8)}
ASYM = {"a1": 1.5, "a2": 0.8, "b1": 1.2, "b2": 0.9, "c1": 2.1, "c2": 0.7}
EXTREME_LO = {k: 0.1 for k in UNIFORM}
EXTREME_HI = {k: 3.0 for k in UNIFORM}


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_asm_counts():
    """At the ice point the partition function is the ASM count."""
    expected = {1: 1, 2: 2, 3: 7, 4: 42, 5: 429}
    for n, want in expected.items():
        got = ExactSampler(n, UNIFORM).partition_function()
        assert abs(got - want) < 1e-9, f"n={n}: got {got}, want {want}"


def test_partition_function_matches_brute_force():
    for w, label in [(UNIFORM, "uniform"), (DELTA_M3, "Delta=-3"),
                     (ASYM, "asym"), (EXTREME_LO, "all 0.1"),
                     (EXTREME_HI, "all 3.0")]:
        for n in (3, 4):
            _, _, z_bf = exact_distribution(n, w)
            z_tm = ExactSampler(n, w).partition_function()
            rel = abs(z_tm - z_bf) / z_bf
            assert rel < 1e-12, f"{label} n={n}: rel err {rel}"


def test_sampled_distribution_matches_exact():
    """The whole point: sampling frequencies must match the true measure."""
    for w, label in [(UNIFORM, "uniform"), (DELTA_M3, "Delta=-3"), (ASYM, "asym")]:
        n = 4
        cfgs, ep, _ = exact_distribution(n, w)
        keys = {C.astype(np.int64).tobytes(): i for i, C in enumerate(cfgs)}
        S = ExactSampler(n, w)
        rng = np.random.default_rng(12345)
        cnt = np.zeros(len(cfgs))
        N = 40000
        for _ in range(N):
            H = S.sample(rng)
            idx = keys.get(H.astype(np.int64).tobytes())
            assert idx is not None, f"{label}: sampled an INVALID configuration"
            cnt[idx] += 1
        emp = cnt / cnt.sum()
        # 4/sqrt(N) is a loose but non-vacuous bound on max deviation
        tol = 4.0 / math.sqrt(N)
        assert np.max(np.abs(emp - ep)) < tol, f"{label}: max dev too large"


def test_samples_are_valid_height_functions():
    for w in (UNIFORM, DELTA_M3, ASYM):
        for n in (4, 7, 9):
            H, _ = exact_sample(n, a1=w["a1"], a2=w["a2"], b1=w["b1"],
                                b2=w["b2"], c1=w["c1"], c2=w["c2"], seed=7)
            for i in range(n + 1):
                for j in range(n):
                    assert abs(int(H[i, j + 1]) - int(H[i, j])) == 1
            for i in range(n):
                for j in range(n + 1):
                    assert abs(int(H[i + 1, j]) - int(H[i, j])) == 1
            # DWBC boundary
            for j in range(n + 1):
                assert H[0, j] == j and H[n, j] == n - j
            for i in range(n + 1):
                assert H[i, 0] == i and H[i, n] == n - i


def test_conservation_law_not_a_bug():
    """N_a1-N_a2, N_b1-N_b2 and N_c1-N_c2 are CONSTANT under DWBC.

    Consequence: swapping a1<->a2 (or c1<->c2) leaves the distribution
    exactly unchanged. This looks like a wiring bug in naive "boost one
    weight, see which type grows" tests -- it is not. Do not "fix" it.
    """
    n = 5
    diffs = set()
    for H in enumerate_height_functions(n):
        c = classify_vertices(H, n)
        diffs.add((c["a1"] - c["a2"], c["b1"] - c["b2"], c["c1"] - c["c2"]))
    assert len(diffs) == 1, f"expected one invariant triple, got {diffs}"

    _, d1, _ = exact_distribution(n, {**UNIFORM, "a1": 5.0})
    _, d2, _ = exact_distribution(n, {**UNIFORM, "a2": 5.0})
    assert np.max(np.abs(d1 - d2)) < 1e-12
    # sanity: the test can detect a real difference
    _, d3, _ = exact_distribution(n, {**UNIFORM, "b1": 5.0})
    assert np.max(np.abs(d1 - d3)) > 1e-3


def test_rejects_invalid_weights():
    """Non-positive / non-finite weights must fail loudly, not return junk.

    Regression: negative weights previously produced a 'valid-looking'
    sample from a distribution that does not exist.
    """
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            ExactSampler(4, {**UNIFORM, "c1": bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"weight {bad} should have been rejected")


def test_rejects_invalid_n():
    for bad in (0, -3, 2.5, "4"):
        try:
            ExactSampler(bad, UNIFORM)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"n={bad!r} should have been rejected")


def test_seed_reproducibility():
    a, _ = exact_sample(6, c1=2.0, c2=1.0, seed=42)
    b, _ = exact_sample(6, c1=2.0, c2=1.0, seed=42)
    c, _ = exact_sample(6, c1=2.0, c2=1.0, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cftp_only_correct_at_uniform_point():
    """Documents the bug that motivated the rewrite.

    CFTP's p_up = c1/(c1+c2) move is fine at the uniform point and badly
    wrong away from it. This test pins that down so nobody reinstates it as
    a general-purpose method.
    """
    from sixvertex.cftp import _half_sweep
    from sixvertex.sampler import SixVertexSampler

    n = 4
    def cftp_style_empirical(c1, c2, N=25000):
        w = {**UNIFORM, "c1": c1, "c2": c2}
        cfgs, ep, _ = exact_distribution(n, w)
        keys = {C.astype(np.int64).tobytes(): i for i, C in enumerate(cfgs)}
        p_up = c1 / (c1 + c2)
        i_idx, j_idx = np.meshgrid(np.arange(n+1), np.arange(n+1), indexing="ij")
        interior = (i_idx > 0) & (i_idx < n) & (j_idx > 0) & (j_idx < n)
        parity = (i_idx + j_idx) % 2
        masks = [interior & (parity == 0), interior & (parity == 1)]
        H = SixVertexSampler.extremal_height(n, "lo").astype(np.float64)
        rng = np.random.default_rng(4)
        for _ in range(1500):
            for m in masks:
                H = _half_sweep(H, m, rng.random(H.shape).astype(np.float32), p_up)
        cnt = np.zeros(len(cfgs))
        for _ in range(N):
            for m in masks:
                H = _half_sweep(H, m, rng.random(H.shape).astype(np.float32), p_up)
            idx = keys.get(H.astype(np.int64).tobytes())
            if idx is not None:
                cnt[idx] += 1
        return np.max(np.abs(cnt / cnt.sum() - ep))

    assert cftp_style_empirical(1.0, 1.0) < 0.02, "should be fine at uniform"
    assert cftp_style_empirical(2.0, 1.0) > 0.05, "should be visibly wrong away from it"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
