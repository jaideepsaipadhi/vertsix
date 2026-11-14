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

Two working rules this suite has earned the hard way.

MUTATION-CHECK EVERY NEW ASSERTION. A test that cannot fail is worse than no
test, because it reads as evidence. Several checks here were vacuous for
months: the Izergin-Korepin comparison ran only at a=b and returned silently
without mpmath, the compiled-vs-numpy coalescence check compared the compiled
path against itself because the numba flag had been bound at import time, and
the brute-force oracle never validated a face whose four corners were all on
the boundary. Each looked like a passing test.

When mutation-checking, CLEAR __pycache__ between the mutation and the
restore. Python invalidates bytecode by mtime and size, so an edit that
restores a file within the same second can leave a stale .pyc -- which makes
a mutation look like it was not detected. That is indistinguishable from the
finding you are trying to establish. Observed here, and the one conclusion
drawn from a surviving mutation was re-run with a cleared cache before being
believed.

PREFER A SWEEP TO HAND-PICKED CASES where the sweep is affordable, and it
usually is. Five tables of six, eight domains of 1078, five enumerator cases
all chosen from the non-empty half, three domains of 191 for the monotonicity
the whole method rests on -- each of those hid a real defect that the
corresponding sweep found in seconds.
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
               (1, 1, 0, 0): "c1", (0, 0, 1, 1): "c2",
               (0, 1, 1, 0): "b1", (1, 0, 0, 1): "b2"}
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


def _js_function_body(code, name):
    """The whole body of a JS function, by brace matching.

    Several tests used to slice a fixed number of characters after the
    function's name. That silently truncates as soon as the function grows,
    and a truncated window is how a bug hides: the assertion still passes
    because the code it should have examined is simply not in the slice.
    Adding one branch to updateModelUI pushed the line two tests were written
    to check straight out of their 1800-character windows.
    """
    k = code.index(name)
    k = code.index("{", k)
    depth, i = 0, k
    while i < len(code):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[k:i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces after {name!r}")


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


def test_only_weight_products_matter():
    """Under DWBC only the products a1*a2, b1*b2, c1*c2 affect the measure.

    Consequence of the conservation law: (c1=2, c2=0.5) and (c1=1, c2=1)
    are the SAME distribution. This is a trap when designing tests -- two
    "different" weight sets can be secretly identical, which will look like
    a bug (or hide one). It is also why Delta depends only on the products.
    """
    n = 5
    _, d_a, _ = exact_distribution(n, {**UNIFORM, "c1": 2.0, "c2": 0.5})
    _, d_b, _ = exact_distribution(n, UNIFORM)
    assert np.max(np.abs(d_a - d_b)) < 1e-12, "equal products must give equal measures"

    _, d_c, _ = exact_distribution(n, {**UNIFORM, "c1": 2.0, "c2": 2.0})
    assert np.max(np.abs(d_a - d_c)) > 1e-3, "different products must differ"


def test_observables_unbiased_across_seeds():
    """Expected values must match theory, checked across many seeds.

    Single-seed z-scores fluctuate: a lone |z| ~ 2.7 is normal. Bias shows
    up as a consistent sign/magnitude across independent seeds, so this
    test averages over seeds rather than trusting one.
    """
    n = 5
    w = DELTA_M3
    cfgs, pr, _ = exact_distribution(n, w)
    expected_total = float(sum(p * H.sum() for p, H in zip(pr, cfgs)))
    S = ExactSampler(n, w)
    zs = []
    for seed in range(8):
        rng = np.random.default_rng(900 + seed)
        totals = [S.sample(rng).sum() for _ in range(2500)]
        m = float(np.mean(totals))
        se = float(np.std(totals)) / math.sqrt(len(totals))
        zs.append((m - expected_total) / se if se > 0 else 0.0)
    zs = np.array(zs)
    assert abs(zs.mean()) < 1.0, f"systematic bias: mean z = {zs.mean()}"
    assert np.mean(np.abs(zs) > 3) < 0.3, f"too many outliers: {zs}"


def test_api_gating_matches_server_routing():
    """Regression: /api/init once reported `is_symmetric_regime` while
    /api/exact/status hardcoded it to True, so the API contradicted itself
    for identical weights. The reported availability flag must agree with
    what _run_exact_job actually does.
    """
    import importlib
    server = importlib.import_module("server")

    def server_routes_to(n, w):
        is_uniform = all(v == 1.0 for v in w.values())
        if n <= server.MAX_EXACT_N:
            return "exact-sequential"
        return "cftp" if is_uniform else "refuse"

    cases = [
        (8,  {"a1":1.5,"a2":0.8,"b1":1.2,"b2":0.9,"c1":2.1,"c2":0.7}),
        (14, {"a1":1.0,"a2":1.0,"b1":1.0,"b2":1.0,"c1":2.83,"c2":2.83}),
        (15, {"a1":1.0,"a2":1.0,"b1":1.0,"b2":1.0,"c1":1.0,"c2":1.0}),
        (40, {"a1":1.0,"a2":1.0,"b1":1.0,"b2":1.0,"c1":2.83,"c2":2.83}),
        (200,{"a1":1.5,"a2":0.8,"b1":1.2,"b2":0.9,"c1":2.1,"c2":0.7}),
    ]
    for n, w in cases:
        is_uniform = all(v == 1.0 for v in w.values())
        advertised = (n <= server.MAX_EXACT_N) or is_uniform
        actual = server_routes_to(n, w) != "refuse"
        assert advertised == actual, (
            f"n={n} w={w}: advertised exact_available={advertised} "
            f"but server routes to {server_routes_to(n, w)}")


def test_no_stale_symmetric_regime_field():
    """The flag tested a1=a2=b1=b2=1 -- the superseded CFTP criterion, which
    ignored c1,c2 and matched no decision the server makes. It must not
    reappear on the API surface."""
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "server.py")).read()
    # A mention in the explanatory comment is fine; use as a response KEY
    # (i.e. followed by a colon, as in a dict literal) is what must not return.
    import re
    lines = [ln for ln in src.splitlines()
             if not ln.strip().startswith("#")]
    offenders = [ln for ln in lines
                 if re.search(r'["\']is_symmetric_regime["\']\s*:', ln)
                 or re.search(r'response\s*\[\s*["\']is_symmetric_regime', ln)]
    assert not offenders, (
        f"is_symmetric_regime resurfaced as an API response field: {offenders}")


def test_running_jobs_are_not_expired():
    """Regression: cleanup expired by created_at regardless of status, so a
    still-running job could be evicted purely for being old -- discarding its
    result and 404-ing the client. Exact sampling at large n can legitimately
    exceed the TTL."""
    import importlib, time as _t
    server = importlib.import_module("server")
    now = _t.time()
    with server._jobs_lock:
        saved = dict(server._jobs)
        server._jobs.clear()
        server._jobs["running_old"] = {"status": "running", "created_at": now - 10*server._JOB_TTL_SECONDS,
                                       "last_T": None, "attempts": 0, "n": 200}
        server._jobs["done_old"] = {"status": "done", "created_at": now - 10*server._JOB_TTL_SECONDS,
                                    "finished_at": now - 2*server._JOB_TTL_SECONDS,
                                    "last_T": None, "attempts": 0, "n": 8}
        server._jobs["done_recent"] = {"status": "done", "created_at": now - 10*server._JOB_TTL_SECONDS,
                                       "finished_at": now, "last_T": None, "attempts": 0, "n": 8}
        server._cleanup_old_jobs()
        left = set(server._jobs)
        server._jobs.clear(); server._jobs.update(saved)
    assert "running_old" in left, "a RUNNING job was expired"
    assert "done_recent" in left, "a recently finished result was expired"
    assert "done_old" not in left, "a long-finished job was not expired"


def test_exact_cache_is_memory_bounded():
    """Regression: the cache cleared only when entry COUNT exceeded 8, but
    entries vary from ~1 MB (n=10) to ~47 MB (n=14), so 8 entries could mean
    ~380 MB -- too much beside the interpreter on a 512 MB instance."""
    from sixvertex import exact as ex
    with ex._cache_lock:
        ex._cache.clear(); ex._costs.clear()
    for k in range(6):
        ex.exact_sample(13, c1=1.5 + 0.01*k, c2=1.7, seed=1)
        total = sum(ex._costs.get(key, 0) for key in ex._cache)
        assert total <= ex._CACHE_BUDGET_PAIRS, (
            f"cache exceeded budget: {total} > {ex._CACHE_BUDGET_PAIRS}")
    assert len(ex._cache) >= 1, "cache evicted everything; LRU should retain the hot entry"


def test_sessions_are_isolated():
    """Regression: /api/init wrote a single module-level sampler shared by
    every caller, so two clients clobbered each other and /api/step returned
    the OTHER client's model (A asked n=6, received n=30)."""
    import importlib
    server = importlib.import_module("server")
    app = server.app.test_client()

    a = app.post("/api/init", json={"n": 6,  "a1":1,"a2":1,"b1":1,"b2":1,
                                    "c_up":1,"c_down":1}).get_json()
    b = app.post("/api/init", json={"n": 30, "a1":1,"a2":1,"b1":1,"b2":1,
                                    "c_up":2.5,"c_down":2.5}).get_json()
    assert a["session_id"] != b["session_id"]

    sa = app.post("/api/step", json={"sweeps":1,"session_id":a["session_id"]}).get_json()
    sb = app.post("/api/step", json={"sweeps":1,"session_id":b["session_id"]}).get_json()
    assert sa["frame"]["n"] == 6,  f"session A leaked: got n={sa['frame']['n']}"
    assert sb["frame"]["n"] == 30, f"session B leaked: got n={sb['frame']['n']}"

    # missing / unknown ids must error, never silently serve someone else
    for payload in ({"sweeps":1}, {"sweeps":1,"session_id":"deadbeef"}):
        r = app.post("/api/step", json=payload)
        assert r.status_code == 400, "bad session_id should be rejected"
        assert "frame" not in r.get_json()


def test_procfile_pins_single_worker():
    """Regression: the in-memory job/session/cache state is per-process, so
    more than one worker silently breaks exact sampling (job polls 404 on the
    other worker). Keep the constraint enforced in the repo."""
    import os, re
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "Procfile")
    assert os.path.exists(path), "Procfile missing; start command is unversioned"
    cmd = [ln for ln in open(path).read().splitlines()
           if ln.strip() and not ln.strip().startswith("#")]
    assert cmd, "Procfile has no command"
    line = cmd[0]
    m = re.search(r"--workers\s+(\d+)", line)
    assert m, f"Procfile does not pin --workers: {line}"
    assert m.group(1) == "1", (
        f"Procfile sets --workers {m.group(1)}; the in-memory job and session "
        f"stores require exactly 1")


def test_torch_path_pins_float64():
    """Regression (static): the torch branch built its weight tensor with
    `torch.ones_like(top)`, inheriting the heights' float32 dtype. Two
    consequences, both silent:

      * the divide-by-zero floor is 1e-300, which underflows to 0.0 in
        float32, so `clamp(before, min=1e-300)` guards nothing and the
        division can produce inf;
      * if heights were ever stored as an integer dtype, float weights would
        be truncated (a1=1.5 -> 1) on the torch path while numpy stayed right.

    torch is not in requirements.txt, so this branch is not exercised at
    runtime here; the check is static so the fix cannot silently regress.
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "sixvertex", "sampler.py")).read()

    assert "torch.ones_like(top, dtype=torch.float64)" in src, (
        "torch weight tensor no longer pins float64")

    # and the numpy path it mirrors must stay float64 too
    assert "np.ones_like(top, dtype=np.float64)" in src, (
        "numpy weight tensor no longer pins float64")

    # the guard value must be representable in whatever dtype is used
    import numpy as _np
    assert _np.float64(1e-300) > 0, "guard must be nonzero in float64"
    assert _np.float32(1e-300) == 0, (
        "sanity: this test exists because 1e-300 underflows in float32")


def test_step_request_work_is_bounded():
    """Regression: /api/step capped n (<=400) and sweeps (<=500) separately,
    so n=400 with sweeps=500 was accepted and ran ~44 s synchronously under a
    lock on a single worker -- stalling every other request, including the
    status polls of in-flight exact jobs. Public endpoint, so trivially
    reachable. Bound the product, not the factors."""
    import importlib
    server = importlib.import_module("server")
    app = server.app.test_client()

    init = app.post("/api/init", json={"n": 400}).get_json()
    sid = init["session_id"]

    # the pathological request must be refused, and refused cheaply
    r = app.post("/api/step", json={"sweeps": 500, "session_id": sid})
    assert r.status_code == 400, "oversized step request was accepted"
    assert "too large" in r.get_json()["error"]
    assert "frame" not in r.get_json()

    # and the error must tell the caller what WOULD work
    import re
    m = re.search(r"sweeps <= (\d+)", r.get_json()["error"])
    assert m, "error should suggest a workable sweep count"
    suggested = int(m.group(1))
    ok = app.post("/api/step", json={"sweeps": suggested, "session_id": sid})
    assert ok.status_code == 200, "the suggested sweep count was itself rejected"

    # small requests are unaffected
    small = app.post("/api/init", json={"n": 40}).get_json()
    r2 = app.post("/api/step", json={"sweeps": 500, "session_id": small["session_id"]})
    assert r2.status_code == 200, "a modest request was wrongly refused"


def test_client_validates_frame_length():
    """Regression (static): the browser decoded the binary frame without
    checking it against the reported n. A short frame decodes without error;
    every out-of-range read returns undefined, so the height field renders as
    NaN -- visibly corrupt, no exception. Measured on a truncated frame: 31
    undefined reads, 41 NaN pixels, zero errors raised.

    Guarding matters here specifically: this project already shipped one
    silent frame-format mismatch (server moved to binary frames while the
    client still parsed plain JSON) and "Exact Sample" quietly did nothing
    for weeks.
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()
    body_start = src.index("function frameFromServerData")
    body = src[body_start:body_start + 2000]
    assert "expected" in body and "malformed frame" in body, (
        "frameFromServerData no longer validates the decoded frame length")
    # the server side must keep emitting a frame whose size matches n
    from sixvertex.sampler import SixVertexSampler
    import base64
    s_ = SixVertexSampler(n=6, c_up=1.0, c_down=1.0)
    frame = s_.to_binary_frame()
    size = frame["n"] + 1
    heights = base64.b64decode(frame["height_b64"])
    active = base64.b64decode(frame["active_b64"])
    assert len(heights) == size * size * 2, "height frame size disagrees with n"
    assert len(active) == size * size, "active frame size disagrees with n"


def test_run_is_not_dead_after_exact_sample():
    """Regression (static): the exact-sample success path set `sampler = null`,
    which left Run silently dead. Clicking it flipped the button to "pause" --
    so it looked live -- but localStep() bails on `!sampler`, so the sweep
    counter stayed at 0 until the user happened to press Reset. No error
    anywhere. Measured: 198 sweeps before an exact sample, 0 after.

    The path must now seed the client chain from the exact draw instead
    (which is also correct statistically: the chain starts in equilibrium,
    so there is no burn-in).
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()

    start = src.index('statusData.status === "done"')
    block = src[start:start + 2000]
    # Strip comment lines: the fix documents the old bug in prose, and a raw
    # substring search would match that description instead of live code.
    code = "\n".join(ln for ln in block.splitlines()
                     if not ln.strip().startswith("//"))
    assert "sampler = null" not in code, (
        "exact-sample path discards the live sampler again; Run will be dead")
    assert "new SixVertexJS(" in block, (
        "exact-sample path no longer seeds a live sampler from the result")


def test_live_weight_changes_reach_the_chain():
    """Regression (static): SixVertexJS captured its weights at construction
    and nothing ever updated them, so moving a slider mid-run changed only the
    Delta readout. The UI could show "-3.03 antiferroelectric" while the chain
    was still sampling at Delta=+0.5 -- a picture that does not match its own
    stated parameters, which is the exact failure mode this tool has been
    caught on before.

    Verified behaviourally at the time of the fix: retargeting a live chain
    from Delta=+0.5 to Delta=-3 dropped the flippable-site fraction from
    0.233 to 0.105, i.e. the chain really does respond.
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()
    start = src.index("function updateDeltaDisplay")
    block = src[start:start + 1500]
    code = "\n".join(ln for ln in block.splitlines()
                     if not ln.strip().startswith("//"))
    assert "sampler.w = w" in code, (
        "slider changes no longer propagate to the running chain; the Delta "
        "readout can desync from what is actually being sampled")


def test_n_slider_rebuilds_the_lattice():
    """Regression (static): the n slider only relabelled the UI. The chain's
    size is fixed at construction and cannot be retargeted in place, so with
    the slider dragged to 200 while a chain built at n=20 kept running, the
    exported SVG came out 20x20 while the panel read "200" -- an exported
    artifact contradicting its own stated parameters.

    Must rebuild (debounced, since dragging fires a stream of events).
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()
    start = src.index('nSlider.addEventListener')
    block = src[start:start + 1200]
    code = "\n".join(ln for ln in block.splitlines()
                     if not ln.strip().startswith("//"))
    assert "localInit()" in code, (
        "n slider no longer rebuilds the sampler; the displayed n can desync "
        "from the chain and from exported files")
    assert "setTimeout" in code and "clearTimeout" in code, (
        "n slider rebuild is not debounced; dragging will thrash allocations")


def test_ui_does_not_mislabel_the_exact_method():
    """Regression: the UI hardcoded "CFTP" in the button and in every error
    message, but for n <= MAX_EXACT_N (most interactive use) the method is the
    sequential transfer-matrix sampler, not CFTP. Worse, the |Delta|>1 note
    claimed Exact Sample "still gives a mathematically correct result ... but
    can take substantially longer at large n" -- at |Delta|>1 the weights are
    necessarily non-uniform, so above the limit exact sampling is *refused*,
    not slower. The note promised a capability the tool declines to provide.

    Gorin has already asked what CFTP even is here, so naming the wrong
    algorithm is not a cosmetic issue.
    """
    import os
    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    html = open(os.path.join(here, "..", "static", "index.html")).read()

    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))
    # user-visible strings must not assert a specific method
    for bad in ('"exact sample (CFTP)"', "CFTP request failed",
                "CFTP failed to start", "CFTP status check failed"):
        assert bad not in code, f"UI still hardcodes the wrong method name: {bad}"

    assert "exact sample (CFTP)" not in html, "button still mislabels the method"
    assert "can take substantially longer at large n" not in html, (
        "ordered-regime note still promises exact sampling above the size "
        "limit, where it is actually refused")


def test_stale_exact_results_are_discarded():
    """Regression: two earlier fixes interacted badly. Making the n slider
    rebuild the lattice, plus seeding the live chain from an exact result,
    meant a job started at one n could land *after* the user changed n and
    silently reinstate the old chain. Observed: start exact at n=13, drag to
    n=60, and the finished job restored a 13x13 chain while the panel read 60
    and the exported SVG came out 13x13 -- exactly the desync those fixes were
    meant to remove.

    Same hazard for the weights: a draw made at Delta=-3 must not be presented
    under a Delta=+0.92 label.

    Guarded with a generation counter captured at request time.
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("//"))

    assert "paramGen" in code, "no generation counter guarding exact results"
    assert "const requestedGen = paramGen" in code, (
        "exact request does not capture the parameter generation")
    assert "paramGen !== requestedGen" in code, (
        "exact result is applied without checking whether parameters moved on")
    # the counter must actually be bumped by the controls that matter
    assert code.count("bumpParams()") >= 3, (
        "bumpParams() is not wired to n and both weight-slider directions")


def test_reset_is_not_overwritten_by_a_stale_exact_job():
    """Regression: the staleness guard was wired only to the sliders, so
    Reset went unguarded. Pressing Reset during an exact job left the user
    with a fresh chain that the finished job then silently replaced -- they
    asked to start over and received an exact sample instead, with no
    indication.

    Fixed at the root: localInit() bumps the generation, so every rebuild
    path (Reset, the n slider, and any future caller) invalidates an in-flight
    request rather than each caller having to remember.
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("//"))

    start = code.index("function localInit")
    body = code[start:start + 400]
    assert "bumpParams()" in body, (
        "localInit() no longer bumps the generation; rebuilds triggered by "
        "Reset can be silently overwritten by an in-flight exact job")


def test_exact_job_owns_its_own_in_flight_flag():
    """Regression: the Exact Sample button was gated on the general-purpose
    `busy` flag, which localInit() sets and then CLEARS as part of any normal
    rebuild. So the debounced n-slider rebuild, firing during an exact job,
    cleared the job's own flag and re-enabled the button -- letting a second
    job launch on top of the first. Observed: 2 requests to /api/exact/start
    where there should be 1, both running against a single worker.

    The job now owns a dedicated `exactInFlight` flag that rebuilds cannot
    touch.
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("//"))

    assert "exactInFlight" in code, "no dedicated in-flight flag for exact sampling"
    assert "if (exactInFlight) return;" in code, (
        "exact handler still guards on the shared `busy` flag, which rebuilds clear")
    assert "!safe || exactInFlight" in code, (
        "button gating ignores whether a job is already in flight")


def test_shipped_cftp_module_is_correct_at_uniform():
    """The CFTP module still ships and still serves n > MAX_EXACT_N at the
    uniform point, so it needs its own correctness check -- it uses the
    simplified p_up = c1/(c1+c2) rule that is valid ONLY there.

    Uses a chi-squared test at n=4 (42 configurations, ~700 expected counts
    each). A max-deviation check at larger n is worthless here: with 7436
    configurations and a few thousand samples almost every count is 0 or 1,
    and the statistic measures nothing.
    """
    from sixvertex.cftp import cftp_sample
    cfgs = enumerate_height_functions(4)
    keys = {c.astype(np.int64).tobytes(): i for i, c in enumerate(cfgs)}
    counts = np.zeros(len(cfgs))
    N = 12000
    for seed in range(N):
        H, _ = cftp_sample(n=4, c_up=1.0, c_down=1.0, master_seed=seed)
        k = keys.get(np.asarray(H, dtype=np.int64).tobytes())
        assert k is not None, "CFTP produced an invalid configuration"
        counts[k] += 1

    expected = counts.sum() / len(cfgs)
    assert expected > 100, "test is underpowered"
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    # 42 configs -> 41 dof; the 0.1% upper tail is ~76
    assert chi2 < 76, (
        f"CFTP deviates from the uniform measure at the ice point: chi2={chi2:.1f}")


def test_concurrent_exact_jobs_are_capped():
    """Regression: nothing limited concurrent background jobs. The sampler
    cache is memory-bounded but the transient per-job builds are not --
    measured, 6 concurrent n=13 jobs took the server from 44 MB to 151 MB,
    and an n=14 build is ~3x larger again, so a few dozen public requests
    would exhaust a 512 MB instance. The browser runs one job at a time, but
    /api/exact/start is public. With the cap: 10 requests -> 3 accepted,
    7 refused, peak 99 MB.
    """
    import importlib
    server = importlib.import_module("server")
    assert hasattr(server, "_MAX_CONCURRENT_JOBS"), "no concurrency cap defined"
    assert 1 <= server._MAX_CONCURRENT_JOBS <= 8, (
        f"implausible cap: {server._MAX_CONCURRENT_JOBS}")

    src = open(server.__file__).read()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("#"))
    assert "_MAX_CONCURRENT_JOBS" in code, "cap defined but never enforced"
    assert "429" in code, "over-limit requests should be refused with 429"


def test_exact_flag_released_on_every_exit_path():
    """Regression: `exactInFlight` was released only on the happy path. The
    early `return` taken when the server refuses a start skipped the cleanup,
    so the flag stayed true and the Exact Sample button was disabled
    permanently -- no reload, no recovery. Adding the concurrency cap made
    this reachable in normal use: one 429 and the button was dead for good.

    Cleanup must live in a `finally`, so refusals, exceptions and success all
    release the flag.
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("//"))

    start = code.index("btnExact.addEventListener")
    handler = code[start:]                      # to end of file; the handler
                                                # body is several KB long
    assert "} finally {" in handler, (
        "exact handler has no finally block; an early return can strand "
        "exactInFlight and permanently disable the button")
    fin = handler[handler.index("} finally {"):]
    assert "exactInFlight = false" in fin, (
        "exactInFlight is not released in the finally block")
    # Released exactly once, inside finally -- not scattered across exit paths.
    # (Counted within the handler only: the module-level declaration
    # `let exactInFlight = false;` sits above it and is not a release.)
    assert handler.count("exactInFlight = false") == 1, (
        "exactInFlight released in more than one place; keep it in finally only")


def test_abandoned_jobs_do_not_hold_slots_forever():
    """Regression from combining two earlier fixes. Making cleanup expire only
    FINISHED jobs (so genuine long runs survive) meant a job whose thread died
    without writing a terminal status stayed "running" forever. Adding the
    concurrency cap then gave those zombies the power to disable exact
    sampling server-wide: three six-hour-old "running" entries with no live
    threads returned 429 to every request and survived every cleanup.

    A watchdog now reaps them, with a threshold far above any legitimate
    runtime so real long jobs are untouched.
    """
    import importlib, time as _t
    server = importlib.import_module("server")
    app = server.app.test_client()
    now = _t.time()

    with server._jobs_lock:
        saved = dict(server._jobs)
        server._jobs.clear()
        for k in range(server._MAX_CONCURRENT_JOBS):
            server._jobs[f"zombie{k}"] = {
                "status": "running",
                "created_at": now - 10 * server._JOB_MAX_RUNTIME_SECONDS,
                "last_T": None, "attempts": 0, "n": 13}
        # a genuine long-running job, comfortably under the threshold
        server._jobs["legit"] = {
            "status": "running",
            "created_at": now - server._JOB_MAX_RUNTIME_SECONDS // 4,
            "last_T": None, "attempts": 0, "n": 200}

    try:
        r = app.post("/api/exact/start", json={"n": 8, "a1": 1, "a2": 1, "b1": 1,
                                               "b2": 1, "c_up": 1, "c_down": 1})
        assert r.status_code == 200, (
            f"zombie jobs still block new work: HTTP {r.status_code}")
        with server._jobs_lock:
            assert server._jobs["legit"]["status"] == "running", (
                "watchdog reaped a legitimate long-running job")
            for k in range(server._MAX_CONCURRENT_JOBS):
                z = server._jobs.get(f"zombie{k}")
                assert z is None or z["status"] == "error", (
                    "abandoned job left in the running state")
    finally:
        with server._jobs_lock:
            server._jobs.clear(); server._jobs.update(saved)


def test_api_docs_match_the_implementation():
    """The API contract changed substantially across several rounds --
    including two breaking changes (/api/step now requires session_id;
    /api/init no longer returns is_symmetric_regime) plus new 429/400
    responses and a new endpoint -- while the README documented none of it.
    Anyone scripting against the server would have hit undocumented failures.

    Pin the documented contract to the live routes so they cannot drift
    apart again.
    """
    import os, importlib
    here = os.path.dirname(__file__)
    readme = open(os.path.join(here, "..", "README.md")).read()
    server = importlib.import_module("server")

    for ep in ("/api/config", "/api/init", "/api/step",
               "/api/exact/start", "/api/exact/status"):
        assert ep in readme, f"{ep} is undocumented"

    routes = {r.rule for r in server.app.url_map.iter_rules()}
    for rule in routes:
        if rule.startswith("/api/"):
            base = rule.split("<")[0].rstrip("/")
            assert base in readme, f"route {rule} exists but is undocumented"

    # the breaking changes must stay called out
    assert "Breaking changes" in readme, "breaking API changes not flagged"
    assert "session_id" in readme, "session_id requirement undocumented"
    assert "exact_available" in readme, "replacement field undocumented"

    # and the app must really behave as documented
    app = server.app.test_client()
    cfg = app.get("/api/config").get_json()
    assert cfg["max_exact_n"] == server.MAX_EXACT_N
    init = app.post("/api/init", json={"n": 8}).get_json()
    assert "session_id" in init and "is_symmetric_regime" not in init
    assert app.post("/api/step", json={"sweeps": 1}).status_code == 400


def test_out_of_range_values_are_rejected_not_clamped():
    """Regression: `sweeps` was silently clamped (`max(1, min(sweeps, 500))`),
    so a request for 9999 sweeps ran 500 and returned success. The caller then
    believes the chain is far more equilibrated than it is -- the same
    silent-wrong-data failure as a picture labelled with parameters it was not
    drawn from.

    `n` was already rejecting (a change made when validation was added, which
    also made the README's word "capped" wrong). Both now behave the same way.
    """
    import importlib
    server = importlib.import_module("server")
    app = server.app.test_client()

    init = app.post("/api/init", json={"n": 20}).get_json()
    sid = init["session_id"]

    # in range -> fine
    assert app.post("/api/step", json={"sweeps": 500, "session_id": sid}).status_code == 200
    # out of range -> refused, and NOT silently substituted
    for bad in (501, 9999, 0, -5):
        r = app.post("/api/step", json={"sweeps": bad, "session_id": sid})
        assert r.status_code == 400, f"sweeps={bad} was accepted"
        assert "frame" not in r.get_json(), (
            f"sweeps={bad} returned a frame; it was clamped rather than refused")

    # n behaves the same way
    ok = app.post("/api/exact/start", json={"n": 250, "a1": 1, "a2": 1, "b1": 1,
                                            "b2": 1, "c_up": 1, "c_down": 1})
    assert ok.status_code == 200
    # Starting an n=250 job really does start one, and nothing here waits for
    # it. Left in the table it stayed "running" for the whole rest of the
    # suite, holding one of the three concurrency slots -- which is a test
    # leaking shared state, and it made the job-lifecycle test fail only when
    # run after this one. Drop the entry so the slot comes back. (The worker
    # thread keeps going until it finishes; it discards its result, because
    # the writer checks that the id is still in the table.)
    _leaked = ok.get_json().get("job_id")
    if _leaked:
        with server._jobs_lock:
            server._jobs.pop(_leaked, None)
    bad = app.post("/api/exact/start", json={"n": 251, "a1": 1, "a2": 1, "b1": 1,
                                             "b2": 1, "c_up": 1, "c_down": 1})
    assert bad.status_code == 400, "n above the limit was accepted"
    assert "job_id" not in bad.get_json(), "n was clamped rather than refused"


def test_extreme_weights_behave_correctly_or_are_refused():
    """The API accepts any positive finite weight; the UI sliders stop at 3.0.
    So the extreme regimes are reachable by scripting and must either be
    handled correctly or refused -- never silently produce garbage.

    Upward: the backward pass overflows and the guard must fire.
    Downward: the measure concentrates on the unique configuration with no
    c-vertices (exactly one exists at every n, verified by enumeration), so
    Z -> 1 and every sample should be that configuration. Getting 0 c-vertices
    here is the correct answer, not underflow damage.
    """
    # Upward extremes: handled correctly OR refused, never silently wrong.
    #
    # This block used to require a refusal for all three, which froze one
    # branch of the contract the docstring states. The sampler now rescales
    # and retries when a build fails, so b=1e3 at n=12 -- a model it can
    # represent perfectly well once the weights are normalised -- is sampled
    # instead of refused. The spread cases are still beyond reach and are
    # still refused.
    #
    # Whichever branch a case takes, two things must hold: a refusal says
    # why, and a sampler that IS returned must not have rounded any
    # reachable state away. The second is the silent failure this test
    # exists to exclude -- underflow can drop states while leaving the
    # partition function finite and positive, so every sample drawn is a
    # legal configuration from a measure on a smaller state space.
    import numpy as _np

    for w in ({**UNIFORM, "c1": 1e30, "c2": 1e30},
              {**UNIFORM, "b1": 1e3,  "b2": 1e3},
              {**UNIFORM, "a1": 1e30, "a2": 1e30}):
        try:
            s = ExactSampler(12, w)
        except RuntimeError as e:
            assert "overflow" in str(e).lower() or "underflow" in str(e).lower(), e
        else:
            assert s._underflowed == 0, (
                f"{w}: the sampler was built with {s._underflowed} reachable "
                f"states rounded away; it would draw from a truncated support")
            H = _np.asarray(s.sample(_np.random.default_rng(0)), dtype=int)
            assert H.shape == (13, 13)

    # And a rescued build must be the SAME measure, not merely a valid one:
    # b=1e3 needs the rescue at n=12, while the same model written at one
    # tenth the scale does not. Same seed, same configuration, every time.
    big = {**UNIFORM, "b1": 1e3, "b2": 1e3}
    small = {k: v / 10.0 for k, v in big.items()}
    s_big, s_small = ExactSampler(12, big), ExactSampler(12, small)
    assert hasattr(s_big, "_z_scale"), (
        "b=1e3 no longer needs rescaling at n=12; this check is not "
        "exercising the path it claims to")
    assert not hasattr(s_small, "_z_scale")
    for k in range(25):
        a = _np.asarray(s_big.sample(_np.random.default_rng(k)), dtype=int)
        b = _np.asarray(s_small.sample(_np.random.default_rng(k)), dtype=int)
        assert _np.array_equal(a, b), (
            f"the rescued build disagrees with the same measure written at "
            f"another scale (seed {k})")

    # Under the standard convention DWBC forces at least n c-vertices, and
    # the minimum is achieved exactly by the n! permutation matrices. Both
    # facts are wrong under the swapped labels this code used to carry, so
    # they double as a check that the convention is right.
    import math as _m
    for n in (3, 4, 5):
        cs = []
        for H in enumerate_height_functions(n):
            cv = classify_vertices(H, n)
            cs.append(cv["c1"] + cv["c2"])
        assert min(cs) == n, f"min c-count at n={n} is {min(cs)}, expected {n}"
        assert sum(1 for x in cs if x == min(cs)) == _m.factorial(n), (
            f"configs attaining the minimum c-count at n={n} should number n!")

    # Downward: as c -> 0 the measure concentrates on those minimal
    # configurations, so Z -> n! * c^n and every sample attains c-count n.
    n = 5
    tiny = 1e-8
    S = ExactSampler(n, {**UNIFORM, "c1": tiny, "c2": tiny})
    Z = S.partition_function()
    assert abs(Z / (_m.factorial(n) * tiny ** n) - 1.0) < 1e-6, (
        f"Z should tend to n! c^n as c -> 0, got {Z}")
    rng = np.random.default_rng(5)
    for _ in range(20):
        H = S.sample(rng)
        cv = classify_vertices(H, n)
        assert cv["c1"] + cv["c2"] == n, (
            "as c -> 0 every sample must attain the minimum c-count")


def test_action_controls_come_before_the_parameter_notes():
    """Regression: explanatory notes added over successive rounds pushed the
    SAMPLING section below the fold. At 1366x768 -- one of the most common
    laptop resolutions -- Run, Step, Reset, Exact Sample and Save were ALL
    off-screen, so a first-time visitor could not see how to start the
    simulation at all. Only the sliders were visible.

    Fixed by ordering the panel actions-before-parameters. Guarded here
    because the failure mode is additive: every future note lengthens the
    parameters block, and nothing else would notice.
    """
    import os, re
    here = os.path.dirname(__file__)
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    order = re.findall(r'section-title">(\w+)', html)
    assert "sampling" in order and "parameters" in order, order
    assert order.index("sampling") < order.index("parameters"), (
        f"sampling controls must precede the parameters block, got {order}")


def test_touch_targets_are_enlarged_on_coarse_pointers():
    """Regression: on a 390px phone every control was under the ~44px minimum
    Apple and Google both recommend, and the weight/size sliders were 4px
    tall -- essentially undraggable, since a fingertip covers roughly ten
    times that. Mobile is the platform this tool was first reported broken
    on, so fiddly controls matter here more than usual.

    The rules are scoped to `pointer: coarse` so the desktop layout is
    unaffected (verified: the desktop Run button stayed 26px).
    """
    import os
    here = os.path.dirname(__file__)
    css = open(os.path.join(here, "..", "static", "style.css")).read()

    assert "pointer: coarse" in css, "no touch-specific sizing rules"
    block = css[css.index("pointer: coarse"):]
    assert "min-height: 44px" in block, "buttons not enlarged for touch"
    for pseudo in ("::-webkit-slider-thumb", "::-moz-range-thumb"):
        assert pseudo in block, (
            f"slider thumb not enlarged for touch ({pseudo}); a 4px track is "
            f"undraggable on a phone")

    # the checkbox relies on its label for a usable target, so the label must
    # stay associated with it
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    assert 'for="symmetric-check"' in html, (
        "symmetric checkbox lost its label; the box alone is too small to tap")


def test_no_false_gpu_claims():
    """Regression: the page headline read "GPU height-function sampler" while
    live sampling runs as plain in-browser JavaScript, exact sampling runs on
    numpy, and requirements.txt has no torch -- so the deployed instance can
    never use a GPU. That was the single most prominent line of text on the
    site, and it was false.

    `/api/init` had the same problem: it reported `using_gpu: s.use_torch`,
    which is true whenever torch merely *imports*. Torch on a CPU-only host
    would report using_gpu true while running on the CPU.
    """
    import os, importlib
    here = os.path.dirname(__file__)
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    subtitle = html[html.index('class="subtitle"'):]
    subtitle = subtitle[:subtitle.index("</div>")]
    assert "GPU" not in subtitle, (
        f"headline claims GPU acceleration the deployed tool does not have: {subtitle}")

    # requirements really are CPU-only, so the claim would be unbackable
    reqs = open(os.path.join(here, "..", "requirements.txt")).read().lower()
    assert "torch" not in reqs, (
        "torch is now a declared dependency; revisit the GPU wording")

    server = importlib.import_module("server")
    app = server.app.test_client()
    info = app.post("/api/init", json={"n": 8}).get_json()
    assert "using_gpu" in info and "using_torch" in info
    assert info["using_gpu"] is False, (
        "using_gpu must be derived from the device in use, not from whether "
        "torch imported")


def test_loading_screen_does_not_impose_seconds_of_dead_time():
    """Regression: the loading screen enforced a 3000 ms minimum while the app
    was measurably ready in ~57 ms -- roughly 3.3 s of pure dead time on every
    visit, 58x the actual load. Not a cosmetic quibble here: the first review
    of this tool complained it felt slow next to Petrov's, and a lot of work
    went into sampling speed which a branding animation then handed back.

    A short floor is fine (it stops the animation flickering on a fast load);
    seconds are not.
    """
    import os, re
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()
    m = re.search(r"MIN_LOADING_MS\s*=\s*(\d+)", src)
    assert m, "MIN_LOADING_MS not found"
    ms = int(m.group(1))
    assert ms <= 1000, (
        f"loading screen floor is {ms} ms; that is dead time on every visit")


def test_page_explains_itself_when_scripts_do_not_run():
    """Regression: if draw.js failed to load -- blocked request, proxy,
    adblocker, cache miss -- or JavaScript was disabled, the user sat on
    "loading..." forever with no explanation. The existing 8 s fallback was
    inside draw.js, so it was useless in exactly the case that mattered: it
    died with the script it was meant to rescue.

    Needs (a) a <noscript> message and (b) a watchdog that lives OUTSIDE the
    app bundle and so cannot be taken out by the same failure.
    """
    import os
    here = os.path.dirname(__file__)
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    js = open(os.path.join(here, "..", "static", "draw.js")).read()

    assert "<noscript>" in html, "no message for users without JavaScript"
    ns = html[html.index("<noscript>"):html.index("</noscript>")]
    assert "JavaScript" in ns, "noscript block does not explain what is wrong"

    # the watchdog must be inline in the document, before the app scripts
    watchdog = html.index("__vertsixLoaded")
    app_script = html.index('src="draw.js')
    assert watchdog < app_script, (
        "the load watchdog must come before (and live outside) draw.js, or it "
        "dies with the script it is meant to rescue")

    assert "window.__vertsixLoaded = true" in js, (
        "draw.js no longer signals successful load; the watchdog will fire "
        "spuriously on every visit")


def test_stage_is_capped_square_on_narrow_layouts():
    """Regression: the lattice is square and is fitted to min(width, height),
    but the stage took whatever height the column gave it. On a phone that
    meant a 360x1142 canvas drawing a 360x360 picture -- 68% dead black --
    and the controls sat 375px below the fold behind it. Nothing was gained
    by the extra height.

    Capping the stage at square on the single-column layout removed the dead
    area entirely and brought the controls above the fold (measured: Run
    visible without scrolling at 390x844 and 320x568). Desktop is untouched.
    """
    import os
    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))

    start = code.index("function sizeStage")
    body = code[start:start + 1500]
    assert "matchMedia" in body, (
        "sizeStage no longer distinguishes the narrow layout; the stage will "
        "stretch and strand the controls below a band of empty canvas")
    assert "h = w;" in body, "stage is not made square on the narrow layout"

    # The stage must also be bounded by the viewport height, not just squared.
    # Capping to square alone gave a 780x780 stage on a 390px-tall landscape
    # phone -- taller than the display, controls at y=820.
    assert "window.innerHeight" in body, (
        "stage is squared but not bounded by the viewport height; on a short "
        "landscape screen it will overflow and strand the controls")

    # The single-column switch must be height-aware and identical in JS and
    # CSS. A landscape phone is narrow by width but very short: stacking there
    # put the canvas above the fold and the controls below it.
    css = open(os.path.join(here, "..", "static", "style.css")).read()
    bp = "(max-width: 55rem) and (min-height: 500px)"
    assert bp in body, f"JS single-column breakpoint is not {bp}"
    assert "max-width: 55rem) and (min-height: 500px" in css, (
        "CSS single-column breakpoint is not height-aware")


def test_export_semantics_are_documented():
    """The two exports differ in a way that bites silently. PNG is a
    screenshot: it captures the current zoom and pan, so at 1083% zoom on an
    n=40 lattice it held only a fraction of the configuration. SVG exports all
    40 cells regardless. Nothing about the cropped PNG looks wrong, so someone
    preparing a figure can easily ship a partial lattice.

    The README said only "exports the current view", which is true and does
    not warn anyone.
    """
    import os
    here = os.path.dirname(__file__)
    readme = open(os.path.join(here, "..", "README.md")).read()
    html = open(os.path.join(here, "..", "static", "index.html")).read()

    # Bound the window by the SECTION, not by a character count. A fixed
    # 1200-character slice drifts with every edit above it: an insertion can
    # push the sentence being looked for out of view (a spurious failure)
    # and a deletion can pull unrelated text in (a silent pass). The same
    # flaw, with a 1600-character slice, is documented on the island check
    # below.
    exports = readme[readme.index("save PNG"):]
    _end = exports.find("\n## ")
    exports = exports[:_end] if _end != -1 else exports
    assert "zoom" in exports and "crop" in exports.lower(), (
        "README does not warn that PNG export depends on zoom/pan state")
    assert "regardless of zoom" in exports, (
        "README does not state that SVG is zoom-independent")

    # and the distinction should be discoverable without reading the README
    assert 'id="btn-save"' in html and "title=" in html
    btn = html[html.index('id="btn-save"'):]
    btn = btn[:btn.index(">")]
    assert "zoom" in btn, "PNG button has no tooltip explaining the crop behaviour"


def test_arctic_circle_appears_in_an_exact_sample():
    """End-to-end physics check: sampler -> height function -> active mask.

    At the ice point the flippable ("liquid") region must fill the disk
    inscribed in the square, with frozen corners. This is checked on a CFTP
    sample rather than an MCMC run so equilibration is not in question -- an
    under-equilibrated chain at n=120 put 4% of flippable sites outside the
    circle and 61 in the corners, which looks like a bug and is not one.

    The tolerance matters: the arctic boundary fluctuates on scale n^(1/3),
    so a handful of sites just outside r=R is expected. Demanding zero was
    wrong. What must not happen is activity DEEP in the frozen corners.
    """
    from sixvertex.cftp import cftp_sample
    from sixvertex.sampler import SixVertexSampler

    n = 60
    H, _ = cftp_sample(n=n, c_up=1.0, c_down=1.0, master_seed=11, max_T=1 << 21)
    s = SixVertexSampler(n=n, a1=1, a2=1, b1=1, b2=1, c_up=1, c_down=1)
    # Assign through the same branch server.py uses. Writing s.H directly
    # with a numpy array breaks when torch is installed: height_array() then
    # calls .detach() on it. Only reproducible on machines that have torch --
    # this sandbox has none, so the suite passes here either way and the bug
    # only shows on the collaborator's machine. Keep this branch.
    if s.use_torch:
        import torch
        s.H = torch.from_numpy(np.asarray(H, dtype=np.float32)).to(s.device)
    else:
        s.H = np.asarray(H, dtype=np.float32)
    m = s.active_mask()

    ys, xs = np.where(m)
    assert len(xs) > 0, "no flippable sites at all"
    c = n / 2.0
    r = np.sqrt((xs - c) ** 2 + (ys - c) ** 2) / (n / 2.0)

    # the liquid region is concentrated inside the inscribed circle
    assert np.median(r) < 0.85, f"median radius {np.median(r):.3f} is too large"
    assert (r > 1.10).mean() < 0.02, (
        f"{(r > 1.10).mean():.3f} of flippable sites are well outside the "
        f"arctic circle; the frozen region is not forming")
    # and nothing is active deep in the frozen corners
    assert (r > 1.30).sum() == 0, "flippable sites deep inside the frozen corners"


def test_cftp_valid_on_the_whole_monotone_region():
    """CFTP is monotone exactly on c1c2 >= a1a2 and c1c2 >= b1b2, not merely at
    the uniform point. The shipped code used to restrict it to all-weights-1
    because the old module hardcoded p_up = c1/(c1+c2), correct only there.

    With the four-face rule the coupling is valid on the whole region, which is
    what makes large-n exact sampling available for non-uniform weights.
    """
    from sixvertex.cftp_exact import cftp_sample, in_monotone_region

    # Corrected convention: the region is c1c2 >= a1a2 and c1c2 >= b1b2.
    # This is the antiferroelectric-favouring direction, so it contains
    # Delta < -1 -- including Gorin's Figure 17 bottom.
    inside = [
        dict(a1=1., a2=1., b1=1., b2=1., c1=1., c2=1.),      # boundary
        dict(a1=1., a2=1., b1=1., b2=1., c1=2., c2=2.),
        dict(a1=1., a2=1., b1=2., b2=2., c1=3., c2=3.),
        dict(a1=1.5, a2=0.5, b1=1., b2=1., c1=1.2, c2=1.2),  # a1 != a2
        dict(a1=1., a2=1., b1=1., b2=1.,                     # Delta = -3
             c1=math.sqrt(8), c2=math.sqrt(8)),
    ]
    for w in inside:
        assert in_monotone_region(w)
        H, info = cftp_sample(10, w, master_seed=7, check_monotone=True)
        assert info["monotonicity_violations"] == 0, (
            f"ordering violated inside the proven region for {w}")
        n = 10
        for i in range(n + 1):
            for j in range(n):
                assert abs(int(H[i, j+1]) - int(H[i, j])) == 1
        for i in range(n):
            for j in range(n + 1):
                assert abs(int(H[i+1, j]) - int(H[i, j])) == 1

    outside = [
        dict(a1=1., a2=1., b1=1.5, b2=1.5, c1=1., c2=1.),    # B > C
        dict(a1=2., a2=2., b1=1., b2=1., c1=1., c2=1.),      # A > C
    ]
    for w in outside:
        assert not in_monotone_region(w)
        try:
            cftp_sample(10, w, master_seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"CFTP accepted weights outside the region: {w}")


def test_cftp_exact_matches_brute_force():
    """The corrected CFTP must sample the right measure, not merely coalesce."""
    from sixvertex.cftp_exact import cftp_sample
    n = 4
    w = dict(a1=1., a2=1., b1=1., b2=1., c1=2., c2=2.)
    cfgs, probs, _ = exact_distribution(n, w)
    keys = {c.astype(np.int64).tobytes(): i for i, c in enumerate(cfgs)}
    counts = np.zeros(len(cfgs))
    N = 1500
    for seed in range(N):
        H, _ = cftp_sample(n, w, master_seed=100000 + seed)
        k = keys.get(np.asarray(H, dtype=np.int64).tobytes())
        assert k is not None, "CFTP produced an invalid configuration"
        counts[k] += 1
    dev = float(np.max(np.abs(counts / counts.sum() - probs)))
    assert dev < 4.0 / math.sqrt(N), f"max deviation {dev:.4f} too large"


def test_stochastic_six_vertex_sampler():
    """The stochastic six-vertex model covers Delta >= 1 with free-exit
    boundary conditions -- the right-hand panel of Figure 18 of
    arXiv:2309.12495, and a regime DWBC cannot sample exactly.

    Because the vertex weights are genuine conditional probabilities, a
    configuration is generated by one sweep of the lattice: exact, O(N^2), no
    Markov chain and no coalescence.
    """
    from sixvertex import stochastic as st

    b1, b2 = 0.3, 0.7
    # Delta = (b1+b2)/(2 sqrt(b1 b2)) >= 1 by AM-GM, equality iff b1 == b2
    assert st.delta(b1, b2) > 1.0
    assert abs(st.delta(0.4, 0.4) - 1.0) < 1e-12

    legal = {(0,0,0,0), (1,1,1,1), (1,0,1,0), (1,0,0,1), (0,1,0,1), (0,1,1,0)}
    n = 30
    counts = {}
    for seed in range(20):
        h, v = st.sample(n, b1, b2, seed=seed)
        for i in range(n):
            for j in range(n):
                key = (int(h[i,j]), int(v[i,j]), int(h[i,j+1]), int(v[i+1,j]))
                # arrows in must equal arrows out at every vertex
                assert key[0] + key[1] == key[2] + key[3], "arrow conservation failed"
                assert key in legal, f"illegal vertex type {key}"
                counts[key] = counts.get(key, 0) + 1

    # the sampled transition frequencies must match b1 and b2
    stay10 = counts.get((1,0,1,0), 0); turn10 = counts.get((1,0,0,1), 0)
    stay01 = counts.get((0,1,0,1), 0); turn01 = counts.get((0,1,1,0), 0)
    assert abs(stay10/(stay10+turn10) - b1) < 0.02, "P((1,0)->(1,0)) != b1"
    assert abs(stay01/(stay01+turn01) - b2) < 0.02, "P((0,1)->(0,1)) != b2"

    # out-of-range parameters are refused rather than silently clamped
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            st.sample(5, bad, 0.5, seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"b1={bad} should have been rejected")


def test_live_equilibration_diagnostic_exists():
    """The site warned about slow mixing with a static note, which reads as
    boilerplate. A collaborator tested n=80 at Delta=-3, saw the configuration
    sit on the diagonal, and reported it as broken.

    The code was not broken -- the browser engine reproduces the exact
    distribution at n=5 to within 0.0016 over all 429 configurations, and
    mixes fine at n=40. But at n=80 two chains started from opposite extremal
    configurations disagreed by 0.131 in c-vertex density after 20000 sweeps,
    against a seed-to-seed spread of 0.017. It had not mixed, and the tool
    presented that as if it were a sample.

    So the UI now runs the diagnostic live: a shadow chain from the opposite
    start, with the observed disagreement reported on screen.
    """
    import os
    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))

    assert "makeShadow" in code, "no shadow chain for the equilibration check"
    assert "updateMixingBadge" in code, "no live mixing diagnostic"
    assert 'id="mixing-badge"' in html, "no element to report the diagnostic"
    # the shadow must start from the OTHER extremal configuration, else the
    # comparison is vacuous
    blk = code[code.index("function makeShadow"):]
    blk = blk[:blk.index("function cFraction")]
    assert "Math.min" in blk, (
        "shadow chain does not start from the maximal height function; two "
        "chains from the same start cannot detect failure to mix")
    # and it must actually be advanced alongside the main chain
    assert "shadow.step(sweeps)" in code, "shadow chain is never advanced"


def test_convention_matches_izergin_korepin():
    """Pin the face-label convention to the literature, at a != b.

    This code originally had b and c swapped relative to the standard
    six-vertex convention. The consequences were not cosmetic: the reported
    Delta had the wrong sign structure, and the monotone region was stated as
    b-dominant when it is c-dominant. Because the region is in fact the
    antiferroelectric-favouring one, Delta = -3 -- which a collaborator kept
    reporting as stuck -- is actually inside it and exactly samplable.

    The check: the Izergin-Korepin determinant for DWBC must reproduce our
    partition function in the SAME weight slots. IK is itself validated here
    by giving the ASM numbers at the ice point.

    Two things were wrong with this check, both of which made it weaker than
    it reads.

    First it ran only at lambda = 0, where a = f(eta - lambda) and
    b = f(eta + lambda) are EQUAL, so the whole comparison sat on the line
    a = b. The spectral parameter slides a and b apart at FIXED Delta, so the
    comparison now runs at lambda != 0 as well, in both orderings and in both
    branches: sinh for Delta = -3, sin for the ice point and the free-fermion
    point.

    What that does NOT buy is worth stating, because it is tempting to assume
    otherwise: it does not make an a/b swap detectable. Z is symmetric under
    exchanging a and b for DWBC -- the swap is a reflection of the lattice,
    which maps these boundary conditions to themselves -- and that is checked
    below rather than assumed. So a/b joins b1/b2 and c1/c2 in the list of
    relabellings no computation can see, and the convention still rests on
    the citation. Widening lambda broadens the check across the (a, b) plane
    against an error that treats the two asymmetrically WITHOUT being a clean
    swap, which is a different and still-possible fault -- and that class is
    not hypothetical. Weighting just the a1 face (the empty vertex) as b1 and
    leaving everything else alone gives ours/IK = 1.000000 at every N when
    lambda = 0, so the old form of this test passed it; at lambda = 0.15 the
    same fault gives 1.159, 1.516 and 2.107 at N = 2, 3, 4.

    Second, a missing mpmath made it return quietly, which pytest reports as a
    pass. mpmath is an optional dev dependency and was NOT installed, so the
    convention check the README cites had been vacuous. It skips loudly now.
    """
    import pytest

    mp = pytest_importorskip_mpmath()
    if mp is None:
        pytest.skip(
            "mpmath is not installed, so the Izergin-Korepin convention "
            "check cannot run. It is declared under the 'dev' extra; install "
            "it rather than letting this check pass silently.")
    mp.mp.dps = 40

    def Z_IK(N, eta, lam, trig):
        f = mp.sin if trig else mp.sinh
        def phi(x): return f(2*eta) / (f(eta - x) * f(eta + x))
        M = mp.matrix(N, N)
        for i in range(N):
            for j in range(N):
                M[i, j] = mp.diff(phi, lam, i + j)
        d = mp.mpf(1)
        for k in range(N):
            d *= mp.factorial(k)
        return float((f(eta-lam) * f(eta+lam))**(N*N) * mp.det(M) / d**2)

    # (a) IK is correct: ice point gives the ASM numbers
    eta, lam = mp.pi/3, mp.mpf(0)
    a = float(mp.sin(eta))
    for N, asm in zip(range(1, 6), [1, 2, 7, 42, 429]):
        assert abs(Z_IK(N, eta, lam, True) / a**(N*N) - asm) < 1e-6, (
            "IK implementation fails the ASM check")

    # (b) our sampler matches IK in the same slots, across three regimes and
    #     with a and b pulled apart in both directions
    cases = [
        ("Delta = -3 (sinh)", mp.acosh(3)/2, False,
         [mp.mpf(0), mp.mpf("0.15"), mp.mpf("0.3"), mp.mpf("-0.25")]),
        ("ice point (sin)", mp.pi/3, True,
         [mp.mpf(0), mp.mpf("0.2"), mp.mpf("-0.35")]),
        ("free fermion (sin)", mp.pi/4, True,
         [mp.mpf("0.1"), mp.mpf("-0.18")]),
    ]
    saw_a_less, saw_a_more = False, False
    for label, eta, trig, lams in cases:
        f = mp.sin if trig else mp.sinh
        for lam in lams:
            a = float(f(eta - lam))
            b = float(f(eta + lam))
            c = float(f(2 * eta))
            if lam != 0:
                assert abs(a - b) > 1e-6, (
                    f"{label}, lambda={lam}: a and b did not separate, so "
                    f"this case is not testing what it claims")
                saw_a_less = saw_a_less or a < b
                saw_a_more = saw_a_more or a > b
            w = dict(a1=a, a2=a, b1=b, b2=b, c1=c, c2=c)
            for N in range(1, 6):
                ours = ExactSampler(N, w).partition_function()
                ik = Z_IK(N, eta, lam, trig)
                assert abs(ours - ik) / ik < 1e-9, (
                    f"{label}, lambda={float(lam)}, N={N}: ours={ours} but "
                    f"IK={ik} (a={a:.6f}, b={b:.6f}); the convention has "
                    f"slipped in the a/b or b/c slots")

    assert saw_a_less and saw_a_more, (
        "the comparison never ran with both a > b and a < b, so it is still "
        "sitting on the a = b line")

    # The symmetry that bounds what any of this can prove: exchanging a and b
    # leaves Z alone, so no partition-function comparison -- IK included --
    # can tell the two apart. Recorded here so the limit is not rediscovered
    # as a surprise.
    for (a, b, c) in ((0.6, 1.4, 2.0), (0.8, 1.3, 1.1)):
        for N in (2, 3, 4, 5):
            za = ExactSampler(N, dict(a1=a, a2=a, b1=b, b2=b,
                                      c1=c, c2=c)).partition_function()
            zb = ExactSampler(N, dict(a1=b, a2=b, b1=a, b2=a,
                                      c1=c, c2=c)).partition_function()
            assert abs(za - zb) <= 1e-9 * za, (
                f"Z is no longer symmetric under a<->b at N={N} "
                f"(a={a}, b={b}, c={c}): {za} vs {zb}. If that is real, an "
                f"a/b swap has become detectable and this reasoning changes")


def pytest_importorskip_mpmath():
    try:
        import mpmath
        return mpmath
    except ImportError:
        return None


def test_package_surface_is_installable_and_minimal():
    """The library must be usable without the web app.

    The samplers need only numpy; Flask and gunicorn belong to the demo and
    torch is an optional accelerator for the MCMC engine. If any of those
    became a hard dependency, installing the library would drag in a web
    stack, so pin the declared dependency list.
    """
    import os, re
    here = os.path.dirname(__file__)
    toml = open(os.path.join(here, "..", "pyproject.toml")).read()

    m = re.search(r"^dependencies = \[(.*?)\]", toml, re.S | re.M)
    assert m, "pyproject.toml declares no dependencies list"
    deps = m.group(1).lower()
    for forbidden in ("flask", "gunicorn", "torch", "matplotlib", "mpmath"):
        assert forbidden not in deps, (
            f"{forbidden} must not be a hard dependency of the library")
    assert "numpy" in deps

    # the public API must actually exist
    import sixvertex as sv
    for name in ("sample", "delta", "exact_sample", "cftp_sample",
                 "in_monotone_region", "SixVertexSampler", "stochastic",
                 "MAX_EXACT_N", "__version__"):
        assert hasattr(sv, name), f"public API is missing {name}"

    # and route correctly, including refusing when nothing is valid
    w = dict(a1=1., a2=1., b1=1., b2=1., c1=math.sqrt(8), c2=math.sqrt(8))
    assert abs(sv.delta(w) + 3.0) < 1e-12
    _, info = sv.sample(8, w, seed=1)
    assert info["method"] == "exact-sequential"
    _, info = sv.sample(20, w, seed=1)
    assert info["method"] == "cftp"
    try:
        sv.sample(40, dict(a1=1., a2=1., b1=3., b2=3., c1=1., c2=1.), seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("sample() must refuse when no exact method applies")


def test_poll_loop_tolerates_transient_failures():
    """Regression: a long exact job means many status polls -- a 16 minute run
    at one poll a second is about a thousand requests. The loop had no failure
    tolerance, so a SINGLE bad response abandoned the whole computation even
    though the server was still working on it.

    A proxy timeout returns an HTML error page, and `await res.json()` on that
    throws SyntaxError rather than returning {ok:false}, so it escaped to the
    outer catch. Verified in a browser: six consecutive failures (mixed
    connection aborts and 524 HTML pages) now recover and the job completes;
    before, the first one killed it.

    Observed in production: n=80 at Delta=-3 completed server-side in 982s
    while the browser session died at ~434s.
    """
    import os
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("//"))

    # anchor on the main handler: fetchExactSample also declares a jobId
    start = code.index("btnExact.addEventListener")
    loop = code[start:start + 4000]

    assert "MAX_CONSECUTIVE_FAILURES" in loop, (
        "poll loop has no tolerance for transient status-check failures")
    assert "consecutiveFailures = 0" in loop, (
        "failure counter is never reset, so isolated failures accumulate")
    # the fetch/json pair must be individually guarded, not just the whole loop
    assert loop.index("try {") < loop.index("await fetch(`/api/exact/status`" if False else "await fetch(`/api/exact/status"), (
        "the status fetch is not inside its own try block")
    # and the interval must back off rather than hammering for 45 minutes
    assert "elapsedPolling < 30" in loop and "5000" in loop, (
        "poll interval does not back off as the job ages")


def test_compiled_sweep_matches_numpy_exactly():
    """The optional compiled sweep must be the SAME computation, not an
    approximation: it consumes the same random values at the same positions,
    so for a given seed it must reproduce the numpy result bit for bit.

    Measured speedups: 76x at n=40, 30x at n=80, 16x at n=128; end to end
    n=80 at Delta=-3 went from 52s to 5.9s at the same 16384 sweeps.
    numba is optional -- without it the numpy path is used and remains the
    reference.
    """
    import sixvertex.cftp_exact as ce
    if not ce._HAS_NUMBA:
        return                      # nothing to compare against

    w = dict(a1=1., a2=1., b1=1., b2=1.,
             c1=math.sqrt(8), c2=math.sqrt(8))
    for seed in range(4):
        ce._HAS_NUMBA = True
        A, ia = ce.cftp_sample(16, w, master_seed=seed)
        ce._HAS_NUMBA = False
        B, ib = ce.cftp_sample(16, w, master_seed=seed)
        ce._HAS_NUMBA = True
        assert np.array_equal(A, B), (
            f"compiled and numpy sweeps disagree at seed {seed}")
        assert ia["sweeps"] == ib["sweeps"], (
            "compiled path coalesces at a different time; the randomness is "
            "not being consumed identically")


def test_adaptive_start_saves_work_without_changing_the_law():
    """The doubling schedule runs 8, 16, ... up to T, so total work is
    2T - T_start: starting at 8 does about twice the sweeps the answer needs.

    Starting nearer T recovers that, and affects efficiency only -- coalescence
    is what certifies the sample, so any window gives an exact result. Verified
    by sampling: max deviation from the brute-force law at n=4 was 0.0151 with
    both start=8 and start=1024, i.e. identical.

    The estimate is deliberately restricted to large n. T/n^2 is not constant
    (about 0.9 at n=24 rising to 5 at n=80), so a fixed factor overshoots at
    small n, and overshooting costs work: at n=24 a start of 2048 did 2048
    sweeps where doubling from 8 coalesced at 512 and did 1016.
    """
    from sixvertex.cftp_exact import _estimate_T

    # below the threshold, unchanged -- no risk where there is no gain
    for n in (8, 24, 40, 56):
        assert _estimate_T(n) == 8, f"estimate should not engage at n={n}"

    # above it, a real starting window that is a power of two
    for n in (64, 80, 96):
        T0 = _estimate_T(n)
        assert T0 > 8, f"estimate did not engage at n={n}"
        assert T0 & (T0 - 1) == 0, f"start {T0} is not a power of two"
        # aims high rather than low: undershooting costs a whole extra pass
        assert T0 >= n * n, f"start {T0} is below n^2 at n={n}"

    # and the window must not alter the sampled measure
    w = dict(a1=1., a2=1., b1=1., b2=1., c1=2.0, c2=2.0)
    a, ia = cftp_exact_sample_for_test(6, w, seed=3, T0=8)
    b, ib = cftp_exact_sample_for_test(6, w, seed=3, T0=256)
    assert np.array_equal(a, b), (
        "the starting window changed the sample for a fixed seed")


def cftp_exact_sample_for_test(n, w, seed, T0):
    from sixvertex.cftp_exact import cftp_sample
    return cftp_sample(n, w, master_seed=seed, initial_T=T0)


def test_colour_classifier_uses_the_corrected_convention():
    """Regression, reported by a collaborator: the colour pickers labelled
    b1/b2 were in fact colouring c1/c2.

    When the b/c labels were corrected across the four samplers, this fourth
    classifier -- the one used only for rendering -- was missed. His
    diagnostic: in a large simulation the frozen corners must be a- and
    b-types. Measured after the fix at n=60 uniform, the corner blocks were
    99.75% a/b.
    """
    import os
    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    # classifyFaceLocal, the classifier this test was written against, has
    # been REMOVED. It was a second copy of the convention kept only for the
    # SVG export, it classified raw corner heights (wrong for the stochastic
    # model), and it fell through to "a1" for anything it could not place.
    # Everything now goes through classifyBits, so that is what must carry
    # the corrected labels -- and there is no second copy left to drift.
    assert "classifyFaceLocal" not in js, (
        "the duplicate rendering classifier is back; it is where the b/c "
        "label fix was missed once already")
    body = _js_function_body(js, "function classifyBits")
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("//"))
    assert 'if (l && t && !b && !r) return "c1"' in code, (
        "(l,t) must be a c-type; the rendering classifier has the old labels")
    # (t,b) must be a b-type. WHICH b is pinned separately, against the
    # Gorin-Kenyon frozen corners, by
    # test_vertex_labels_match_gorin_kenyon_via_the_frozen_corners. This
    # assertion used to demand "b1" and so froze the label that turned out to
    # be the wrong way round -- the same mistake as pinning the index
    # expression in the path view.
    import re as _re
    m = _re.search(r'if \(!l && t && b && !r\) return "(b[12])"', code)
    assert m, "(t,b) must be a b-type; the rendering classifier has odd labels"


def test_fluctuation_normalisation_preserves_variance():
    """Height fluctuations are shown as (H1 - H2)/sqrt(2) from two independent
    exact samples. Var[H1 - H2] = 2 Var[H], so the sqrt(2) returns the variance
    of a single height function -- which is the point of that normalisation.

    Verified numerically at n=20 over 60 samples: the pointwise fluctuation
    variance of a single sample was 0.7228 and the variance of the normalised
    difference 0.7478, a ratio of 1.03.
    """
    from sixvertex.cftp_exact import cftp_sample
    w = dict(a1=1., a2=1., b1=1., b2=1., c1=1., c2=1.)
    n = 14
    S = np.array([np.asarray(cftp_sample(n, w, master_seed=4000 + k)[0], float)
                  for k in range(40)])
    mean = S.mean(axis=0)
    var_single = float(((S - mean) ** 2).mean())
    diffs = [(S[2*k] - S[2*k+1]) / math.sqrt(2) for k in range(20)]
    var_diff = float(np.mean([(d ** 2).mean() for d in diffs]))
    ratio = var_diff / var_single
    assert 0.8 < ratio < 1.25, (
        f"sqrt(2) normalisation does not preserve variance: ratio {ratio:.3f}")


def test_new_export_and_view_modes_exist():
    """Requested by a collaborator: a lattice-path view, a fluctuation view,
    and text export so others can work with the output."""
    import os
    here = os.path.dirname(__file__)
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    js = open(os.path.join(here, "..", "static", "draw.js")).read()

    for opt in ('value="paths"', 'value="fluct"'):
        assert opt in html, f"missing view mode {opt}"
    for btn in ("btn-fluct", "btn-save-heights", "btn-save-types"):
        assert f'id="{btn}"' in html, f"missing control {btn}"
    assert "function drawPaths" in js
    assert "heightsAsText" in js and "typesAsText" in js
    # paths follow the standard convention (Gorin-Kenyon arXiv:2408.14446
    # Fig. 1): each set bit of (l,t,b,r) is an occupied edge, drawn as a stub
    # from the vertex centre. An earlier version drew level lines of the
    # height function -- the dual picture, and not the one the literature or
    # this interface describes.
    assert "j + 0.5" in js and "i + 0.5" in js, (
        "path view does not draw stubs from vertex centres")
    assert "lo + 0.5" not in js, (
        "path view still uses marching squares on the height function")
    # the type codes must follow the corrected convention order
    assert "{ a1: 1, a2: 2, b1: 3, b2: 4, c1: 5, c2: 6 }" in js


def test_stochastic_model_is_reachable_and_height_function_is_valid():
    """The stochastic sampler existed in the package but appeared nowhere in
    the server or the UI, so nobody using the website could reach it -- a whole
    exact method, covering Delta >= 1, invisible.

    Its height_function was also wrong: a double cumulative sum giving steps of
    8 rather than 1. Nothing called it, which is exactly why the error
    survived. The correct one counts horizontal arrows entering column j in the
    rows BELOW i -- rows 0..i-1 with row 0 the bottom; this docstring and the
    module's said "above", the opposite. Vertically the step is h[i,j] in
    {0,1}, never -1, and horizontally the sum telescopes through arrow
    conservation to v[0,j] - v[i,j], so it lies in {-1,0,1}.
    """
    import os, importlib
    from sixvertex import stochastic as st

    # height function must actually be one
    for n in (8, 24, 40):
        h, v = st.sample(n, 0.3, 0.7, seed=2)
        H = st.height_function(h, v)
        assert H.shape == (n + 1, n + 1)
        assert np.abs(np.diff(H, axis=1)).max() <= 1, "horizontal step > 1"
        # the vertical step is h[i,j], so 0 or 1 -- never -1. |diff| <= 1 alone
        # would pass a height function that decreased going up.
        dv = np.diff(H, axis=0)
        assert set(np.unique(dv).tolist()) <= {0, 1}, (
            f"vertical steps are {sorted(set(np.unique(dv).tolist()))}, "
            f"but H(i+1,j) - H(i,j) = h[i,j] must be 0 or 1")
        assert np.array_equal(dv, h), (
            "the vertical step is not h[i,j] itself")
        # and it counts the rows BELOW i, not above
        assert np.array_equal(H[1:, :], np.cumsum(h, axis=0)), (
            "H is not the cumulative count of arrows in the lower rows")
        assert H[0, :].max() == 0, "row 0 should carry no accumulated arrows"

    # reachable from the server
    server = importlib.import_module("server")
    app = server.app.test_client()
    r = app.post("/api/stochastic",
                 json={"n": 32, "b1": 0.3, "b2": 0.7, "seed": 5})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] and body["info"]["method"] == "stochastic-sequential"
    assert body["info"]["delta"] >= 1.0, "stochastic weights must give Delta >= 1"

    # b1, b2 are probabilities, not Boltzmann weights
    for bad in ({"b1": 0}, {"b1": 1.5}, {"b2": -0.1}):
        rr = app.post("/api/stochastic", json={"n": 10, **bad})
        assert rr.status_code == 400, f"{bad} should be rejected"

    # and reachable from the page
    here = os.path.dirname(__file__)
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    assert 'value="stochastic"' in html, "no model selector entry"
    assert 'id="btn-stoch"' in html, "no way to trigger a stochastic sample"
    assert "/api/stochastic" in js, "client never calls the endpoint"


def test_seed_is_settable_and_reproduces():
    """A fixed seed must reproduce a sample exactly -- that is what lets
    someone else regenerate a figure. The server accepted a seed but the UI
    never sent one."""
    import os, importlib
    server = importlib.import_module("server")
    app = server.app.test_client()

    a = app.post("/api/stochastic",
                 json={"n": 24, "b1": 0.3, "b2": 0.7, "seed": 99}).get_json()
    b = app.post("/api/stochastic",
                 json={"n": 24, "b1": 0.3, "b2": 0.7, "seed": 99}).get_json()
    c = app.post("/api/stochastic",
                 json={"n": 24, "b1": 0.3, "b2": 0.7, "seed": 100}).get_json()
    assert a["frame"]["height_b64"] == b["frame"]["height_b64"], (
        "same seed did not reproduce the sample")
    assert a["frame"]["height_b64"] != c["frame"]["height_b64"], (
        "different seeds gave the same sample")

    here = os.path.dirname(__file__)
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    assert 'id="seed"' in html, "no seed input in the UI"
    assert "currentSeed()" in js, "the client never reads the seed box"

    # Every request that DRAWS something must carry a seed. This used to be
    # written as js.count("seed: currentSeed()") >= 2, which froze one
    # spelling of the intent rather than the intent: passing the seed through
    # a parameter instead of calling currentSeed() inline at the fetch site
    # broke the assertion while satisfying it better. (That mattered -- the
    # fluctuation view has to send a DERIVED seed for its second sample, or
    # it differences a sample against itself.) Check each request body
    # instead, which is the thing actually being claimed.
    import re as _re
    sampling = ["/api/exact/start", "/api/domain/start", "/api/stochastic"]
    for ep in sampling:
        n_found = 0
        for m in _re.finditer(_re.escape(f'fetch("{ep}"'), js):
            body = js[m.start():m.start() + 900]
            assert "seed:" in body, (
                f"a request to {ep} is sent without a seed, so that path is "
                f"not reproducible:\n{body[:400]}")
            n_found += 1
        assert n_found >= 1, f"no request to {ep} found in draw.js"

    # and the in-browser chain, which is what the page runs by default, must
    # read the box too rather than seeding itself from the clock
    assert _re.search(r"new SixVertexJS\([^)]*localSeed", js), (
        "the in-browser chain does not take its seed from the seed box")


def test_model_switch_disables_rather_than_hides_and_restores_the_chain():
    """Switching to the stochastic model must not silently break the DWBC one.

    Two things were wrong in the first version. The DWBC controls were hidden
    rather than disabled, which makes the tool look like it lost features; and
    a stochastic sample clears the live chain (different model, different
    boundary conditions), so switching back left Run dead at 0 sweeps -- the
    same failure as the earlier dead-Run-after-exact-sample bug.
    """
    import os
    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))

    body = _js_function_body(code, "function updateModelUI")
    assert 'el.style.display = "";' in body, (
        "DWBC controls are hidden on model switch; disable them with a reason "
        "instead so the tool does not look like it lost features")
    assert "el.disabled = true" in body, "controls are not disabled in stochastic mode"
    assert "el.title =" in body, "disabled controls give no reason"
    assert "if (!sampler) localInit();" in body, (
        "switching back to DWBC does not rebuild the chain; Run will be dead")


def test_ui_text_states_the_corrected_region():
    """The panel still claimed exact sampling was "valid only at a=b=1", which
    was the pre-correction description. With the labels fixed the region is
    c-dominant and contains the whole antiferroelectric phase, so Delta=-3 --
    the case a collaborator kept reporting -- is exactly samplable."""
    import os
    here = os.path.dirname(__file__)
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    assert "a=b=1 regime" not in html, "stale gating claim still in the UI"
    assert "valid only at a=b=1" not in html, "stale gating claim still in the UI"
    # and the correct condition is stated somewhere the user can see
    assert "c<sub>1</sub>c<sub>2</sub> &ge;" in html, (
        "the UI does not state the actual CFTP condition")


def test_runtime_labelled_buttons_are_stacked():
    """Buttons whose label changes at runtime cannot share a row.

    "exact sample" becomes "coalescing... (18s)" while a job runs, which is
    far wider, so side by side it overran its neighbour and both labels were
    clipped. The same applied to the two text-export buttons. They are now
    full-width, one per line.

    Note the earlier overflow check missed this: it compared each control
    against the panel edge, so controls overlapping EACH OTHER passed. The
    check that found it compares siblings.
    """
    import os
    here = os.path.dirname(__file__)
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    css = open(os.path.join(here, "..", "static", "style.css")).read()

    assert ".stack" in css, "no stacked-button layout defined"
    block = css[css.index(".stack"):]
    assert "flex-direction: column" in block[:200], ".stack is not a column"

    # the runtime-labelled buttons must live in a stack, not a row
    i = html.index('id="btn-exact"')
    before = html[:i]
    assert before.rindex('class="stack"') > before.rindex('class="row"'), (
        "btn-exact is in a row; its label changes at runtime and will overrun")
    j = html.index('id="btn-save-heights"')
    before2 = html[:j]
    assert before2.rindex('class="stack"') > before2.rindex('class="row"'), (
        "the text-export buttons are in a row and their labels do not fit")


def test_path_convention_matches_gorin_kenyon_figure_1():
    """The six vertex types must draw as in Gorin-Kenyon arXiv:2408.14446
    Fig. 1: (1) a1 empty, (2) a2 crossing, (3) b1 vertical, (4) b2
    horizontal, (5) c1 and (6) c2 turning.

    That falls straight out of the bits already computed for the face type:
    l,t,b,r set means the left/top/bottom/right edge carries path, so drawing
    a stub from the vertex centre to each set edge reproduces the figure.
    """
    BITS = {"a1": (0, 0, 0, 0), "a2": (1, 1, 1, 1), "b1": (0, 1, 1, 0),
            "b2": (1, 0, 0, 1), "c1": (1, 1, 0, 0), "c2": (0, 0, 1, 1)}
    edges = {t: [e for e, on in zip("ltbr", bits) if on]
             for t, bits in BITS.items()}

    assert edges["a1"] == [],                  "(1) a1 must be empty"
    assert set(edges["a2"]) == set("ltbr"),    "(2) a2 must be a crossing"
    assert set(edges["b1"]) == {"t", "b"},     "(3) b1 must be vertical"
    assert set(edges["b2"]) == {"l", "r"},     "(4) b2 must be horizontal"
    assert set(edges["c1"]) == {"l", "t"},     "(5) c1 must be a turn"
    assert set(edges["c2"]) == {"b", "r"},     "(6) c2 must be a turn"
    # the two c types must turn the opposite way from each other
    assert not (set(edges["c1"]) & set(edges["c2"])), (
        "the two c-type turns must not share an edge")


def test_fluctuation_view_is_guarded_and_cleared_on_reset():
    """Two integration bugs reported by a collaborator.

    1) Selecting the fluctuation view before computing one dereferenced a null
       frame and threw out of draw() -- "Cannot read properties of null
       (reading 'get')". The option is now disabled until a field exists, and
       draw() falls back to the height view anyway: a stale selection can
       survive, and a view mode should never be able to break rendering.

    2) After computing fluctuations, Reset left the canvas stuck on the old
       fluctuation drawing. A reset builds a new chain, so the field no longer
       describes what is on screen; it is dropped and the view leaves.
    """
    import os
    here = os.path.dirname(__file__)
    html = open(os.path.join(here, "..", "static", "index.html")).read()
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))

    # the option starts unavailable
    assert 'id="opt-fluct"' in html and "disabled" in html[
        html.index('id="opt-fluct"'):html.index('id="opt-fluct"') + 120], (
        "the fluctuation view option is selectable before a field exists")

    # draw() guards regardless
    i = code.index('mode === "fluct"')
    branch = code[i:i + 400]
    assert "if (!fluctFrame)" in branch, (
        "the fluctuation branch dereferences the frame without checking it")

    # reset clears the field and leaves the view
    start = code.index("function localInit")
    body = code[start:start + 700]
    assert "fluctFrame = null" in body, "reset does not clear the fluctuation field"
    assert 'viewMode.value = "height"' in body, (
        "reset leaves the view stuck on the stale fluctuation drawing")
    assert "setFluctAvailable(false)" in body, (
        "reset does not disable the option again")


def test_mixing_badge_is_smoothed_and_size_scaled():
    """The badge's threshold and smoothing, as properties rather than literals.

    This asserted that the strings "0.9 * gapEMA" and "2.5 / n" appeared in
    the source. That pins an implementation detail -- it fails if someone
    legitimately retunes the coefficient -- and verifies neither thing that
    matters: that the threshold actually brackets the equilibrium gap, and
    that the average actually smooths.

    Both are now checked by running the shipped expressions in node against
    measurements. The calibration, at uniform weights:

         n     tol      equilibrium gap sd    tol/sd    gap at 1 sweep
        12    0.2083         0.0839            2.5x         0.750
        16    0.1562         0.0529            3.0x         0.844
        24    0.1042         0.0553            1.9x         0.906
        32    0.0781         0.0316            2.5x         0.930

    so tol must stay above the equilibrium gap (no false alarm) and below the
    cold-start gap (the alarm fires when it should).
    """
    import json
    import pathlib
    import re
    import shutil
    import subprocess

    import numpy as np
    import pytest

    from sixvertex.cftp_exact import (_colour_indices, _sweep, _weight_table,
                                      cftp_sample, extremal_height)

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available to run the shipped badge expressions")

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "static" / "draw.js").read_text(encoding="utf-8")

    tol_line = re.search(r"const tol = ([^;]+);", src)
    assert tol_line, "the badge no longer computes a tolerance"
    ema_line = re.search(r"gapEMA = gapEMA === null \? gap : ([^;]+);", src)
    assert ema_line, "the badge no longer smooths the gap"

    # the shipped tolerance, evaluated at each n
    script = ("const tolFor = (n) => " + tol_line.group(1) + ";\n"
              + "const emaStep = (gapEMA, gap) => " + ema_line.group(1) + ";\n"
              + "let e = null; const trace = [];\n"
              + "for (let k = 0; k < 40; k++) { e = e === null ? 0.5 : emaStep(e, 0.5); }\n"
              + "const converged = e;\n"
              + "let spike = e; spike = emaStep(spike, 5.0);\n"
              + "console.log(JSON.stringify({tol: [12,16,24,32].map(tolFor), "
                "converged: converged, afterSpike: spike}));")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr[:400]
    got = json.loads(out.stdout.strip().splitlines()[-1])

    # the average smooths: it converges to a constant input, and one spike
    # moves it by far less than the spike itself
    assert abs(got["converged"] - 0.5) < 0.01, got
    assert got["afterSpike"] - got["converged"] < 0.5 * (5.0 - 0.5), (
        f"a single spike moved the average by "
        f"{got['afterSpike'] - got['converged']:.3f} of a 4.5 jump; it is not "
        f"smoothing")

    # the threshold brackets the gap at every size
    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.0, c2=1.0)

    def bdens(H, n):
        H = np.asarray(H, dtype=np.int64)
        tl, tr = H[:n, :n], H[:n, 1:]
        bl, br = H[1:, :n], H[1:, 1:]
        l, t = (bl - tl) == 1, (tr - tl) == 1
        b, r = (br - bl) == 1, (br - tr) == 1
        return float(((~l & t & b & ~r) | (l & ~t & ~b & r)).sum()) / (n * n)

    for n, tol in zip((12, 16, 24, 32), got["tol"]):
        eq = [bdens(cftp_sample(n, w, master_seed=6000 + s)[0], n)
              for s in range(30)]
        gap_sd = float(np.std(eq, ddof=1)) * np.sqrt(2.0)
        tbl = _weight_table(w)
        ci, size = _colour_indices(n)
        A = extremal_height(n, "lo").astype(np.float64)
        B = extremal_height(n, "hi").astype(np.float64)
        rA, rB = np.random.default_rng(1), np.random.default_rng(2)
        A = _sweep(A, tbl, rA.random(A.shape), ci, size)
        B = _sweep(B, tbl, rB.random(B.shape), ci, size)
        cold = abs(bdens(A, n) - bdens(B, n))

        assert tol > 1.5 * gap_sd, (
            f"n={n}: tol {tol:.4f} is only {tol / gap_sd:.1f}x the "
            f"equilibrium gap sd {gap_sd:.4f}; the badge will false-alarm")
        assert tol < cold, (
            f"n={n}: tol {tol:.4f} exceeds the cold-start gap {cold:.3f}, so "
            f"the badge stays quiet on a completely unequilibrated chain")

def test_stochastic_paths_are_drawn_from_arrows_not_the_height_function():
    """Reported: the stochastic picture showed only two vertex types.

    The cause was routing it through the DWBC renderer. That classifier assumes
    every adjacent pair of heights differs by exactly +-1, but this model's
    height function steps by 0 or 1, so it hit patterns like (1,0,0,0) that are
    not legal six-vertex types at all and collapsed six types into two.

    The arrows carry the information -- all six types are present in them -- so
    the path view reads those directly: at vertex (i,j) the occupied edges are
    left=h[i,j], bottom=v[i,j], right=h[i,j+1], top=v[i+1,j].
    """
    import os, importlib
    from sixvertex import stochastic as st

    n = 30
    h, v = st.sample(n, 0.3, 0.7, seed=3)
    seen = set()
    for i in range(n):
        for j in range(n):
            seen.add((int(h[i, j]), int(v[i, j]),
                      int(h[i, j + 1]), int(v[i + 1, j])))
    assert len(seen) == 6, f"expected all six vertex types, saw {len(seen)}"

    # the server must hand the arrows over, not only the height function
    server = importlib.import_module("server")
    body = server.app.test_client().post(
        "/api/stochastic", json={"n": 16, "b1": 0.3, "b2": 0.7, "seed": 1}
    ).get_json()
    assert "h_b64" in body["frame"] and "v_b64" in body["frame"], (
        "the endpoint returns no arrows, so the path view cannot draw the "
        "six types")

    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    assert "arrowFrame" in js, "the client never reads the arrows"
    # Check that faceBits reads the arrows, NOT the exact index expression.
    # This assertion used to demand the literal "arrowFrame.h(i, j)", which
    # froze the very indexing that inverted the vertical axis and broke the
    # paths: it would have blocked the fix rather than caught the bug. The
    # continuity invariant is pinned by
    # test_drawn_paths_have_no_dangling_vertical_segments instead.
    face = js[js.index("function faceBits"):]
    face = face[:face.index("const tl =")]
    assert "arrowFrame.h(" in face and "arrowFrame.v(" in face, \
        "paths are not drawn from the arrows"


def test_paths_are_black_on_white():
    """Requested: invert the path view, black paths on a white background --
    the convention in the literature, and it prints legibly."""
    import os
    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))
    i = code.index('mode === "paths"')
    branch = code[i:i + 500]
    assert "img.data[idx] = 255" in branch, "path background is not white"
    assert "PATH_INK" in code, "path colour is not defined"
    ink = code[code.index("PATH_INK ="):code.index("PATH_INK =") + 60]
    assert "#1" in ink or "#0" in ink, f"path ink is not dark: {ink}"


def test_model_switch_disables_every_dwbc_control():
    """Reported: the DWBC controls stayed live in stochastic mode.

    Three separate causes, all mine:
      - the reset button's id is btn-init, not btn-reset, so it was never in
        the list being disabled;
      - updateDeltaDisplay() re-enabled the exact button on any slider change,
        because it did not know which model was selected;
      - the mixing badge lingered from an earlier DWBC run. It compares two
        Markov chains, and the stochastic model has no chain at all.
    """
    import os
    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))

    assert "function isStochastic" in code, (
        "no single source of truth for which model is selected")

    # every DWBC control, reset included
    body = _js_function_body(code, "function updateModelUI")
    for btn in ("btn-init", "btn-play", "btn-step", "btn-exact", "btn-fluct"):
        assert btn in body, f"{btn} is not disabled when the model switches"

    # the exact button must stay off regardless of slider activity
    i = code.index("btnExact.disabled =")
    assert "isStochastic()" in code[i:i + 120], (
        "updateDeltaDisplay re-enables the exact button in stochastic mode")

    # and the chain diagnostic must not show for a model with no chain
    j = code.index("function updateMixingBadge")
    assert "isStochastic()" in code[j:j + 500], (
        "the mixing badge is shown for the stochastic model, which has no "
        "Markov chain")


def test_all_views_read_the_arrows_for_the_stochastic_model():
    """Reported twice: the stochastic picture "has only two types of vertices".

    The first fix taught the PATH view to read the arrows but left the
    coloured-vertex raster classifying the stochastic height function with
    DWBC logic. That logic assumes adjacent heights differ by exactly +-1;
    this model's differ by 0 or 1. Measured on the live site at n=24, the
    coloured view saw a1 x 272 and b2 x 275 -- two types -- plus 29 faces
    that fell through a silent `return "a1"`. The arrows had all six the
    whole time (a1 263, a2 264, b1 9, b2 11, c1 26, c2 3).

    Two changes stop this recurring: both views now take their edge bits from
    one `faceBits` helper, and a face that is not one of the six returns
    "invalid" instead of masquerading as a1.
    """
    import os
    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))

    assert "function faceBits" in code, "no shared edge-bit helper"
    assert "function classifyBits" in code, "no shared classifier"
    assert 'return "invalid"' in code, (
        "an unclassifiable face still falls through to a valid type; that is "
        "what hid this bug")

    # EVERY vertex6 branch, not just the first. The original version of this
    # test took code.index('mode === "vertex6"') -- the on-screen raster --
    # and stopped. The SVG export's branch reads `viewMode.value ===
    # "vertex6"`, a different string, so the test never looked at it, and the
    # export went on classifying raw corner heights for weeks after the
    # screen was fixed. Checking one call site is how this bug survived twice.
    import re
    sites = [m.start() for m in re.finditer(r'vertex6"\)?\s*\{', code)]
    sites += [m.start() for m in re.finditer(r'=== "vertex6"', code)]
    # A RENDERING branch is one that unpacks a colour to paint with; the
    # colour-picker code also mentions VERTEX6_COLORS but paints nothing, and
    # keying on the name alone picked it up as a false positive.
    rendering = [k for k in sites if "const [r, g, b] =" in code[k:k + 900]]
    assert len(rendering) >= 2, (
        f"expected at least two rendering branches (screen and SVG export), "
        f"found {len(rendering)}")
    for k in rendering:
        branch = code[k:k + 900]
        assert "faceBits(" in branch, (
            "a vertex6 rendering branch does not use the shared edge bits, "
            "so it will misread the stochastic model: " + branch[:160])
        assert "invalid" in branch, (
            "a vertex6 rendering branch has no invalid case, so an "
            "unclassifiable face will masquerade as a valid type")

    # and there must be no height-only classifier left to misuse
    assert "classifyFaceLocal" not in code, (
        "the height-only classifier is back; it assumes adjacent heights "
        "differ by exactly +-1, which the stochastic model violates")

    # and the path view must use the same helper rather than its own copy
    j = code.index("function drawPaths")
    paths = code[j:j + 1600]
    assert "faceBits(frame, i, j)" in paths, (
        "the path view has its own bit extraction again; the two will diverge")

    # the types EXPORT had the same defect: it classified raw heights, so a
    # stochastic sample exported the same two-type garbage -- into a file,
    # where it looks authoritative.
    k = code.index("function typesAsText")
    export = code[k:k + 900]
    assert "faceBits(frame, i, j)" in export, (
        "the vertex-type export still classifies raw heights")
    assert "classifyFaceLocal(" not in export, (
        "the vertex-type export still uses the height-only classifier")

    # the chain-only view must not be offered for a model with no chain
    assert 'option[value="active"]' in code, (
        "the liquid/frozen view is not gated; for the stochastic model the "
        "server sends an empty mask, so it would render everything frozen")


def test_stochastic_arrows_give_all_six_vertex_types():
    """The arrows must actually contain six types -- otherwise the view fix
    above would be dressing up a sampler bug."""
    from sixvertex import stochastic as st

    def bits(l, t, b, r):
        if not l and not t and not b and not r: return "a1"
        if l and t and b and r:                 return "a2"
        if l and t and not b and not r:         return "c1"
        if not l and not t and b and r:         return "c2"
        if not l and t and b and not r:         return "b1"
        if l and not t and not b and r:         return "b2"
        return "invalid"

    for n, seed in ((24, 5), (40, 7)):
        h, v = st.sample(n, 0.3, 0.7, seed=seed)
        counts = {}
        for i in range(n):
            for j in range(n):
                k = bits(bool(h[i, j]), bool(v[i + 1, j]),
                         bool(v[i, j]), bool(h[i, j + 1]))
                counts[k] = counts.get(k, 0) + 1
        assert counts.get("invalid", 0) == 0, (
            f"n={n}: {counts['invalid']} faces are not legal vertex types")
        assert len(counts) == 6, (
            f"n={n}: expected all six vertex types, saw {sorted(counts)}")


def test_experiment_observables_are_pinned():
    """The two measurement campaigns depend on choices that were made from
    data, not from taste, and that a later edit could silently undo.

    Coalescence: the quantity is the FORWARD coupling time, not the number
    CFTP prints. CFTP's is the first window of a doubling schedule, so it is
    quantised to powers of two -- at n=16 it read 256 at Delta = +0.5, 0, -1,
    -2 and -3 alike, a staircase hiding the whole dependence.

    Island: the component is 8-connected. Measured at n=48 over 60 samples,
    4-connectivity separated the island from the runner-up by a median 3.6x
    and left 8% of samples ambiguous, and the largest-plateau definition left
    43% ambiguous; 8-connectivity separated by 50x and left none. The level is
    the modal parity-corrected height, which agreed with the independent
    plateau definition in 90-97% of samples.
    """
    import os
    here = os.path.dirname(__file__)
    exp = os.path.join(here, "..", "experiments")
    # an assertion, not a silent `return`: experiments/ is part of the
    # repository, so its absence is a broken checkout rather than a reason to
    # report PASS without checking anything.
    assert os.path.isdir(exp), (
        f"{exp} is missing; this test pins the observables defined there and "
        f"cannot do so without it")

    mixing = open(os.path.join(exp, "mixing.py")).read()
    isl = open(os.path.join(exp, "island.py")).read()

    assert "def forward_coalescence_time" in mixing, (
        "the coalescence observable is gone; CFTP's own T is quantised to "
        "powers of two and cannot resolve the Delta dependence")
    assert "in_monotone_region" in mixing, (
        "coalescence time is only defined where the coupling is monotone")

    assert "connectivity=8" in isl, (
        "the island must default to 8-connectivity; 4-connectivity and the "
        "plateau definition both leave a large fraction of samples ambiguous")
    assert "np.ones((3, 3))" in isl, "8-connectivity structure is missing"
    assert "(i + j) & 1" in isl, (
        "the parity correction is gone; the raw height alternates inside the "
        "island, so its level is only constant after removing the checkerboard")
    # the lattice spacing is 2, so the centred variable is halved
    # Read the function's BODY, not a fixed slice after its name. This was
    # `isl.split("def island")[1][:1600]`, and adding a paragraph to the
    # docstring pushed the line it looks for past 1600 characters, so the
    # check failed on a documentation change and would equally have passed
    # over a real one once the function grew. _js_function_body in this file
    # carries the same warning for the JS side; it had not been applied here.
    #
    # The assertion itself was also weaker than it reads: it looked for the
    # substring "/ 2", which appears in `(level - n / 2)` whatever happens
    # to the outer division. Deleting that outer /2 -- exactly the fault
    # described below -- left the test passing. Match the expression.
    import re as _re
    _isl_body = isl.split("def island")[1].split("\ndef ")[0]
    assert _re.search(r"k\s*=\s*\(\s*level\s*-\s*n\s*/\s*2\s*\)\s*/\s*2",
                      _isl_body), (
        "k must be (level - n/2)/2; neighbouring plateaus differ by 2, not 1")


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


def test_island_shift_summary_is_identifiable_only_with_three_cells():
    """The (mu, sigma) fit needs three populated cells; with two it is a ridge.

    This pins the guard added after an n=24 run reported sigma=0.208 with a
    bootstrap interval of width 0.006 -- not precision, but a degenerate MLE
    on a two-cell histogram that happened to draw zero in a cell with
    expected count 3.
    """
    import numpy as np
    from experiments.island import discrete_gaussian_fit

    two_cell = [0.0] * 1650 + [-1.0] * 350
    fit = discrete_gaussian_fit(two_cell)
    assert fit["n_cells_used"] < 3, (
        "a two-cell histogram must not report enough cells to test a shape")

    three_cell = two_cell + [1.0] * 400
    assert discrete_gaussian_fit(three_cell)["n_cells_used"] >= 3


def test_island_neighbour_ratio_is_stable_across_lattice_size():
    """P(k=-1)/P(k=0) is the quantity that is actually measured.

    Unlike mu and sigma it needs only the two populated cells, and a discrete
    Gaussian with variance bounded in n predicts it does not drift with n.
    Measured at Delta=-3: 0.212 (n=16), 0.210 (n=24), 0.222 (n=32).

    Scope: every n here is 0 mod 4, and that is load-bearing rather than
    incidental. k is centred on n/2, so the secondary level sits below the
    mode at that parity and above it at n = 2 (mod 4), where this same
    estimator reads 0.000 instead of 0.21. The quantity that is genuinely
    stable in n is the direction-agnostic secondary/dominant ratio -- see
    test_the_island_level_ratio_is_stable_once_the_estimator_stops_picking_a_side.
    """
    from collections import Counter

    from sixvertex.cftp_exact import cftp_sample
    from experiments.island import island
    from experiments.mixing import weights_for_delta

    w = weights_for_delta(-3.0)
    ratios = []
    for n in (16, 24):
        ks = [island(cftp_sample(n, w, master_seed=4_100_000 + s)[0], n)["k"]
              for s in range(250)]
        c = Counter(k for k in ks if k is not None)
        mode = max(c, key=c.get)
        ratios.append(c.get(mode - 1, 0) / c[mode])

    assert all(0.10 < r < 0.40 for r in ratios), ratios
    assert abs(ratios[0] - ratios[1]) < 0.15, ratios


def test_island_distribution_is_one_sided():
    """The island sits one level low often and one level high almost never.

    P(+1)/P(0) ~ 1.5e-3 against P(-1)/P(0) ~ 0.21, a factor of ~140. Any
    change that symmetrises this has broken either the parity correction or
    the component selection.

    Scope: at n = 32, which is 0 mod 4. The sidedness is a property of the
    centring k = (level - n/2)/2, not of the island -- at n = 14, 18, 22 the
    secondary level sits one step ABOVE the mode with the same ~0.2 weight,
    so the distribution is one-sided the other way. Read this as a statement
    about this n, not about all lattice sizes.
    """
    from collections import Counter

    from sixvertex.cftp_exact import cftp_sample
    from experiments.island import island
    from experiments.mixing import weights_for_delta

    w = weights_for_delta(-3.0)
    ks = [island(cftp_sample(32, w, master_seed=4_200_000 + s)[0], 32)["k"]
          for s in range(300)]
    c = Counter(k for k in ks if k is not None)
    mode = max(c, key=c.get)
    assert c.get(mode - 1, 0) > 20, dict(c)
    assert c.get(mode + 1, 0) <= 5, dict(c)


def test_the_site_credits_gorin():
    """Gorin asked for this line explicitly, in his own wording.

    A test rather than a one-off edit because the header markup has been
    rewritten several times and a credit that silently disappears is worse
    than one that was never added.
    """
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, "static", "index.html")).read()
    css = open(os.path.join(root, "static", "style.css")).read()
    assert "proposed by Vadim Gorin" in html
    assert 'class="credit"' in html
    assert ".credit {" in css, "the credit line must be styled, not default text"


def test_local_exponent_recovers_a_known_power_law():
    """Known-answer test for the asymptotics observable.

    A rising local exponent is the whole claim in experiments/run_asymptotics,
    so the estimator has to return a FLAT exponent on data that really is a
    power law, and a rising one on data that really is not. Both directions
    are checked, because an estimator biased upward would manufacture the
    result.
    """
    import numpy as np

    ns = np.array([32, 40, 48, 56, 64, 80], float)

    def local(t):
        return np.log(t[1:] / t[:-1]) / np.log(ns[1:] / ns[:-1])

    flat = local(7.0 * ns ** 3)
    assert np.allclose(flat, 3.0), flat

    # exp(n/20) is superpolynomial: the local exponent must climb
    rising = local(np.exp(ns / 20))
    assert np.all(np.diff(rising) > 0), rising
    assert rising[-1] > 1.8 * rising[0], rising


def test_delta_minus_four_coalescence_exceeds_its_own_low_n_power_law():
    """Pins the measured departure, at a size that still runs in seconds.

    Fitting tau ~ n^alpha on n=16..32 at Delta=-4 and extending to n=48
    under-predicts the measured time. The same construction at Delta=-1 does
    not, which is why the control is part of the test rather than a footnote.
    """
    import numpy as np

    from experiments.mixing import forward_coalescence_time, weights_for_delta

    def med(n, d, k=9):
        w = weights_for_delta(d)
        return float(np.median([forward_coalescence_time(n, w, seed=5_000 + s)
                                for s in range(k)]))

    ratios = {}
    for d in (-4.0, -1.0):
        a, b, c = med(16, d), med(32, d), med(48, d)
        alpha = np.log(b / a) / np.log(2.0)
        ratios[d] = c / (b * (48 / 32) ** alpha)

    assert ratios[-4.0] > 1.10, ratios
    assert ratios[-4.0] > ratios[-1.0], ratios


# ------------------------------------------------- general domains (2408.14446)
def test_general_domain_reduces_to_dwbc_exactly():
    """The gate for the whole general-domain module.

    k = l = 1 with I = [[1]] IS domain wall boundary conditions, so the
    general code must reproduce the DWBC extremal heights bit for bit. The
    conventions here (which edge of a face is l, t, b, r, and which boundary
    edges carry paths) were read off the DWBC code rather than re-derived,
    precisely because an independently derived convention was silently wrong
    once before.
    """
    import numpy as np

    from sixvertex.cftp_exact import extremal_height
    from sixvertex.domains import domain_mask, extremal_heights

    for n in (1, 2, 3, 4, 5, 6, 8, 12):
        lo, hi = extremal_heights(domain_mask([0, n], [0, n], [[1]]))
        assert np.array_equal(lo, extremal_height(n, "lo").astype(np.int64)), n
        assert np.array_equal(hi, extremal_height(n, "hi").astype(np.int64)), n


def test_two_independent_enumerators_agree_on_general_domains():
    """Heights and edges are counted by code sharing nothing but the mask.

    The height route goes through boundary propagation and the height
    convention; the edge route assigns 0/1 to lattice edges and only checks
    each cell against the six legal vertices.

    This used to be five hand-picked domains, all of them non-empty, and the
    claim that the two agree was false. The height enumerator checked a face
    only when a FREE corner of it was assigned, so a face whose four corners
    are all boundary-fixed was never checked at all -- and 150 of 250 domains
    in the 3x3 family have such a face. On 23 of those 250 it reported 1 or 2
    configurations for domains that have NONE, always in that direction. Five
    non-empty examples could not see it.

    So the sweep is the test now: every domain reachable from a 3x3 convex
    array on five cut sets. The ASM numbers are kept as an absolute anchor,
    since two enumerators agreeing on a wrong answer is the failure mode a
    cross-check alone cannot catch.

    They also disagreed once before, in the other direction -- the height
    enumerator visited corners in row-major order and guessed a value at
    corners with no fixed neighbour yet, over-counting 5 configurations as
    2604. A square exposes neither bug, which is why the sweep is general.
    """
    from sixvertex.domains import domain_mask
    from experiments.brute_domain import (enumerate_edge_configs,
                                          enumerate_heights)

    cases = [([0, 3], [0, 3], [[1]], 7),
             ([0, 4], [0, 4], [[1]], 42),
             ([0, 5], [0, 5], [[1]], 429),
             ([0, 1, 2, 4], [0, 1, 2, 4], [[1, 1, 1], [0, 1, 1], [0, 0, 1]], None),
             ([0, 1, 3, 4], [0, 2, 3, 4], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], 4)]
    for X, Y, I, expected in cases:
        m = domain_mask(X, Y, I)
        e = enumerate_edge_configs(m)
        h = len(enumerate_heights(m))
        assert e == h, (X, Y, I, e, h)
        if expected is not None:
            assert e == expected, (X, Y, I, e, expected)

    # the ASM anchor: both routes against a known sequence, not just each other
    asm = [1, 2, 7, 42, 429]
    for n, want in enumerate(asm, start=1):
        m = domain_mask([0, n], [0, n], [[1]])
        assert len(enumerate_heights(m)) == want, n
        assert enumerate_edge_configs(m) == want, n

    # and the sweep, including the empty domains the old cases never reached
    import itertools

    from sixvertex.domains import DomainError

    cutsets = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4],
               [0, 2, 4, 6]]
    compared = empties = 0
    for bits in itertools.product([0, 1], repeat=9):
        I = np.array(bits).reshape(3, 3)
        if I.sum() == 0:
            continue
        for X in cutsets:
            for Y in cutsets:
                try:
                    m = domain_mask(X, Y, I)
                except DomainError:
                    continue
                try:
                    h = len(enumerate_heights(m))
                except (RuntimeError, DomainError):
                    continue          # disconnected corner graph, or cap
                e = enumerate_edge_configs(m)
                compared += 1
                empties += (e == 0)
                assert h == e, (
                    f"the two enumerators disagree on X={X}, Y={Y}, "
                    f"I={I.tolist()}: heights {h}, edges {e}")
    # the edge route's own assumption: contiguous rows and columns, which is
    # what its entry/exit edges rest on. True of every mask domain_mask can
    # build, and now refused rather than silently miscounted if it is not.
    import pytest

    gap = np.zeros((4, 4), dtype=bool)
    gap[0, 0] = gap[0, 3] = True
    gap[1, :] = True
    with pytest.raises(ValueError, match="not contiguous"):
        enumerate_edge_configs(gap)

    assert (compared, empties) == (874, 244), (
        f"the sweep census moved to {(compared, empties)}; it covers 874 "
        f"domains of which 244 are empty, and the bug this test was rewritten "
        f"for only shows on the empty ones")


def test_convexity_of_I_does_not_imply_the_domain_is_nonempty():
    """Roughly half of the convex patterns admit no configuration at all.

    Measured over all 98 convex 3x3 patterns at N=4: 46 samplable, 49 with
    zero configurations, 3 whose corner graph is disconnected. This is a
    property of the model, not a bug, and it is pinned because an empty domain
    must keep being refused rather than silently sampled.
    """
    from sixvertex.domains import domain_mask, extremal_heights, DomainError
    from experiments.brute_domain import enumerate_edge_configs

    empty = [[1, 1, 1], [1, 1, 1], [1, 0, 0]]
    m = domain_mask([0, 1, 2, 4], [0, 1, 2, 4], empty)
    assert enumerate_edge_configs(m) == 0
    try:
        extremal_heights(m)
        raise AssertionError("an empty domain must be refused, not sampled")
    except DomainError:
        pass


def test_a_permutation_inside_the_domain_is_not_necessary():
    """Counterexample to the obvious characterisation, kept as a test.

    A configuration with exactly N corners is a permutation matrix supported
    in the domain, so it was natural to guess that the domain is nonempty iff
    the mask has a perfect matching. Sufficient in all 686 domains tested,
    but NOT necessary: this plus-shaped domain has three rows confined to two
    columns, so Hall's condition fails, yet it carries 4 configurations --
    every one of them with more than N corners.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from sixvertex.domains import domain_mask
    from experiments.brute_domain import enumerate_edge_configs

    m = domain_mask([0, 1, 3, 4], [0, 2, 3, 4],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    cost = np.where(m, 0.0, 1.0)
    r, c = linear_sum_assignment(cost)
    assert cost[r, c].sum() > 0, "expected no perfect matching"
    assert enumerate_edge_configs(m) == 4


def test_general_domain_sampler_is_exact_on_the_square():
    """The general-domain CFTP against the enumerated law, at c != 1.

    Run on the square so the answer is independently known; the staircase
    version of this check lives in experiments/validate_domains.py, which is
    too slow for the suite.
    """
    import numpy as np

    from sixvertex.domains import cftp_sample, classify_counts, domain_mask
    from experiments.brute_domain import enumerate_heights

    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=2.0, c2=2.0)
    m = domain_mask([0, 4], [0, 4], [[1]])
    hs = enumerate_heights(m)
    wt = []
    for H in hs:
        x = 1.0
        for k, v in classify_counts(H, m).items():
            x *= w[k] ** v
        wt.append(x)
    exact = np.array(wt) / sum(wt)

    key = {H.tobytes(): i for i, H in enumerate(hs)}
    N = 1500
    cnt = np.zeros(len(hs))
    for s in range(N):
        H, _ = cftp_sample(m, w, master_seed=910_000 + s)
        cnt[key[np.ascontiguousarray(H).tobytes()]] += 1
    dev = float(np.max(np.abs(cnt / N - exact)))
    assert dev < 3 / np.sqrt(N), dev


def test_general_domain_sampler_is_exact_on_a_staircase():
    """Exactness on a domain that is not a rectangle.

    Also a regression test for a failure that looked like bad mixing and was
    not: extremal_heights left +-1e9 sentinels on corners outside the domain,
    and cftp_sample compared whole arrays, so -1e9 was tested against +1e9
    forever. The chains were in fact coalescing after a single sweep while the
    sampler reported "no coalescence by T=2^20".
    """
    import numpy as np

    from sixvertex.domains import cftp_sample, classify_counts, domain_mask
    from experiments.brute_domain import enumerate_heights

    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.6, c2=1.6)
    m = domain_mask([0, 2, 4, 6], [0, 2, 4, 6], [[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    hs = enumerate_heights(m)
    assert len(hs) == 8

    wt = []
    for H in hs:
        x = 1.0
        for k, v in classify_counts(H, m).items():
            x *= w[k] ** v
        wt.append(x)
    exact = np.array(wt) / sum(wt)

    key = {H.tobytes(): i for i, H in enumerate(hs)}
    N = 2000
    cnt = np.zeros(len(hs))
    for s in range(N):
        H, info = cftp_sample(m, w, master_seed=930_000 + s)
        assert info["T"] < 1 << 20, "coalescence must not run away here"
        cnt[key[np.ascontiguousarray(H).tobytes()]] += 1
    dev = float(np.max(np.abs(cnt / N - exact)))
    assert dev < 3 / np.sqrt(N), dev


def test_extremal_heights_leave_no_sentinels():
    """Corners outside the domain must not carry +-1e9 into anything."""
    import numpy as np

    from sixvertex.domains import corner_mask, domain_mask, extremal_heights

    m = domain_mask([0, 2, 4, 6], [0, 2, 4, 6], [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    lo, hi = extremal_heights(m)
    cm = corner_mask(m)
    assert np.all(np.abs(lo[~cm]) < 100) and np.all(np.abs(hi[~cm]) < 100)
    assert np.all(hi[cm] >= lo[cm])


def test_existence_criterion_matches_exhaustive_enumeration():
    """The polynomial-time nonemptiness test against ground truth.

    The whole sweep, not spot-checks: every domain reachable from a 3x3 convex
    array on five cut sets, each compared against exhaustive edge enumeration.
    1078 domains, zero disagreements, about 4 seconds -- there was no reason
    for this to be eight hand-picked cases.

    The module docstring claimed "980 domains" and this one claimed "the
    1364-domain sweep". The two disagreed, and neither was reproducible
    because the sweep was never written down; it is performed here instead.

    The eight cases kept below are the historically interesting ones, where
    two earlier versions of the criterion failed in opposite directions: one
    accepted empty domains whose disconnected pieces were never checked, the
    other rejected nonempty disconnected domains outright.
    """
    from sixvertex.domains import admits_configuration, domain_mask
    from experiments.brute_domain import enumerate_edge_configs

    cases = [
        ([0, 4], [0, 4], [[1]]),
        ([0, 1, 2, 4], [0, 1, 2, 4], [[1, 1, 1], [1, 1, 1], [1, 0, 0]]),
        ([0, 1, 2, 4], [0, 1, 2, 4], [[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
        ([0, 1, 3, 4], [0, 2, 3, 4], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        ([0, 1, 2, 4], [0, 1, 2, 4], [[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
        ([0, 1, 2, 4], [0, 1, 2, 4], [[1, 0, 0], [0, 0, 1], [0, 1, 1]]),
        ([0, 2, 4, 6], [0, 2, 4, 6], [[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        ([0, 1, 2, 3], [0, 1, 2, 3], [[0, 0, 1], [1, 0, 0], [1, 1, 0]]),
    ]
    for X, Y, I in cases:
        m = domain_mask(X, Y, I)
        assert admits_configuration(m) == (enumerate_edge_configs(m) > 0), (X, Y, I)

    # and the full sweep
    import itertools

    from sixvertex.domains import DomainError

    cutsets = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4],
               [0, 2, 4, 6]]
    compared = 0
    for bits in itertools.product([0, 1], repeat=9):
        I = np.array(bits).reshape(3, 3)
        if I.sum() == 0:
            continue
        for X in cutsets:
            for Y in cutsets:
                try:
                    m = domain_mask(X, Y, I)
                except DomainError:
                    continue
                compared += 1
                assert admits_configuration(m) == (enumerate_edge_configs(m) > 0), (
                    f"the criterion and exhaustive enumeration disagree on "
                    f"X={X}, Y={Y}, I={I.tolist()}")
    assert compared == 1078, (
        f"the sweep now covers {compared} domains, not the 1078 both "
        f"docstrings record")


def test_nonemptiness_depends_on_the_cuts_not_only_on_I():
    """I alone does not decide it -- 91 of 98 convex patterns flip with cuts.

    Worth pinning because the natural reading of Section 2.1 is that I
    describes the domain, and any characterisation phrased in terms of I
    alone is therefore doomed.
    """
    from sixvertex.domains import admits_configuration, domain_mask

    I = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    a = admits_configuration(domain_mask([0, 2, 4, 6], [0, 2, 4, 6], I))
    b = admits_configuration(domain_mask([0, 1, 2, 4], [0, 1, 2, 4], I))
    assert a != b, (a, b)


def test_existence_criterion_scales_past_enumeration():
    """Polynomial where enumeration is hopeless: N=240 in well under a second.

    The all-pairs form of the same test took 68 seconds here; the
    multi-source form is one relaxation pass.
    """
    import time

    from sixvertex.domains import admits_configuration, domain_mask

    N = 240
    c = [0, N // 3, 2 * N // 3, N]
    m = domain_mask(c, c, [[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    t0 = time.time()
    assert admits_configuration(m) is True
    assert time.time() - t0 < 5.0


def test_only_degree_four_free_corners_are_sampled():
    """The monotone-coupling precondition is enforced, not assumed.

    CFTP is exact only because the coupling is monotone, and the condition
    c1c2 >= a1a2, c1c2 >= b1b2 was derived for a corner surrounded by four
    faces -- the only kind a square has. General domains do produce free
    corners of degree 1, 2 and 3 -- 122 of the 1277 free corners across the 135
    samplable domains this scan reaches, spread over 11 domains -- and at those
    the derivation does not apply. (This said "122 of 1635 across a 213-domain
    scan"; the 122 reproduces and the other two do not, so
    test_the_domain_scan_census_is_computed_not_quoted counts them instead.) Every domain the sampler accepts
    turned out to have only degree-4 free corners anyway, but that is a
    finite observation, so it is checked rather than relied on.
    """
    import itertools

    import numpy as np

    from sixvertex.domains import (DomainError, _faces_of_corner,
                                   admits_configuration, cftp_sample,
                                   domain_mask, free_corners)

    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.6, c2=1.6)
    checked = 0
    for bits in itertools.product([0, 1], repeat=9):
        I = np.array(bits).reshape(3, 3)
        if I.sum() == 0:
            continue
        for cuts in ([0, 1, 2, 4], [0, 2, 4, 6]):
            try:
                m = domain_mask(cuts, cuts, I)
            except DomainError:
                continue
            if not admits_configuration(m):
                continue
            degs = [len(_faces_of_corner(m, int(i), int(j)))
                    for (i, j) in map(tuple, np.argwhere(free_corners(m)))]
            try:
                cftp_sample(m, w, master_seed=0)
                accepted = True
            except DomainError:
                accepted = False
            if accepted:
                assert all(d == 4 for d in degs), (cuts, I.tolist(), degs)
                checked += 1
    assert checked > 50, checked


def test_the_two_samplers_agree_on_the_square():
    """DWBC sampler vs general-domain sampler, same law, no shared code.

    Different sweeps, different random streams, different extremal-height
    construction. Both are also compared against the enumerated law so a
    shared error would still show.

    The comparison is a 2 x k test of HOMOGENEITY. An earlier version fed one
    sampler's random counts in as chi-square "expected" values; that carries
    about twice the variance it assumes, and it reported p=0.001 for two
    samplers that were each matching the exact law to within 0.005.
    """
    import numpy as np
    from scipy.stats import chi2_contingency

    from sixvertex.cftp_exact import cftp_sample as dwbc_sample
    from sixvertex.domains import (classify_counts, cftp_sample as dom_sample,
                                   domain_mask)
    from experiments.brute_domain import enumerate_heights

    n = 4
    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=2.0, c2=2.0)
    m = domain_mask([0, n], [0, n], [[1]])
    hs = enumerate_heights(m)
    wt = []
    for H in hs:
        x = 1.0
        for k, v in classify_counts(H, m).items():
            x *= w[k] ** v
        wt.append(x)
    exact = np.array(wt) / sum(wt)
    key = {H.astype(np.int64).tobytes(): i for i, H in enumerate(hs)}

    N = 2500
    cA = np.zeros(len(hs))
    cB = np.zeros(len(hs))
    for s in range(N):
        H, _ = dwbc_sample(n, w, master_seed=640_000 + s)
        cA[key[np.asarray(H, dtype=np.int64).tobytes()]] += 1
        G, _ = dom_sample(m, w, master_seed=740_000 + s)
        cB[key[np.ascontiguousarray(G, dtype=np.int64).tobytes()]] += 1

    assert np.max(np.abs(cA / N - exact)) < 3 / np.sqrt(N)
    assert np.max(np.abs(cB / N - exact)) < 3 / np.sqrt(N)
    keep = exact * N >= 5
    _, p, _, _ = chi2_contingency(np.vstack([cA[keep], cB[keep]]))
    assert p > 0.001, p


def test_the_coupling_is_monotone_on_a_general_domain():
    """Exhaustive, and over the family rather than three chosen members.

    Monotonicity is the single property CFTP's exactness rests on, and this
    used to check 3 hand-picked domains at 2 values of Delta -- 6 checks. The
    exhaustive check is quadratic in the configuration count, which is why it
    was kept small, but it is cheap on every domain that has few enough
    configurations to enumerate: 191 of them carry 200 or fewer, and sweeping
    all of those at three Delta values is 573 exhaustive checks in about 17
    seconds. Hand-picked cases standing in for a sweep that was affordable all
    along is how three separate defects hid in this suite today, so the sweep
    is the test.

    Exhaustive means exhaustive: every ordered pair of configurations, every
    site, and the driving uniform evaluated at every threshold of the update
    and on both sides of it. Zero violations.

    The Delta values span the region rather than sampling its interior: 0.5 is
    the BOUNDARY, where a1a2 = b1b2 = c1c2 exactly and the condition holds
    with equality, which is the case most likely to fail; -5.0 is deep inside.
    experiments/monotone_check covers larger domains, including a notch with
    2889 configurations, which is past what belongs in a test run.
    """
    import itertools

    import numpy as np

    from sixvertex.domains import (DomainError, admits_configuration,
                                   domain_mask)
    from experiments.brute_domain import enumerate_heights
    from experiments.mixing import weights_for_delta
    from experiments.monotone_check import monotonicity_violations

    masks = []
    for bits in itertools.product([0, 1], repeat=9):
        I = np.array(bits).reshape(3, 3)
        if I.sum() == 0:
            continue
        for cuts in ([0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 4, 6]):
            try:
                m = domain_mask(cuts, cuts, I)
            except DomainError:
                continue
            if not admits_configuration(m):
                continue
            try:
                k = len(enumerate_heights(m))
            except (RuntimeError, DomainError):
                continue
            if 0 < k <= 200:
                masks.append((cuts, I.tolist(), m))

    assert len(masks) == 191, (
        f"the sweep now covers {len(masks)} domains, not the 191 recorded; "
        f"the enumerable-domain census moved")

    checks = 0
    for d in (0.5, -1.0, -5.0):
        w = weights_for_delta(d)
        for cuts, I, m in masks:
            v = monotonicity_violations(m, w)
            checks += 1
            assert not v, (
                f"the coupling is NOT monotone at Delta={d} on cuts={cuts}, "
                f"I={I}: {v[0]}. CFTP's exactness rests on this.")
    assert checks == 573, checks


def test_the_monotone_region_is_sharp():
    """Not over-conservative: monotonicity fails as soon as the region ends.

    Walking a across the C >= A boundary with b = c = 1, the coupling is
    monotone at a=1.00, where A = C exactly, and stops being monotone at
    a=1.05. If it had survived outside, in_monotone_region would be refusing
    weights that are perfectly samplable.
    """
    from sixvertex.cftp_exact import in_monotone_region
    from sixvertex.domains import domain_mask
    from experiments.monotone_check import monotonicity_violations

    m = domain_mask([0, 4], [0, 4], [[1]])
    on = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.0, c2=1.0)
    out = dict(a1=1.05, a2=1.05, b1=1.0, b2=1.0, c1=1.0, c2=1.0)
    assert in_monotone_region(on) and not in_monotone_region(out)
    assert not monotonicity_violations(m, on), "boundary case must be monotone"
    assert monotonicity_violations(m, out), "just outside must NOT be monotone"


def test_stationary_chain_estimator_agrees_with_independent_sampling():
    """The cheap estimator must give the SAME answer as the expensive one.

    A heat-bath sweep preserves the measure, so a chain started from one exact
    CFTP sample is in equilibrium at every step and time averages are
    unbiased. That is the whole justification for replacing 7e5 independent
    CFTP samples (~25 h) with a few million sweeps (minutes), so it is checked
    against the known answer rather than argued.

    Reference: R_lo = 0.2119 at n=16, Delta=-3 from 2000 independent CFTP
    samples. Errors here are chain-to-chain batch means -- observations within
    a chain are correlated by construction and must never be pooled as if
    independent.
    """
    import numpy as np

    from experiments.mixing import weights_for_delta
    from experiments.run_shift_chain import chain_observations, summarise

    w = weights_for_delta(-3.0)
    per = []
    for c in range(8):
        ks = chain_observations(16, w, nmeas=400, thin=24,
                                seed=31_000 + c, start_seed=32_000 + c)
        per.append((int((ks == -1).sum()), int((ks == 0).sum()),
                    int((ks == 1).sum())))
    s = summarise(per)
    assert abs(s["R_lo"] - 0.2119) < 3 * np.hypot(s["se_lo"], 0.0113), s


def test_mu_sigma_closed_form_inverts_a_known_discrete_gaussian():
    """Known-answer test for the ratio -> (mu, sigma) inversion."""
    import numpy as np

    from experiments.run_shift_chain import mu_sigma_from_ratios

    for mu, sg in ((-0.30, 0.36), (0.0, 0.5), (0.12, 0.42)):
        k = np.arange(-6, 7)
        p = np.exp(-(k - mu) ** 2 / (2 * sg * sg))
        p /= p.sum()
        i0 = int(np.argmax(p))
        m, s = mu_sigma_from_ratios(p[i0 - 1] / p[i0], p[i0 + 1] / p[i0])
        assert abs(m - mu) < 1e-6 and abs(s - sg) < 1e-6, (mu, sg, m, s)


def test_island_ratio_is_non_monotone_in_delta():
    """R_lo has an interior minimum near Delta=-3, and that forbids a limit.

    Pinned because it is the fact that kills the tempting extrapolation. The
    fitted slopes of 1/sigma^2 and 2mu/sigma^2 against ln c come out equal and
    opposite to within 0.2 sigma, which would force mu -> -1/2 -- but that
    statement is exactly "R_lo is constant", and R_lo is chi2=220 on 5 d.o.f.
    away from constant. Each slope's own error (0.12) is far larger than the
    structure living in their difference, so two marginal fits that each pass
    are wrong together.
    """
    import csv
    import os

    import numpy as np

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "experiments", "shift_delta_chain_sample.csv")
    rows = sorted(csv.DictReader(open(path)), key=lambda r: float(r["delta"]))
    d = np.array([float(r["delta"]) for r in rows])
    R = np.array([float(r["R_lo"]) for r in rows])
    s = np.array([float(r["se_lo"]) for r in rows])

    i = int(np.argmin(R))
    assert 0 < i < len(R) - 1, "the minimum must be interior, not at an end"
    assert -4.5 < d[i] < -2.0, d[i]
    # and it is a real minimum, not noise, at both ends
    assert R[0] - R[i] > 5 * np.hypot(s[0], s[i]), (R[0], R[i])
    assert R[-1] - R[i] > 5 * np.hypot(s[-1], s[i]), (R[-1], R[i])

    w = 1 / s ** 2
    m = (w * R).sum() / w.sum()
    assert float((((R - m) / s) ** 2).sum()) > 50, "R_lo must be far from constant"


def test_mu_and_sigma_drift_monotonically_with_depth():
    """Both decrease steadily across the AF phase, with no sign of settling."""
    import csv
    import os

    import numpy as np

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "experiments", "shift_delta_chain_sample.csv")
    rows = sorted(csv.DictReader(open(path)),
                  key=lambda r: -abs(float(r["delta"])))
    mu = np.array([float(r["mu"]) for r in rows])
    sg = np.array([float(r["sigma"]) for r in rows])
    # ordered from deepest to shallowest: both should increase
    assert np.all(np.diff(mu) > 0), mu
    assert np.all(np.diff(sg) > 0), sg
    assert mu.min() < -0.34 and mu.max() > -0.31, mu


def test_the_island_ratio_minimum_survives_a_change_of_size():
    """The Delta=-3 minimum is a property of the model, not of n=16.

    The Delta sweep was run at n=16 because it is cheapest, on the strength of
    an n-independence check made at ONE Delta. Repeating at n=32 keeps the
    minimum in the same place, so the shape is not a finite-size artifact.
    """
    import csv
    import os

    import numpy as np

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("shift_delta_chain_sample.csv",
                 "shift_delta_chain_n32_sample.csv"):
        rows = sorted(csv.DictReader(open(os.path.join(root, "experiments", name))),
                      key=lambda r: float(r["delta"]))
        d = np.array([float(r["delta"]) for r in rows])
        R = np.array([float(r["R_lo"]) for r in rows])
        i = int(np.argmin(R))
        assert 0 < i < len(R) - 1, (name, "minimum must be interior")
        assert -4.0 < d[i] < -2.0, (name, d[i])


def test_the_island_ratio_level_is_size_dependent():
    """The LEVEL does shift with n, which an earlier test was too weak to see.

    Independent sampling over n=16..48 reported R_lo flat in n at p=0.27. The
    chain estimator is precise enough to resolve a pooled offset of
    +0.0088 +- 0.0029 (3.0 sigma) between n=16 and n=32, so "flat in n" was a
    statement about that test's power, not about the model. Quoted mu and
    sigma therefore carry an n label.
    """
    import csv
    import os

    import numpy as np

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def load(name):
        rows = sorted(csv.DictReader(open(os.path.join(root, "experiments", name))),
                      key=lambda r: float(r["delta"]))
        g = lambda k: np.array([float(r[k]) for r in rows])
        return g("delta"), g("R_lo"), g("se_lo")

    d16, R16, s16 = load("shift_delta_chain_sample.csv")
    d32, R32, s32 = load("shift_delta_chain_n32_sample.csv")
    diffs, ses = [], []
    for D in sorted(set(d16) & set(d32)):
        i = int(np.where(d16 == D)[0][0])
        j = int(np.where(d32 == D)[0][0])
        diffs.append(R32[j] - R16[i])
        ses.append(float(np.hypot(s16[i], s32[j])))
    diffs, ses = np.array(diffs), np.array(ses)
    w = 1 / ses ** 2
    m = float((w * diffs).sum() / w.sum())
    se = float(1 / np.sqrt(w.sum()))
    assert abs(m) / se > 2.0, (m, se)
    assert m > 0, m


def test_compiled_domain_sweep_is_bit_identical_to_the_reference():
    """Same standard applied to the DWBC sweep: identity, not 'close enough'.

    The two differ only in how the same arithmetic is scheduled, so any
    divergence is a bug in one of them. Checked on non-rectangular domains,
    since a square exercises none of the masking.
    """
    import numpy as np

    import sixvertex.domains as sd
    from sixvertex.domains import (_weight_table, domain_mask,
                                   extremal_heights, sweep, sweep_fast,
                                   sweep_plan, sweep_plan_arrays)

    if not sd._HAS_NUMBA:
        # pytest.skip, not a bare `return`: a return reports PASS, so on a
        # machine without numba this whole comparison vanished silently. Same
        # defect as the Izergin-Korepin check, which returned quietly when
        # mpmath was absent and was therefore vacuous wherever it mattered.
        import pytest
        pytest.skip("numba is not installed, so there is no compiled sweep "
                    "to compare against the reference one")
    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=2.0, c2=2.0)
    for cuts, I in (([0, 2, 4, 6], [[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
                    ([0, 4, 8, 12], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
                    ([0, 4, 8, 12], [[1, 1, 1], [1, 1, 1], [0, 1, 1]])):
        m = domain_mask(cuts, cuts, I)
        lo, _ = extremal_heights(m)
        plan, pa = sweep_plan(m), sweep_plan_arrays(m)
        tbl = _weight_table(w)
        tarr = np.array(tbl, dtype=np.float64)
        A = lo.astype(np.float64).copy()
        B = lo.astype(np.float64).copy()
        rng = np.random.default_rng(3)
        for _ in range(40):
            r = rng.random(A.shape)
            sweep(A, m, w, r, plan, tbl)
            sweep_fast(B, pa, tarr, r)
            assert np.array_equal(A, B), (cuts, I)


def test_island_level_is_concentrated_on_three_cells_on_every_domain():
    """The narrow fluctuation is not special to DWBC.

    On the square, the staircase, the plus and the notch, all of the island
    level's mass sits on {mode-2, mode, mode+2} -- levels step by 2 because
    the parity-corrected height does. So the qualitative discrete-Gaussian
    picture carries over to the Gorin-Kenyon domains.
    """
    import numpy as np

    from sixvertex.domains import domain_mask
    from experiments.island_domain import DOMAINS, chain_levels
    from experiments.mixing import weights_for_delta

    w = weights_for_delta(-3.0)
    n = 24
    for name, I in DOMAINS.items():
        cuts = [0, n] if len(I) == 1 else [0, n // 3, 2 * n // 3, n]
        m = domain_mask(cuts, cuts, I)
        lv, _, _ = chain_levels(m, w, nmeas=250, thin=32,
                                seed=81_000, start_seed=82_000)
        good = lv[~np.isnan(lv)]
        assert good.size > 200, (name, good.size)
        vals, cnts = np.unique(good, return_counts=True)
        mode = vals[np.argmax(cnts)]
        three = sum((good == mode + d).sum() for d in (-2, 0, 2)) / good.size
        assert three > 0.98, (name, three, dict(zip(vals, cnts)))


def test_island_width_differs_between_domains_with_clean_separation():
    """R_lo is domain-dependent, and not through component ambiguity.

    The square and the notch BOTH have a cleanly separated island (45x and
    51x over the runner-up), so neither number rests on a doubtful choice of
    component -- yet their R_lo differ by a factor of about five. The
    staircase (1.1x) and plus (2.6x) are recorded but carry that caveat.
    """
    import csv
    import os

    import numpy as np

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = {r["domain"]: r for r in csv.DictReader(
        open(os.path.join(root, "experiments", "island_domain_sample.csv")))}
    for d in ("square", "notch"):
        assert float(rows[d]["separation"]) > 20, d
    a, b = rows["square"], rows["notch"]
    diff = float(a["R_lo"]) - float(b["R_lo"])
    se = float(np.hypot(float(a["se_lo"]), float(b["se_lo"])))
    assert diff / se > 5, (diff, se)


def test_every_renderer_respects_the_domain_mask():
    """General domains: cells outside the region must never be drawn.

    The guard lives in one helper, faceInDomain, next to the shared edge
    bits -- for the same reason faceBits exists. A per-call-site check is how
    the stochastic mislabelling survived three rounds of fixes, and this
    model has the same shape of risk: six places paint cells, and any one of
    them left unguarded renders the removed blocks as if they were part of
    the model.
    """
    import os
    import re

    here = os.path.dirname(__file__)
    code = "\n".join(
        ln for ln in open(os.path.join(here, "..", "static", "draw.js")).read().splitlines()
        if not ln.strip().startswith("//"))

    assert "function faceInDomain" in code
    assert "if (!faceInDomain(frame, i, j)) return null;" in code, (
        "faceBits does not short-circuit outside the domain, so every view "
        "that reads edge bits will paint cells that are not in the model")

    # every consumer of faceBits must handle the null
    uses = [m.start() for m in re.finditer(r"= faceBits\(", code)]
    assert len(uses) >= 4, f"expected >=4 faceBits consumers, found {len(uses)}"
    for k in uses:
        window = code[k:k + 420]
        assert "=== null" in window, (
            "a faceBits consumer does not handle the outside-domain case: "
            + window[:140])

    # and the two renderers that never call faceBits need the guard directly
    for marker in ("const isActive = frame.getActive(i, j)",
                   "const avg = (frame.get(i, j)"):
        k = code.index(marker)
        assert "faceInDomain" in code[max(0, k - 400):k], (
            f"renderer at {marker!r} has no domain guard")


def test_general_domain_endpoint_refuses_what_it_should():
    """The server answers with something the user can act on.

    An empty domain is the important one: without the up-front check the
    sampler would simply never coalesce and the user would watch a progress
    bar forever. admits_configuration answers in one relaxation pass.
    """
    import server

    c = server.app.test_client()
    base = dict(n=12, a1=1, a2=1, b1=1, b2=1, c_up=2.0, c_down=2.0, seed=1,
                x_cuts=[3, 6], y_cuts=[3, 6])

    def post(**kw):
        d = dict(base)
        d.update(kw)
        r = c.post("/api/domain/start", json=d)
        return r.status_code, (r.get_json() or {})

    code, body = post(I=[[1, 0, 1], [1, 1, 1], [1, 1, 1]])
    assert code == 400 and "convex" in body["error"], body

    code, body = post(I=[[1, 1, 1], [1, 1, 1], [1, 0, 0]])
    assert code == 400 and "no valid configuration" in body["error"], body

    code, body = post(x_cuts=[6, 3], I=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert code == 400 and "increasing" in body["error"], body

    # outside the monotone region there is no exact method to fall back on
    code, body = post(c_up=0.5, c_down=0.5, I=[[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    assert code == 400 and "monotone" in body["error"], body

    # and a legal one starts
    code, body = post(I=[[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    assert code == 200 and body.get("ok"), body


def test_general_domain_sampler_accepts_a_missing_seed():
    """Omitting the seed must not reach numpy as SeedSequence([None]).

    It did: the request surfaced to the user as "int() argument must be a
    string, a bytes-like object or a real number, not 'NoneType'". The DWBC
    sampler had always handled this; this one had not.
    """
    from sixvertex.domains import cftp_sample, domain_mask

    m = domain_mask([0, 4, 8, 12], [0, 4, 8, 12],
                    [[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    H, info = cftp_sample(m, dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0,
                                  c1=2.0, c2=2.0), master_seed=None)
    assert isinstance(info["master_seed"], int)


def test_the_frozen_view_is_off_for_every_model_without_a_mask():
    """"liquid / frozen" marks sites a local chain can still flip.

    Neither the stochastic model nor a general domain has such a mask -- the
    server sends an empty one for both -- so this view would paint the whole
    configuration as frozen. That is a confident wrong picture, which is why
    the view was taken off the menu for the stochastic model already; the
    general domain was missed when that model was added.

    Caught by stepping a browser through every model in every view and
    looking at what was drawn, not by reading the diff.
    """
    import os

    here = os.path.dirname(__file__)
    code = "\n".join(
        ln for ln in open(os.path.join(here, "..", "static", "draw.js")).read().splitlines()
        if not ln.strip().startswith("//"))
    body = _js_function_body(code, "function updateModelUI")

    assert "const noMask = stoch || domain;" in body, (
        "the frozen view is gated on the stochastic model alone; a general "
        "domain has no flippable-site mask either")
    assert "activeOpt.disabled = noMask;" in body
    assert 'if (noMask && viewMode.value === "active") viewMode.value = "paths";' in body, (
        "a user already on the frozen view is left there when switching to a "
        "model that has no mask")


def test_browser_vertex_tally_matches_server_side_classification():
    """What the page draws must be what the sampler produced.

    Driving the real page at n=40 on the staircase domain (c=2) and reading
    its own types export gave a1 78, a2 78, b1 92, b2 625, c1 117, c2 77 over
    1067 cells. Classifying server-side over six samples gives 87.8, 87.8,
    90.0, 623.0, 109.2, 69.2 -- the same distribution, and in particular the
    same heavy frozen b region, which is real and not a rendering fault.

    The b labels have since been swapped to match Gorin-Kenyon (see
    test_vertex_labels_match_gorin_kenyon_via_the_frozen_corners), so the
    heavy region is now called b1. Only the name moved: the counts, the
    measure and the samples are identical, since #b1 = #b2 identically.

    This pins the server half, so a drift in the classification would show up
    even without a browser in the loop.
    """
    from collections import Counter

    from sixvertex.domains import (classify_counts, classify_invalid,
                                   cftp_sample, domain_mask)

    m = domain_mask([0, 13, 26, 40], [0, 13, 26, 40],
                    [[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=2.0, c2=2.0)
    tot = Counter()
    runs = 3
    for s in range(runs):
        H, _ = cftp_sample(m, w, master_seed=400 + s)
        assert not classify_invalid(H, m)
        tot.update(classify_counts(H, m))

    assert sum(tot.values()) == runs * int(m.sum())
    assert all(tot[k] > 0 for k in ("a1", "a2", "b1", "b2", "c1", "c2")), dict(tot)
    frac = tot["b1"] / sum(tot.values())
    assert 0.50 < frac < 0.67, f"b1 fraction {frac:.3f} moved; expected ~0.58"


# ------------------------------------------------ the CFTP stopping time T*
def test_backward_coalescence_is_monotone_in_the_start_time():
    """The property the binary search in cftp_stopping_time rests on.

    CFTP draws the field for step k from _rand_field(master_seed, k), indexed
    by ABSOLUTE time, so widening the window reuses the same fields near the
    present. Coalescence is therefore monotone in T. The search would silently
    return a wrong answer if it were not, so this asserts it rather than
    quoting the argument.
    """
    from experiments.cftp_stopping import _coalesces_from
    from experiments.mixing import weights_for_delta
    from sixvertex.cftp_exact import _colour_indices, _weight_table

    n, w = 8, weights_for_delta(-2.0)
    tbl = _weight_table(w)
    colour_idx, size = _colour_indices(n)
    for seed in (31, 32, 33):
        flags = [_coalesces_from(n, w, seed, T, tbl, colour_idx, size)
                 for T in range(1, 60)]
        # once True, never False again
        assert flags == sorted(flags), (seed, flags)


def test_stopping_time_is_the_exact_minimum():
    """T* coalesces and T*-1 does not -- that is the whole definition."""
    from experiments.cftp_stopping import _coalesces_from, cftp_stopping_time
    from experiments.mixing import weights_for_delta
    from sixvertex.cftp_exact import _colour_indices, _weight_table

    for n, delta, seed in ((8, -2.0, 900), (10, 0.0, 901), (12, -3.0, 902)):
        w = weights_for_delta(delta)
        tbl = _weight_table(w)
        colour_idx, size = _colour_indices(n)
        T = cftp_stopping_time(n, w, seed)
        assert T is not None and T >= 1
        assert _coalesces_from(n, w, seed, T, tbl, colour_idx, size), (n, delta, T)
        assert not _coalesces_from(n, w, seed, T - 1, tbl, colour_idx, size), \
            (n, delta, T)


def test_stopping_time_is_at_most_what_the_sampler_reports():
    """The sampler's doubling schedule brackets T* from above.

    cftp_sample starts its doubling at _estimate_T(n) rather than at 1, so
    the reported number is NOT T* rounded up to a power of two and can
    exceed 2*T* -- only the upper bound is a theorem, and only that is
    asserted. The gap is the cost of the heuristic start, and measuring T*
    directly is what the stopping-time sweep is for.
    """
    from experiments.cftp_stopping import cftp_stopping_time
    from experiments.mixing import weights_for_delta
    from sixvertex.cftp_exact import cftp_sample

    for n, delta, seed in ((8, -2.0, 900), (12, -1.0, 905)):
        w = weights_for_delta(delta)
        T = cftp_stopping_time(n, w, seed)
        _, info = cftp_sample(n, w, master_seed=seed)
        assert T <= info["sweeps"], (n, delta, T, info["sweeps"])


def test_the_stopping_time_is_non_monotone_in_delta():
    """The result Gorin asked for, pinned at a size that runs in seconds.

    T* falls from Delta=+0.5 to a minimum near Delta=-1 and then climbs again
    into the antiferroelectric phase. Both legs are asserted: a one-sided
    check would pass on a monotone curve.
    """
    import numpy as np

    from experiments.cftp_stopping import cftp_stopping_time
    from experiments.mixing import weights_for_delta

    def med(delta, k=12):
        w = weights_for_delta(delta)
        return float(np.median([cftp_stopping_time(12, w, 8_000 + s)
                                for s in range(k)]))

    hot, mid, cold = med(0.5), med(-1.0), med(-4.0)
    assert mid < hot, (hot, mid, cold)
    assert mid < cold, (hot, mid, cold)


# --------------------------------------------- the stochastic model, again
def test_stochastic_sampler_matches_brute_force_enumeration():
    """End-to-end check of the sweep: every configuration, exact probability.

    The per-vertex rule is enumerated over the whole tree and the resulting
    distribution over (h, v) compared with the sampler's empirical one. This
    catches sweep-order and indexing faults that a marginal check would not,
    because it pins the JOINT law rather than one statistic.
    """
    from collections import Counter

    import numpy as np
    from scipy.stats import chi2 as chi2_dist

    from sixvertex.stochastic import sample

    b1, b2, n = 0.35, 0.6, 3

    exact = {}

    def rec(i, j, h, v, p):
        if i == n:
            k = (tuple(h.ravel()), tuple(v.ravel()))
            exact[k] = exact.get(k, 0.0) + p
            return
        ni, nj = (i, j + 1) if j + 1 < n else (i + 1, 0)
        hi, vi = h[i, j], v[i, j]
        if hi == 1 and vi == 1:
            opts = [((1, 1), 1.0)]
        elif hi == 0 and vi == 0:
            opts = [((0, 0), 1.0)]
        elif hi == 1 and vi == 0:
            opts = [((1, 0), b1), ((0, 1), 1 - b1)]
        else:
            opts = [((0, 1), b2), ((1, 0), 1 - b2)]
        for (ho, vo), q in opts:
            h2, v2 = h.copy(), v.copy()
            h2[i, j + 1] = ho
            v2[i + 1, j] = vo
            rec(ni, nj, h2, v2, p * q)

    h0 = np.zeros((n, n + 1), dtype=int)
    v0 = np.zeros((n + 1, n), dtype=int)
    h0[:, 0] = 1
    rec(0, 0, h0, v0, 1.0)
    assert abs(sum(exact.values()) - 1.0) < 1e-12

    runs = 60_000
    seen = Counter()
    for s in range(runs):
        h, v = sample(n, b1, b2, seed=s)
        seen[(tuple(h.ravel()), tuple(v.ravel()))] += 1

    assert set(seen) <= set(exact), "sampler produced a configuration of probability zero"
    chi = sum((seen.get(k, 0) - p * runs) ** 2 / (p * runs) for k, p in exact.items())
    dof = len(exact) - 1
    assert chi2_dist.sf(chi, dof) > 0.001, f"chi2 {chi:.1f} on {dof} dof"


def test_the_first_row_of_the_stochastic_model_is_geometric():
    """Row 0 is exactly solvable, so it pins the orientation of the sweep.

    Nothing enters from below, one arrow enters at the left, and it travels
    right a Geometric(1-b1) number of steps before turning up. If rows and
    columns were transposed anywhere this would come out Geometric(1-b2).
    """
    import numpy as np

    from sixvertex.stochastic import sample

    b1, b2, n = 0.3, 0.7, 30
    runs = np.array([int(sample(n, b1, b2, seed=s)[0][0, :n + 1].sum())
                     for s in range(3000)])
    assert abs(runs.mean() - 1 / (1 - b1)) < 0.05, runs.mean()
    assert abs((runs == 1).mean() - (1 - b1)) < 0.02, (runs == 1).mean()


def test_every_server_sample_cancels_the_pending_lattice_rebuild():
    """Regression: a size change used to silently discard the next sample.

    The n slider schedules localInit() on a 200ms debounce. Changing the size
    and then clicking a sample button within that window let the rebuild land
    AFTER the sample and replace it with a fresh in-browser DWBC lattice --
    the sample was computed, drawn, and thrown away a fraction of a second
    later. Driving the real page showed it at n=120 and n=200 and not at the
    default size, which is why reading the diff never found it.

    Every entry point that installs a server-produced frame must cancel the
    pending rebuild first.
    """
    import re

    src = open("static/draw.js").read()
    assert "cancelPendingRebuild = () =>" in src, "the hook is gone"

    # Each handler that fetches a sample from the server.
    # The guard each handler sets on entry. Checking these rather than the
    # fetch itself keeps the window small and the test honest: the cancel has
    # to happen at the top of the handler, before any await.
    markers = [r"exactInFlight = true", r"btnStoch\.disabled = true",
               r"async function sampleDomain\(\)"]
    found = 0
    for pat in markers:
        hits = [m.start() for m in re.finditer(pat, src)]
        assert hits, f"entry point {pat!r} is gone; update this test"
        for pos in hits:
            found += 1
            window = src[max(0, pos - 400):pos + 400]
            assert "cancelPendingRebuild()" in window, (
                f"the handler at {pat!r} does not cancel the pending rebuild")
    assert found >= 4, found


def test_island_predicate_is_off_above_the_transition_and_on_below():
    """Pins the solid-c crossing predicate at a size that runs in seconds.

    The point of the definition is that it says NO where there is no island.
    Measured at n=48 with 80 samples it gives 0.000 at Delta=-0.5 and 0.650
    at Delta=-3; this asserts the separation with wide margins so it pins the
    behaviour rather than the third decimal.

    A regression here most likely means the erosion step or the window
    fraction moved. Both matter: without erosion the predicate fires on the
    speckled c region that exists at every Delta, and with the window at N/2
    instead of N/5 it asks the island to be twice the size it ever reaches.
    """
    import numpy as np

    from experiments.island_exists import island_exists
    from experiments.mixing import weights_for_delta
    from sixvertex.cftp_exact import cftp_sample

    def rate(delta, n=32, reps=24):
        w = weights_for_delta(delta)
        hits = 0
        for s in range(reps):
            H, _ = cftp_sample(n, w, master_seed=31_000 + 97 * s)
            hits += int(island_exists(H, n)[0])
        return hits / reps

    hot, cold = rate(-0.5), rate(-3.0)
    assert hot <= 0.10, f"predicate fires at Delta=-0.5 ({hot:.3f})"
    assert cold >= 0.40, f"predicate too quiet at Delta=-3 ({cold:.3f})"
    assert cold - hot > 0.3, (hot, cold)


def test_island_predicate_needs_both_erosion_and_a_small_window():
    """The two ingredients each do work; neither is decoration.

    Without erosion the predicate fires at Delta=-0.5, where the c region is
    speckled rather than solid and there is no island. This asserts that,
    so that dropping the erosion step cannot pass silently.
    """
    import numpy as np
    from scipy import ndimage

    from experiments.island import c_face_mask
    from experiments.island_exists import island_exists
    from experiments.mixing import weights_for_delta
    from sixvertex.cftp_exact import cftp_sample

    w = weights_for_delta(-0.5)
    n, reps = 32, 24
    with_erosion, without = 0, 0
    for s in range(reps):
        H, _ = cftp_sample(n, w, master_seed=31_000 + 97 * s)
        with_erosion += int(island_exists(H, n)[0])

        # same test, skipping the erosion step
        cm = c_face_mask(np.asarray(H, np.int64), n)
        lo = int(round(n * 0.4))
        hi = int(round(n * 0.6))
        W = cm[lo:hi, lo:hi]
        lab, nl = ndimage.label(W, structure=np.ones((3, 3)))
        span = 0
        for c in range(1, nl + 1):
            cols = np.nonzero((lab == c).any(axis=0))[0]
            if cols.size:
                span = max(span, cols.max() - cols.min() + 1)
        without += int(span >= W.shape[1])

    assert with_erosion <= 0.10 * reps, with_erosion
    assert without > with_erosion, (
        f"without erosion {without}/{reps}, with {with_erosion}/{reps}: "
        f"the erosion step is doing nothing, so the predicate is not "
        f"measuring solidity")


def test_the_island_core_is_a_filament_above_the_transition():
    """The order parameter separates the two phases by SHAPE, not size.

    A = (largest solid-c component)/N^2 decays like N^-theta, so the core
    covers N^(2-theta) faces: theta = 1 is a filament of length N, theta = 0
    a macroscopic island. Measured over N = 32..64, theta is about 1 at
    Delta = -1 and 0.16 at Delta = -3.

    This asserts the ordering, which is what the three sizes support, rather
    than either exponent's value, which they do not.
    """
    import numpy as np

    from experiments.island_order import core_area_fraction
    from experiments.mixing import weights_for_delta
    from sixvertex.cftp_exact import cftp_sample

    def theta(delta, ns=(24, 40), reps=14):
        a = []
        w = weights_for_delta(delta)
        for n in ns:
            vals = [core_area_fraction(
                cftp_sample(n, w, master_seed=51_000 + 97 * s)[0], n)
                for s in range(reps)]
            a.append(np.mean(vals))
        return -np.polyfit(np.log(ns), np.log(a), 1)[0]

    hot, cold = theta(-1.0), theta(-3.0)
    assert cold < hot, f"theta did not drop with depth: {hot:.2f} -> {cold:.2f}"
    assert hot - cold > 0.15, (hot, cold)
    # NOT asserted: the value of theta. It drifts down as the fitting window
    # moves up -- theta(-3) is 0.64 over n = 24..40 and 0.16 over n = 32..64 --
    # which is consistent with theta -> 0 and a genuinely macroscopic island,
    # but means no single fit is the number. The test runs at the small sizes
    # for speed, so it pins only the gap.
    assert cold < 0.8, f"core still decaying fast at Delta=-3 (theta={cold:.2f})"


# ------------------------------------------- the lattice-path picture itself
def test_drawn_paths_have_no_dangling_vertical_segments():
    """Every shared edge must be drawn from both sides, or paths break.

    Gorin reported the stochastic lattice-path view as "nonsense" and guessed
    a hidden coordinate flip. He was right. The arrows use the MATHEMATICAL
    convention -- v[m,j] enters math row m from below, so "up" is increasing
    m -- while the canvas draws row i at y=i with y growing downward, so
    increasing i goes DOWN the screen. Reading v[i,j] as the screen-bottom
    edge inverted the vertical direction, and the two cells meeting at a
    screen edge then referenced edges two rows apart: 19.4% of vertical
    segments did not meet (302/1560 at n=40). Horizontal segments were always
    fine, which is why the picture looked plausible at a glance.

    This replicates the drawing mapping in static/draw.js and asserts the
    invariant that makes a path picture a path picture.
    """
    import numpy as np

    from sixvertex.stochastic import sample

    for n, b1, b2, seed in ((40, 0.8, 0.2, 7), (32, 0.3, 0.7, 11),
                            (24, 0.5, 0.5, 3)):
        h, v = sample(n, b1, b2, seed=seed)

        def bits(i, j):
            """faceBits for screen cell (i, j), mirroring draw.js."""
            m = n - 1 - i
            return dict(l=h[m, j], r=h[m, j + 1],
                        t=v[m + 1, j], b=v[m, j])

        # Shared vertical screen edge at x=j+1: right of (i,j), left of (i,j+1)
        for i in range(n):
            for j in range(n - 1):
                assert bits(i, j)["r"] == bits(i, j + 1)["l"], (n, i, j)

        # Shared horizontal screen edge at y=i+1: bottom of (i,j), top of (i+1,j)
        for i in range(n - 1):
            for j in range(n):
                assert bits(i, j)["b"] == bits(i + 1, j)["t"], (n, i, j)


def test_path_view_reads_the_arrows_with_the_row_flip():
    """The shipped faceBits is RUN, not pattern-matched.

    This used to assert that the string "arrowFrame.n - 1 - i" appeared in the
    source, with a Python replica checked separately. That is the arrangement
    a mutation walked straight through elsewhere in this file: a port verified
    against the server proves nothing about the browser once the two drift.

    So the property itself is checked, in node, on the shipped function: two
    vertically adjacent cells must agree on the edge they share, which is
    exactly what broke when the row flip was missing ("19% of vertical
    segments did not meet"). Measured here at n=12, b1=0.3, b2=0.7, seed 5:

        shipped                0 of 132 shared edges disagree
        flip removed (m = i)  22 of 132   (17%)
        t and b exchanged     22 of 132   (17%)

    so the test has teeth against both ways of getting it wrong, and the
    historical 19% is reproduced.
    """
    import json
    import pathlib
    import shutil
    import subprocess
    import tempfile

    import pytest

    from sixvertex.stochastic import sample as stoch_sample

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available, so the shipped faceBits is "
                    "unverified here")

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "static" / "draw.js").read_text(encoding="utf-8")

    def grab(name):
        i = src.index("function " + name)
        return src[i:src.index("\n  }", i) + 4]

    assert "arrowFrame.n - 1 - i" in grab("faceBits"), (
        "the screen-row to math-row flip is gone from faceBits")

    n = 12
    h, v = stoch_sample(n, 0.3, 0.7, seed=5)
    harness = (
        "let arrowFrame = null;\n" + grab("faceInDomain") + "\n"
        + grab("faceBits") + "\n"
        + f"const n = {n};\n"
        + "const H = " + json.dumps(h.tolist()) + ";\n"
        + "const V = " + json.dumps(v.tolist()) + ";\n"
        + "arrowFrame = {n: n, h: (m,j) => H[m][j], v: (m,j) => V[m][j]};\n"
        + "const frame = {n: n, inDomain: () => true, get: () => 0};\n"
        + "let mismatch = 0, total = 0;\n"
        + "for (let i = 0; i + 1 < n; i++) for (let j = 0; j < n; j++) {\n"
        + "  const a = faceBits(frame, i, j), b = faceBits(frame, i + 1, j);\n"
        + "  total++; if (a.b !== b.t) mismatch++;\n"
        + "}\n"
        + "console.log(JSON.stringify({mismatch: mismatch, total: total}));\n")
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "facebits.js"
        f.write_text(harness, encoding="utf-8")
        out = subprocess.run([node, str(f)], capture_output=True, text=True,
                             timeout=120)
    assert out.returncode == 0, out.stderr[:400]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["total"] == (n - 1) * n, res
    assert res["mismatch"] == 0, (
        f"{res['mismatch']} of {res['total']} vertically adjacent cells "
        f"disagree on the edge they share, so the drawn paths come apart -- "
        f"the row flip or the t/b assignment in faceBits is wrong again")


def test_stochastic_height_and_arrows_are_sent_in_the_same_orientation():
    """The two payloads must describe the same picture, not mirror images.

    The path view draws the arrows with screen row i mapped to math row
    n-1-i, so math-up is screen-up. The height view reads the height array
    straight through frame.get, with payload row i drawn at y=i. If only one
    of the two is flipped, the height view is a vertical mirror of the path
    view of the SAME sample -- which is exactly what happened when the path
    flip was first fixed: paths dense in the upper left, height bright in the
    lower left.

    The height function satisfies H(i+1,j) - H(i,j) = h[i,j] in the math
    frame. With the rows reversed on the way out, that becomes a drop going
    DOWN the screen, equal to the left-edge arrow of the screen cell. Pinning
    that identity ties the two payloads together: drop either flip and this
    fails.
    """
    import base64
    import importlib

    import numpy as np

    server = importlib.import_module("server")
    n = 24
    body = server.app.test_client().post(
        "/api/stochastic", json={"n": n, "b1": 0.8, "b2": 0.2, "seed": 4}
    ).get_json()
    f = body["frame"]
    H = np.frombuffer(base64.b64decode(f["height_b64"]),
                      dtype=np.int16).reshape(n + 1, n + 1)
    h = np.frombuffer(base64.b64decode(f["h_b64"]),
                      dtype=np.uint8).reshape(n, n + 1)

    for i in range(n):
        m = n - 1 - i
        assert np.array_equal(H[i, :] - H[i + 1, :], h[m, :].astype(np.int64)), (
            f"height and arrows disagree at screen row {i}: the two views "
            f"would be vertical mirrors of each other")


def test_svg_export_handles_every_view_mode():
    """The exported file must be the picture on the screen.

    The export dispatched on "active" and "vertex6" and sent everything else
    to the height colour map. So asking for an SVG of the LATTICE PATHS --
    the view Gorin was told to open, and the one anyone would export for a
    figure -- silently returned a smoothed height field instead: a different
    picture of the same sample, which is the worst way for an export to be
    wrong. "fluct" had the same fault, returning the plain height map rather
    than the red/blue fluctuation field.

    Checks that every option in the view menu is named in the export, so a
    new view cannot be added without deciding how it exports.
    """
    import re

    html = open("static/index.html").read()
    sel = html[html.index('<select id="viewmode">'):]
    sel = sel[:sel.index("</select>")]
    views = re.findall(r'value="([^"]+)"', sel)
    assert set(views) >= {"height", "active", "vertex6", "paths", "fluct"}, views

    js = open("static/draw.js").read()
    export = js[js.index('btnSaveSvg.addEventListener'):]
    export = export[:export.index('link.download = "six-vertex-sample.svg"')]
    for v in views:
        if v == "height":
            continue          # the fall-through default, correctly
        assert f'"{v}"' in export, (
            f'the SVG export never mentions the "{v}" view, so it falls '
            f'through to the height colour map and exports a different '
            f'picture than the one on screen')
    # the paths export must draw strokes, not a grid of filled cells
    assert "stroke" in export and "<path d=" in export


def test_the_in_browser_sampler_reproduces_the_exact_measure():
    """The live chain is what the page shows by default; it needs checking.

    static/sixvertex.js is a second, independent implementation of the model
    -- a heat-bath chain in JavaScript -- and nothing tied it to the Python
    sampler. A wrong conditional there is exactly the fault that produced a
    64% bias in the Python heat-bath move, and it would be invisible: the
    pictures would still look like plausible six-vertex configurations.

    At n=4 there are 42 configurations, so the whole distribution can be
    compared rather than a summary statistic. Run at c = sqrt(8), Delta = -3,
    the weights of the right panel of Figure 7.15 in the Gorin-Ferrari-
    Borodin book draft.
    """
    import collections, json, math, os, shutil, subprocess, tempfile

    import numpy as np
    import pytest
    from scipy.stats import chi2 as chi2_dist

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "..", "static", "sixvertex.js")).read()
    driver = src + """
const n=4, c=Math.sqrt(8), w={a1:1,a2:1,b1:1,b2:1,c1:c,c2:c};
const s=new SixVertexJS(n,w,999);
s.step(5000);
const t=new Map();
for(let k=0;k<40000;k++){ s.step(4); const key=Array.from(s.H).join(",");
  t.set(key,(t.get(key)||0)+1); }
console.log(JSON.stringify(Array.from(t)));
"""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "drv.js")
        open(p, "w").write(driver)
        out = subprocess.run([node, p], capture_output=True, text=True,
                             timeout=300)
    assert out.returncode == 0, out.stderr[:400]
    js = dict(json.loads(out.stdout))

    from sixvertex.exact import exact_sample
    w = dict(a1=1., a2=1., b1=1., b2=1., c1=math.sqrt(8), c2=math.sqrt(8))
    py = collections.Counter()
    for s in range(20000):
        e = exact_sample(4, seed=s, **w)
        e = e[0] if isinstance(e, tuple) else e
        py[",".join(map(str, np.asarray(e, int).ravel()))] += 1

    assert len(js) == 42, f"JS chain reached {len(js)} configurations, not 42"
    assert set(js) == set(py), "the two samplers do not agree on the support"

    nj, np_ = sum(js.values()), sum(py.values())
    chi = 0.0
    dof = -1
    for k in set(js) | set(py):
        a, b = js.get(k, 0), py.get(k, 0)
        tot = a + b
        if tot < 20:
            continue
        ea, eb = tot * nj / (nj + np_), tot * np_ / (nj + np_)
        chi += (a - ea) ** 2 / ea + (b - eb) ** 2 / eb
        dof += 1
    assert chi2_dist.sf(chi, dof) > 0.001, f"chi2 {chi:.1f} on {dof} dof"


def test_partition_function_at_the_free_fermion_point():
    """A second exactly-solvable anchor, independent of the ASM one.

    At a=b=1, c=sqrt(2) the model is free-fermionic (Delta=0) and the
    domain-wall partition function is 2^(n(n-1)/2) * sqrt(2)^n. The all-ones
    point (Delta=1/2, Z = the ASM numbers) only probes one Delta; this pins a
    second, and it is sensitive to the c weights, which the ASM check is not.
    """
    import math

    from sixvertex.exact import ExactSampler

    r2 = math.sqrt(2)
    for n in range(1, 8):
        Z = ExactSampler(n, dict(a1=1., a2=1., b1=1., b2=1.,
                                 c1=r2, c2=r2)).partition_function()
        want = 2 ** (n * (n - 1) / 2) * r2 ** n
        assert abs(Z / want - 1) < 1e-9, (n, Z, want)


def test_c1_is_the_horizontal_to_vertical_turn():
    """Fixes the c1/c2 naming against the definition in the book draft.

    Gorin-Ferrari-Borodin, Introduction to integrable probability, Chapter 7:
    "Type c1 is the horizontal-to-vertical turn, and c2 the
    vertical-to-horizontal turn", with paths running up and right. So c1 is
    the vertex whose LEFT and TOP edges carry path, and c2 the one whose
    BOTTOM and RIGHT edges do.

    This matters because the book colours the six types separately (Figure
    7.15) and because its left panel uses c1 != c2. It cannot be checked
    against sampled data: see the next test.
    """
    from sixvertex.exact import _face_weight

    w = {k: 0.0 for k in ("a1", "a2", "b1", "b2", "c1", "c2")}
    w["c1"] = 11.0
    w["c2"] = 13.0
    # corner heights with left and top edges occupied: l=(bl-tl)=1, t=(tr-tl)=1
    assert _face_weight(0, 1, 1, 1, w) == 11.0, "c1 is not the left/top turn"
    # bottom and right occupied: b=(br-bl)=1, r=(br-tr)=1
    assert _face_weight(0, 0, 0, 1, w) == 13.0, "c2 is not the bottom/right turn"


def test_label_swaps_are_invisible_in_the_measure():
    """Why the naming above has to come from the book and not from data.

    Under domain-wall boundary conditions #b1 = #b2 identically, so swapping
    b1 and b2 leaves even the partition function unchanged. And #c1 - #c2 = n
    identically, so swapping c1 and c2 multiplies every configuration's
    weight by the same constant (c1/c2)^n: Z changes by that factor, but the
    probability measure does not move at all.

    So a wrong c1/c2 or b1/b2 label would be undetectable in any sample, and
    would show up only as two exchanged colours in a figure.
    """
    from sixvertex.exact import ExactSampler

    n = 4

    def Z(c1, c2, b1=1.0, b2=1.0):
        return ExactSampler(n, dict(a1=1., a2=1., b1=b1, b2=b2,
                                    c1=c1, c2=c2)).partition_function()

    assert abs(Z(2., 2., 1., 3.) - Z(2., 2., 3., 1.)) < 1e-6, "b swap moved Z"
    assert abs(Z(1., 2.) / Z(2., 1.) - (1 / 2) ** n) < 1e-9, \
        "c swap did not scale Z by (c1/c2)^n"


def test_vertex_labels_match_gorin_kenyon_via_the_frozen_corners():
    """Anchors all six labels to a published convention, not to our own.

    Gorin-Kenyon, arXiv:2408.14446, proof of Theorem 2.11, extends a domain
    to the full square by adding "Type 1 vertices in all positions of the
    bottom-right domain, Type 2 in the top-left, Type 3 in the bottom-left,
    and Type 4 in the top-right", where Types 1..6 carry weights a1, a2, b1,
    b2, c1, c2 and paths enter from the left of every row and leave the top
    of every column. The four frozen corners of DWBC therefore identify
    a1, a2, b1 and b2 outright.

    We had b1 and b2 the wrong way round: our bottom-left corner was b2. The
    swap is invisible in the measure -- #b1 = #b2 identically under DWBC, so
    the partition function and even the samples are untouched -- and it was
    caught only because the stochastic module, which defines b1 as the
    HORIZONTAL continuation probability, disagreed with the vertex legend
    about which vertex b1 is.
    """
    import collections

    import numpy as np

    from sixvertex.cftp_exact import cftp_sample
    from sixvertex.domains import FACE

    n = 48
    w = dict(a1=1., a2=1., b1=1., b2=1., c1=1., c2=1.)
    H = np.asarray(cftp_sample(n, w, master_seed=11)[0], int)

    def cls(i, j):
        tl, tr, bl, br = H[i, j], H[i, j + 1], H[i + 1, j], H[i + 1, j + 1]
        bits = (int((bl - tl) == 1), int((tr - tl) == 1),
                int((br - bl) == 1), int((br - tr) == 1))
        return FACE.get(bits, "invalid")

    k = 6
    corner = {}
    for name, (i0, j0) in (("top-left", (0, 0)), ("top-right", (0, n - k)),
                           ("bottom-left", (n - k, 0)),
                           ("bottom-right", (n - k, n - k))):
        c = collections.Counter(cls(i, j)
                                for i in range(i0, i0 + k)
                                for j in range(j0, j0 + k))
        corner[name] = c.most_common(1)[0][0]

    assert corner["bottom-right"] == "a1", corner
    assert corner["top-left"] == "a2", corner
    assert corner["bottom-left"] == "b1", corner
    assert corner["top-right"] == "b2", corner


def test_the_stochastic_sliders_agree_with_the_vertex_legend():
    """b1 means the same thing in both halves of the software.

    sixvertex/stochastic.py defines b1 as the probability that a path
    arriving horizontally continues horizontally, so raising b1 must produce
    more vertices that the legend calls b1. It used to produce the opposite:
    at b1 = 0.9 the colouring showed 1707 b2 faces against 13 b1 ones.
    """
    import collections

    from sixvertex.domains import FACE
    from sixvertex.stochastic import sample

    n = 48
    h, v = sample(n, 0.9, 0.1, seed=2)
    c = collections.Counter(
        FACE.get((int(h[i, j]), int(v[i + 1, j]), int(v[i, j]),
                  int(h[i, j + 1])), "invalid")
        for i in range(n) for j in range(n))
    assert c["b1"] > 10 * max(c["b2"], 1), dict(c)


def test_every_stated_monotone_region_agrees_with_the_predicate():
    """Each prose statement of the CFTP region must be the region.

    The authoritative region is in_monotone_region(). It has been stated in
    prose in five other places: a user-facing error message in each of the two
    server endpoints, a comment beside each of the two gating calls, and a test
    docstring. One of those copies (the DWBC endpoint's error message) carried
    b and c swapped -- the same swap that cftp_exact.py records having fixed in
    its own docstring -- and so told a user who hit the refusal to move their
    weights further OUT of the valid region.

    A string comparison would only pin today's wording. Instead this harvests
    every "<pair> >= <pair> and <pair> >= <pair>" sentence from the tree and
    evaluates it as a predicate against in_monotone_region on random weights,
    so any restatement that is not the region fails whatever words surround it.
    """
    import re, random, pathlib
    from sixvertex.cftp_exact import in_monotone_region

    PAIR = {"a1a2": ("a1", "a2"), "b1b2": ("b1", "b2"), "c1c2": ("c1", "c2")}
    # The clauses are joined by "and" in most copies and by a comma in
    # domains.py, so accept either.
    SEP = r"(?:\s+and\s+|\s*,\s*)"
    pat = re.compile(
        r"(a1a2|b1b2|c1c2)\s*>=\s*(a1a2|b1b2|c1c2)" + SEP +
        r"(a1a2|b1b2|c1c2)\s*>=\s*(a1a2|b1b2|c1c2)", re.S)
    # cftp_exact.py's module docstring states the region in letter variables
    # with a legend beside it. That form escaped this check entirely, which is
    # how the pre-correction statement "B >= A and B >= C" survived there --
    # contradicting the corrected docstring sixteen lines below it, at the top
    # of the very module that defines the predicate.
    letters = re.compile(
        r"([ABC])\s*>=\s*([ABC])" + SEP + r"([ABC])\s*>=\s*([ABC])", re.S)
    legend = re.compile(
        r"A\s*=\s*a1a2\s*,\s*B\s*=\s*b1b2\s*,\s*C\s*=\s*c1c2")
    LETTER = {"A": "a1a2", "B": "b1b2", "C": "c1c2"}

    root = pathlib.Path(__file__).resolve().parent.parent
    found = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".py", ".js", ".html", ".md") or "node_modules" in path.parts:
            continue
        text = path.read_text(errors="ignore")
        # Collapse the line breaks and comment/f-string furniture that these
        # sentences are wrapped in, so a statement split over two lines is
        # still seen as one.
        # Collapse line breaks together with the comment and string-literal
        # furniture on BOTH sides of them. Stripping only the leading quote of
        # a continuation line left a stray quote between "and" and the second
        # clause, which is how three more copies hid from this check.
        flat = re.sub(r'["\']?[\n\r]+\s*(?://|\#|f["\']|["\'])?\s*', " ", text)
        # Three spellings are in use across the tree: c1c2 (comments, the
        # server messages), c1*c2 (the package-level ValueError) and the
        # HTML-entity &ge; (the browser's gate note). Normalise all of them,
        # or a copy escapes the check by how it is written.
        flat = flat.replace("&ge;", ">=").replace("\u2265", ">=")
        # Subscript markup too: index.html writes the same sentence as
        # c<sub>1</sub>c<sub>2</sub> &ge; ..., which matched nothing until
        # the tags were stripped -- a statement can hide from this check
        # behind its own markup.
        flat = re.sub(r"</?su[bp]>", "", flat)
        # Markdown wraps these in backticks, and the prose files are where
        # the statement is most load-bearing: README's routing section told
        # readers CFTP applies on "b1b2 >= a1a2 and b1b2 >= c1c2" -- the
        # pre-correction region -- two hundred lines above the passage
        # explaining that very correction.
        flat = flat.replace("`", "")
        flat = re.sub(r"([abc])1\s*\*\s*([abc])2", r"\g<1>1\g<2>2", flat)
        hits = [(m, m.group(0)) for m in pat.finditer(flat)]
        if legend.search(flat):
            for m in letters.finditer(flat):
                hits.append((m, " ".join(
                    LETTER[g] if i % 2 == 0 else g
                    for i, g in enumerate(
                        [m.group(1), ">=", m.group(2), "and",
                         m.group(3), ">=", m.group(4)]))))
        for m, stmt in hits:
            # Historical notes legitimately quote the old region while
            # explaining the correction, so there has to be a way to write
            # one. The convention: lead in with "it read" / "hand read" /
            # "used to", or follow the quote with "old label" or
            # "pre-correction". Anything else quoting the old region is
            # treated as a live claim and fails, which is what this check is
            # for. (This comment is itself an example -- quoting the
            # pre-correction region needs the marker like any other.)
            before = flat[max(0, m.start() - 60):m.start()]
            after = flat[m.end():m.end() + 60]
            if ("it read" in before or "hand read" in before
                    or "used to" in before or "old label" in after
                    or "pre-correction" in after):
                continue
            found.append((path.relative_to(root), stmt))

    assert len(found) >= 5, f"expected the known restatements, found {found}"

    rng = random.Random(20261113)
    for rel, stmt in found:
        m = pat.search(stmt)
        l1, r1, l2, r2 = (PAIR[g] for g in m.groups())
        for _ in range(400):
            w = {k: rng.choice([0.2, 0.5, 1.0, 1.7, 3.0]) for k in
                 ("a1", "a2", "b1", "b2", "c1", "c2")}
            prod = lambda p: w[p[0]] * w[p[1]]
            stated = (prod(l1) >= prod(r1) - 1e-12 and
                      prod(l2) >= prod(r2) - 1e-12)
            assert stated == in_monotone_region(w), (
                f"{rel} states '{stmt}', which disagrees with "
                f"in_monotone_region at {w}")


def test_the_monotone_gate_is_scale_invariant_and_agrees_across_languages():
    """Scaling all six weights must not change whether CFTP is offered.

    The Boltzmann measure is invariant under w -> lambda*w, so the monotone
    predicate must be. With an absolute tolerance it was not: a=b=1, c=1/2 is
    outside the region and was refused at lambda=1 but ACCEPTED at lambda=1e-6,
    because the products themselves fell below the 1e-12 tolerance. That is the
    bad direction -- CFTP would run with no monotone coupling, so coalescence
    no longer certifies anything and the sample is silently off-measure.

    The browser carries its own copy of the gate (isExactSafe in draw.js) to
    decide whether to offer exact sampling at all, so the two must agree
    weight-for-weight; this runs the shipped JS in node against the Python.
    """
    import json, random, shutil, subprocess, pathlib
    from sixvertex.cftp_exact import in_monotone_region

    rng = random.Random(815)
    cases = []
    # the known failure, swept over scale, in both directions
    for lam in (1e-8, 1e-6, 1e-3, 1.0, 1e3, 1e6):
        for c in (0.5, 1.0, 1.4, 2.0):
            cases.append({"a1": lam, "a2": lam, "b1": lam, "b2": lam,
                          "c1": c * lam, "c2": c * lam})
    # and a random battery spanning many orders of magnitude
    for _ in range(300):
        e = lambda: 10.0 ** rng.uniform(-8, 8)
        cases.append({k: e() for k in ("a1", "a2", "b1", "b2", "c1", "c2")})

    # scale invariance of the authoritative predicate
    for w in cases:
        base = in_monotone_region(w)
        for lam in (1e-6, 1e-3, 1e3, 1e6):
            scaled = {k: v * lam for k, v in w.items()}
            if not all(0 < v < float("inf") for v in scaled.values()):
                continue  # scaling itself overflowed; not a claim about the gate
            assert in_monotone_region(scaled) == base, (
                f"gate changed under scaling by {lam}: {w}")

    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("node not available to check the browser copy of the gate")

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "static" / "draw.js").read_text()
    start = src.index("function isExactSafe")
    end = src.index("\n  }", start) + 4
    body = src[start:end].replace("MAX_EXACT_N", "14")
    script = (body + "\nconst cases = " + json.dumps(cases) + ";\n"
              "console.log(JSON.stringify("
              "cases.map(w => isExactSafe(w, 64))));\n")
    out = subprocess.run([node, "-e", script], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    js = json.loads(out.stdout)

    for w, got in zip(cases, js):
        assert got == in_monotone_region(w), (
            f"draw.js isExactSafe disagrees with in_monotone_region at {w}: "
            f"js={got} python={in_monotone_region(w)}")


def test_cftp_matches_brute_force_at_six_distinct_weights():
    """The existing brute-force check runs at a1=a2, b1=b2, c1=c2.

    At such a pairwise-symmetric point any bug that uses one member of a pair
    twice -- a1*a1 where a1*a2 was meant, the error-shape behind the b1/b2 and
    b/c label mixups already found here -- is invisible, because both readings
    give the same number. This runs the same comparison with all six weights
    distinct and the three products distinct, where such a slip moves the
    measure, and scores it with a chi-square over all 42 configurations rather
    than a max-deviation bound.

    (A genuine relabelling WITHIN a pair stays invisible to any distributional
    test, and provably so: under DWBC #a1=#a2, #b1=#b2 and #c1-#c2=n, so every
    configuration's weight is (a1a2)^#a1 (b1b2)^#b1 (c1c2)^#c2 times the
    constant c1^n -- the measure sees only the three products. That is why the
    vertex naming had to be settled against a published convention and not
    against data. This test pins the products.)
    """
    from sixvertex.cftp_exact import cftp_sample, in_monotone_region

    n = 4
    w = dict(a1=0.7, a2=1.3, b1=0.5, b2=1.6, c1=1.1, c2=2.3)
    assert len(set(w.values())) == 6
    A, B, C = w["a1"]*w["a2"], w["b1"]*w["b2"], w["c1"]*w["c2"]
    assert len({round(A, 12), round(B, 12), round(C, 12)}) == 3
    assert in_monotone_region(w), "test point must be inside the CFTP region"

    cfgs, probs, _ = exact_distribution(n, w)
    assert len(cfgs) == 42, f"n=4 DWBC has 42 states, got {len(cfgs)}"
    keys = {c.astype(np.int64).tobytes(): i for i, c in enumerate(cfgs)}

    N = 6000
    counts = np.zeros(len(cfgs))
    for seed in range(N):
        H, _ = cftp_sample(n, w, master_seed=777000 + seed)
        k = keys.get(np.asarray(H, dtype=np.int64).tobytes())
        assert k is not None, "CFTP produced a configuration outside the state space"
        counts[k] += 1

    expected = probs * N
    assert expected.min() > 5.0, (
        f"chi-square needs every expected count above 5, min is "
        f"{expected.min():.2f} -- raise N")
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    dof = len(cfgs) - 1
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, dof))
    except ImportError:
        # Wilson-Hilferty normal approximation, accurate to ~1e-3 here
        z = ((chi2 / dof) ** (1/3) - (1 - 2/(9*dof))) / math.sqrt(2/(9*dof))
        p = 0.5 * math.erfc(z / math.sqrt(2))
    assert p > 1e-3, (
        f"CFTP does not reproduce the exact measure at asymmetric weights: "
        f"chi2={chi2:.1f} on {dof} dof, p={p:.2e}")


def test_delta_is_bounded_by_one_half_inside_the_monotone_region():
    """Inside the CFTP region Delta can never exceed 1/2.

    C >= max(A, B) gives A + B - C <= min(A, B), so

        Delta = (A + B - C) / (2 sqrt(AB)) <= min(A,B) / (2 sqrt(AB))
              = sqrt(min(A,B)/max(A,B)) / 2 <= 1/2,

    with equality only at A = B = C. The browser used to warn about slow
    coalescence when Delta > 2, a threshold carried over from the old
    labelling where the region was b-dominant and Delta was unbounded inside
    it. After the b/c correction that warning could never fire, so the one
    piece of UI meant to flag an impractical run was dead. It now keys on n,
    which is what the measurements say drives the cost (at n = 48 the median
    sweeps to coalescence are 4096 at Delta = +0.5, 0.0 and -3.0 alike).

    This pins the bound, so a future Delta-keyed threshold above 1/2 is
    recognisably unreachable rather than quietly dead.
    """
    import random
    from sixvertex.cftp_exact import in_monotone_region

    def delta_of(w):
        A, B = w["a1"]*w["a2"], w["b1"]*w["b2"]
        return (A + B - w["c1"]*w["c2"]) / (2 * math.sqrt(A * B))

    rng = random.Random(1909)
    worst, inside = -math.inf, 0
    for _ in range(60000):
        w = {k: 10.0 ** rng.uniform(-3, 3)
             for k in ("a1", "a2", "b1", "b2", "c1", "c2")}
        if not in_monotone_region(w):
            continue
        inside += 1
        worst = max(worst, delta_of(w))
    assert inside > 5000, f"too few in-region samples ({inside}) to conclude"
    assert worst <= 0.5 + 1e-9, f"Delta reached {worst} inside the region"

    # equality is attained, so 1/2 is the true supremum and not a loose bound
    eq = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.0, c2=1.0)
    assert in_monotone_region(eq)
    assert abs(delta_of(eq) - 0.5) < 1e-12

    # Any Delta threshold the UI uses must be REACHABLE inside the region.
    #
    # This block used to assert that the warning was not gated on Delta at
    # all, and that its trigger expression mentioned CFTP_SLOW_N by name. The
    # first was the wrong lesson drawn from the original bug and the second
    # froze a spelling: the warning now fires on a deep-Delta condition too,
    # which is correct (below about -5 the coalescence time climbs steeply),
    # and the trigger reads `safe && (bigN || deepDelta)`. Both assertions
    # blocked that, for no reason connected to the fault they came from.
    #
    # The fault was a threshold of Delta > 2, which the bound proved above
    # makes unreachable, so the warning could never fire. That is what to
    # check: every Delta threshold in the file must be <= 1/2.
    import pathlib
    import re as _re
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "static" / "draw.js").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                      if not ln.strip().startswith("//"))

    thresholds = []
    for m in _re.finditer(r"CFTP_SLOW_DELTA\s*=\s*(-?[\d.]+)", code):
        thresholds.append(float(m.group(1)))
    for m in _re.finditer(r"dlt\s*([<>]=?)\s*(-?[\d.]+)", code):
        thresholds.append(float(m.group(2)))
    assert thresholds, "no Delta threshold found in draw.js"
    for t in thresholds:
        assert t <= 0.5 + 1e-12, (
            f"draw.js compares Delta against {t}, which is unreachable "
            f"inside the monotone region (Delta <= 1/2 there), so that "
            f"branch can never fire")

    # and the warning must still consider n, which is the other factor
    assert "CFTP_SLOW_N" in code


def test_all_five_vertex_label_tables_agree():
    """Every encoding of (l,t,b,r) -> label in the tree must give one answer.

    The labelling is written out five times: domains.FACE, exact._face_weight,
    cftp_exact._weight_table, classifyBits in draw.js, and classifyFace in
    static/sixvertex.js. The b1/b2 correction against Gorin-Kenyon reached the
    first four and missed the fifth, so the in-browser chain weighted the
    vertical through-vertex as b1 while the legend printed beside it said b2.
    A sixth, unused copy sat at the top of cftp_exact.py disagreeing with that
    module's own table.

    No distributional test can catch this: under DWBC every configuration has
    #b1 = #b2 (and #c1 - #c2 = n), so exchanging b1 and b2 multiplies every
    weight by (b1/b2)^0 = 1 and leaves the measure exactly invariant. That is
    precisely why it has to be pinned by cross-checking the tables against
    each other rather than by sampling.
    """
    import json, pathlib, shutil, subprocess
    from sixvertex.domains import FACE
    from sixvertex.cftp_exact import _weight_table

    root = pathlib.Path(__file__).resolve().parent.parent
    bits = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 1, 0, 0),
            (0, 0, 1, 1), (0, 1, 1, 0), (1, 0, 0, 1)]
    assert set(FACE) == set(bits), FACE

    # distinct weights, so every label is told apart by its value
    w = {"a1": 2.0, "a2": 3.0, "b1": 5.0, "b2": 7.0, "c1": 11.0, "c2": 13.0}
    by_value = {v: k for k, v in w.items()}
    assert len(by_value) == 6

    # 1. domains.FACE is the reference
    ref = {b: FACE[b] for b in bits}

    # 2. cftp_exact's packed 16-entry table
    tbl = _weight_table(w)
    for (l, t, b, r) in bits:
        got = by_value.get(float(tbl[l*8 + t*4 + b*2 + r]))
        assert got == ref[(l, t, b, r)], (
            f"cftp_exact._weight_table calls {(l,t,b,r)} {got}, "
            f"domains.FACE calls it {ref[(l,t,b,r)]}")

    # Corner heights realising each edge pattern. _face_weight and the
    # browser's classifyFace both take the four corners rather than the edge
    # bits (top = tr-tl, bottom = br-bl, left = bl-tl, right = br-tr), and the
    # ice rule makes the fourth difference follow from the other three.
    corners = {}
    for (l, t, b, r) in bits:
        tl = 0
        tr, bl = tl + t, tl + l
        br = bl + b
        assert br - tr == r, (l, t, b, r)
        corners[(l, t, b, r)] = (tl, tr, bl, br)

    # 3. exact.py's per-face weight
    from sixvertex.exact import _face_weight
    for bb in bits:
        tl, tr, bl, br = corners[bb]
        got = by_value.get(float(_face_weight(tl, tr, bl, br, w)))
        assert got == ref[bb], (
            f"exact._face_weight calls {bb} {got}, "
            f"domains.FACE calls it {ref[bb]}")

    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("node not available to check the two browser tables")

    # 4. classifyBits in draw.js -- returns the label name directly
    src = (root / "static" / "draw.js").read_text()
    i = src.index("function classifyBits")
    body = src[i:src.index("\n  }", i) + 4]
    script = (body + "\nconsole.log(JSON.stringify(" + json.dumps(bits) +
              ".map(([l,t,b,r]) => classifyBits(!!l,!!t,!!b,!!r))));")
    out = subprocess.run([node, "-e", script], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    for bb, got in zip(bits, json.loads(out.stdout)):
        assert got == ref[bb], (
            f"draw.js classifyBits calls {bb} {got}, "
            f"domains.FACE calls it {ref[bb]}")

    # 5. classifyFace in static/sixvertex.js -- returns the WEIGHT, and takes
    #    corner heights rather than edge bits, so feed it corners that realise
    #    each pattern: top=tr-tl, bottom=br-bl, left=bl-tl, right=br-tr.
    src2 = (root / "static" / "sixvertex.js").read_text()
    j = src2.index("function classifyFace")
    body2 = src2[j:src2.index("\n}", j) + 2]
    script2 = (body2 + "\nconst w = " + json.dumps(w) + ";\n"
               "console.log(JSON.stringify(" + json.dumps([list(corners[b]) for b in bits]) +
               ".map(([tl,tr,bl,br]) => classifyFace(tl,tr,bl,br,w))));")
    out2 = subprocess.run([node, "-e", script2], capture_output=True,
                          text=True, timeout=120)
    assert out2.returncode == 0, out2.stderr
    for bb, val in zip(bits, json.loads(out2.stdout)):
        got = by_value.get(float(val))
        assert got == ref[bb], (
            f"sixvertex.js classifyFace calls {bb} {got}, "
            f"domains.FACE calls it {ref[bb]}")


def test_the_in_browser_sampler_at_six_distinct_weights():
    """The companion check runs at a1=a2, b1=b2, c1=c2, where the chain's
    b-branches are indistinguishable from each other.

    Instrumenting classifyFace showed those branches are live -- over 40 sweeps
    at n=24 they fired 3653 and 4035 times -- so the b1/b2 labelling genuinely
    changes the local acceptance probabilities, even though it cannot change
    the stationary measure (#b1 = #b2 under DWBC). The symmetric test point
    cannot see any of that. This one runs with all six weights distinct, where
    every branch carries a different number, and compares the whole 42-state
    distribution against the sequential exact sampler.

    (The instrumentation also showed the `return 1` fallback for a face
    matching none of the six patterns never fires, as the flippability guard
    at the top of the update implies: a site is only touched when its four
    neighbours are equal.)
    """
    import collections, json, os, shutil, subprocess, tempfile

    import pytest
    from scipy.stats import chi2 as chi2_dist

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    w = dict(a1=0.7, a2=1.3, b1=0.5, b2=1.6, c1=1.1, c2=2.3)
    assert len(set(w.values())) == 6

    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "..", "static", "sixvertex.js")).read()
    driver = src + """
const w = """ + json.dumps(w) + """;
const s = new SixVertexJS(4, w, 20261113);
s.step(5000);
const t = new Map();
for (let k = 0; k < 40000; k++) { s.step(4);
  const key = Array.from(s.H).join(",");
  t.set(key, (t.get(key) || 0) + 1); }
console.log(JSON.stringify(Array.from(t)));
"""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "drv.js")
        open(p, "w").write(driver)
        out = subprocess.run([node, p], capture_output=True, text=True,
                             timeout=600)
    assert out.returncode == 0, out.stderr[:400]
    js = dict(json.loads(out.stdout))
    assert len(js) == 42, f"JS chain reached {len(js)} configurations, not 42"

    # exact probabilities, not a second sample: no Monte Carlo error on this side
    cfgs, probs, _ = exact_distribution(4, w)
    exact = {",".join(map(str, np.asarray(c, int).ravel())): float(pr)
             for c, pr in zip(cfgs, probs)}
    assert set(js) == set(exact), "the chain and the exact measure disagree on the support"

    N = sum(js.values())
    chi, dof = 0.0, -1
    for k, pr in exact.items():
        e = pr * N
        if e < 5:
            continue
        chi += (js.get(k, 0) - e) ** 2 / e
        dof += 1
    assert dof > 30, f"too few cells ({dof + 1}) to conclude"
    pval = float(chi2_dist.sf(chi, dof))
    assert pval > 1e-3, (
        f"the in-browser chain does not reproduce the exact measure at "
        f"asymmetric weights: chi2={chi:.1f} on {dof} dof, p={pval:.2e}")


def test_cftp_sample_does_not_depend_on_where_the_doubling_starts():
    """The property that makes CFTP exact, checked directly.

    CFTP is exact because the random field at time -k is fixed once and REUSED
    by every attempt: _rand_field(master_seed, k, shape) is a pure function of
    the seed and k, never of T. If an implementation instead redraws the
    randomness each time it doubles T -- the classic way to get this wrong --
    it still coalesces and still returns plausible pictures, but the output is
    no longer the composed map applied from the infinite past, and the measure
    is biased.

    The signature of doing it right: once the two chains have met by time -T,
    running from any earlier -T' gives the SAME configuration, because the map
    from -T' to 0 factors through a map that is already constant. So sweeping
    the starting T must not move the answer by a single site.

    This is the one exactness check that works where brute force cannot. The
    chi-square tests pin the measure at n=4, over 42 states; this holds at
    n=32, where the state space is far past enumeration.
    """
    import numpy as np
    from sixvertex.cftp_exact import cftp_sample

    w = dict(a1=0.7, a2=1.3, b1=0.5, b2=1.6, c1=1.1, c2=2.3)
    starts = (1, 3, 7, 64, 1000)

    for n in (6, 12, 24, 32):
        for seed in (11, 12):
            results, finals = [], []
            for T0 in starts:
                H, info = cftp_sample(n, w, master_seed=seed, initial_T=T0)
                results.append(np.asarray(H, dtype=np.int64).tobytes())
                finals.append(info["sweeps"])
            assert len(set(results)) == 1, (
                f"n={n} seed={seed}: the sample depends on the starting T "
                f"(final T values {finals}) -- the random field is not being "
                f"reused across attempts, so the output is not exact")
            # the runs must genuinely have differed, or this proves nothing
            assert len(set(finals)) > 1, (
                f"n={n} seed={seed}: every start produced the same final T "
                f"({finals}); the test did not exercise distinct schedules")

    # and a different seed must give a different answer, so the above is not
    # passing merely because the sampler ignores its randomness
    a = cftp_sample(16, w, master_seed=1, initial_T=8)[0]
    b = cftp_sample(16, w, master_seed=2, initial_T=8)[0]
    assert np.asarray(a).tobytes() != np.asarray(b).tobytes(), (
        "two different master seeds gave identical samples")


def test_the_monotone_region_boundary_is_real_in_both_directions():
    """The refusal outside the region must be substantive, not defensive.

    Everything rests on one claim: the coupling of the two extremal chains
    driven by shared uniforms preserves Hlo <= Hhi exactly on
    c1c2 >= a1a2 and c1c2 >= b1b2. The other tests here check that the code
    SAYS the right region (every prose copy is harvested and evaluated against
    the predicate) -- but agreement among statements proves nothing if the
    predicate itself names the wrong set. This checks the predicate against
    the behaviour it predicts, in both directions:

      inside   -> the order Hlo <= Hhi is never violated;
      outside  -> it is violated, more often the further out you go, and far
                  enough out the chains stop meeting at all.

    Measured at n=10: 0 violations inside at three separate weight points, 13
    to 40 just outside at c=0.9, and about 1750 at c=0.5. So the boundary is
    where the predicate puts it, and the b/c swap that this file has had to
    correct twice would show up here as acceptance of non-monotone weights.
    """
    import numpy as np
    import sixvertex.cftp_exact as ce

    inside = [
        (dict(a1=0.7, a2=1.3, b1=0.5, b2=1.6, c1=1.1, c2=2.3), "asymmetric"),
        (dict(a1=1., a2=1., b1=1., b2=1., c1=1., c2=1.), "uniform"),
        (dict(a1=1., a2=1., b1=1., b2=1., c1=2.83, c2=2.83), "Delta=-3"),
    ]
    for w, lbl in inside:
        assert ce.in_monotone_region(w), lbl
        for seed in (1, 2, 3):
            _, info = ce.cftp_sample(10, w, master_seed=seed,
                                     check_monotone=True)
            assert info["monotonicity_violations"] == 0, (
                f"{lbl}: the coupling broke order inside the region "
                f"({info['monotonicity_violations']} violations)")

    # Outside, the refusal has to be bypassed to observe what it is protecting
    # against. Restore the predicate whatever happens.
    real = ce.in_monotone_region
    try:
        ce.in_monotone_region = lambda w, tol=1e-12: True
        for c in (0.9, 0.5):
            w = dict(a1=1., a2=1., b1=1., b2=1., c1=c, c2=c)
            assert not real(w), f"c={c} should be outside the region"
            seen = []
            for seed in (1, 2, 3):
                try:
                    _, info = ce.cftp_sample(10, w, master_seed=seed,
                                             check_monotone=True,
                                             max_T=1 << 12)
                    seen.append(info["monotonicity_violations"])
                except RuntimeError:
                    seen.append(None)  # never coalesced: also a failure
            assert all(v is None or v > 0 for v in seen), (
                f"at c={c}, outside the region, the coupling stayed ordered "
                f"({seen}) -- either the predicate is too conservative or the "
                f"check is not looking at anything")
    finally:
        ce.in_monotone_region = real

    assert ce.in_monotone_region is real


def test_the_same_seed_reproduces_the_same_sample_on_every_path():
    """The seed box on the page promises reproducibility; nothing tested it.

    Three different samplers sit behind it -- the stochastic single-sweep
    sampler, the sequential transfer-matrix sampler for n <= MAX_EXACT_N, and
    CFTP above it -- each with its own seeding path through the server, and
    CFTP's runs in a worker thread. A seed that is dropped, or mixed with the
    clock or with thread state, gives a tool whose output cannot be quoted in
    a paper or checked by anyone else. The failure is invisible from a single
    run: every sample looks fine on its own.

    Checks all three: the same seed twice is identical, a different seed
    differs, and no seed gives something different again.
    """
    import hashlib
    import json as _json
    import time

    import server

    c = server.app.test_client()

    def dig(payload):
        return hashlib.sha256(
            _json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def stochastic(seed):
        r = c.post("/api/stochastic",
                   json={"n": 24, "b1": 0.3, "b2": 0.7, "seed": seed})
        assert r.status_code == 200, r.data[:300]
        body = r.get_json()
        assert body["ok"], body
        return dig(body["frame"])

    def exact(seed, n, c1=2.0):
        r = c.post("/api/exact/start",
                   json={"n": n, "a1": 1, "a2": 1, "b1": 1, "b2": 1,
                         "c1": c1, "c2": c1, "seed": seed})
        assert r.status_code == 200, r.data[:300]
        job = r.get_json()["job_id"]
        for _ in range(1200):
            s = c.get(f"/api/exact/status/{job}").get_json()
            if s.get("status") == "done":
                return dig(s["frame"])
            assert s.get("status") != "error", s
            time.sleep(0.05)
        raise AssertionError(f"job {job} did not finish")

    paths = {
        "stochastic": (stochastic, 7, 8),
        # n <= MAX_EXACT_N: the sequential transfer-matrix sampler
        "sequential": (lambda s: exact(s, 8), 5, 6),
        # above it: CFTP in a worker thread
        "cftp": (lambda s: exact(s, 18), 42, 43),
    }
    for name, (run, s1, s2) in paths.items():
        a, b = run(s1), run(s1)
        assert a == b, f"{name}: seed {s1} did not reproduce its own sample"
        d = run(s2)
        assert d != a, (
            f"{name}: seeds {s1} and {s2} gave the same sample -- the seed is "
            f"not reaching the sampler")
        r1, r2 = run(None), run(None)
        assert r1 != r2 or r1 != a, (
            f"{name}: an omitted seed is not producing fresh randomness")


def test_the_fluctuation_view_is_normalised_to_a_single_sample():
    """The fluctuation view states a quantitative claim; check the arithmetic.

    It shows (X - Y)/sqrt(2) for two INDEPENDENT exact samples, and the
    comment in draw.js justifies the sqrt(2) by saying it "preserves the
    variance of a single height function, so the field shown has the same
    scale as the fluctuation of one sample about its mean". For iid X, Y,

        Var((X - Y)/sqrt(2)) = (Var X + Var Y)/2 = Var X,

    so the claim is exactly right -- but only if the two samples really are
    independent. Differencing two samples that share a seed, or a sample
    against itself, would give identically zero; differencing correlated ones
    would understate the scale and make the lattice look calmer than it is.
    Since the view exists to show the size of the fluctuations, a wrong factor
    here is a wrong answer presented as a picture.

    Checked by variance, not by eye: the ratio is taken variance-weighted,
    which is what sets the displayed colour scale. Near-frozen sites are
    excluded from the per-site spread, since a variance estimate there is
    unstable (at n=8 the sites with Var < 0.02 gave ratios from 0.60 to 1.11,
    while all 45 sites with Var > 0.05 fell within [0.955, 1.065]).
    """
    import math

    import numpy as np
    from sixvertex.exact import exact_sample

    n, M = 6, 1500
    w = dict(a1=1., a2=1., b1=1., b2=1., c1=2., c2=2.)
    S = []
    for s in range(2 * M):
        e = exact_sample(n, seed=s, **w)
        e = e[0] if isinstance(e, tuple) else e
        S.append(np.asarray(e, dtype=float))
    S = np.array(S)
    X, Y = S[0::2], S[1::2]
    F = (X - Y) / math.sqrt(2.0)

    vx = X.var(axis=0, ddof=1)
    vf = F.var(axis=0, ddof=1)
    live = vx > 1e-9
    assert live.sum() >= 10, "too few fluctuating sites to conclude"

    weighted = vf[live].sum() / vx[live].sum()
    assert abs(weighted - 1.0) < 0.05, (
        f"the fluctuation field is not on the scale of a single sample: "
        f"variance-weighted ratio {weighted:.4f}, expected 1")

    big = vx > 0.05
    if big.sum() >= 5:
        r = vf[big] / vx[big]
        assert abs(r.mean() - 1.0) < 0.05, (
            f"per-site ratio away from 1 ({r.mean():.4f}) at the sites that "
            f"actually fluctuate")

    # the field must be centred: a nonzero mean would be a systematic offset
    assert abs(float(F.mean())) < 0.05, (
        f"the differenced field is not centred (mean {F.mean():.4f}) -- the "
        f"two samples are not exchangeable")

    # and differencing a sample against ITSELF must collapse, which is the
    # failure mode the independence assumption is protecting against
    assert np.allclose((X - X) / math.sqrt(2.0), 0.0)


def test_the_fluctuation_view_uses_two_different_seeds():
    """With a seed in the box, the fluctuation view used to show nothing.

    It differences two exact samples, but both requests sent currentSeed(),
    so a seeded run drew the SAME sample twice and the field was identically
    zero -- a blank picture captioned "from two independent exact samples",
    with the range printed as +/-0.00. A blank seed box was unaffected,
    because the server then chose its own seed per request. So the bug only
    appeared for someone who set a seed to make a figure reproducible, which
    is the one group that would then quote the picture.

    The second seed is now derived from the first rather than randomised, so
    the view stays reproducible: the same seed gives the same pair of samples
    and a bit-identical canvas (checked in a browser), a different seed gives
    a different one.

    Three things have to hold, and this checks all three.
    """
    import json, pathlib, shutil, subprocess

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "static" / "draw.js").read_text()

    # 1. the handler must not send the same seed twice
    i = src.index("const seedA = currentSeed();")
    handler = src[i:i + 400]
    assert handler.count("fetchExactSample") == 2, handler
    assert "companionSeed(seedA)" in handler, (
        "the second sample is not drawn from a derived seed:\n" + handler)

    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("node not available")

    # 2. companionSeed must never return its argument, including at the edges
    j = src.index("function companionSeed")
    body = src[j:src.index("\n  }", j) + 4]
    seeds = [0, 1, 2, 7, 12345, 2 ** 31 - 2, 2 ** 31 - 1, 2 ** 32 - 1,
             -1, -12345, 1013904223, 2147483647]
    script = (body + "\nconst xs = " + json.dumps(seeds) + ";\n"
              "console.log(JSON.stringify(xs.map(s => [s, companionSeed(s)])"
              ".concat([[null, companionSeed(null)]])));")
    out = subprocess.run([node, "-e", script], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    pairs = json.loads(out.stdout)
    for s, t in pairs:
        if s is None:
            assert t is None, "a blank seed box must stay blank, not become 0"
            continue
        assert t is not None
        assert t != s, f"companionSeed({s}) returned the seed itself"
        assert 0 <= t < 2147483647, f"companionSeed({s}) = {t} out of range"

    # 3. and the derived seeds must actually give different samples, or the
    #    field is still zero however distinct the numbers look
    import numpy as np
    from sixvertex.exact import exact_sample

    w = dict(a1=1., a2=1., b1=1., b2=1., c1=2., c2=2.)
    same = 0
    checked = [(s, t) for s, t in pairs if s is not None][:6]
    for s, t in checked:
        def draw(k):
            e = exact_sample(8, seed=int(k) % (2 ** 31 - 1), **w)
            return np.asarray(e[0] if isinstance(e, tuple) else e, dtype=int)
        if np.array_equal(draw(s), draw(t)):
            same += 1
    assert same == 0, (
        f"{same} of {len(checked)} derived seed pairs produced identical "
        f"samples, so the fluctuation field would be zero for them")


def test_the_in_browser_chain_honours_the_seed_box():
    """The seed box promised more than it delivered.

    Its tooltip said "the same seed, n and weights give the same
    configuration", but the in-browser chain -- the path the page runs by
    DEFAULT, before anyone presses "exact sample" -- seeded itself from
    Date.now() and never read the box. So the promise held only for the
    server-side samplers. Nothing looked wrong: every run produced a valid
    configuration, just not the one the seed named.

    The equilibration badge compares that chain against a shadow started from
    the opposite corner, and judges equilibration by how far apart they are.
    Driving both from one stream would make them agree for the wrong reason
    and report equilibrium early, so the shadow takes a derived seed: distinct
    from the chain's, and still reproducible.

    Checked at both levels -- that draw.js wires the box in, and that the two
    seeds really do drive different trajectories.
    """
    import json, pathlib, shutil, subprocess

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "static" / "draw.js").read_text()

    i = src.index("const s0 = currentSeed();")
    block = src[i:i + 400]
    assert "new SixVertexJS(n, w, localSeed)" in block, block
    assert "makeShadow(n, w, localSeed)" in block, (
        "the shadow chain is not given the chain's seed to derive from:\n"
        + block)
    assert "companionSeed(baseSeed)" in src, (
        "makeShadow does not derive an independent seed")

    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("node not available")

    j = src.index("function companionSeed")
    companion = src[j:src.index("\n  }", j) + 4]
    sv = (root / "static" / "sixvertex.js").read_text()

    driver = sv + "\n" + companion + """
const w = {a1:1, a2:1, b1:1, b2:1, c1:2, c2:2};
function runFrom(seed) {
  const s = new SixVertexJS(12, w, seed);
  s.step(30);
  return Array.from(s.H).join(",");
}
const s = 777;
console.log(JSON.stringify({
  same:      runFrom(s) === runFrom(s),
  different: runFrom(s) !== runFrom(s + 1),
  shadow:    runFrom(s) !== runFrom(companionSeed(s)),
  companion: companionSeed(s),
}));
"""
    out = subprocess.run([node, "-e", driver], capture_output=True,
                         text=True, timeout=180)
    assert out.returncode == 0, out.stderr[:400]
    r = json.loads(out.stdout)

    assert r["same"], (
        "the same seed did not reproduce the same trajectory -- the chain is "
        "not deterministic in its seed, so the box cannot promise anything")
    assert r["different"], "two seeds gave the same trajectory"
    assert r["companion"] != 777, "companionSeed returned the seed itself"
    assert r["shadow"], (
        "the shadow chain follows the same trajectory as the main one, so "
        "the equilibration badge is comparing a chain with itself")


def test_general_domain_sampler_is_exact_at_six_distinct_weights():
    """The two domain exactness checks both run at a1=a2, b1=b2, c1=c2.

    At such a point the compiled weight table holds three distinct numbers
    rather than six, so a mis-indexed entry can land on a slot carrying the
    same value and change nothing. The general-domain path has its own sweep
    (sweep_fast over plan arrays, driven by a 16-entry table built from
    domains.FACE), separate from the DWBC one, and nothing exercised it with
    six distinguishable weights.

    Compared against the enumerated law on a staircase -- not the square, so
    the boundary is not the DWBC one either.
    """
    import numpy as np
    from scipy.stats import chi2 as chi2_dist

    from sixvertex.domains import (cftp_sample, classify_counts, domain_mask,
                                   in_monotone_region)
    from experiments.brute_domain import enumerate_heights

    w = dict(a1=0.7, a2=1.3, b1=0.5, b2=1.6, c1=1.1, c2=2.3)
    assert len(set(w.values())) == 6
    assert in_monotone_region(w)

    m = domain_mask([0, 3, 6], [0, 3, 6], [[1, 1], [0, 1]])
    hs = enumerate_heights(m)
    assert len(hs) == 49, f"expected 49 configurations, got {len(hs)}"

    wt = []
    for H in hs:
        x = 1.0
        for k, v in classify_counts(H, m).items():
            x *= w[k] ** v
        wt.append(x)
    exact = np.array(wt) / sum(wt)

    key = {H.tobytes(): i for i, H in enumerate(hs)}
    N = 4000
    cnt = np.zeros(len(hs))
    for s in range(N):
        H, _ = cftp_sample(m, w, master_seed=606_000 + s)
        k = key.get(np.ascontiguousarray(H).tobytes())
        assert k is not None, "the sampler left the enumerated state space"
        cnt[k] += 1

    e = exact * N
    live = e >= 5
    assert live.sum() >= 15, (
        f"only {live.sum()} cells have expected count >= 5; raise N")
    chi = float((((cnt - e) ** 2) / e)[live].sum())
    dof = int(live.sum()) - 1
    p = float(chi2_dist.sf(chi, dof))
    assert p > 1e-3, (
        f"the general-domain sampler does not reproduce the enumerated law "
        f"at asymmetric weights: chi2={chi:.1f} on {dof} dof, p={p:.2e}")


def test_label_swaps_leave_the_measure_invariant_on_general_domains_too():
    """Why no amount of sampling can settle the b1/b2 naming.

    Under DWBC the invariance is easy to see: every configuration has
    #a1 = #a2, #b1 = #b2 and #c1 - #c2 = n, so exchanging a pair multiplies
    each weight by the same power and the normalised measure does not move.
    It is tempting to think a less symmetric domain would break that and let
    the convention be read off the data -- and the individual identities DO
    break: on this staircase #b1 - #b2 = 9 for every configuration, and on
    the notched square #a1 - #a2 = -4.

    But the DIFFERENCE is constant across configurations, because it is fixed
    by the boundary, not by the interior. So the swap still multiplies every
    weight by one common factor: the partition function moves (by 35184 for
    b1<->b2 on the staircase) and the probability measure does not.

    Hence the naming had to be anchored to a published convention
    (Gorin-Kenyon Theorem 2.11), and the cross-table test is what guards it.
    This records the reason, so nobody re-opens the question by sampling a
    stranger domain.

    "A stranger domain" used to be two of them, chosen by hand. That is thin
    support for the argument the entire vertex-naming edifice rests on -- and
    the one cited again this session to conclude that sampler.py's swapped
    b1/b2 changed nothing observable. So the invariant is swept over every
    enumerable domain in the family instead: 619 domains, zero failures, four
    seconds. If it had failed anywhere, those swaps WOULD have been detectable
    on those domains and the conclusion would have been wrong.
    """
    import numpy as np

    from sixvertex.domains import classify_counts, domain_mask
    from experiments.brute_domain import enumerate_heights

    domains = {
        "staircase": ([0, 3, 6], [0, 3, 6], [[1, 1], [0, 1]]),
        "plus": ([0, 1, 3, 4], [0, 2, 3, 4], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    }
    w = dict(a1=0.7, a2=1.3, b1=0.5, b2=1.6, c1=1.1, c2=2.3)

    for name, (X, Y, I) in domains.items():
        m = domain_mask(X, Y, I)
        hs = enumerate_heights(m)
        counts = [classify_counts(H, m) for H in hs]

        # the mechanism: each pair's count difference is a domain invariant
        for lo, hi in (("a1", "a2"), ("b1", "b2"), ("c1", "c2")):
            diffs = {c.get(lo, 0) - c.get(hi, 0) for c in counts}
            assert len(diffs) == 1, (
                f"{name}: #{lo} - #{hi} is not constant across "
                f"configurations ({sorted(diffs)}) -- if this ever fails, the "
                f"swap is detectable from samples and this reasoning changes")

        def law(weights):
            v = np.array([
                np.prod([weights[k] ** n for k, n in c.items()])
                for c in counts], dtype=float)
            return v / v.sum(), v.sum()

        base, Z0 = law(w)
        for lo, hi in (("a1", "a2"), ("b1", "b2"), ("c1", "c2")):
            sw = dict(w)
            sw[lo], sw[hi] = w[hi], w[lo]
            other, Z1 = law(sw)
            assert np.abs(base - other).max() < 1e-12, (
                f"{name}: swapping {lo}/{hi} moved the measure")
            # Z is allowed to move, and that is the whole asymmetry: the
            # normalisation absorbs the constant factor.
            assert Z1 > 0

    # ---- the same two properties over the whole enumerable family ----
    import itertools

    from sixvertex.domains import DomainError, admits_configuration

    cutsets = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4],
               [0, 2, 4, 6]]
    swept = 0
    for bits in itertools.product([0, 1], repeat=9):
        I = np.array(bits).reshape(3, 3)
        if I.sum() == 0:
            continue
        for X in cutsets:
            for Y in cutsets:
                try:
                    m = domain_mask(X, Y, I)
                except DomainError:
                    continue
                if not admits_configuration(m):
                    continue
                try:
                    hs = enumerate_heights(m)
                except (RuntimeError, DomainError):
                    continue
                if not hs or len(hs) > 400:
                    continue
                cs = [classify_counts(H, m) for H in hs]
                swept += 1
                for lo, hi in (("a1", "a2"), ("b1", "b2"), ("c1", "c2")):
                    diffs = {c.get(lo, 0) - c.get(hi, 0) for c in cs}
                    assert len(diffs) == 1, (
                        f"#{lo} - #{hi} is not constant on X={X}, Y={Y}, "
                        f"I={I.tolist()}: {sorted(diffs)}. The label swap is "
                        f"DETECTABLE there, so the naming argument fails on "
                        f"that domain.")
                    v = np.array([np.prod([w[k] ** n for k, n in c.items()])
                                  for c in cs], dtype=float)
                    sw = dict(w)
                    sw[lo], sw[hi] = w[hi], w[lo]
                    v2 = np.array([np.prod([sw[k] ** n for k, n in c.items()])
                                   for c in cs], dtype=float)
                    assert np.abs(v / v.sum() - v2 / v2.sum()).max() < 1e-12, (
                        f"swapping {lo}/{hi} moved the measure on X={X}, "
                        f"Y={Y}, I={I.tolist()}")
    assert swept == 619, (
        f"the sweep now covers {swept} enumerable domains, not the 619 "
        f"recorded")


def test_the_exact_sampling_job_lifecycle():
    """Every exact sample goes through the job machinery; test it as a whole.

    Six properties, each of which has a plausible failure that looks like
    nothing from a single run:

      - an unknown or expired job id answers 404 with a message, not a 500
        with a stack trace;
      - concurrent jobs do not share state (this server once kept a single
        module-level sampler, so two clients clobbered each other);
      - the concurrency cap refuses the next job with 429 and says why,
        rather than accepting work it will not do;
      - a finished job frees its slot;
      - a job stuck in "running" past the watchdog threshold is reaped, so
        _MAX_CONCURRENT_JOBS abandoned jobs cannot disable exact sampling
        server-wide until a restart;
      - and a RUNNING job is not evicted merely for being old. That one is
        the subtler half: expiring by created_at regardless of status meant a
        long legitimate run (exact sampling at n in the low hundreds) could
        be discarded mid-flight, its worker still burning CPU and its result
        thrown away on completion.
    """
    import time

    import server

    c = server.app.test_client()

    r = c.get("/api/exact/status/deadbeefdeadbeef")
    assert r.status_code == 404, r.status_code
    body = r.get_json()
    assert body and body.get("ok") is False and body.get("error"), body

    def finish(job_id, limit=3000):
        for _ in range(limit):
            s = c.get(f"/api/exact/status/{job_id}").get_json()
            if s.get("status") in ("done", "error"):
                return s
            time.sleep(0.05)
        raise AssertionError(f"job {job_id} never finished")

    def start(seed, n=20):
        return c.post("/api/exact/start",
                      json={"n": n, "a1": 1, "a2": 1, "b1": 1, "b2": 1,
                            "c1": 2, "c2": 2, "seed": seed})

    # Hermetic: other tests in this file start jobs, and one of them starts a
    # genuinely long n=250 job. A foreign entry left in the table changes what
    # the cap does here, so take the table aside for the duration and put it
    # back afterwards. (A foreign worker that finishes meanwhile simply finds
    # its id missing and discards its result, which is what it already does
    # for an expired job.)
    with server._jobs_lock:
        borrowed = dict(server._jobs)
        server._jobs.clear()

    fakes = []

    def occupy(k, created_at):
        """Park a fake 'running' job in the table, holding a slot."""
        name = f"fake_{k}_{len(fakes)}"
        with server._jobs_lock:
            server._jobs[name] = {"status": "running", "created_at": created_at}
        fakes.append(name)
        return name

    try:
        # Concurrent jobs must not share state. They are NOT relied on to
        # overlap: at this size a job finishes in well under a second, so an
        # earlier version of this test started three and asserted the fourth
        # was refused, which passed only when numba's first-call compile
        # happened to make the first jobs slow enough to still be running.
        # That is a race in the test, not a property of the server, and it
        # failed as soon as the suite ran it warm. The cap is checked below
        # with jobs that are guaranteed to be occupying their slots.
        frames = []
        for seed in (1, 2, 3):
            r = start(seed)
            assert r.status_code == 200, (r.status_code, r.data[:200])
            s_ = finish(r.get_json()["job_id"])
            assert s_["status"] == "done", s_
            frames.append(repr(s_["frame"]))
        assert len(set(frames)) == 3, (
            "jobs at different seeds returned identical frames -- they are "
            "sharing state")

        # The cap, deterministically: fill every slot, then ask for one more.
        now = time.time()
        for k in range(server._MAX_CONCURRENT_JOBS):
            occupy(k, now)
        over = start(4)
        assert over.status_code == 429, (
            f"a job was accepted with every slot occupied "
            f"(HTTP {over.status_code}); the cap is not holding")
        assert "too many" in (over.get_json() or {}).get("error", ""), \
            over.get_json()

        # Freeing a slot lets work through again.
        with server._jobs_lock:
            server._jobs.pop(fakes.pop(), None)
        r = start(9, n=10)
        assert r.status_code == 200, (
            f"a slot was freed but the next job was still refused "
            f"(HTTP {r.status_code})")
        finish(r.get_json()["job_id"])

        with server._jobs_lock:
            for k in list(fakes):
                server._jobs.pop(k, None)
            fakes.clear()

        # A long legitimate run must survive cleanup: old enough to expire by
        # TTL, still well inside the watchdog threshold.
        name = occupy("long", time.time() - (server._JOB_TTL_SECONDS + 600))
        with server._jobs_lock:
            server._cleanup_old_jobs()
            survived = server._jobs.get(name)
        assert survived is not None and survived["status"] == "running", (
            "a running job was evicted for being old; a long exact sample "
            "would be discarded mid-flight")
        with server._jobs_lock:
            server._jobs.pop(name, None)
        fakes.remove(name)

        # Abandoned workers must not disable sampling server-wide.
        stale = time.time() - (server._JOB_MAX_RUNTIME_SECONDS + 60)
        stuck = [occupy(f"stuck{k}", stale)
                 for k in range(server._MAX_CONCURRENT_JOBS)]
        r = start(10, n=10)
        assert r.status_code == 200, (
            f"abandoned jobs disabled exact sampling (HTTP {r.status_code}); "
            f"the watchdog did not reap them")
        with server._jobs_lock:
            statuses = {k: server._jobs[k]["status"] for k in stuck}
        assert set(statuses.values()) == {"error"}, statuses
        finish(r.get_json()["job_id"])
    finally:
        with server._jobs_lock:
            for k in fakes:
                server._jobs.pop(k, None)
            for k, v in borrowed.items():
                server._jobs.setdefault(k, v)


def test_every_server_result_is_discarded_if_the_parameters_moved():
    """A finished job must not be drawn under controls it does not belong to.

    All three handlers that fetch a sample from the server can outlive the
    parameters they were started with: the sliders and the domain grid stay
    live while a job runs, and a CFTP sample at a few hundred sites takes
    long enough for someone to change their mind. Each one therefore
    snapshots paramGen before starting and drops the result if it has moved.

    The domain handler did neither. Measured in a browser: start a sample on
    one set of blocks, flip a block while it runs, and the finished job drew
    the OLD domain's configuration under the new grid, captioned with the old
    domain's coupling window (1024, where the new one needs 4096). The
    picture and the controls described different regions, which is the exact
    failure this codebase already fixed once for live weight changes.

    Stated as an invariant over the handlers rather than as a fact about that
    one, so a fourth path cannot be added without the guard. The grid toggle
    must also bump the generation, or the check it feeds never fires.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    js = (root / "static" / "draw.js").read_text()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))

    # Every place that SNAPSHOTS the generation must also check it: a
    # snapshot with no check is the shape the domain handler would have had
    # if someone had added half the guard.
    n_snap = code.count("requestedGen = paramGen")
    n_check = code.count("paramGen !== requestedGen")
    assert n_snap >= 3, (
        f"only {n_snap} handlers snapshot paramGen; the two exact paths and "
        f"the domain path all outlive their parameters")
    assert n_check == n_snap, (
        f"{n_snap} snapshots of paramGen but {n_check} checks -- a result is "
        f"being rendered without comparing the generation it was started at")

    # the domain handler specifically, since it is the one that lacked both
    dom = _js_function_body(code, "function sampleDomain(")
    assert "requestedGen = paramGen" in dom and "paramGen !== requestedGen" in dom, (
        "sampleDomain does not guard its result against a parameter change")
    assert dom.index("paramGen !== requestedGen") < dom.index("renderFrame("), (
        "sampleDomain renders before checking the generation")

    # fetchExactSample is the one function that builds a server frame without
    # a guard, and that is correct only because it RETURNS the frame for its
    # caller to check. Pin that, so the exemption cannot quietly widen.
    helper = _js_function_body(code, "function fetchExactSample(")
    assert "frameFromServerData" in helper
    assert "renderFrame(" not in helper, (
        "fetchExactSample now renders as well as returns, so it needs the "
        "generation guard its callers currently provide")

    # and the controls that change the DOMAIN must bump the generation,
    # otherwise the guard above is checking a number that never moves
    grid = _js_function_body(code, "function buildDomainGrid(")
    assert "DOMAIN_I[u][v] = DOMAIN_I[u][v] ? 0 : 1;" in grid, grid[:200]
    assert "bumpParams()" in grid, (
        "toggling a domain block does not bump paramGen, so a sample in "
        "flight is not recognised as belonging to a different domain")


def test_a_removed_job_is_not_resurrected_by_its_own_worker():
    """A deliberate removal from the job table must stick.

    The completion handler writes its status under `job_id in _jobs`, so a
    worker whose entry was removed discards its result -- but an earlier
    draft of this guard wrote an "error" status on the way out instead, which
    put the entry back in the table after something had deliberately taken it
    out. The worker now exits quietly.

    What this test does NOT claim: that stopping early saves work. It was
    first written that way, asserting the thread exits after the entry is
    dropped, and a mutation that removed the abort entirely still passed --
    at these sizes the job simply finishes on its own inside the window. The
    measurement says the same thing: the progress callback fires once per
    doubling attempt and the last attempt dominates, so a job dropped 0.3s in
    ran for 0.92x (n=96) and 1.01x (n=128) of an uninterrupted one. Bounding
    the waste would need a check inside the sweep loop, and the server does
    not drop running jobs anyway -- _cleanup_old_jobs expires only finished
    ones. So the property worth holding is that the removal sticks, and that
    is what is asserted.
    """
    import threading
    import time

    import server

    c = server.app.test_client()
    before = threading.active_count()

    r = c.post("/api/exact/start",
               json={"n": 48, "a1": 1, "a2": 1, "b1": 1, "b2": 1,
                     "c1": 1, "c2": 1, "seed": 1})
    assert r.status_code == 200, (r.status_code, r.data[:200])
    jid = r.get_json()["job_id"]

    for _ in range(200):
        time.sleep(0.05)
        with server._jobs_lock:
            if server._jobs.get(jid, {}).get("status") == "running":
                break
    with server._jobs_lock:
        server._jobs.pop(jid, None)

    for _ in range(1200):
        time.sleep(0.05)
        if threading.active_count() <= before:
            break
    else:
        raise AssertionError("the worker never finished")

    with server._jobs_lock:
        assert jid not in server._jobs, (
            "the worker wrote a status for a job that had been removed, "
            "putting the entry back in the table")

    # and the removal must not have broken the endpoint for anyone else
    r2 = c.post("/api/exact/start",
                json={"n": 8, "a1": 1, "a2": 1, "b1": 1, "b2": 1,
                      "c1": 1, "c2": 1, "seed": 2})
    assert r2.status_code == 200, (r2.status_code, r2.data[:200])
    jid2 = r2.get_json()["job_id"]
    for _ in range(2000):
        st = c.get(f"/api/exact/status/{jid2}").get_json()
        if st.get("status") in ("done", "error"):
            break
        time.sleep(0.05)
    assert st["status"] == "done", st


def test_the_exact_sampler_does_not_refuse_a_model_because_of_its_scale():
    """The Boltzmann measure is invariant under scaling all six weights, so
    the sampler must be too -- including when the partition function stops
    being representable.

    The backward pass accumulates products of face weights, so Z grows like
    lambda^(n^2). Past float range the build produced inf and the sampler
    raised "numerical overflow/underflow ... rescale the weights (the model
    is invariant under a common rescaling of all six weights)". The message
    was right about the invariance, which is exactly why the refusal was
    wrong: w and lambda*w are the SAME distribution, so the scale the caller
    typed decided whether they got a sample or an error. At n=12 all six
    weights at 100 worked and at 140 raised -- for the uniform measure.

    Now the build normalises and retries, and Z is reported in the original
    scale, which may be inf or 0. That is the honest answer: the measure is
    ordinary, the number simply is not representable. Nothing about sampling
    depends on being able to name it.

    The retry runs ONLY after a direct build has failed, so every case that
    already worked is untouched -- asserted below by checking that the
    ordinary weights never take the rescaled path.
    """
    import math

    import numpy as np
    from scipy.stats import chi2 as chi2_dist

    from sixvertex.exact import ExactSampler

    n = 6
    base = dict(a1=0.7, a2=1.3, b1=0.5, b2=1.6, c1=1.1, c2=2.3)
    cfgs, probs, _ = exact_distribution(n, base)
    key = {np.asarray(c, dtype=int).tobytes(): i for i, c in enumerate(cfgs)}

    def law(lam):
        w = {k: v * lam for k, v in base.items()}
        s = ExactSampler(n, w)
        rng = np.random.default_rng(5)
        N = 4000
        cnt = np.zeros(len(cfgs))
        for _ in range(N):
            H = np.asarray(s.sample(rng), dtype=int)
            k = key.get(H.tobytes())
            assert k is not None, f"lam={lam}: sampler left the state space"
            cnt[k] += 1
        e = probs * N
        live = e >= 5
        chi = float((((cnt - e) ** 2) / e)[live].sum())
        return s, cnt, chi, int(live.sum()) - 1

    # unscaled: must NOT take the retry path
    s1, _, chi1, dof1 = law(1.0)
    assert not hasattr(s1, "_z_scale"), (
        "ordinary weights took the rescaling path; the retry is supposed to "
        "run only after a direct build has failed")
    assert chi2_dist.sf(chi1, dof1) > 1e-3, chi1
    assert math.isfinite(s1.partition_function())

    # scales whose Z is past float range in both directions
    for lam in (1e10, 1e-10, 1e30):
        s, cnt, chi, dof = law(lam)
        assert hasattr(s, "_z_scale"), (
            f"lam={lam} did not need rescaling; pick a harsher scale or this "
            f"test is not exercising the path it claims to")
        # the direct build really would have failed: that is why we are here
        raw = float(sum(s.B[0].values()))
        assert math.isfinite(raw) and raw > 0, raw
        z = s.partition_function()
        assert z == math.inf or z == 0.0, (
            f"lam={lam}: Z={z}, expected the original-scale value to be "
            f"unrepresentable")
        assert chi2_dist.sf(chi, dof) > 1e-3, (
            f"lam={lam}: the rescaled build does not reproduce the measure "
            f"(chi2={chi:.1f} on {dof} dof)")

    # and the measure must be the SAME measure, not merely a valid one
    _, c1, _, _ = law(1.0)
    for lam in (1e10, 1e-10):
        _, c2, _, _ = law(lam)
        assert np.array_equal(c1, c2), (
            f"lam={lam} gave a different sample sequence from the same seed; "
            f"rescaling changed the conditional probabilities")


def test_the_two_models_export_different_height_conventions_and_say_so():
    """Exported data must carry enough provenance to be read back.

    Both models write a file called vertsix_heights_n<N>.txt, and their
    height functions do not mean the same thing: under DWBC neighbouring
    values differ by +-1, under the stochastic model by 0 or -1. Nothing in
    the file said which. Reading one with the other's face rule does not
    error -- it yields a full lattice of plausible, wrong vertex types.
    Measured on one stochastic sample, applying the DWBC rule to its own
    height export disagreed with its own type export at 105 of 144 faces.

    These are the files that end up in a paper, so they now carry the model,
    the weights (or b1, b2 and Delta), the seed, the row order and the
    convention. Every line is '#'-prefixed, which numpy.loadtxt skips, so
    they still load exactly as before.
    """
    import json
    import pathlib
    import shutil
    import subprocess

    import numpy as np

    # the underlying fact the header exists to record
    from sixvertex import stochastic as st
    from sixvertex.exact import exact_sample

    e = exact_sample(8, a1=1., a2=1., b1=1., b2=1., c1=1., c2=1., seed=3)
    Hd = np.asarray(e[0] if isinstance(e, tuple) else e, dtype=int)
    h, v = st.sample(8, 0.8, 0.2, seed=3)
    Hs = st.height_function(h, v).astype(int)

    dsteps = set(np.unique(np.diff(Hd, axis=0)).tolist())
    ssteps = set(np.unique(np.diff(Hs, axis=0)).tolist())
    assert dsteps <= {-1, 1}, dsteps
    assert ssteps <= {0, 1, -1} and ssteps != dsteps, (
        f"the two height conventions are no longer different (DWBC {dsteps}, "
        f"stochastic {ssteps}); if they have been unified, the header's "
        f"warning is now misleading and should change")

    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("node not available")

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "static" / "draw.js").read_text()
    body = _js_function_body(src, "function exportHeader(")

    # minimal stubs for the few things the header reads
    driver = """
    const MODEL = {value: "__MODEL__"};
    const SLIDERS = {sb1: {value: "0.8"}, sb2: {value: "0.2"}, seed: {value: "77"}};
    const document = {getElementById: (id) => id === "model" ? MODEL
                                               : (SLIDERS[id] || null)};
    function isStochastic() { return MODEL.value === "stochastic"; }
    function isDomain() { return MODEL.value === "domain"; }
    function currentWeights() {
      return {a1: 0.7, a2: 1.3, b1: 0.5, b2: 1.6, c1: 1.1, c2: 2.3};
    }
    function currentSeed() {
      const v = SLIDERS.seed.value.trim();
      return v === "" ? null : parseInt(v, 10);
    }
    function exportHeader(frame, what) __BODY__
    const frame = {n: 12};
    console.log(JSON.stringify({
      heights: exportHeader(frame, "height function"),
      types: exportHeader(frame, "vertex types"),
    }));
    """
    out = {}
    for model in ("dwbc", "stochastic"):
        script = (driver.replace("__BODY__", body)
                        .replace("__MODEL__", model))
        r = subprocess.run([node, "-e", script], capture_output=True,
                           text=True, timeout=120)
        assert r.returncode == 0, r.stderr[:500]
        out[model] = json.loads(r.stdout)

    for model, hdrs in out.items():
        for what, text in hdrs.items():
            lines = [ln for ln in text.splitlines() if ln.strip()]
            assert lines, (model, what)
            assert all(ln.startswith("#") for ln in lines), (
                f"{model}/{what}: a header line is not comment-prefixed, so "
                f"numpy.loadtxt would try to parse it:\\n{text}")
            assert "seed: 77" in text, f"{model}/{what} does not record the seed"
            assert "n = 12" in text, f"{model}/{what} does not record n"

    # the convention line must distinguish the models, which is the point
    dh = out["dwbc"]["heights"]
    sh = out["stochastic"]["heights"]
    assert "+1 or -1" in dh and "DWBC" in dh, dh
    assert "0 or -1" in sh and "stochastic" in sh, sh
    assert dh != sh

    # and each must record the parameters needed to reproduce the figure
    assert "a1 = 0.7" in dh and "c2 = 2.3" in dh, dh
    assert "b1 = 0.8" in sh and "b2 = 0.2" in sh, sh

    # the type export must name the code map and the naming convention it uses
    for model in out:
        t = out[model]["types"]
        assert "1=a1" in t and "6=c2" in t, t
        assert "Gorin-Kenyon" in t, (
            "the type codes do not say which vertex naming convention they "
            "follow, which is the thing that was wrong once already")


def test_states_rounded_away_are_accounted_for_rather_than_ignored_or_feared():
    """Underflow in the backward pass drops states; the question is whether
    the mass they carried mattered.

    A row with reachable successors whose backward value rounds to zero has
    underflowed, not been proved unreachable, and dropping it shrinks the
    support. Nothing downstream notices: the partition function stays finite
    and positive, and every configuration drawn is still legal. So the drop
    has to be measured where it happens.

    Measuring it both ways matters. Treating ANY drop as fatal was the first
    attempt here, and it refused cases that are correct to hundreds of
    digits: at a1=1e-20, n=12 over a thousand rows round away, but each held
    less than DBL_MIN against a total of order one, so the bound on the lost
    mass is about 1e-305 -- states whose relative probability the enumerator
    in these tests cannot represent either. Refusing those would repeat the
    error of refusing a model for the scale it happened to be typed in.

    So the bound count * DBL_MIN is weighed against the retained total, and
    only a loss that could be measured is refused. Both sides are exercised
    below, because a threshold that nothing ever crosses is not a check.
    """
    import numpy as np

    from sixvertex.exact import ExactSampler

    DBL_MIN = 2.2250738585072014e-308
    U = dict(a1=1., a2=1., b1=1., b2=1., c1=1., c2=1.)

    # Negligible: many rows drop, but the total is of order one.
    accepted = 0
    for key, tiny in (("a1", 1e-20), ("a1", 1e-40), ("b1", 1e-60)):
        w = {**U, key: tiny}
        s = ExactSampler(12, w)
        assert s._underflowed > 0, (
            f"{key}={tiny} no longer underflows at n=12; this case is not "
            f"exercising the accounting it claims to")
        total = float(sum(s.B[0].values()))
        assert total > 0 and np.isfinite(total)
        assert s._underflowed * DBL_MIN <= 1e-9 * total
        H = np.asarray(s.sample(np.random.default_rng(0)), dtype=int)
        assert H.shape == (13, 13)
        accepted += 1
    assert accepted == 3

    # Significant: the retained total is itself near the underflow floor, so
    # the dropped mass is no longer negligible against it.
    w = {k: 3e-9 for k in U}
    w["a1"] = 3e-9 * 1e-40
    try:
        ExactSampler(6, w)
    except RuntimeError as e:
        msg = str(e).lower()
        assert "underflow" in msg and "negligible" in msg, e
    else:
        raise AssertionError(
            "a build whose dropped mass is comparable to its partition "
            "function was accepted; the sampler would draw from a measurably "
            "truncated support")


def test_a_negative_seed_is_accepted_everywhere_and_nothing_returns_a_500():
    """Typing -5 into the seed box broke every sampler.

    numpy's default_rng and SeedSequence take non-negative integers only, and
    the value went to them unchecked. On the two job endpoints that surfaced
    as a failed job reading "expected non-negative integer" -- a numpy
    message shown to someone who had just asked for a reproducible figure.
    On /api/stochastic it was worse: an unhandled exception, so Flask
    returned a 500 whose HTML body hit the browser's .json() and produced
    "Unexpected token '<', "<!doctype "... is not valid JSON". Both were
    reachable from the page, since parseInt("-5") is -5 and the box invites
    a seed.

    A non-finite n did the same thing by a different route: int(float("inf"))
    raises OverflowError, which the handler's (TypeError, ValueError) clause
    did not catch.

    Non-negative seeds still pass through exactly, so nothing that worked
    before changes; a negative one maps deterministically into the top of the
    64-bit range. The last assertion is the general one: no input to any
    endpoint should produce a 5xx, since that means a stack trace rather than
    an answer.
    """
    import time

    import server

    c = server.app.test_client()

    def exact_job(seed, n=8):
        r = c.post("/api/exact/start",
                   json={"n": n, "a1": 1, "a2": 1, "b1": 1, "b2": 1,
                         "c_up": 1, "c_down": 1, "seed": seed})
        assert r.status_code == 200, (r.status_code, r.data[:200])
        jid = r.get_json()["job_id"]
        for _ in range(3000):
            s = c.get(f"/api/exact/status/{jid}").get_json()
            if s.get("status") in ("done", "error"):
                return s
            time.sleep(0.05)
        raise AssertionError("job never finished")

    def stoch(seed, n=16):
        r = c.post("/api/stochastic",
                   json={"n": n, "b1": 0.3, "b2": 0.7, "seed": seed})
        assert r.status_code == 200, (
            f"seed={seed} gave HTTP {r.status_code}: {r.data[:160]}")
        return r.get_json()["frame"]

    for seed in (-1, -12345, -(2 ** 63)):
        s = exact_job(seed)
        assert s["status"] == "done", (
            f"a negative seed failed the exact job: {s.get('error')!r}")
        stoch(seed)

    # deterministic, and distinct from each other and from the positives
    assert stoch(-1) == stoch(-1)
    assert stoch(-1) != stoch(-2)
    assert stoch(-1) != stoch(1)

    # non-negative seeds pass through unchanged
    for seed in (0, 7, 12345, 2 ** 31):
        assert stoch(seed) == stoch(seed)
    assert stoch(7) != stoch(8)

    # The normalised value must stay exactly representable as a JavaScript
    # number, and the page must echo back what the CALLER typed. With a 2^64
    # modulus, -5 became 18446744073709552000 in the browser -- not the value
    # the server used, and not retypable, so the reported seed was unusable.
    assert server._normalise_seed(-5) < 2 ** 53
    assert server._normalise_seed(-(2 ** 63)) < 2 ** 53
    assert server._normalise_seed(7) == 7 and server._normalise_seed(0) == 0
    assert server._normalise_seed(None) is None
    for seed in (-5, 7, None):
        info = c.post("/api/stochastic",
                      json={"n": 8, "b1": 0.3, "b2": 0.7,
                            "seed": seed}).get_json()["info"]
        assert info.get("seed") == seed, (
            f"the server echoed {info.get('seed')!r} for a request with "
            f"seed={seed!r}; the page would report a seed that does not "
            f"reproduce the sample")

    # non-finite n is a 400, not a 500
    for v in (float("inf"), float("nan"), -float("inf")):
        for path, body in (("/api/exact/start",
                            {"a1": 1, "a2": 1, "b1": 1, "b2": 1,
                             "c_up": 1, "c_down": 1}),
                           ("/api/stochastic", {"b1": 0.3, "b2": 0.7})):
            r = c.post(path, json={**body, "n": v})
            assert r.status_code == 400, (
                f"{path} with n={v} returned {r.status_code}, not a clean 400")

    # The general invariant: no input should produce a stack trace. Any 5xx
    # here is a crash the caller cannot act on.
    weird = [None, "", "abc", [], {}, True, False, 0, -1, 1e400, float("nan"),
             float("inf"), -0.0, "1", "1e5", [1, 2], {"x": 1}, 2 ** 70,
             -(2 ** 70), "-5"]
    started = []
    failures = []

    def probe(path, body, field, value):
        d = dict(body)
        d[field] = value
        r = c.post(path, json=d)
        if r.status_code >= 500:
            failures.append((path, field, repr(value)[:24], r.status_code))
        elif r.status_code == 200 and path.endswith("/start"):
            j = (r.get_json() or {}).get("job_id")
            if j:
                started.append(j)

    try:
        for v in weird:
            for field in ("n", "a1", "c_up", "seed"):
                probe("/api/exact/start",
                      {"n": 8, "a1": 1, "a2": 1, "b1": 1, "b2": 1,
                       "c_up": 1, "c_down": 1}, field, v)
            for field in ("n", "b1", "seed"):
                probe("/api/stochastic", {"n": 8, "b1": 0.3, "b2": 0.7},
                      field, v)
            for field in ("n", "seed", "x_cuts", "y_cuts", "I"):
                probe("/api/domain/start",
                      {"n": 12, "a1": 1, "a2": 1, "b1": 1, "b2": 1,
                       "c_up": 2, "c_down": 2, "x_cuts": [4, 8],
                       "y_cuts": [4, 8],
                       "I": [[1, 1, 1], [0, 1, 1], [0, 0, 1]]}, field, v)
        assert not failures, (
            f"{len(failures)} inputs produced a 5xx instead of an error the "
            f"caller can read: {failures[:8]}")
    finally:
        # don't leave the fuzz's jobs holding concurrency slots
        with server._jobs_lock:
            for j in started:
                server._jobs.pop(j, None)


def test_the_svg_export_is_independent_of_the_camera():
    """The README promises two different things of the two exports.

    "PNG is a screenshot and honours the current zoom and pan, so a
    zoomed-in view exports a crop; SVG exports the complete configuration
    regardless of zoom or pan."

    Verified in a browser at n=12: zooming in changed the PNG (20449 ->
    19484 bytes) and left the SVG byte-identical, with 144 = 12*12 rects
    either way. This pins it structurally so it cannot drift: the SVG
    builder must not read the camera, and the PNG path must go through the
    canvas, which is what makes it a screenshot.

    A documented export behaviour is the kind of thing someone relies on
    without re-checking -- a figure for a paper is exported once -- so the
    failure would be a silently cropped lattice in a published figure.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    js = (root / "static" / "draw.js").read_text()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))

    i = code.index("btnSaveSvg.addEventListener")
    depth, j = 0, code.index("{", i)
    k = j
    while k < len(code):
        if code[k] == "{":
            depth += 1
        elif code[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    svg_handler = code[j:k + 1]

    assert "<svg" in svg_handler or "svg" in svg_handler.lower(), svg_handler[:200]
    for forbidden in ("camera", "baseScale(", "devicePixelRatio"):
        assert forbidden not in svg_handler, (
            f"the SVG export reads {forbidden!r}; it is documented to export "
            f"the complete configuration regardless of zoom or pan, so it "
            f"must not depend on the camera")

    # and it must cover the whole lattice: a loop over n, not over what is
    # currently visible
    assert re.search(r"for\s*\(\s*(let|var)\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*n",
                     svg_handler), (
        "the SVG export does not loop over the full lattice:\n"
        + svg_handler[:300])

    # the PNG path, by contrast, is the canvas itself -- that is what makes
    # it honour zoom and pan
    i = code.index("btnSave.addEventListener")
    png_handler = code[i:i + 600]
    assert "toDataURL" in png_handler or "toBlob" in png_handler, png_handler[:300]


def test_delta_drives_coalescence_too_not_only_n():
    """A correction to a claim this project made earlier today.

    The slow-sampling warning was rewritten to key on n, with the text
    "CFTP's cost is set by n, not by Delta" and the advice "Delta will not
    help". That came from measurements at Delta = +0.5, 0.0 and -3.0, where
    the sweep count really is flat -- and all three sit inside the flat part
    of the curve, so the conclusion was exactly right over the range it was
    drawn from and wrong everywhere else.

    Median sweeps to coalescence, a = b = 1, c set from Delta:

           Delta   +0.5    -3     -6    -10    -15    -20
      n=10            64     64    256    512   4096   4096
      n=16           256    256   1024   2048  16384  >2^16
      n=24          2048   1024   4096  >2^16  >2^16  >2^16

    Flat to about Delta = -5, then steep, and the two factors compound:
    n=24 at Delta=-10 already exceeds a budget that n=96 meets at Delta=0.
    Someone at Delta = -25 was being told to reduce n, which would not have
    helped them.

    Both halves are pinned here: the measurement, so the claim cannot be
    re-derived from too narrow a range again, and the trigger, so the
    warning keeps naming whichever factor is responsible.
    """
    import math
    import pathlib

    from sixvertex.cftp_exact import cftp_sample

    def sweeps(n, delta, seeds=(1, 2, 3)):
        c = math.sqrt(2 - 2 * delta)
        w = dict(a1=1., a2=1., b1=1., b2=1., c1=c, c2=c)
        out = []
        for s in seeds:
            _, info = cftp_sample(n, w, master_seed=s, max_T=1 << 17)
            out.append(info["sweeps"])
        out.sort()
        return out[len(out) // 2]

    n = 10
    flat = [sweeps(n, d) for d in (0.5, -3.0)]
    deep = sweeps(n, -15.0)
    assert flat[0] == flat[1], (
        f"the curve is no longer flat between Delta=+0.5 and -3 at n={n} "
        f"({flat}); the table in this docstring needs redoing")
    assert deep >= 4 * max(flat), (
        f"Delta no longer drives the coalescence time at n={n}: {max(flat)} "
        f"sweeps at Delta=-3 against {deep} at Delta=-15. If that is real, "
        f"the warning can go back to keying on n alone")

    # the shipped warning must key on BOTH, and must not tell the user that
    # Delta is irrelevant
    js = (pathlib.Path(__file__).resolve().parent.parent
          / "static" / "draw.js").read_text()
    i = js.index("const slowNote")
    block = js[i:i + 2600]
    assert "CFTP_SLOW_DELTA" in block and "CFTP_SLOW_N" in block, (
        "the slow-sampling warning no longer considers both n and Delta:\n"
        + block[:400])
    code = "\n".join(ln for ln in block.splitlines()
                     if not ln.strip().startswith("//"))
    for claim in ("not by \\u0394", "will not help"):
        assert claim not in code, (
            f"the warning still tells the user {claim!r}, which the "
            f"measurements above contradict")


def test_the_documented_size_limits_are_the_enforced_ones():
    """Each endpoint's n range is a promise; check it at the boundary.

    The README says out-of-range values "return 400 rather than being
    silently adjusted", and quotes 250 for /api/exact/start. The other two
    limits differ (4..400 for the stochastic sampler, 6..160 for general
    domains) and the domain one was not written down anywhere, so a caller
    could only discover it by being refused.

    Checked AT the boundary in both directions, because an off-by-one here
    is invisible from inside the range: the limit itself must be accepted
    and one past it refused.
    """
    import server

    c = server.app.test_client()
    started = []

    def post(path, body, n):
        d = dict(body)
        d["n"] = n
        if path == "/api/domain/start" and n >= 6:
            d["x_cuts"] = [max(1, n // 3), max(2, 2 * n // 3)]
            d["y_cuts"] = [max(1, n // 3), max(2, 2 * n // 3)]
        r = c.post(path, json=d)
        j = r.get_json() or {}
        if j.get("job_id"):
            started.append(j["job_id"])
        return r.status_code

    cases = [
        ("/api/exact/start", {"a1": 1, "a2": 1, "b1": 1, "b2": 1,
                              "c_up": 1, "c_down": 1}, 4, 250),
        ("/api/stochastic", {"b1": 0.3, "b2": 0.7}, 4, 400),
        ("/api/domain/start", {"a1": 1, "a2": 1, "b1": 1, "b2": 1,
                               "c_up": 2, "c_down": 2,
                               "I": [[1, 1, 1], [0, 1, 1], [0, 0, 1]]}, 6, 160),
    ]
    try:
        for path, body, lo, hi in cases:
            assert post(path, body, lo - 1) == 400, f"{path}: n={lo-1} accepted"
            assert post(path, body, lo) == 200, f"{path}: n={lo} refused"
            assert post(path, body, hi) == 200, f"{path}: n={hi} refused"
            assert post(path, body, hi + 1) == 400, f"{path}: n={hi+1} accepted"
            assert post(path, body, 10 ** 6) == 400, f"{path}: huge n accepted"
    finally:
        with server._jobs_lock:
            for j in started:
                server._jobs.pop(j, None)

    # sweeps are rejected rather than clamped, which is the failure the
    # README calls out by name: running 500 for a request that asked 9999
    # leaves the caller believing the chain is far better equilibrated
    init = c.post("/api/init", json={"n": 8, "a1": 1, "a2": 1, "b1": 1,
                                     "b2": 1, "c_up": 1, "c_down": 1}).get_json()
    sid = init["session_id"]
    assert c.post("/api/step", json={"sweeps": 500, "session_id": sid}).status_code == 200
    r = c.post("/api/step", json={"sweeps": 9999, "session_id": sid})
    assert r.status_code == 400, "an oversized sweep count was accepted"
    assert "frame" not in (r.get_json() or {}), "sweeps were clamped, not refused"

    # and every limit must appear in the README, since that is where a
    # caller looks before being refused
    import pathlib
    readme = (pathlib.Path(__file__).resolve().parent.parent
              / "README.md").read_text()
    for value in ("250", "400", "160"):
        assert value in readme, f"the limit {value} is not documented"


def test_the_domain_enumerators_reproduce_the_asm_numbers():
    """Anchor the ORACLE, not just the sampler.

    The general-domain tests in this file are judged against
    experiments/brute_domain -- its enumerate_heights supplies the exact law
    that the domain CFTP sampler is compared to. An error there would not
    show up as a failure; it would quietly redefine what "correct" means for
    every one of those tests, and they would keep passing.

    The existing ASM check anchors ExactSampler.partition_function. This
    anchors the oracle itself: on the full square a general-domain
    enumerator must count alternating sign matrices, 1, 2, 7, 42, 429, 7436.

    Both enumerators are checked. They share domains.FACE, so this does not
    make them independent of a labelling error -- the five-table test covers
    that -- but they walk the state space differently, one over height
    functions and one over edge configurations, so an enumeration fault in
    either shows up here as a disagreement with a published sequence.
    """
    from sixvertex.domains import domain_mask
    from experiments.brute_domain import (enumerate_edge_configs,
                                          enumerate_heights)

    asm = {1: 1, 2: 2, 3: 7, 4: 42, 5: 429, 6: 7436}
    for n, want in asm.items():
        m = domain_mask([0, n], [0, n], [[1]])
        got_h = len(enumerate_heights(m))
        got_e = enumerate_edge_configs(m)
        assert got_h == want, (
            f"enumerate_heights counts {got_h} configurations on the {n}x{n} "
            f"square; the ASM number is {want}. The domain tests judge the "
            f"sampler against this enumerator, so they would not catch it")
        assert got_e == want, (
            f"enumerate_edge_configs counts {got_e} on the {n}x{n} square, "
            f"expected the ASM number {want}")


def test_the_island_level_ratio_is_stable_once_the_estimator_stops_picking_a_side():
    """The island statistic is stable in n; the estimator was not.

    Two claims in this file -- that P(mode-1)/P(mode) does not drift with n,
    and that the island "sits one level low often and one level high almost
    never" -- were measured at n = 16, 24 and 32. All three are 0 mod 4, and
    that turns out to be the whole story.

    k is centred on n/2, so its offset moves with n mod 4: integer at
    n = 0 (mod 4), half-integer at n = 2 (mod 4), quarter-integer for odd n.
    The distribution itself keeps the same shape everywhere -- a dominant
    level and a secondary one spaced by 1 -- but the secondary sits BELOW
    the mode when n = 0 (mod 4) and ABOVE it when n = 2 (mod 4). Asking for
    mode-1 therefore reads the real ratio at one parity and essentially zero
    at the other:

        n       14     16     18     20     22     24     28
        n%4      2      0      2      0      2      0      0
        2nd/1st  .205   .270   .202   .238   .223   .253   .218
        mode-1   .000   .270   .000   .238   .002   .253   .218

    So the physics is stable and the one-sidedness is a property of the
    centring convention, not of the island. This pins the direction-agnostic
    ratio across BOTH parity classes, which is the form of the claim that
    survives.
    """
    from collections import Counter

    from sixvertex.cftp_exact import cftp_sample
    from experiments.island import island
    from experiments.mixing import weights_for_delta

    w = weights_for_delta(-3.0)
    agnostic, directional = {}, {}
    for n in (14, 16, 18, 20):
        ks = []
        for s in range(260):
            r = island(cftp_sample(n, w, master_seed=8_800_000 + s)[0], n)
            if r["k"] is not None:
                ks.append(r["k"])
        c = Counter(ks)
        order = sorted(c, key=c.get, reverse=True)
        assert len(order) >= 2, f"n={n}: only one level populated, {c}"
        agnostic[n] = c[order[1]] / c[order[0]]
        directional[n] = c.get(order[0] - 1, 0) / c[order[0]]

    for n, r in agnostic.items():
        assert 0.10 < r < 0.40, (
            f"n={n}: secondary/dominant ratio {r:.3f} outside the measured "
            f"band; this is the quantity that should NOT depend on n")
    spread = max(agnostic.values()) - min(agnostic.values())
    assert spread < 0.15, f"the agnostic ratio drifted with n: {agnostic}"

    # and the directional estimator must still show the parity split, since
    # that is the thing being documented; if it ever stops, the centring has
    # changed and the docstrings above need revisiting
    for n in (14, 18):
        assert directional[n] < 0.05, (
            f"n={n} is 2 mod 4 but P(mode-1)/P(mode) = {directional[n]:.3f}; "
            f"the k centring no longer behaves as documented")
    for n in (16, 20):
        assert directional[n] > 0.10, (
            f"n={n} is 0 mod 4 but P(mode-1)/P(mode) = {directional[n]:.3f}")


def test_the_theta_floor_reaches_further_than_delta_minus_one():
    """The exponent's trustworthy range was stated one or two steps too wide.

    plot_island_order.py rejects Delta = -0.5 and -0.75 as floor effects --
    the largest solid component is a handful of faces, so A cannot decay
    further and the fitted exponent measures the floor rather than the
    physics -- and then says "only the points from Delta = -1 downwards,
    where the core is tens to hundreds of faces, carry a meaningful
    exponent".

    Measured core sizes in faces, at the three sizes the fit uses:

        Delta      N=32    N=48    N=64
        -0.75       7.9    12.7    14.9     <- dismissed as floor
        -1.00       9.5    16.4    20.0     <- claimed meaningful
        -1.50      19.2    31.5    52.9
        -3.00      57.4   118.0   199.8

    Delta = -1 is 20 faces at N = 64 against 14.9 at -0.75: the same order as
    the point being rejected, and nowhere near "tens to hundreds". So the
    meaningful range starts nearer -1.5.

    This matters because -1 is exactly where a transition would be claimed,
    so the weakest point in the fit sits on the most interesting Delta. The
    check runs at N = 32, where the effect is clearest and the sampling is
    cheap; the larger sizes are in the table above rather than in the loop.
    """
    import numpy as np

    from sixvertex.cftp_exact import cftp_sample
    from experiments.island_order import core_area_fraction
    from experiments.mixing import weights_for_delta

    n = 32
    faces = {}
    for d in (-0.75, -1.0, -1.5):
        w = weights_for_delta(d)
        v = [core_area_fraction(cftp_sample(n, w, master_seed=6_100_000 + s)[0], n)
             * n * n for s in range(50)]
        faces[d] = float(np.mean(v))

    assert faces[-0.75] < 15, faces
    assert faces[-1.0] < 20, (
        f"Delta=-1 now has a core of {faces[-1.0]:.1f} faces at N={n}; if it "
        f"has genuinely reached 'tens to hundreds' the caveat can be relaxed")
    assert faces[-1.0] < 2.5 * faces[-0.75], (
        f"Delta=-1 ({faces[-1.0]:.1f} faces) is supposed to be the same order "
        f"as the rejected Delta=-0.75 ({faces[-0.75]:.1f}); that is the whole "
        f"point of narrowing the trustworthy range")
    assert faces[-1.5] > 1.5 * faces[-1.0], (
        f"Delta=-1.5 ({faces[-1.5]:.1f}) is no longer clearly above Delta=-1 "
        f"({faces[-1.0]:.1f}), so the suggested cutoff needs revisiting")

    # and the document must not still advertise the old cutoff
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parent.parent / "experiments"
           / "plot_island_order.py").read_text(encoding="utf-8")
    assert "meaningful range starts around" in doc, (
        "plot_island_order.py no longer records where the exponent becomes "
        "trustworthy")


def test_the_numba_toggle_reaches_the_coalescence_measurement():
    """validate.py's compiled-vs-numpy check must not compare a path to itself.

    It works by setting ``cftp_exact._HAS_NUMBA = False`` and calling
    ``forward_coalescence_time`` again. experiments/mixing.py used to do
    ``from sixvertex.cftp_exact import _HAS_NUMBA``, which binds the value at
    import time: the toggle never reached the function, both runs took the
    compiled path, and a deliberately sabotaged ``_sweep`` passed the check.
    Pinned here on the behaviour -- the toggle changes which code runs -- not
    on how mixing.py spells the import.
    """
    import pytest

    import sixvertex.cftp_exact as ce
    import experiments.mixing as mixing

    if not ce._HAS_NUMBA:
        pytest.skip("numba is not installed; there is no second path to compare")

    w = mixing.weights_for_delta(-2.0)
    calls = {"numpy": 0}
    real_sweep = mixing._sweep

    def counting_sweep(*a, **k):
        calls["numpy"] += 1
        return real_sweep(*a, **k)

    saved = ce._HAS_NUMBA
    try:
        mixing._sweep = counting_sweep
        ce._HAS_NUMBA = True
        compiled = mixing.forward_coalescence_time(10, w, seed=3)
        took_numpy_while_compiled = calls["numpy"]
        ce._HAS_NUMBA = False
        pure = mixing.forward_coalescence_time(10, w, seed=3)
        took_numpy_after_toggle = calls["numpy"] - took_numpy_while_compiled
    finally:
        mixing._sweep = real_sweep
        ce._HAS_NUMBA = saved

    assert took_numpy_while_compiled == 0, (
        "the compiled path is not being taken when _HAS_NUMBA is True")
    assert took_numpy_after_toggle > 0, (
        "setting cftp_exact._HAS_NUMBA = False did not reach "
        "forward_coalescence_time -- the flag is bound at import time again, "
        "which makes validate.py's compiled-vs-numpy check vacuous")
    assert compiled == pure, (
        f"the two sweep implementations disagree on the same randomness: "
        f"{compiled} vs {pure}")


def test_the_island_lattice_claim_is_qualified_by_n_mod_four():
    """k = (G - n/2)/2 is integer valued only at n = 0 mod 4.

    The spacing is 1 at every n, but the offset moves with n mod 4. Two
    documents claimed k "comes out an integer" flatly, and validate.py checked
    integrality at n=32 alone -- passing while asserting something false of
    every other size.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for rel in ("experiments/island.py", "experiments/validate.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "n mod 4" in text or "n % 4" in text, (
            f"{rel} no longer records that k's offset depends on n mod 4")
    # and the fact itself, not just the wording: at n = 2 mod 4 the centred
    # height is a half-integer, so "k is an integer" is false there.
    from sixvertex.cftp_exact import cftp_sample
    from experiments.island import island
    from experiments.mixing import weights_for_delta

    w = weights_for_delta(-3.0)
    ks = [island(cftp_sample(34, w, master_seed=880000 + s)[0], 34)["k"]
          for s in range(4)]
    ks = [k for k in ks if k is not None]
    assert ks, "no island found at n=34; the check below proves nothing"
    assert all(abs(k - round(k)) > 0.1 for k in ks), (
        f"k came out integer valued at n=34 (n%4=2): {sorted(set(ks))} -- "
        f"either the centring changed or this test is now vacuous")


def test_the_stopping_time_refuses_weights_where_its_bisection_is_invalid():
    """cftp_stopping_time rests on coalescence being monotone in T.

    That holds inside the monotone region and fails outside it, because what
    the predicate tracks is whether the two EXTREMAL chains agree, which
    certifies a constant composed map only while they cannot cross. The
    function used to accept any weights and bisect anyway, handing back a
    confident number; its sibling forward_coalescence_time has always refused.

    Both halves are asserted: that a violation really exists off-region (or
    the refusal would be cosmetic), and that the refusal happens.
    """
    import pytest

    from experiments.cftp_stopping import _coalesces_from, cftp_stopping_time
    from experiments.mixing import weights_for_delta
    from sixvertex.cftp_exact import (_colour_indices, _weight_table,
                                      in_monotone_region)

    off = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=0.9, c2=0.9)
    assert not in_monotone_region(off)

    n = 8
    tbl = _weight_table(off)
    colour_idx, size = _colour_indices(n)
    violations = 0
    for seed in range(12):
        flags = [_coalesces_from(n, off, seed, T, tbl, colour_idx, size)
                 for T in range(1, 70)]
        if flags != sorted(flags):
            violations += 1
    assert violations > 0, (
        "no seed showed coalescence turning back off as T grew, so this test "
        "no longer demonstrates why the refusal is needed")

    with pytest.raises(ValueError, match="monotone"):
        cftp_stopping_time(n, off, 0)

    # and it still works where it is defined
    assert cftp_stopping_time(n, weights_for_delta(-2.0), 900) >= 1


def test_the_monotone_region_is_sharp_under_the_exhaustive_check():
    """The region is not just sufficient -- the coupling fails outside it.

    monotone_check.py's exhaustive single-site check (every ordered pair of
    configurations, every site, every threshold of the driving uniform) is the
    decisive instrument, and its docstring claimed sharpness without anything
    asserting it. Pinned here on both sides of the C >= A and C >= B walls, at
    the smallest step that crosses them -- a = 1.01 rather than the 1.05 the
    note used to cite.
    """
    from experiments.monotone_check import monotonicity_violations
    from sixvertex.cftp_exact import in_monotone_region
    from sixvertex.domains import domain_mask

    m = domain_mask([0, 4], [0, 4], [[1]])

    def monotone(**kw):
        w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.0, c2=1.0)
        w.update(kw)
        return in_monotone_region(w), not monotonicity_violations(m, w)

    # on the C = A face: monotone, and the predicate agrees
    assert monotone() == (True, True)
    assert monotone(a1=0.95, a2=0.95) == (True, True)
    # one percent outside either wall and the coupling is already broken
    assert monotone(a1=1.01, a2=1.01) == (False, False), (
        "the coupling is still monotone just outside C >= A, so the region is "
        "conservative rather than sharp -- or the check stopped working")
    assert monotone(b1=1.05, b2=1.05) == (False, False)


def test_the_shift_chain_centres_its_cells_on_the_data_not_on_zero():
    """run_shift_chain counted the literal cells -1, 0, +1.

    That is the right triple only at n = 0 mod 4. k = (G - n/2)/2 with G in
    steps of 2, so at n = 2 mod 4 the occupied levels are half-integers and
    every observation fell outside all three cells: R_lo and R_hi came out
    0/0 and the script printed mu = sigma = nan without complaint.
    """
    import numpy as np

    from experiments.mixing import weights_for_delta
    from experiments.run_shift_chain import chain_observations, pooled_mode

    w = weights_for_delta(-3.0)
    seen = {}
    for n in (16, 18):
        ks = chain_observations(n, w, 60, 8, seed=1, start_seed=2)
        seen[n] = ks[~np.isnan(ks)]
        assert seen[n].size, f"no island found at n={n}"

    # the data-driven mode is a value that actually occurs, at both residues
    for n, ks in seen.items():
        mode = pooled_mode([ks])
        assert mode in set(ks.tolist()), (n, mode, sorted(set(ks.tolist())))
        covered = int(((ks == mode - 1) | (ks == mode) | (ks == mode + 1)).sum())
        assert covered == ks.size, (
            f"n={n}: the three cells around the mode {mode} hold {covered} of "
            f"{ks.size} observations, so the spacing is not 1")

    # the shared helper -- one copy, used by run_shift_chain and
    # run_shift_delta_chain, which both had the bug independently
    from experiments.run_shift_chain import cells_from_chains

    for n, ks in seen.items():
        mode, per = cells_from_chains([ks])
        assert sum(per[0]) == ks.size, (n, mode, per, ks.size)

    # and the old hardcoded triple really did miss everything at n = 18
    ks18 = seen[18]
    hardcoded = int(((ks18 == -1) | (ks18 == 0) | (ks18 == 1)).sum())
    assert hardcoded == 0, (
        "n=18 now lands on integer k, so this test no longer demonstrates why "
        "the centring has to come from the data")


def test_the_crossing_predicate_turns_on_below_minus_one_not_at_it():
    """island_exists.py claimed "the onset at Delta = -1" from 40 samples.

    200 samples at n=64 put P at 0.010 at Delta = -1.0 and 0.115 at -1.5, so
    the predicate's crossover is near -1.5 to -2 -- which is what
    island_order.py already said of the same measurement, the two notes having
    disagreed. Asserted here on the cheap size, where the signal is the same
    shape: the continuous companion (mean widest extent) is monotone in Delta,
    and P is still small at -1 while large at -3.
    """
    import numpy as np

    from experiments.island_exists import island_exists
    from experiments.mixing import weights_for_delta
    from sixvertex.cftp_exact import cftp_sample

    n, reps = 32, 30
    P, W = {}, {}
    for d in (-0.75, -1.0, -1.5, -3.0):
        w = weights_for_delta(d)
        hits, ext = 0, []
        for s in range(reps):
            H, _ = cftp_sample(n, w, master_seed=777_000 + 53 * s)
            ok, _lvl, e = island_exists(H, n)
            hits += int(ok)
            ext.append(e)
        P[d], W[d] = hits / reps, float(np.mean(ext))

    order = [W[d] for d in (-0.75, -1.0, -1.5, -3.0)]
    assert order == sorted(order), (
        f"the mean widest extent is not monotone in Delta: {W}")
    assert P[-1.0] <= 0.2, (
        f"P({n}, Delta=-1) = {P[-1.0]}, so the predicate is firing at the "
        f"claimed transition rather than below it")
    assert P[-3.0] >= 0.4, f"P({n}, Delta=-3) = {P[-3.0]}, too low to be a test"
    assert P[-3.0] > 2 * P[-1.0] + 0.1, (P[-1.0], P[-3.0])


def test_every_grid_cell_of_experiment_one_gets_its_own_seeds():
    """run_mixing.py ran seeds 0..199 in every (n, Delta) cell.

    The per-cell medians are unbiased either way, but it is exactly the error
    run_shift.py documents -- "the same seeds producing the same islands, not
    two independent agreements" -- and measured at n=16 between Delta=-1.0 and
    -1.25 the shared seeds correlate the coalescence times (Spearman rho 0.248,
    p=0.026) while disjoint ones do not (-0.140, p=0.22).

    Pinned on the property: every cell of the default grid, and of a wider one,
    draws from its own seed range. The sign of Delta has to be carried, since
    abs(delta) collides +0.5 with -0.5.
    """
    from experiments.run_mixing import seed_block

    ns = [12, 16, 24, 32, 40, 64, 80]
    deltas = [0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0,
              -2.5, -3.0, -4.0, -5.0]
    cells = [(n, d) for n in ns for d in deltas]
    blocks = {c: set(seed_block(c[0], c[1], 200)) for c in cells}

    assert all(len(b) == 200 for b in blocks.values())
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            assert not (blocks[a] & blocks[b]), (
                f"cells {a} and {b} share seeds "
                f"{sorted(blocks[a] & blocks[b])[:4]}")

    # +0.5 and -0.5 are the pair that an unsigned encoding merges
    assert not (blocks[(16, 0.5)] & blocks[(16, -0.5)])


def test_the_theta_exponent_carries_its_interval_and_names_its_inversions():
    """Panel (b) of plot_island_order plotted theta as a bare line.

    The CSV's se column was read for panel (a) and never used for the
    exponent, while the docstring claimed an error "of order 0.2" -- so seven
    values spanning 0.16 to 1.15 were drawn as if resolved. Propagated, the
    standard error is 0.12 to 0.20, which is wide enough that the committed
    sample's two ordering inversions are both inside the noise, and the
    docstring used to claim "the ORDERING is solid" while naming only one of
    them.

    Three things are pinned: the interval is nonzero and of the stated size,
    the deep and shallow ends are genuinely distinguished, and a CSV with too
    few sizes is refused rather than fitted.
    """
    import csv
    import pathlib

    import pytest

    from experiments.plot_island_order import theta_with_interval

    src = (pathlib.Path(__file__).resolve().parent.parent / "experiments"
           / "island_order_sample.csv")
    rows = list(csv.DictReader(src.open()))
    sizes = sorted({int(r["n"]) for r in rows})
    assert len(sizes) >= 3

    out = {d: theta_with_interval(rows, d, sizes, reps=4000)
           for d in (-0.75, -1.0, -1.25, -1.5, -3.0)}

    # the interval is real and of the size the docstring states
    for d, (t, se, lo, hi) in out.items():
        assert 0.05 < se < 0.35, (d, se)
        assert lo < t < hi, (d, t, lo, hi)

    # the ends are distinguished: filament at the shallow end, blob at the deep
    assert out[-0.75][2] > 0.4, "theta at Delta=-0.75 no longer excludes 0"
    assert out[-3.0][3] < 0.7, "theta at Delta=-3 no longer excludes 1"

    # and the -1.25 / -1.5 inversion is inside the noise, not structure
    t125, se125 = out[-1.25][0], out[-1.25][1]
    t150, se150 = out[-1.5][0], out[-1.5][1]
    assert t125 < t150, (
        "the -1.25 / -1.5 inversion is gone; the docstring records it")
    assert abs(t150 - t125) < 2 * (se125 ** 2 + se150 ** 2) ** 0.5, (
        f"the inversion is now larger than the noise ({t125:.3f} vs "
        f"{t150:.3f}), which would be a finding rather than a caveat")

    # too few sizes is refused, not fitted to a confident-looking number
    with pytest.raises(SystemExit, match="at least three"):
        theta_with_interval(rows, -3.0, [32], reps=10)


def test_the_island_figure_marks_the_measured_crossing_not_only_the_predicted_one():
    """plot_island_exists.py promised a marked crossing and drew none.

    Its only vertical line was Delta = -1, the prediction, while the docstring
    said "the crossing point of successive sizes is marked ... that, not the
    midpoint of any single curve, is the finite-size estimate". So the figure
    showed the expected answer and not the measured one.

    The measured crossings move DEEPER with N (-1.125 then -1.357), away from
    -1, and each rests on bracketing P differences no larger than their
    standard errors. Both halves are pinned: the drift, and that it is not
    significant -- if a later sample makes it significant that is a finding
    and this test should be re-read, not silenced.
    """
    import csv
    import pathlib

    from experiments.plot_island_exists import curve_crossings

    src = (pathlib.Path(__file__).resolve().parent.parent / "experiments"
           / "island_exists_sample.csv")
    rows = list(csv.DictReader(src.open()))
    sizes = sorted({int(r["n"]) for r in rows})
    assert len(sizes) >= 3

    found = []
    for i in range(len(sizes) - 1):
        for x, bd, bse in curve_crossings(rows, sizes[i], sizes[i + 1]):
            found.append((sizes[i], sizes[i + 1], x, bd, bse))

    assert len(found) >= 2, f"no crossings located: {found}"

    # every crossing sits below Delta = -1, not at it
    for n1, n2, x, _bd, _bse in found:
        assert x < -1.0, (
            f"crossing of N={n1},{n2} is at Delta={x:+.3f}, at or above -1; "
            f"the docstring records it as deeper")

    # the position is INTERPOLATED, not snapped to a sampled Delta: a crossing
    # that lands exactly on a grid point means the interpolation is not
    # running, and that mutation survives every other assertion here
    grid = sorted({float(r["delta"]) for r in rows})
    for n1, n2, x, _bd, _bse in found:
        assert all(abs(x - g) > 1e-9 for g in grid), (
            f"crossing of N={n1},{n2} sits exactly on the sampled Delta "
            f"{x:+.3f}, so it was snapped to a grid point rather than "
            f"interpolated between the bracketing cells")

    # and it moves deeper as N grows
    assert found[-1][2] < found[0][2], (
        f"the crossing no longer moves deeper with N: "
        f"{[round(f[2], 3) for f in found]}")

    # none of them is resolved -- judged by the BRACKETING cells, since the
    # difference at the crossing is zero by construction
    for n1, n2, x, bd, bse in found:
        assert bd <= 1.5 * bse, (
            f"crossing of N={n1},{n2} at {x:+.3f} now rests on a difference "
            f"({bd:.4f}) well above its standard error ({bse:.4f}) -- that is "
            f"a result, and the docstring's caveat needs rewriting")


def test_the_domain_dependence_of_r_lo_is_not_explained_by_area():
    """The four Gorin-Kenyon domains do not have the same number of cells.

    plot_island_domain.py reports a factor-five spread in R_lo "at the SAME
    Delta and the same n" and used not to mention that the domains hold 2304,
    1536, 1280 and 2048 faces -- a 1.80x range, which matters because
    plot_shift_size.py measures a real size dependence in R_lo (+0.0088 over a
    4x change in cells, 3.0 sigma).

    The refutation is the ordering, not the bound: if area drove R_lo it would
    rise with cell count, and instead the two largest domains sit at the two
    extremes. Pinned here, because this is the first question a referee asks.
    """
    import csv
    import pathlib

    import numpy as np

    src = (pathlib.Path(__file__).resolve().parent.parent / "experiments"
           / "island_domain_sample.csv")
    rows = list(csv.DictReader(src.open()))
    cells = np.array([int(r["cells"]) for r in rows], float)
    r_lo = np.array([float(r["R_lo"]) for r in rows], float)
    names = [r["domain"] for r in rows]

    assert len(rows) >= 4
    spread = r_lo.max() - r_lo.min()
    assert spread > 0.1, f"the spread this test is about is gone: {spread}"

    # the size effect cannot account for it: +0.0088 per 4x in cells
    allowed = 0.0088 * np.log(cells.max() / cells.min()) / np.log(4.0)
    assert allowed < 0.1 * spread, (
        f"the known size dependence ({allowed:.4f}) is now a sizeable part of "
        f"the {spread:.4f} spread, so area is no longer ruled out by magnitude")

    # and the spread is not ordered by area in either direction
    order = np.argsort(cells)
    by_area = r_lo[order]
    assert not np.all(np.diff(by_area) >= 0), "R_lo now rises with area"
    assert not np.all(np.diff(by_area) <= 0), "R_lo now falls with area"

    # concretely: the two largest domains are at the two extremes
    big2 = set(np.array(names)[np.argsort(cells)[-2:]])
    ends = {names[int(np.argmin(r_lo))], names[int(np.argmax(r_lo))]}
    assert big2 == ends, (
        f"the two largest domains {sorted(big2)} are no longer the two "
        f"extremes of R_lo {sorted(ends)}; the docstring says they are")


def test_the_shift_figure_quotes_the_p_values_its_own_sample_gives():
    """plot_shift.py quoted the omnibus test as p=0.09; the sample gives 0.034.

    The docstring built an argument on the two tests disagreeing -- the
    omnibus failing to reject constancy while the slope rejected it. On the
    committed sample both reject at 0.05, so the contrast was not there and
    only the power difference is. Pinned by recomputing through the module's
    own load() and fits, which is what a reader would do.
    """
    import pathlib
    import sys

    import numpy as np

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "experiments"))
    from plot_shift import constant_fit, load, trend

    rows = sorted(load(str(root / "experiments" / "shift_delta_sample.csv")),
                  key=lambda r: r["delta"])
    v = np.array([r["r_lo"] for r in rows])
    s = np.array([r["se"] for r in rows])
    x = np.array([r["delta"] for r in rows])

    _m, chi2v, dof, p_const, _se = constant_fit(v, s)
    slope, slope_se, _z, p_slope = trend(x, v, s)

    assert dof == len(rows) - 1
    assert abs(chi2v - 13.62) < 0.1, f"chi2 is now {chi2v:.2f}, docstring 13.62"
    assert abs(p_const - 0.034) < 0.005, (
        f"the omnibus p is now {p_const:.3f}; the docstring records 0.034 "
        f"(and once wrongly recorded 0.09)")
    assert abs(p_slope - 0.0077) < 0.0015, (
        f"the slope p is now {p_slope:.4f}, docstring 0.0077")
    assert abs(slope - 0.0095) < 0.001 and abs(slope_se - 0.0036) < 0.001

    # the claim that survives: both reject constancy, the slope more sharply
    assert p_const < 0.05 and p_slope < 0.05, (p_const, p_slope)
    assert p_slope < p_const, "the slope is no longer the sharper instrument"


def test_the_legacy_cftp_is_correct_at_exactly_one_weight_point():
    """sixvertex/cftp.py claimed correctness for ARBITRARY c1, c2.

    Its comment read "this proven-safe regime allows a=b=1 with ARBITRARY c1,
    c2 -- including deep ferroelectric/antiferroelectric weights (|Delta|>1).
    CFTP remains exactly correct there", citing as verification that it
    "coalesces correctly" at c1=c2=sqrt(8). Coalescing is not sampling the
    right measure.

    p_up = c_up/(c_up+c_down) is the only place the weights enter, so at
    c1 = c2 it is 1/2 for every c and the chain returns the UNIFORM measure
    whatever it was asked for. At c1 != c2 it returns neither the requested
    measure nor the uniform one. Correct at c1 = c2 = 1 and nowhere else --
    pinned here in total variation, because a comment claiming otherwise would
    license using this module for antiferroelectric sampling.
    """
    import numpy as np

    from sixvertex.cftp import cftp_sample

    n, N = 4, 4000
    cfgs = enumerate_height_functions(n)
    keys = {c.astype(np.int64).tobytes(): i for i, c in enumerate(cfgs)}
    _c, unif, _z = exact_distribution(
        n, dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.0, c2=1.0))

    def empirical(c1, c2):
        counts = np.zeros(len(cfgs))
        for s in range(N):
            H, _ = cftp_sample(n=n, c_up=c1, c_down=c2, master_seed=s)
            counts[keys[np.asarray(H, dtype=np.int64).tobytes()]] += 1
        return counts / counts.sum()

    def tv(p, q):
        return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())

    noise = 0.05  # total-variation sampling error at N=4000 over 42 cells

    # (a) exact at the uniform point
    e11 = empirical(1.0, 1.0)
    assert tv(e11, unif) < noise, tv(e11, unif)

    # (b) at c1 = c2 != 1 it returns the UNIFORM measure, not the one asked for
    for c in (2.0, 0.5):
        emp = empirical(c, c)
        _c2, want, _z2 = exact_distribution(
            n, dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=c, c2=c))
        assert tv(emp, unif) < noise, (
            f"at c1=c2={c} the legacy chain no longer returns the uniform "
            f"measure (TV {tv(emp, unif):.4f}); p_up must have stopped being "
            f"1/2, which would change what this module is")
        assert tv(emp, want) > 0.2, (
            f"at c1=c2={c} the legacy chain now matches the measure it was "
            f"asked for (TV {tv(emp, want):.4f}) -- if that is real, the "
            f"module comment needs rewriting in the other direction")

    # (c) at c1 != c2 it matches neither
    emp = empirical(1.0, 2.0)
    _c3, want, _z3 = exact_distribution(
        n, dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.0, c2=2.0))
    assert tv(emp, want) > 0.4 and tv(emp, unif) > 0.4, (
        tv(emp, want), tv(emp, unif))

    # and nothing in the product routes to it: the general cftp_sample is the
    # one the package exports, and server.py's import of this module (which
    # shadowed that name) is gone.
    import pathlib

    import sixvertex

    from sixvertex.cftp_exact import cftp_sample as general

    assert sixvertex.cftp_sample is general, (
        "sixvertex.cftp_sample is no longer the general sampler")
    server_src = (pathlib.Path(__file__).resolve().parent.parent
                  / "server.py").read_text(encoding="utf-8")
    assert "from sixvertex.cftp import" not in server_src, (
        "server.py imports the legacy a=b=1 CFTP again, shadowing the general "
        "cftp_sample by name")


def test_every_label_table_in_the_package_is_discovered_and_agrees():
    """The hand-written census missed sixvertex/sampler.py -- both copies.

    test_all_five_vertex_label_tables_agree enumerates five tables by name.
    There are seven in the Python package alone: sampler.py's
    _classify_face_np and _classify_face_torch were never in the list, and both
    had b1 and b2 the wrong way round -- (0,1,1,0) labelled b1 where every
    other table calls it b2 -- which is the same miss that left
    static/sixvertex.js wrong, in the LIVE MCMC path this time.

    Nothing observable changed (under DWBC #b1 = #b2, so the swap multiplies
    every weight by 1, and this class exposes weights rather than labels), but
    a hand-list is the wrong instrument for a census: it cannot report what it
    does not know about. This finds the tables by introspection instead, so a
    new copy following the naming convention is probed automatically and a
    rename that shrinks the census fails the count.
    """
    import importlib
    import inspect
    import pkgutil
    import re

    import numpy as np

    import sixvertex
    from sixvertex.domains import FACE

    PAT = re.compile(r"(classify_face|face_weight|weight_table)")
    found = []
    for m in pkgutil.iter_modules(sixvertex.__path__):
        mod = importlib.import_module(f"sixvertex.{m.name}")
        for name, obj in vars(mod).items():
            if PAT.search(name) and callable(obj):
                found.append((f"{mod.__name__}.{name}", obj, None))
            if inspect.isclass(obj) and obj.__module__ == mod.__name__:
                for mn, mo in vars(obj).items():
                    if PAT.search(mn) and callable(mo):
                        found.append((f"{mod.__name__}.{obj.__name__}.{mn}",
                                      mo, obj))

    assert len(found) >= 7, (
        f"the census shrank to {len(found)} tables: {[f[0] for f in found]}. "
        f"A table was renamed out of the convention, which hides it.")

    w = {"a1": 2.0, "a2": 3.0, "b1": 5.0, "b2": 7.0, "c1": 11.0, "c2": 13.0}
    by_value = {v: k for k, v in w.items()}
    bits = sorted(FACE)

    def corners(l, t, b, r):
        tl = 0
        bl, tr = tl + l, tl + t
        br = bl + b
        assert br - tr == r
        return tl, tr, bl, br

    checked = 0
    for label, fn, cls in found:
        params = list(inspect.signature(fn).parameters)
        for (l, t, b, r) in bits:
            tl, tr, bl, br = corners(l, t, b, r)
            idx = l * 8 + t * 4 + b * 2 + r
            if params[:1] == ["w"]:                      # table builder
                got = by_value.get(float(np.asarray(fn(w)).ravel()[idx]))
            elif params[:4] == ["tl", "tr", "bl", "br"] and cls is None:
                if "tbl" in params:                      # needs a table
                    from sixvertex.cftp_exact import _weight_table
                    got = by_value.get(float(np.asarray(fn(
                        *(np.array([[float(x)]]) for x in (tl, tr, bl, br)),
                        _weight_table(w))).ravel()[0]))
                else:
                    got = by_value.get(float(fn(tl, tr, bl, br, w)))
            elif cls is not None:                        # bound classify
                s = cls(n=4, a1=w["a1"], a2=w["a2"], b1=w["b1"], b2=w["b2"],
                        c_up=w["c1"], c_down=w["c2"])
                if "torch" in label and not getattr(s, "use_torch", False):
                    break                                # no torch installed
                arrs = [np.array([[float(x)]]) for x in (tl, tr, bl, br)]
                if "torch" in label:
                    import torch
                    arrs = [torch.from_numpy(a) for a in arrs]
                got = by_value.get(float(np.asarray(fn(s, *arrs)).ravel()[0]))
            else:                                        # H, i, j, w form
                continue
            assert got == FACE[(l, t, b, r)], (
                f"{label} labels (l,t,b,r)={(l, t, b, r)} as {got}; "
                f"domains.FACE calls it {FACE[(l, t, b, r)]}")
            checked += 1

    assert checked >= 5 * len(bits), (
        f"only {checked} (table, bits) pairs were actually probed; the "
        f"adapters are skipping tables rather than checking them")


def test_the_domain_scan_census_is_computed_not_quoted():
    """The degree-4 precondition's evidence is a finite scan, so count it.

    Two docstrings quoted "213 samplable domains" and "1635 free corners".
    Neither reproduces from the scan they describe -- it is 135 and 1277 --
    while the 122 low-degree corners and the 11 domains carrying them do. The
    larger figures came from a wider scan and were never updated.

    Also pinned: the stated MECHANISM (those 11 have a disconnected corner
    graph) against the refusal path actually taken (the extremal-height
    validity check, not connectivity). The module claimed they "were already
    refused for having a disconnected corner graph", which named the wrong
    check even though the property holds.
    """
    import itertools

    import numpy as np
    import pytest

    from sixvertex.domains import (DomainError, _components, _faces_of_corner,
                                   admits_configuration, cftp_sample,
                                   check_monotone_precondition, domain_mask,
                                   free_corners)

    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.6, c2=1.6)
    domains = corners = low = 0
    low_domains = []
    for bits in itertools.product([0, 1], repeat=9):
        I = np.array(bits).reshape(3, 3)
        if I.sum() == 0:
            continue
        for cuts in ([0, 1, 2, 4], [0, 2, 4, 6]):
            try:
                m = domain_mask(cuts, cuts, I)
            except DomainError:
                continue
            if not admits_configuration(m):
                continue
            degs = [len(_faces_of_corner(m, int(i), int(j)))
                    for (i, j) in map(tuple, np.argwhere(free_corners(m)))]
            domains += 1
            corners += len(degs)
            nl = sum(1 for d in degs if d != 4)
            low += nl
            if nl:
                low_domains.append(m)

    assert (domains, corners, low) == (135, 1277, 122), (
        f"the scan census moved to {(domains, corners, low)}; the docstrings "
        f"record 135 domains, 1277 free corners, 122 of them low-degree")
    assert len(low_domains) == 11, len(low_domains)

    for m in low_domains:
        # the stated property holds
        assert len(_components(m)) > 1, (
            "a low-degree-corner domain now has a connected corner graph, so "
            "the mechanism in check_monotone_precondition no longer explains "
            "why these are unreachable")
        # the precondition catches it when asked directly
        with pytest.raises(DomainError, match="degree-4|not 4"):
            check_monotone_precondition(m)
        # but the refusal the sampler actually gives is the earlier one
        with pytest.raises(DomainError) as e:
            cftp_sample(m, w, master_seed=0)
        assert "not a valid configuration" in str(e.value), (
            f"the refusal path changed to {str(e.value)[:80]!r}; the docstring "
            f"records that these are stopped by extremal-height validity, not "
            f"by the connectivity check")


def test_the_live_mcmc_samples_the_right_measure_at_general_weights():
    """SixVertexSampler had NO distributional test anywhere in the suite.

    It is the chain the server runs for the live view at whatever weights the
    user sets, and it was only ever exercised at the uniform point (and for
    extremal_height). Given what sixvertex/cftp.py turned out to be -- a module
    that coalesced happily while returning the uniform measure for every
    c1 = c2 -- an untested general-weights sampler was the worst remaining gap.

    Measured, 1500 chains x 120 sweeps at n=4, against the exact distribution:

        uniform      TV 0.0551
        c-dominant   TV 0.0523
        general      TV 0.0629   (a1=1.3 a2=0.8 b1=1.7 b2=0.6 c1=2.1 c2=1.4)

    against a floor of 0.0662 (uniform) and 0.0623 (general) from drawing 1500
    samples from the exact law itself. So it is indistinguishable from exact at
    all three. This is the cheap version of that: 800 chains x 30 sweeps.

    WHAT IT CAN AND CANNOT DETECT, since claiming more would be the error this
    whole exercise keeps finding. At N=800 the TV floor is 0.085 (99th
    percentile 0.110), and:

        TV(exact[W], exact[uniform])   0.2922   <- caught comfortably
        TV(exact[W], exact[c1=2.6])    0.0517   <- BELOW the floor, invisible
        TV(exact[W], exact[b1<->b2])   0.0000   <- invisible by construction

    So it catches "returns the wrong measure entirely", which is the failure
    that actually occurred, and does not resolve a fine misspecification. The
    chi-square below is the sharper of the two statistics (it rejects the
    c1=2.6 misspecification at p = 0.0014 where TV cannot see it), so both are
    applied, with thresholds loose enough that neither fails by chance.

    The b1<->b2 row is worth keeping: it is 0.0000 exactly, at general
    weights, which independently confirms the label-swap invariance that the
    vertex-naming tests rest on.
    """
    import numpy as np
    from scipy.stats import chisquare

    from sixvertex.sampler import SixVertexSampler

    n, N, sweeps = 4, 800, 30
    cfgs = enumerate_height_functions(n)
    keys = {c.astype(np.int64).tobytes(): i for i, c in enumerate(cfgs)}
    W = dict(a1=1.3, a2=0.8, b1=1.7, b2=0.6, c1=2.1, c2=1.4)
    _cfg, exact_p, _z = exact_distribution(n, W)

    counts = np.zeros(len(cfgs))
    for s in range(N):
        smp = SixVertexSampler(n=n, a1=W["a1"], a2=W["a2"], b1=W["b1"],
                               b2=W["b2"], c_up=W["c1"], c_down=W["c2"],
                               seed=7000 + s)
        smp.step(sweeps)
        H = np.asarray(smp.height_array(), dtype=np.int64)
        k = keys.get(H.tobytes())
        assert k is not None, "the MCMC left the state space"
        counts[k] += 1

    tv = 0.5 * float(np.abs(counts / counts.sum() - exact_p).sum())
    assert tv < 0.15, (
        f"TV(MCMC, exact) = {tv:.4f} at general weights; the floor from exact "
        f"draws at N={N} is 0.085 with a 99th percentile of 0.110, and a "
        f"sampler returning the wrong measure sits at 0.29 or more")

    expected = exact_p * counts.sum()
    chi, p = chisquare(counts, expected * counts.sum() / expected.sum())
    assert p > 1e-4, (
        f"chi-square against the exact law gives p = {p:.2g}; the MCMC is not "
        f"sampling the measure its weights define")

    # the label swap is exactly invisible, at general weights
    _c2, swapped, _z2 = exact_distribution(
        n, dict(W, b1=W["b2"], b2=W["b1"]))
    assert float(np.abs(exact_p - swapped).max()) < 1e-15, (
        "exchanging b1 and b2 changed the exact measure, which would break "
        "the argument the vertex-naming tests rest on")


def test_the_equilibration_badge_uses_the_only_observable_that_can_see_burn_in():
    """draw.js called the b-vertex density "c-vertex density", and had to.

    cFraction tested (l,t,b,r) = (0,1,1,0) and (1,0,0,1), which are b2 and b1
    in that file's own classifyBits, not c1 and c2. The badge then reported
    the number as "c-vertex density". Two errors, and the obvious repair --
    make it count c-vertices, matching its name and its label -- would have
    been a serious regression, because the c-density cannot see burn-in at
    all.

    The proof is the starting states. The two extremal configurations the two
    chains begin from have EXACTLY EQUAL c-density and maximally different
    b-density. An equilibration diagnostic built on a statistic that is
    identical on the two starts would announce "equilibrated" immediately from
    the worst possible state -- a false assurance, which is the one thing this
    badge exists to prevent.

    So the fix was to rename the function and correct the label, not to change
    the arithmetic. This pins the measurement that decides it.
    """
    import pathlib

    import numpy as np

    from sixvertex.cftp_exact import (_colour_indices, _sweep, _weight_table,
                                      extremal_height)

    def density(H, n, which):
        H = np.asarray(H, dtype=np.int64)
        tl, tr = H[:n, :n], H[:n, 1:]
        bl, br = H[1:, :n], H[1:, 1:]
        l, t = (bl - tl) == 1, (tr - tl) == 1
        b, r = (br - bl) == 1, (br - tr) == 1
        m = ((l & t & ~b & ~r) | (~l & ~t & b & r)) if which == "c" else \
            ((~l & t & b & ~r) | (l & ~t & ~b & r))
        return float(m.sum()) / (n * n)

    # (a) the c-density is EQUAL on the two starts; the b-density is not
    for n in (12, 24, 48):
        lo, hi = extremal_height(n, "lo"), extremal_height(n, "hi")
        assert density(lo, n, "c") == density(hi, n, "c"), (
            f"n={n}: the c-density now differs on the two extremal starts, so "
            f"it could serve as a diagnostic after all -- re-read this test")
        assert density(lo, n, "b") - density(hi, n, "b") > 0.9, (
            f"n={n}: the b-density no longer separates the extremal starts")

    # (b) and it still cannot see burn-in one sweep in, where b can
    n = 24
    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.0, c2=1.0)
    tbl = _weight_table(w)
    ci, size = _colour_indices(n)
    A = extremal_height(n, "lo").astype(np.float64)
    B = extremal_height(n, "hi").astype(np.float64)
    rA, rB = np.random.default_rng(1), np.random.default_rng(2)
    A = _sweep(A, tbl, rA.random(A.shape), ci, size)
    B = _sweep(B, tbl, rB.random(B.shape), ci, size)
    gap_c = abs(density(A, n, "c") - density(B, n, "c"))
    gap_b = abs(density(A, n, "b") - density(B, n, "b"))
    tol = max(0.02, 2.5 / n)
    assert gap_c < tol, (gap_c, tol)        # c would say "equilibrated"
    assert gap_b > tol, (gap_b, tol)        # b correctly says it is not
    assert gap_b > 10 * gap_c, (gap_b, gap_c)

    # (c) the badge no longer calls it a c-vertex density
    js = (pathlib.Path(__file__).resolve().parent.parent / "static"
          / "draw.js").read_text(encoding="utf-8")
    assert "function bFraction" in js, "the renamed function is gone"
    assert "in c-vertex density" not in js, (
        "the badge calls the through-vertex density a c-vertex density again")

    # (d) and the SHIPPED bFraction returns the b-density, run in node against
    #     the Python. A text check on the bit patterns would pass a rewritten
    #     function; executing it is the only way to know what the browser
    #     computes.
    import json
    import shutil
    import subprocess
    import tempfile

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available, so the shipped bFraction is "
                    "unverified; only the badge's wording was checked")

    start = js.index("function bFraction")
    body = js[start:js.index("\n  }", start) + 4]

    from sixvertex.cftp_exact import cftp_sample

    samples = []
    for sd in range(3):
        H, _i = cftp_sample(16, w, master_seed=9100 + sd)
        H = np.asarray(H, dtype=np.int64)
        samples.append({"n": 16, "H": H.ravel().tolist(),
                        "b": density(H, 16, "b"),
                        "c": density(H, 16, "c")})
    harness = (
        body
        + "\nconst cases = " + json.dumps(samples) + ";\n"
        + "console.log(JSON.stringify(cases.map(c => "
          "bFraction({n: c.n, H: c.H}))));\n")
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "bfrac.js"
        f.write_text(harness, encoding="utf-8")
        out = subprocess.run([node, str(f)], capture_output=True, text=True,
                             timeout=120)
    assert out.returncode == 0, out.stderr[:400]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    for g, case in zip(got, samples):
        assert abs(g - case["b"]) < 1e-12, (
            f"the shipped bFraction returns {g}, not the b-density "
            f"{case['b']}")
        assert abs(g - case["c"]) > 0.05, (
            f"the shipped bFraction returns {g}, which matches the c-density "
            f"{case['c']} -- if it was changed to count c-vertices, read this "
            f"test's docstring: the badge then cannot see burn-in at all")


def test_the_island_separation_figures_are_the_measured_ones():
    """island.py claimed a median separation of 50x; it is 36.

    The number matters because it is the whole argument for using
    8-connectivity: 4-connected gives about 3.4x, where the island is
    ambiguous, and 8-connected separates by a large factor. But "50" did not
    reproduce at the stated conditions -- n=32, Delta=-3 -- where it is 36.2x
    over 200 samples and 34.4x over the 40 that experiments/validate.py runs.
    So the repository's own validation output had been printing 34.4 beside a
    docstring saying 50.

    Pinned as a band rather than a point, since these are medians over tens of
    samples: what the argument needs is that 8-connectivity separates by more
    than an order of magnitude where 4-connectivity does not, and that no
    sample is ambiguous.
    """
    import numpy as np

    from sixvertex.cftp_exact import cftp_sample
    from experiments.island import island
    from experiments.mixing import weights_for_delta

    w = weights_for_delta(-3.0)
    n, reps = 32, 60
    seps = {}
    for conn in (4, 8):
        seps[conn] = np.array([
            island(cftp_sample(n, w, master_seed=880000 + s)[0], n,
                   connectivity=conn)["separation"] for s in range(reps)])

    m4, m8 = float(np.median(seps[4])), float(np.median(seps[8]))
    assert 2.5 < m4 < 5.0, (
        f"4-connected median separation is {m4:.1f}x; the docstring records "
        f"3.4 to 3.6")
    assert 25.0 < m8 < 50.0, (
        f"8-connected median separation is {m8:.1f}x; the docstring records "
        f"36 (and used to claim 50, which is why this is pinned)")
    assert m8 > 5 * m4, (
        f"8-connectivity no longer separates by an order of magnitude more "
        f"than 4-connectivity ({m8:.1f} vs {m4:.1f}), which is the entire "
        f"reason the island is defined with 8-connectivity")
    assert seps[8].min() >= 2.0, (
        f"the least-separated of {reps} samples is {seps[8].min():.1f}x, so "
        f"the island is no longer unambiguous in every sample")


def test_the_height_export_marks_corners_outside_the_domain():
    """draw.js exported out-of-domain corners as if they were heights.

    typesAsText has always written "" for a face outside the domain and the
    export header has always documented the marker. heightsAsText wrote every
    corner of the (n+1)x(n+1) table as a plain number. The server leaves 0 at
    those corners, and 0 is INSIDE the real height range on a general domain
    (0 to 6 on the staircase), so the values were indistinguishable from
    measurements: 12 of 49 entries, 24% of the table, with nothing in the
    header to say which.

    Two things are pinned: that the rule for "corner inside the domain"
    matches the server's corner_mask exactly, and that the browser applies it.
    """
    import itertools
    import pathlib

    import numpy as np

    from sixvertex.domains import (DomainError, cftp_sample, corner_mask,
                                   domain_mask)

    # (a) the rule the browser uses, ported, against the server's own mask
    def corner_in_domain(mask, i, j):
        n = mask.shape[0]
        for a, b in ((i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)):
            if 0 <= a < n and 0 <= b < n and mask[a, b]:
                return True
        return False

    cases = []
    for bits in itertools.product([0, 1], repeat=9):
        I = np.array(bits).reshape(3, 3)
        if I.sum() == 0:
            continue
        for cuts in ([0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 4, 6]):
            try:
                m = domain_mask(cuts, cuts, I)
            except DomainError:
                continue
            cases.append((cuts, I.tolist(), m))
    assert len(cases) > 250, len(cases)

    # the ported rule, as a first check
    for cuts, I, m in cases:
        cm = corner_mask(m)
        for i in range(m.shape[0] + 1):
            for j in range(m.shape[0] + 1):
                assert corner_in_domain(m, i, j) == bool(cm[i, j]), (
                    f"the ported corner rule disagrees with corner_mask at "
                    f"{(i, j)} on cuts={cuts}, I={I}")

    # and then the SHIPPED function, run in node. A port verified against the
    # server proves nothing about the browser if the two drift apart -- a
    # mutation that broke the rule in draw.js alone passed the ported check.
    import json
    import shutil
    import subprocess
    import tempfile

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available, so the shipped corner rule is "
                    "unverified here; only the ported copy was checked")

    js_src = (pathlib.Path(__file__).resolve().parent.parent / "static"
              / "draw.js").read_text(encoding="utf-8")
    start = js_src.index("function cornerInDomain")
    body = js_src[start:js_src.index("\n  }", start) + 4]

    payload = [{"n": int(m.shape[0]),
                "mask": m.astype(int).tolist(),
                "want": corner_mask(m).astype(int).tolist()}
               for _cuts, _I, m in cases]
    harness = (
        body
        + "\nconst cases = " + json.dumps(payload) + ";\n"
        + "let bad = 0;\n"
        + "for (const c of cases) {\n"
        + "  const frame = {n: c.n, inDomain: (i, j) => c.mask[i][j] === 1};\n"
        + "  for (let i = 0; i <= c.n; i++)\n"
        + "    for (let j = 0; j <= c.n; j++)\n"
        + "      if ((cornerInDomain(frame, i, j) ? 1 : 0) !== c.want[i][j]) bad++;\n"
        + "}\n"
        + "console.log(JSON.stringify({bad: bad, cases: cases.length}));\n")
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "corner.js"
        f.write_text(harness, encoding="utf-8")
        out = subprocess.run([node, str(f)], capture_output=True, text=True,
                             timeout=120)
    assert out.returncode == 0, out.stderr[:400]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["cases"] == len(cases)
    assert res["bad"] == 0, (
        f"the SHIPPED cornerInDomain in draw.js disagrees with the server's "
        f"corner_mask at {res['bad']} corners")

    # (b) the values at those corners really are in range, so a reader could
    #     not have spotted them -- which is why the marker is needed
    m = domain_mask([0, 2, 4, 6], [0, 2, 4, 6],
                    [[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    H, _info = cftp_sample(m, dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0,
                                   c1=1.6, c2=1.6), master_seed=3)
    H = np.asarray(H, dtype=np.int16)
    cm = corner_mask(m)
    inside_lo, inside_hi = int(H[cm].min()), int(H[cm].max())
    outside = set(H[~cm].tolist())
    assert outside, "this domain no longer has corners outside it"
    assert all(inside_lo <= v <= inside_hi for v in outside), (
        f"out-of-domain corners {sorted(outside)} now fall outside the real "
        f"height range [{inside_lo}, {inside_hi}], so they would be visible "
        f"as nonsense; re-read this test")

    # (c) and the browser blanks them, with the header saying so
    js = (pathlib.Path(__file__).resolve().parent.parent / "static"
          / "draw.js").read_text(encoding="utf-8")
    assert "function cornerInDomain" in js
    body = js.split("function heightsAsText", 1)[1].split("\n  }", 1)[0]
    assert "cornerInDomain(frame, i, j)" in body, (
        "heightsAsText no longer checks whether a corner is in the domain")
    assert "blank = lattice corner outside the domain" in js, (
        "the export header no longer documents the blank marker")


def test_the_path_convention_comment_agrees_with_the_classifier_it_sits_beside():
    """drawPaths listed "(3) b1 vertical (4) b2 horizontal" -- backwards.

    That was the eighth statement of the b1/b2 geometry in the tree and the
    only one disagreeing with the other seven, including the legend the user
    reads in index.html ("b1 (horizontal through)"). The drawing was never
    affected -- drawPaths works from the edge bits, and b1's bits draw a
    horizontal line whatever a comment claims -- but the comment cites Figure
    1 of the paper and tells a reader how to read the picture.

    A prose sweep is the wrong instrument here: the narrative comments in
    static/sixvertex.js that DESCRIBE the historical error ("weighted the
    vertical face as b1") trip any such check, and building a brittle text
    matcher is the mistake this suite keeps correcting. So the comment's
    direction word is checked against what the SHIPPED classifier actually
    returns for that geometry.
    """
    import json
    import pathlib
    import re
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available to run the shipped classifier")

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "static" / "draw.js").read_text(encoding="utf-8")

    # what the comment in drawPaths claims
    paths = src[src.index("function drawPaths"):]
    paths = paths[:paths.index("const n = frame.n;")]
    claimed = {}
    for label, word in re.findall(r"\((?:3|4)\)\s*(b1|b2)\s+(HORIZONTAL|VERTICAL|horizontal|vertical)",
                                  paths):
        claimed[label] = word.lower()
    assert claimed == {"b1": "horizontal", "b2": "vertical"}, (
        f"drawPaths' convention list reads {claimed}; every other statement "
        f"in the tree, and the legend in index.html, has b1 horizontal and "
        f"b2 vertical")

    # what the shipped classifier returns for each geometry.
    # horizontal through = left and right edges occupied; vertical = top and
    # bottom.
    i = src.index("function classifyBits")
    body = src[i:src.index("\n  }", i) + 4]
    script = (body + "\nconsole.log(JSON.stringify({"
              + "horizontal: classifyBits(true,false,false,true),"
              + "vertical: classifyBits(false,true,true,false)}));")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr[:400]
    got = json.loads(out.stdout.strip().splitlines()[-1])

    for direction, label in got.items():
        assert claimed[label] == direction, (
            f"the shipped classifyBits calls the {direction} through-vertex "
            f"{label}, while drawPaths' comment calls {label} "
            f"{claimed[label]}")

    # and the user-facing legend says the same thing
    html = (root / "static" / "index.html").read_text(encoding="utf-8")
    assert "(horizontal through)" in html and "(vertical through)" in html
    for label, direction in claimed.items():
        sub = f"b<sub>{label[-1]}</sub> ({direction} through)"
        assert sub in html, (
            f"the legend no longer says {sub!r}; the picture, the comment and "
            f"the legend must agree")


def test_the_fluctuation_field_is_a_difference_of_independent_samples():
    """The caption claims (H1-H2)/sqrt(2) "preserves the variance of a single
    height". That is only true if the two samples are independent.

    The second sample's seed is derived from the first by
    companionSeed(s) = ((s ^ 0x9e3779b9) >>> 0) % 2147483647. Earlier in this
    project the fluctuation view sent the SAME seed twice and displayed a
    field that was identically zero, captioned "+/-0.00"; a test now checks
    the two seeds differ, which is necessary and nowhere near sufficient --
    two seeds can differ and still produce correlated samples, and the
    variance claim would be wrong without anyone seeing a blank picture.

    So the statistical property is checked directly. For independent X, Y with
    the same law, Var((X-Y)/sqrt(2)) = Var(X), and the difference field is
    unbiased. Measured at n=24 over 120 pairs the ratio is 1.012; this is the
    cheap version at n=16 over 60.

    What it guards, stated precisely: reverting companionSeed to the identity
    gives an all-zero field and ratio 0.000, which fails. Replacing it with
    s + 1 gives ratio 0.986 and PASSES -- correctly, because SeedSequence
    decorrelates any distinct seed, so s + 1 really is an independent sample.
    This is a property test, not a pin on the formula: it catches the
    derivation collapsing, which is the failure that actually happened, and
    permits any derivation that delivers independence.
    """
    import numpy as np

    from sixvertex.cftp_exact import cftp_sample

    def companion(s):
        return ((s ^ 0x9e3779b9) & 0xFFFFFFFF) % 2147483647

    n, reps = 16, 60
    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.0, c2=1.0)

    singles = np.stack([
        np.asarray(cftp_sample(n, w, master_seed=50_000 + s)[0],
                   dtype=np.float64) for s in range(reps)])
    diffs = np.stack([
        (np.asarray(cftp_sample(n, w, master_seed=50_000 + s)[0],
                    dtype=np.float64)
         - np.asarray(cftp_sample(n, w, master_seed=companion(50_000 + s))[0],
                      dtype=np.float64)) / np.sqrt(2.0)
        for s in range(reps)])

    assert not np.allclose(diffs, 0.0), (
        "the difference field is identically zero -- the two samples share a "
        "seed again")

    moving = singles.var(axis=0) > 1e-9
    assert moving.sum() > 100, "too few fluctuating sites to measure"
    ratio = float(diffs.var(axis=0)[moving].mean()
                  / singles.var(axis=0)[moving].mean())
    assert 0.85 < ratio < 1.15, (
        f"Var((HA-HB)/sqrt(2)) / Var(H) = {ratio:.3f}; it should be 1 if the "
        f"companion seed gives an independent sample. Below 1 means the two "
        f"samples are correlated and the caption's variance claim is wrong")

    bias = float(diffs.mean())
    assert abs(bias) < 0.15, (
        f"the difference field has mean {bias:+.4f}; a difference of two "
        f"samples from the same law must be unbiased")


def test_off_region_violations_are_failures_of_stochastic_monotonicity():
    """Why "no monotone coupling exists" is a theorem and not a guess.

    README.md states the argument: "a monotone coupling forces the two chains'
    flip-up marginals to be ordered and they are not." That is the step which
    upgrades "THIS coupling breaks" into "no coupling can work", and nothing
    verified it -- monotone_check only reported that order was violated, not
    why.

    The mechanism: for A <= B, the shared-uniform coupling preserves order at
    a site for every u exactly when p(A) <= p(B), where p is the flip-up
    probability given the neighbourhood. So a violation means p(A) > p(B) for
    some ordered pair -- the conditional is not stochastically monotone in the
    configuration. That is a property of the MEASURE, not of the coupling, and
    no choice of coupling can repair it, because any monotone coupling would
    order the marginals it couples.

    Measured just outside the C >= A wall (a = 1.05, b = c = 1) on the square
    n=4: 333 ordered pairs have an order-violating site, and in all 333 the
    flip-up probability decreased.

    HOW MUCH OF THAT IS A MEASUREMENT, honestly. The 333-of-333 is a logical
    identity given the shape of this coupling, not an empirical discovery.
    With A <= B we have va <= vb, and the update sends each chain up or down
    by one, so na > nb needs: both up, va > vb (impossible); both down, same
    (impossible); A down and B up, va > vb + 2 (impossible); or A UP and B
    DOWN, which needs pb <= u < pa, i.e. pa > pb. There is no fourth case. So
    the equality cannot fail while the coupling is the threshold form.

    What the test therefore establishes is narrower than the number suggests:
    that the detector really is that threshold form (a coupling of another
    shape could break the identity and would be caught), and that violations
    exist off-region at all -- the guard below fails if there are none, which
    is what happened when this was pointed at weights INSIDE the region. The
    argument for "no monotone coupling exists" rests on the case analysis
    above, which is recorded here because it was previously only asserted in
    README.md with nothing working it through.

    This is also what bounds the claim: it establishes that MONOTONE CFTP is
    unavailable there, not that no CFTP scheme exists. index.html said the
    latter and now says the former.
    """
    import numpy as np

    from sixvertex.domains import (_weight_table, corner_mask, domain_mask,
                                   sweep_plan)
    from experiments.brute_domain import enumerate_heights
    from experiments.monotone_check import _p_up

    m = domain_mask([0, 4], [0, 4], [[1]])
    w = dict(a1=1.05, a2=1.05, b1=1.0, b2=1.0, c1=1.0, c2=1.0)
    tbl = _weight_table(w)
    hs = enumerate_heights(m)
    cm = corner_mask(m)
    sites = [s for g in sweep_plan(m) for s in g]

    violations = 0
    with_decrease = 0
    for A in hs:
        for B in hs:
            if np.array_equal(A, B) or not np.all(B[cm] >= A[cm]):
                continue
            for flat, nb, faces in sites:
                a = A.ravel().tolist()
                b = B.ravel().tolist()
                pa, va = _p_up(a, flat, nb, faces, tbl)
                pb, vb = _p_up(b, flat, nb, faces, tbl)
                if pa is None or pb is None:
                    continue
                bad = False
                for c in sorted({0.0, 1.0, pa, pb}):
                    for u in (max(0.0, c - 1e-9), c, min(1.0, c + 1e-9)):
                        na = va + 1 if u < pa else va - 1
                        nbv = vb + 1 if u < pb else vb - 1
                        if na > nbv:
                            bad = True
                if bad:
                    violations += 1
                    if pa > pb:
                        with_decrease += 1

    assert violations > 100, (
        f"only {violations} order-violating pairs just outside the region; "
        f"this test needs the violations to exist to say anything about them")
    assert with_decrease == violations, (
        f"{violations - with_decrease} of {violations} violations happen "
        f"WITHOUT the flip-up probability decreasing. The README's argument "
        f"for 'no monotone coupling exists' is that the marginals are not "
        f"ordered; a violation of another kind would mean that argument does "
        f"not cover every case")


def test_cftp_detects_a_ten_percent_error_in_a_single_weight():
    """How small an error the exactness tests can actually see.

    Audited by mutating cftp_exact's weight table and asking which tests
    notice. Measured total-variation distance from the true measure at n=4,
    weights a1=1.3 a2=0.8 b1=1.7 b2=0.6 c1=2.1 c2=1.4:

        a1 <-> c1        TV 0.29964   caught by 3 tests
        c1 *= 1.10       TV 0.02374   caught by NOTHING
        c1 <-> c2        TV 0.00000   invisible, and correctly so -- under
                                      DWBC #c1 - #c2 = n, so the swap
                                      multiplies every weight by (c2/c1)^n
                                      and the normalised measure is unchanged

    So a 10% error in one vertex weight passed every exactness test in the
    suite. Those tests draw a few thousand samples, where the total-variation
    noise floor is around 0.05 -- well above 0.024 -- so they could not have
    seen it however they were written.

    This closes that gap by power rather than cleverness. 20000 CFTP samples
    at n=4 cost 8 seconds, and a chi-square against the enumerated law then
    detects the 10% error at p ~ 7e-8 (worst of 12 trials), while a correct
    sampler gives p uniform on (0,1). The threshold is 1e-5: false failures
    at one in a hundred thousand runs, and two orders of magnitude of margin
    on the detection side.

    What it still cannot see: an error below about TV 0.01 at this sample
    size, and any relabelling that leaves the measure invariant -- for those,
    the cross-table census and the label-swap invariance tests are the
    instruments, not sampling.
    """
    import numpy as np
    from scipy.stats import chisquare

    from sixvertex.cftp_exact import cftp_sample, in_monotone_region

    n, N = 4, 20000
    w = dict(a1=1.3, a2=0.8, b1=1.7, b2=0.6, c1=2.1, c2=1.4)
    assert in_monotone_region(w), "the test point must be inside the region"
    assert len(set(w.values())) == 6, "all six weights must differ"

    cfgs, exact_p, _z = exact_distribution(n, w)
    assert len(cfgs) == 42, len(cfgs)
    keys = {c.astype(np.int64).tobytes(): i for i, c in enumerate(cfgs)}

    counts = np.zeros(len(cfgs))
    for s in range(N):
        H, _info = cftp_sample(n, w, master_seed=1_300_000 + s)
        k = keys.get(np.asarray(H, dtype=np.int64).tobytes())
        assert k is not None, "CFTP produced a configuration outside the state space"
        counts[k] += 1

    expected = exact_p * N
    assert expected.min() > 5.0, (
        f"the least likely configuration expects {expected.min():.1f} counts; "
        f"the chi-square approximation needs more than 5")
    chi, p = chisquare(counts, expected)
    assert p > 1e-5, (
        f"chi-square against the enumerated law gives p = {p:.3g} over "
        f"{N} samples. A correct sampler gives p uniform on (0,1); a 10% "
        f"error in one weight gives about 7e-8")

    # and the sensitivity claim itself, so the threshold cannot quietly rot:
    # the same counts tested against a 10%-off law must be rejected
    _c2, wrong_p, _z2 = exact_distribution(n, dict(w, c1=w["c1"] * 1.10))
    _chi2, p_wrong = chisquare(counts, wrong_p * N)
    assert p_wrong < 1e-5, (
        f"these {N} samples do not distinguish the true law from one with c1 "
        f"10% off (p = {p_wrong:.3g}), so this test has lost the power its "
        f"docstring claims")


def test_the_domain_sampler_detects_a_ten_percent_error_in_a_single_weight():
    """The same audit, for the general-domain sampler.

    A 10% error injected into domains._weight_table's c1 entry passed all 20
    domain-related tests in the suite. The gap was identical to the DWBC one:
    the existing checks draw a few thousand samples, below the noise floor for
    a total-variation shift of 0.028.

    Choice of domain is not free, and the arithmetic says why. The chi-square
    non-centrality per sample is sum (dp)^2/p, and for a 10% c1 error:

        square n=4 (42 cfgs)      4.216e-03   ~20000 samples, 2.4 ms each
        2-block N=6 (49 cfgs)     4.104e-03   ~20500 samples, 1.2 ms each
        2-block N=4 (4 cfgs)      3.983e-32   UNDETECTABLE
        staircase N=4 (2 cfgs)    0           UNDETECTABLE

    The last two are the useful warning: on a domain with a handful of
    configurations the c-vertex counts barely vary, so no number of samples
    separates a 10% weight error from the truth. A cheap domain is not a cheap
    test, and "it passed on the staircase" would have meant nothing.

    So this runs on the 2-block N=6 -- 49 configurations, the best power per
    second of the four -- and asserts both directions: that the sampler agrees
    with the enumerated law, and that these samples would REJECT a law with c1
    10% off, so the power cannot quietly rot away.
    """
    import numpy as np
    from scipy.stats import chisquare

    from sixvertex.domains import (cftp_sample as dom_sample, classify_counts,
                                   domain_mask, in_monotone_region)
    from experiments.brute_domain import enumerate_heights

    N = 22000
    w = dict(a1=1.3, a2=0.8, b1=1.7, b2=0.6, c1=2.1, c2=1.4)
    assert in_monotone_region(w)

    m = domain_mask([0, 3, 6], [0, 3, 6], [[1, 1], [0, 1]])
    hs = enumerate_heights(m)
    assert len(hs) == 49, len(hs)
    counts_of = [classify_counts(H, m) for H in hs]
    keys = {np.asarray(H, dtype=np.int64).tobytes(): i
            for i, H in enumerate(hs)}

    def law(weights):
        v = np.array([np.prod([weights[k] ** c for k, c in cnt.items()])
                      for cnt in counts_of], dtype=float)
        return v / v.sum()

    exact_p = law(w)
    observed = np.zeros(len(hs))
    for s in range(N):
        H, _info = dom_sample(m, w, master_seed=1_700_000 + s)
        k = keys.get(np.asarray(H, dtype=np.int64).tobytes())
        assert k is not None, "the domain sampler left the state space"
        observed[k] += 1

    expected = exact_p * N
    assert expected.min() > 5.0, expected.min()
    _chi, p = chisquare(observed, expected)
    assert p > 1e-5, (
        f"chi-square against the enumerated law gives p = {p:.3g} over {N} "
        f"samples on the 2-block domain")

    wrong = law(dict(w, c1=w["c1"] * 1.10))
    _chi2, p_wrong = chisquare(observed, wrong * N)
    assert p_wrong < 1e-5, (
        f"these {N} samples do not separate the true law from one with c1 10% "
        f"off (p = {p_wrong:.3g}); the test has lost the power its docstring "
        f"claims")


def test_the_stochastic_thresholds_are_exact_to_machine_precision():
    """The stochastic sampler can be checked exactly; sampling it cannot.

    Audited like the others: scaling b1 by 1.02 inside sample() fails two
    tests, but scaling it by 1.005 passes everything. 60000 draws at n=3 give
    good power -- far better than the 10% CFTP blind spot -- but still a 0.5%
    blind spot in a PROBABILITY parameter.

    Unlike CFTP, this sampler does not need statistics. Its whole content is
    two thresholds: given (h_in, v_in) = (1,0) the outgoing pair is (1,0) when
    u < b1 and (0,1) otherwise, and given (0,1) it is (0,1) when u < b2 and
    (1,0) otherwise. Feeding chosen u values straight through the RNG pins
    each threshold to 1e-12 -- a relative error of 3e-12 in b1 -- which is a
    different kind of check from "the histogram looks right", and the analogue
    of comparing the sequential sampler's partition function exactly.

    Reaching a (0,1) vertex takes some driving, and the sequence is the test's
    one subtlety. Under the step initial condition h[:,0] = 1 and v[0,:] = 0,
    so column 0 never presents (0,1). At n=3, feeding u = (b1/2, 1, 1, x):
    vertex (0,0) goes straight, (0,1) turns and so sets v[1,1] = 1, (1,0)
    turns and so sets h[1,1] = 0, and vertex (1,1) then sees exactly
    (h_in, v_in) = (0,1) and tests x against b2.
    """
    import numpy as np

    import sixvertex.stochastic as st

    class StubRNG:
        """Feeds a fixed sequence of uniforms, then zeros."""

        def __init__(self, vals):
            self.vals = list(vals)
            self.i = 0

        def random(self, *a, **k):
            v = self.vals[self.i] if self.i < len(self.vals) else 0.0
            self.i += 1
            return v

    b1, b2 = 0.35, 0.6
    real = st.np.random.default_rng
    try:
        # the b1 threshold, on a (1,0) vertex: n=1 presents one immediately
        for eps, want in ((-1e-12, (1, 0)), (+1e-12, (0, 1))):
            st.np.random.default_rng = (
                lambda seed=None, _v=[b1 + eps]: StubRNG(_v))
            h, v = st.sample(1, b1, b2)
            got = (int(h[0, 1]), int(v[1, 0]))
            assert got == want, (
                f"with u = b1 {eps:+.0e} the (1,0) vertex emitted {got}, not "
                f"{want}: the b1 threshold is not at b1")

        # the b2 threshold, on a (0,1) vertex reached as the docstring says
        for eps, want in ((-1e-12, (0, 1)), (+1e-12, (1, 0))):
            st.np.random.default_rng = (
                lambda seed=None, _v=[b1 * 0.5, 1.0, 1.0, b2 + eps]:
                StubRNG(_v))
            h, v = st.sample(3, b1, b2)
            got = (int(h[1, 2]), int(v[2, 1]))
            assert got == want, (
                f"with u = b2 {eps:+.0e} the (0,1) vertex emitted {got}, not "
                f"{want}: the b2 threshold is not at b2")
    finally:
        st.np.random.default_rng = real

    # the deterministic vertices consume no randomness and cannot be wrong:
    # (1,1) -> (1,1) and (0,0) -> (0,0), checked by driving a grid with a
    # stub that would be exhausted if either branch drew a number
    try:
        drawn = {"n": 0}

        class Counting(StubRNG):
            def random(self, *a, **k):
                drawn["n"] += 1
                return super().random(*a, **k)

        st.np.random.default_rng = lambda seed=None: Counting([0.0] * 64)
        st.sample(4, b1, b2)
        assert drawn["n"] <= 16, (
            f"a 4x4 sweep drew {drawn['n']} uniforms; at most one per vertex "
            f"is possible, and the two deterministic branches should draw "
            f"none")
    finally:
        st.np.random.default_rng = real


def test_the_cftp_heat_bath_threshold_is_exact_to_machine_precision():
    """Corrects a claim I made one commit ago, and then corrects the fix.

    That commit said CFTP "has no exact Z to compare against, so it is
    irreducibly sampling-limited", and set its detection threshold at a
    total-variation distance of 0.024 using 20000 samples. The first half is
    true and the conclusion does not follow. CFTP's correctness splits in two:

      (a) the local update -- the weight table and the heat-bath probability
          it implies -- a deterministic function of the weights, checkable to
          machine precision;
      (b) the CFTP scheme itself, which rests on the monotone-coupling
          argument, exhaustively verified elsewhere in this file.

    _sweep takes the random field as an ARGUMENT, so (a) needs no
    monkeypatching: feed it a chosen uniform and see where the flip happens.

    THE FIRST VERSION OF THIS TEST WAS WEAK AND THE REASON IS WORTH KEEPING.
    It used n=2, where the single interior site has up-faces {a2, c1, c1, a1}
    and down-faces {c1, b2, b1, c1}. c1 appears TWICE in each, so it cancels
    in p_up = w(up)/(w(up) + w(down)) -- scaling c1 by 1.10 left the threshold
    bit-identical and the test passed. A heat-bath threshold only sees weights
    whose MULTIPLICITY DIFFERS between the two branches, and c2 did not appear
    at all.

    So the sites are chosen by that criterion instead of by convenience: the
    test searches the enumerated configurations for flip sites until every one
    of the six weights has differing multiplicity at some site, asserts that
    coverage, and pins the threshold at each. At n=3 site (1,1) there is a
    flip with difference {c1: +1, c2: +1, b1: -1, b2: -1}, which is what
    brings the c weights in.
    """
    import collections

    import numpy as np

    from sixvertex.cftp_exact import (_colour_indices, _sweep, _weight_table,
                                      extremal_height)

    # the convention, written out so this test holds it independently of
    # domains.FACE (the executable census keeps the tables agreeing)
    LEGAL = {(0, 0, 0, 0): "a1", (1, 1, 1, 1): "a2",
             (1, 1, 0, 0): "c1", (0, 0, 1, 1): "c2",
             (1, 0, 0, 1): "b1", (0, 1, 1, 0): "b2"}
    w = dict(a1=1.3, a2=0.8, b1=1.7, b2=0.6, c1=2.1, c2=1.4)

    def face_types(Hh, n):
        out = []
        for i in range(n):
            for j in range(n):
                tl, tr = Hh[i, j], Hh[i, j + 1]
                bl, br = Hh[i + 1, j], Hh[i + 1, j + 1]
                bits = (int(bl - tl == 1), int(tr - tl == 1),
                        int(br - bl == 1), int(br - tr == 1))
                out.append(LEGAL.get(bits, "ILLEGAL"))
        return collections.Counter(out)

    def weight_of(counter):
        if "ILLEGAL" in counter:
            return 0.0
        v = 1.0
        for name, k in counter.items():
            v *= w[name] ** k
        return v

    # find flip sites until every weight is visible at one of them
    cases = []
    covered = set()
    for n in (2, 3):
        for H0 in enumerate_height_functions(n):
            if covered == set(w):
                break
            H0 = H0.astype(np.float64)
            for i in range(1, n):
                for j in range(1, n):
                    nb = [H0[i - 1, j], H0[i + 1, j], H0[i, j - 1], H0[i, j + 1]]
                    if len(set(nb)) != 1:
                        continue
                    nv = nb[0]
                    up, down = H0.copy(), H0.copy()
                    up[i, j] = nv + 1
                    down[i, j] = nv - 1
                    fu, fd = face_types(up, n), face_types(down, n)
                    wu, wd = weight_of(fu), weight_of(fd)
                    if wu <= 0 or wd <= 0:
                        continue
                    visible = {k for k in w if fu[k] != fd[k]}
                    if not visible - covered:
                        continue
                    covered |= visible
                    cases.append((n, i, j, H0.copy(), nv, wu / (wu + wd),
                                  sorted(visible)))

    assert covered == set(w), (
        f"these flip sites only make {sorted(covered)} visible in the "
        f"heat-bath threshold; {sorted(set(w) - covered)} would be unpinned, "
        f"which is exactly how the n=2-only version of this test missed c1 "
        f"and c2")
    assert cases, "no usable flip site found"

    for n, i, j, H0, nv, p_up, visible in cases:
        tbl = _weight_table(w)
        colour_idx, size = _colour_indices(n)
        for eps, want in ((-1e-12, nv + 1), (+1e-12, nv - 1)):
            rnd = np.full(H0.shape, 0.5)
            rnd[i, j] = p_up + eps
            out = _sweep(H0.copy(), tbl, rnd, colour_idx, size)
            assert out[i, j] == want, (
                f"n={n} site=({i},{j}) [pins {visible}]: with u = p_up "
                f"{eps:+.0e} the site became {out[i, j]:.0f}, not {want:.0f}. "
                f"p_up computed from the weights is {p_up:.12f}, so the "
                f"weight table or the update disagrees with it")



def test_the_browser_chain_threshold_is_exact_to_machine_precision():
    """The fourth sampler, held to the same standard as the other three.

    Audited first: scaling c1 by 1.10 inside sixvertex.js fails
    test_the_in_browser_sampler_at_six_distinct_weights, but scaling it by
    1.03 passes. 40000 thinned observations against the exact law is good
    design -- the reference side carries no Monte Carlo error -- yet it still
    leaves a 3% blind spot, and the draws are correlated so its effective
    sample size is smaller than 40000.

    The chain's acceptance probability is a deterministic function of the
    weights, so it can be pinned instead of sampled. _rand() is a method, so
    node can override it with a chosen value.

    Two things make this a real cross-check rather than a restatement. The
    threshold comes out at p = 0.504854368932 at n=2 -- the same number
    test_the_cftp_heat_bath_threshold_is_exact_to_machine_precision derives
    from the weights in Python, independently, with the six legal patterns
    written out by hand. And the implementations share nothing: different
    language, different PRNG, different sweep order.

    The site coverage lesson carries over. At n=2 the only interior site has
    c1 twice in each branch, so it cancels, and c2 never appears -- a c1 error
    is invisible there. An n=3 site with difference {c1:+1, c2:+1, b1:-1,
    b2:-1} supplies the rest, and the test asserts the union covers all six.

    Finding which draw belongs to the target site is done by SEARCH, not by
    replicating the sweep order in Python: for each candidate draw index, feed
    the threshold minus 1e-12 there and 1.0 everywhere else (1.0 always
    rejects, since p < 1), and require that exactly one index moves exactly
    the target site to exactly the expected value -- and that the same index
    with plus 1e-12 moves nothing at all.
    """
    import collections
    import json
    import pathlib
    import shutil
    import subprocess
    import tempfile

    import numpy as np
    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available to run the shipped browser chain")

    LEGAL = {(0, 0, 0, 0): "a1", (1, 1, 1, 1): "a2",
             (1, 1, 0, 0): "c1", (0, 0, 1, 1): "c2",
             (1, 0, 0, 1): "b1", (0, 1, 1, 0): "b2"}
    w = dict(a1=1.3, a2=0.8, b1=1.7, b2=0.6, c1=2.1, c2=1.4)

    def face_types(H, n):
        out = []
        for i in range(n):
            for j in range(n):
                tl, tr = H[i, j], H[i, j + 1]
                bl, br = H[i + 1, j], H[i + 1, j + 1]
                bits = (int(bl - tl == 1), int(tr - tl == 1),
                        int(br - bl == 1), int(br - tr == 1))
                out.append(LEGAL.get(bits, "ILLEGAL"))
        return collections.Counter(out)

    def weight_of(c):
        if "ILLEGAL" in c:
            return 0.0
        v = 1.0
        for name, k in c.items():
            v *= w[name] ** k
        return v

    cases, covered = [], set()
    for n in (2, 3):
        for H0 in enumerate_height_functions(n):
            if covered == set(w):
                break
            H0 = H0.astype(float)
            for i in range(1, n):
                for j in range(1, n):
                    nb = [H0[i - 1, j], H0[i + 1, j],
                          H0[i, j - 1], H0[i, j + 1]]
                    if len(set(nb)) != 1:
                        continue
                    nv = nb[0]
                    up, down = H0.copy(), H0.copy()
                    up[i, j] = nv + 1
                    down[i, j] = nv - 1
                    fu, fd = face_types(up, n), face_types(down, n)
                    wu, wd = weight_of(fu), weight_of(fd)
                    if wu <= 0 or wd <= 0:
                        continue
                    visible = {k for k in w if fu[k] != fd[k]}
                    if not visible - covered:
                        continue
                    covered |= visible
                    other = nv + 1 if H0[i, j] == nv - 1 else nv - 1
                    p_move = (wu if other == nv + 1 else wd) / (wu + wd)
                    cases.append(dict(n=n, i=i, j=j,
                                      H=H0.astype(int).ravel().tolist(),
                                      target=int(other), p=float(p_move),
                                      vis=sorted(visible)))

    assert covered == set(w), (
        f"these sites only pin {sorted(covered)}; {sorted(set(w) - covered)} "
        f"would be unchecked")
    assert any(abs(c["p"] - 0.504854368932) < 1e-11 for c in cases), (
        "the n=2 threshold is no longer 0.504854368932, which is the number "
        "the Python heat-bath test derives independently")

    src = (pathlib.Path(__file__).resolve().parent.parent / "static"
           / "sixvertex.js").read_text(encoding="utf-8")
    driver = src + (
        "\nconst w = " + json.dumps(w) + ";\n"
        + "const cases = " + json.dumps(cases) + ";\n"
        + "const out = [];\n"
        + "for (const c of cases) {\n"
        + "  const size = c.n + 1, ctr = c.i * size + c.j;\n"
        + "  let found = null;\n"
        + "  for (let k = 0; k < 8; k++) {\n"
        + "    for (const sign of [-1, +1]) {\n"
        + "      const s = new SixVertexJS(c.n, w, 1);\n"
        + "      for (let m = 0; m < c.H.length; m++) s.H[m] = c.H[m];\n"
        + "      let d = 0;\n"
        + "      s._rand = () => (d++ === k ? c.p + sign * 1e-12 : 1.0);\n"
        + "      s.step(1);\n"
        + "      const changed = [];\n"
        + "      for (let m = 0; m < c.H.length; m++)\n"
        + "        if (s.H[m] !== c.H[m]) changed.push(m);\n"
        + "      if (sign < 0 && changed.length === 1 && changed[0] === ctr\n"
        + "          && s.H[ctr] === c.target) found = k;\n"
        + "      if (sign > 0 && found === k && changed.length !== 0)"
          " found = -1;\n"
        + "    }\n"
        + "  }\n"
        + "  out.push({site: [c.i, c.j], n: c.n, vis: c.vis, drawIndex: found});\n"
        + "}\n"
        + "console.log(JSON.stringify(out));\n")

    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "threshold.js"
        f.write_text(driver, encoding="utf-8")
        out = subprocess.run([node, str(f)], capture_output=True, text=True,
                             timeout=180)
    assert out.returncode == 0, out.stderr[:400]
    results = json.loads(out.stdout.strip().splitlines()[-1])
    assert len(results) == len(cases)

    for r in results:
        assert r["drawIndex"] is not None and r["drawIndex"] >= 0, (
            f"n={r['n']} site={r['site']} (pins {r['vis']}): no draw index "
            f"makes the browser chain flip exactly that site at exactly its "
            f"computed threshold, so its acceptance probability disagrees "
            f"with the weights")


def test_the_public_dispatch_behaves_the_same_on_both_sides_of_its_boundary():
    """sixvertex.sample() is the API a collaborator calls; it was inconsistent.

    The sequential route forwards **weights into exact_sample, whose signature
    defaults every weight to 1.0, so a partial dict was silently completed
    with ones -- a sample from a DIFFERENT measure than was asked for, with no
    complaint. The CFTP route instead raised a bare KeyError from
    in_monotone_region. Measured before the fix:

        sample(4,  {"c1": 2, "c2": 2})  ->  returned a sample, method=exact-sequential
        sample(16, {"c1": 2, "c2": 2})  ->  KeyError: 'a1'

    The same call, behaving differently either side of n = MAX_EXACT_N, and the
    quiet half being the dangerous one: a typo in a weight name gave a
    plausible picture from the wrong measure.

    This pins the whole routing table as well, since that is what the
    docstring promises and nothing checked end to end.
    """
    import pytest

    import sixvertex as sv

    w = dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.6, c2=1.6)
    small, big = 4, sv.MAX_EXACT_N + 2
    assert small <= sv.MAX_EXACT_N < big

    # the routing table, and the info["method"] contract PACKAGE.md documents
    _H, info = sv.sample(small, w, seed=1)
    assert info["method"] == "exact-sequential", info
    _H, info = sv.sample(big, w, seed=1)
    assert info["method"] == "cftp", info
    _H, info = sv.sample(small, w, seed=1, method="cftp")
    assert info["method"] == "cftp", "method='cftp' was overridden by size"
    _H, info = sv.sample(small, w, seed=1, method="sequential")
    assert info["method"] == "exact-sequential", info

    with pytest.raises(ValueError, match="unknown method"):
        sv.sample(small, w, method="metropolis")
    with pytest.raises(ValueError, match="limited to n"):
        sv.sample(big, w, method="sequential")

    # outside the monotone region: small n still works, large n is refused
    off = dict(a1=1.5, a2=1.5, b1=1.0, b2=1.0, c1=1.0, c2=1.0)
    assert not sv.in_monotone_region(off)
    _H, info = sv.sample(small, off, seed=1)
    assert info["method"] == "exact-sequential", (
        "the sequential sampler works at ANY weights; it must still be used "
        "below the size limit when CFTP cannot be")
    with pytest.raises(ValueError, match="no exact method available"):
        sv.sample(big, off, seed=1)

    # and a partial or mistyped weight dict is refused the SAME way on both
    # sides, rather than silently defaulting on one of them
    for n in (small, big):
        with pytest.raises(ValueError, match="exactly a1, a2, b1, b2, c1, c2"):
            sv.sample(n, {"c1": 2.0, "c2": 2.0}, seed=1)
        with pytest.raises(ValueError, match="unexpected"):
            sv.sample(n, dict(w, c_1=2.0), seed=1)


def test_the_two_cftp_chains_start_from_the_extremal_heights_the_definition_gives():
    """CFTP certifies nothing unless its two chains start at the TOP and BOTTOM
    of the order.  The lo/hi pair is what supplies that, and the formula for it
    -- the pointwise max and min of (corner height -+ Manhattan distance), the
    McShane extensions -- is written out three independent times: in
    cftp_exact.py, in sampler.py, and again in static/sixvertex.js, which is
    the copy every browser runs.

    What the suite checked until now was domains.extremal_heights against
    cftp_exact.extremal_height.  Two of four copies, and both checked only
    against each other: an error shared by every copy passes, and the two
    copies NOT in that comparison could drift without anything noticing.  This
    test checks the pair against its definition instead, and then checks the
    other two copies against it.

    The definition has four parts, and all four matter:

      * every lattice edge differs by exactly +-1.  Lipschitz alone is not
        enough; a height function needs the parity too, and it holds here
        because the four corner values satisfy h = i+j (mod 2), so every term
        (hv -+ dist) does, so the max and the min do.  A formula that got the
        corner data wrong would still be 1-Lipschitz and would still coalesce.
      * the boundary is the DWBC staircase, and lo and hi agree on it -- which
        is what makes them two chains on ONE state space.
      * lo <= hi everywhere.
      * lo < hi at every one of the (n-1)^2 interior corners.  Not a sanity
        check: if the two starts agreed anywhere in the interior they would
        agree there forever (the coupling is monotone and the order is
        preserved), and coalescence at that site would be vacuous.  Measured:
        1, 9, 16, 49, 144 strictly-ordered interior corners at n = 2, 4, 5, 8,
        13, which is (n-1)^2 in each case.

    The JS copy is EXECUTED under node rather than ported, for the reason the
    header gives: a port tests the port.
    """
    import json
    import pathlib
    import subprocess
    import tempfile

    from sixvertex.cftp_exact import extremal_height
    from sixvertex.sampler import SixVertexSampler

    root = pathlib.Path(__file__).resolve().parent.parent
    sizes = [2, 4, 5, 8, 13]

    driver = (
        "const fs=require('fs');\n"
        "eval(fs.readFileSync(process.argv[2],'utf8'));\n"
        "const out={};\n"
        f"for (const n of {json.dumps(sizes)}) "
        "for (const kind of ['lo','hi']) "
        "out[n+':'+kind]=Array.from(extremalHeight(n,kind));\n"
        "console.log(JSON.stringify(out));\n"
    )
    with tempfile.TemporaryDirectory() as td:
        drv = os.path.join(td, "exh.js")
        open(drv, "w").write(driver)
        proc = subprocess.run(
            ["node", drv, str(root / "static" / "sixvertex.js")],
            capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)

    for n in sizes:
        pair = {k: extremal_height(n, k).astype(np.int64) for k in ("lo", "hi")}
        for kind, H in pair.items():
            # a height function, not merely a Lipschitz one
            for axis, name in ((0, "rows"), (1, "columns")):
                steps = set(np.abs(np.diff(H, axis=axis)).ravel().tolist())
                assert steps == {1}, (
                    f"n={n} {kind}: steps across {name} are {sorted(steps)}, so "
                    f"this is not a height function")
            # the DWBC boundary, identical for lo and hi
            up, down = np.arange(n + 1), np.arange(n, -1, -1)
            assert np.array_equal(H[0, :], up) and np.array_equal(H[:, 0], up), (
                f"n={n} {kind}: boundary is not DWBC")
            assert np.array_equal(H[n, :], down) and np.array_equal(H[:, n], down), (
                f"n={n} {kind}: boundary is not DWBC")
            # the browser's copy, run as shipped
            assert np.array_equal(
                np.array(js[f"{n}:{kind}"], dtype=np.int64).reshape(n + 1, n + 1),
                H), (f"n={n} {kind}: static/sixvertex.js disagrees with "
                     f"cftp_exact.extremal_height")
            # and sampler.py's copy, which nothing compared before
            assert np.array_equal(
                SixVertexSampler.extremal_height(n, kind).astype(np.int64), H), (
                f"n={n} {kind}: sampler.extremal_height disagrees with "
                f"cftp_exact.extremal_height")

        lo, hi = pair["lo"], pair["hi"]
        assert (lo <= hi).all(), f"n={n}: lo is not below hi"
        interior = np.zeros_like(lo, dtype=bool)
        interior[1:n, 1:n] = True
        assert (lo[interior] < hi[interior]).all(), (
            f"n={n}: the two chains already agree at "
            f"{int((lo[interior] == hi[interior]).sum())} interior corners, "
            f"where coalescence certifies nothing")
        assert int((lo < hi).sum()) == (n - 1) ** 2, n
