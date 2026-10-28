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
    # Upward extremes must be refused, not silently wrong. Under the
    # corrected labels c is the weight the DWBC measure concentrates on, so
    # c=1e3 is still representable (Z ~ 2e216); the guard fires further out,
    # and on the a/b directions much sooner.
    for w in ({**UNIFORM, "c1": 1e30, "c2": 1e30},
              {**UNIFORM, "b1": 1e3,  "b2": 1e3},
              {**UNIFORM, "a1": 1e30, "a2": 1e30}):
        try:
            ExactSampler(12, w)
        except RuntimeError as e:
            assert "overflow" in str(e).lower()
        else:
            raise AssertionError(f"{w} should have tripped the overflow guard")

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

    exports = readme[readme.index("save PNG"):]
    exports = exports[:1200]
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
    """CFTP is monotone exactly on b1b2 >= a1a2 and b1b2 >= c1c2, not merely at
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
    """Pin the face-label convention to the literature.

    This code originally had b and c swapped relative to the standard
    six-vertex convention. The consequences were not cosmetic: the reported
    Delta had the wrong sign structure, and the monotone region was stated as
    b-dominant when it is c-dominant. Because the region is in fact the
    antiferroelectric-favouring one, Delta = -3 -- which a collaborator kept
    reporting as stuck -- is actually inside it and exactly samplable.

    The check: the Izergin-Korepin determinant for DWBC must reproduce our
    partition function in the SAME weight slots. IK is itself validated here
    by giving the ASM numbers at the ice point.
    """
    mp = pytest_importorskip_mpmath()
    if mp is None:
        return
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

    # (b) our sampler matches IK in the same slots at Delta = -3
    eta, lam = mp.acosh(3)/2, mp.mpf(0)
    a = float(mp.sinh(eta-lam)); b = float(mp.sinh(eta+lam)); c = float(mp.sinh(2*eta))
    assert abs((a*a + b*b - c*c) / (2*a*b) + 3.0) < 1e-9
    w = dict(a1=a, a2=a, b1=b, b2=b, c1=c, c2=c)
    for N in range(1, 5):
        ours = ExactSampler(N, w).partition_function()
        ik = Z_IK(N, eta, lam, False)
        assert abs(ours - ik) / ik < 1e-9, (
            f"N={N}: ours={ours} but IK={ik}; the b/c convention has slipped")


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
    assert 'if (!l && t && b && !r) return "b1"' in code, (
        "(t,b) must be a b-type; the rendering classifier has the old labels")


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
    survived. The correct one counts horizontal arrows crossing column j in
    rows above i: vertically the step is h[i,j] in {0,1}, and horizontally the
    sum telescopes through arrow conservation to v[0,j] - v[i,j].
    """
    import os, importlib
    from sixvertex import stochastic as st

    # height function must actually be one
    for n in (8, 24, 40):
        h, v = st.sample(n, 0.3, 0.7, seed=2)
        H = st.height_function(h, v)
        assert H.shape == (n + 1, n + 1)
        assert np.abs(np.diff(H, axis=1)).max() <= 1, "horizontal step > 1"
        assert np.abs(np.diff(H, axis=0)).max() <= 1, "vertical step > 1"

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
    assert js.count("seed: currentSeed()") >= 2, (
        "seed is not passed on every sampling path")


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
    """Reported: the badge flickered between its two messages several times a
    second at small n.

    Two causes. It compared two SINGLE samples and re-evaluated every
    animation frame, so ordinary sampling noise flipped the verdict; and the
    cutoff was a fixed 0.02, while the c-density's standard error falls like
    1/n -- at n=11 the c-fraction already moves in steps of 1/121 = 0.008.

    Now: an exponential moving average, a 600 ms repaint throttle, and a
    threshold scaling as 2.5/n with a floor. Measured after the fix, over six
    seconds of running: n=11 gave 0 verdict flips, n=40 gave 1 (the genuine
    one-way transition as it equilibrates), and a truly torpid case -- n=80,
    b=3, ferroelectric and outside the monotone region -- stayed solidly red
    with gap 0.529 against a threshold of 0.031.
    """
    import os
    here = os.path.dirname(__file__)
    js = open(os.path.join(here, "..", "static", "draw.js")).read()
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))
    start = code.index("function updateMixingBadge")
    body = code[start:start + 1800]

    assert "gapEMA" in body, "the badge uses the instantaneous gap; it will flicker"
    assert "0.9 * gapEMA" in body, "no smoothing of the gap"
    assert "lastBadgeUpdate" in body, "the badge repaints every animation frame"
    assert "2.5 / n" in body, (
        "the threshold does not scale with system size; a fixed cutoff fires "
        "constantly at small n and is too loose at large n")
    # a reset must start a fresh comparison rather than inherit the average
    li = code.index("function localInit")
    assert "gapEMA = null" in code[li:li + 800], (
        "reset does not clear the smoothed gap")


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
    assert "arrowFrame.h(i, j)" in js, "paths are not drawn from the arrows"


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
    if not os.path.isdir(exp):
        return

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
    assert "/ 2" in isl.split("def island")[1][:1600], (
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
    each cell against the six legal vertices. They disagreed once -- the
    height enumerator visited corners in row-major order and guessed a value
    at corners with no fixed neighbour yet, over-counting 5 configurations as
    2604. A square never exposes that, which is why the check runs on
    staircases too.
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

    Spot-checks of the 1364-domain sweep in which admits_configuration and
    exhaustive edge enumeration agreed everywhere, including cases where two
    earlier versions of the criterion failed in opposite directions: one
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
    corners of degree 1, 2 and 3 (122 of 1635 across a 213-domain scan), and
    at those the derivation does not apply. Every domain the sampler accepts
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
    """Exhaustive, not sampled -- see experiments/monotone_check.

    The full run covers 5 domains x 5 Delta values with zero violations,
    including a notch domain carrying 2889 configurations at each Delta. This
    keeps the cheap end of it in the suite.
    """
    from sixvertex.domains import domain_mask
    from experiments.mixing import weights_for_delta
    from experiments.monotone_check import monotonicity_violations

    cases = [([0, 4], [0, 4], [[1]]),
             ([0, 2, 4, 6], [0, 2, 4, 6], [[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
             ([0, 2, 4, 6], [0, 2, 4, 6], [[0, 1, 0], [1, 1, 1], [0, 1, 0]])]
    for d in (0.5, -3.0):
        w = weights_for_delta(d)
        for X, Y, I in cases:
            v = monotonicity_violations(domain_mask(X, Y, I), w)
            assert not v, (d, I, v)


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
        return
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
    same heavy b2 frozen region, which is real and not a rendering fault.

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
    frac = tot["b2"] / sum(tot.values())
    assert 0.50 < frac < 0.67, f"b2 fraction {frac:.3f} moved; expected ~0.58"


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
