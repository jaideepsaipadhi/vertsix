from flask import Flask, request, jsonify, send_from_directory
import base64
import math
from collections import OrderedDict
import os
import threading
import time
import uuid

from sixvertex.sampler import SixVertexSampler
import numpy as np

from sixvertex.cftp_exact import (cftp_sample as cftp_exact_sample,
                                  in_monotone_region)
from sixvertex.exact import exact_sample, MAX_EXACT_N
from sixvertex import stochastic as _stochastic
from sixvertex import domains as _domains

app = Flask(__name__, static_folder="static", static_url_path="")

_lock = threading.Lock()

# Session-scoped sampler state.
#
# This used to be a single module-level sampler shared by every caller:
# `_state = {"sampler": None}`. Two clients hitting /api/init would clobber
# one another, and /api/step would hand back the OTHER client's model --
# demonstrably so (A initialised n=6, B initialised n=30, A's next /api/step
# returned n=30). The browser UI does not use these endpoints (live sampling
# is client-side), but they are public and scriptable, so returning another
# caller's data is a correctness bug regardless.
_SESSION_TTL_SECONDS = 30 * 60
_MAX_SESSIONS = 32
_MAX_CONCURRENT_JOBS = 3
_JOB_MAX_RUNTIME_SECONDS = 2 * 3600
_sessions = OrderedDict()          # session_id -> {"sampler":..., "touched":...}


def _touch_session(sid):
    entry = _sessions.get(sid)
    if entry is not None:
        entry["touched"] = time.time()
        _sessions.move_to_end(sid)
    return entry


def _prune_sessions():
    now = time.time()
    for sid in [k for k, v in _sessions.items()
                if now - v["touched"] > _SESSION_TTL_SECONDS]:
        del _sessions[sid]
    while len(_sessions) > _MAX_SESSIONS:
        _sessions.popitem(last=False)

# Background-job store for /api/exact. Rendered on a free/shared hosting
# tier, a single CFTP request can legitimately take minutes at large n in
# deep ferroelectric/antiferroelectric regimes -- long enough that the
# platform's own request timeout kills the connection before our code can
# respond, even though the computation itself is completely correct and
# would have finished (verified directly: n=140 at Delta=-3 coalesces in
# ~2 minutes on ordinary hardware, well past what a synchronous HTTP
# request can wait for on typical free-tier hosts).
#
# So /api/exact/start returns almost immediately with a job id, the actual
# CFTP run happens in a background thread, and the frontend polls
# /api/exact/status/<job_id> every second or two -- each individual HTTP
# request stays short regardless of how long the underlying computation
# takes.
_jobs_lock = threading.Lock()
_jobs = {}
_JOB_TTL_SECONDS = 30 * 60


def _cleanup_old_jobs():
    """Expire only FINISHED jobs.

    Previously this expired by `created_at` regardless of status, so a job
    that was still running could be evicted purely for being old -- its
    worker thread would keep burning CPU, its result would be discarded on
    completion (the writer checks `job_id in _jobs`), and the client polling
    it would get "unknown or expired job". That is not hypothetical for the
    large-n use case: exact sampling at n in the low hundreds can legitimately
    exceed the TTL.

    Terminal jobs are expired relative to when they finished, so a result
    stays retrievable for the full TTL after it becomes available rather than
    from when the work started.
    """
    now = time.time()

    # Watchdog: reap jobs stuck in "running".
    #
    # Only expiring FINISHED jobs (above) means a job whose thread died
    # without writing a terminal status -- an OOM kill, or any exit that
    # skips both handlers -- stays "running" forever. Combined with the
    # concurrency cap that is fatal rather than untidy: such a job holds a
    # slot permanently, and _MAX_CONCURRENT_JOBS of them disable exact
    # sampling server-wide until a restart. Verified: three six-hour-old
    # "running" entries with no live threads returned 429 to every new
    # request and survived every cleanup.
    #
    # The threshold is deliberately far above any legitimate runtime (the
    # browser itself gives up after 45 minutes), so this only ever catches
    # jobs that really are abandoned.
    for jid, j in list(_jobs.items()):
        if (j.get("status") == "running"
                and (now - j["created_at"]) > _JOB_MAX_RUNTIME_SECONDS):
            j["status"] = "error"
            j["finished_at"] = now
            j["error"] = ("job abandoned: no result after "
                          f"{_JOB_MAX_RUNTIME_SECONDS // 3600}h "
                          "(the worker is presumed dead)")

    stale = [
        jid for jid, j in _jobs.items()
        if j.get("status") in ("done", "error")
        and (now - j.get("finished_at", j["created_at"])) > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        del _jobs[jid]


class _JobDropped(Exception):
    """Raised inside a worker when its job entry has been removed."""


def _run_exact_job(job_id, n, a1, a2, b1, b2, c1, c2, seed):
    """Run exact sampling in the background.

    Routing:
      - n <= MAX_EXACT_N: use the exact SEQUENTIAL sampler (exact.py). This
        is correct for ARBITRARY weights and has no mixing-time or
        monotonicity requirement, so it is always preferred when affordable.
      - larger n inside the monotone region: CFTP. This description used to
        say "weights at the uniform point: fall back to CFTP, which is only
        valid there", which was the superseded criterion -- the code below
        has called in_monotone_region since the four-face rule replaced the
        hardcoded p_up = c1/(c1+c2), and the region c1c2 >= a1a2,
        c1c2 >= b1b2 is far larger than the point.
      - larger n outside that region: refuse, rather than return a result we
        cannot justify. No coupling of the update is monotone there, so
        coalescence would certify nothing.
    """
    def progress_cb(T, attempts, coalesced):
        # A worker whose entry has been removed stops here rather than
        # running on and writing a terminal status, which would put the
        # entry back in the table after something deliberately took it out.
        #
        # This is hygiene, not a CPU saving, and the distinction is worth
        # recording because it is easy to assume otherwise: the callback
        # fires once per DOUBLING ATTEMPT, and the last attempt dominates the
        # total, so dropping an entry part-way through one buys almost
        # nothing. Measured at n=96 and n=128, a job dropped 0.3s in ran for
        # 0.92x and 1.01x the time of an uninterrupted one. Bounding the
        # waste would need a check inside the sweep loop.
        #
        # Nor does the server currently drop running jobs: _cleanup_old_jobs
        # expires only FINISHED ones, on purpose. So this path is reached by
        # a deliberate removal, and the property it protects is that such a
        # removal sticks.
        with _jobs_lock:
            if job_id not in _jobs:
                raise _JobDropped(job_id)
            _jobs[job_id]["last_T"] = T
            _jobs[job_id]["attempts"] = attempts

    try:
        w = {"a1": a1, "a2": a2, "b1": b1, "b2": b2, "c1": c1, "c2": c2}

        if n <= MAX_EXACT_N:
            H, info = exact_sample(n, a1=a1, a2=a2, b1=b1, b2=b2,
                                   c1=c1, c2=c2, seed=seed)
            H = H.astype(np.float32)
        elif in_monotone_region(w):
            # CFTP with the correct four-face rule is valid on the whole
            # region c1c2 >= a1a2 and c1c2 >= b1b2, not merely at the uniform
            # point. Outside that region no coupling of the single-site update
            # is monotone (the marginals are not ordered), so there is no CFTP
            # scheme to fall back on -- hence the refusal below rather than a
            # slower method.
            H, info = cftp_exact_sample(n, w, master_seed=seed,
                                        max_T=1 << 21, progress_cb=progress_cb)
        else:
            A, B, C = w["a1"]*w["a2"], w["b1"]*w["b2"], w["c1"]*w["c2"]
            raise RuntimeError(
                f"No exact method is available here. The sequential sampler is "
                f"limited to n <= {MAX_EXACT_N} (its cost grows exponentially "
                f"with n), and CFTP requires c1c2 >= a1a2 and c1c2 >= b1b2 "
                f"(here A={A:.4g}, B={B:.4g}, C={C:.4g}); outside that region "
                f"no monotone coupling of the update exists. Reduce n to "
                f"{MAX_EXACT_N} or below, or choose weights with c1c2 "
                f"dominant."
            )

        s = SixVertexSampler(n=n, c_up=c1, c_down=c2, a1=a1, a2=a2, b1=b1, b2=b2)
        if s.use_torch:
            import torch
            s.H = torch.from_numpy(np.asarray(H, dtype=np.float32)).to(s.device)
        else:
            s.H = np.asarray(H, dtype=np.float32)
        frame = s.to_binary_frame()
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["finished_at"] = time.time()
                _jobs[job_id]["frame"] = frame
                _jobs[job_id]["info"] = info
    except _JobDropped:
        # The entry went away while this ran; nothing to report it to, and
        # nothing to report. Exit quietly rather than writing a status that
        # would resurrect the job in the table.
        return
    except (RuntimeError, ValueError) as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["finished_at"] = time.time()
                _jobs[job_id]["error"] = str(e)
    except Exception as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["finished_at"] = time.time()
                _jobs[job_id]["error"] = f"unexpected error: {e}"


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


class BadRequest(Exception):
    """Client sent something we can't act on. Turned into a 400."""


def _normalise_seed(raw):
    """Accept any integer seed, including a negative one.

    numpy's default_rng and SeedSequence take non-negative integers only, and
    the raw value went straight to them: a seed of -5 raised "expected
    non-negative integer" deep inside numpy. That surfaced as a failed job on
    the two job endpoints and, on /api/stochastic, as an unhandled exception
    -- a 500 whose HTML body reached the browser's .json() and produced
    "Unexpected token '<'". Typing a negative number into a box captioned
    "seed (blank = random)" is not an abuse of the API.

    Non-negative seeds are passed through exactly, so nothing that worked
    before changes. A negative one is mapped into [0, 2^32), deterministically
    -- the same negative seed always gives the same sample.

    The modulus is 2^32 rather than 2^64 so the result stays below 2^53 and
    is therefore exact as a JavaScript number. With 2^64, -5 came back as
    18446744073709552000, which is not the value the server used and not a
    number anyone could retype to reproduce the figure: the page would have
    been reporting an unusable seed.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise BadRequest(f"seed must be an integer, got {raw!r}")
    if isinstance(raw, float) and not math.isfinite(raw):
        raise BadRequest(f"seed must be a finite integer, got {raw!r}")
    try:
        v = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise BadRequest(f"seed must be an integer, got {raw!r}")
    return v if v >= 0 else v % (1 << 32)


def _parse_request(data, n_min=4, n_max=400, n_default=40):
    """Parse and validate n + the six weights from a request body.

    Without this, a non-numeric field raised a ValueError inside the view and
    surfaced as a 500 with a stack trace, and non-positive weights were
    accepted outright -- the Boltzmann distribution isn't defined for those,
    so the sampler would return a configuration that looked fine but meant
    nothing.
    """
    if not isinstance(data, dict):
        raise BadRequest("request body must be a JSON object")

    def num(name, default):
        v = data.get(name, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise BadRequest(f"{name} must be a number, got {v!r}")
        if not math.isfinite(v):
            raise BadRequest(f"{name} must be finite, got {v!r}")
        return v

    try:
        n_raw = data.get("n", n_default)
        # int(float("inf")) raises OverflowError, not ValueError, so an
        # unchecked inf escaped this handler and became a 500.
        if isinstance(n_raw, float) and not math.isfinite(n_raw):
            raise BadRequest(f"n must be a finite integer, got {n_raw!r}")
        n = int(n_raw)
    except (TypeError, ValueError, OverflowError):
        raise BadRequest(f"n must be an integer, got {data.get('n')!r}")
    if n < n_min or n > n_max:
        raise BadRequest(f"n must be between {n_min} and {n_max}, got {n}")

    weights = {
        "a1": num("a1", 1.0), "a2": num("a2", 1.0),
        "b1": num("b1", 1.0), "b2": num("b2", 1.0),
        "c1": num("c_up", 1.0), "c2": num("c_down", 1.0),
    }
    for name, v in weights.items():
        if v <= 0.0:
            raise BadRequest(
                f"weight {name} must be strictly positive (the Boltzmann "
                f"distribution is undefined otherwise), got {v}"
            )
    return n, weights, _normalise_seed(data.get("seed"))


@app.route("/api/stochastic", methods=["POST"])
def api_stochastic():
    """Stochastic six-vertex model, step initial condition, free exit.

    Synchronous rather than a background job: the weights are conditional
    probabilities, so one sweep of the lattice gives an exact sample in
    O(n^2) with no Markov chain -- n=300 takes about 0.05 s. Queueing it
    would cost more than running it.
    """
    data = request.get_json(silent=True) or {}
    n_raw = data.get("n", 64)
    try:
        if isinstance(n_raw, float) and not math.isfinite(n_raw):
            raise ValueError(n_raw)
        n = int(n_raw)
    except (TypeError, ValueError, OverflowError):
        return jsonify({"ok": False,
                        "error": f"n must be a finite integer, got {n_raw!r}"}), 400
    if not 4 <= n <= 400:
        return jsonify({"ok": False, "error": "n must be between 4 and 400"}), 400
    try:
        b1 = float(data.get("b1", 0.3))
        b2 = float(data.get("b2", 0.7))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "b1, b2 must be numbers"}), 400
    if not (0.0 < b1 < 1.0 and 0.0 < b2 < 1.0):
        return jsonify({"ok": False, "error":
                        "b1 and b2 must lie strictly between 0 and 1; they "
                        "are conditional probabilities, not Boltzmann "
                        f"weights (got b1={b1}, b2={b2})"}), 400
    try:
        seed = _normalise_seed(data.get("seed"))
    except BadRequest as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    h, v = _stochastic.sample(n, b1, b2, seed=seed)
    # Rows reversed so that math row 0 (the bottom, where no arrows enter) is
    # sent as the LAST row and therefore drawn at the bottom of the canvas.
    # The client draws row i at y=i with y growing downward, and the path view
    # applies the same flip to the arrows; without this the height view came
    # out as a vertical mirror of the path view of the very same sample.
    H = _stochastic.height_function(h, v).astype(np.int16)[::-1]
    lo, hi = int(H.min()), int(H.max())
    # The arrows, not just the height function. This model's height function
    # steps by 0 or 1, whereas the DWBC one steps by exactly +-1, so the DWBC
    # face classifier cannot read it -- routing it through that renderer
    # collapsed six vertex types down to two. The path view draws from these
    # instead: at vertex (i,j) the occupied edges are left=h[i,j],
    # bottom=v[i,j], right=h[i,j+1], top=v[i+1,j].
    return jsonify({
        "ok": True,
        "frame": {
            "n": n,
            "height_b64": base64.b64encode(H.tobytes()).decode("ascii"),
            "active_b64": base64.b64encode(
                np.zeros((n + 1) * (n + 1), dtype=np.uint8).tobytes()
            ).decode("ascii"),
            "h_b64": base64.b64encode(
                np.ascontiguousarray(h, dtype=np.uint8).tobytes()).decode("ascii"),
            "v_b64": base64.b64encode(
                np.ascontiguousarray(v, dtype=np.uint8).tobytes()).decode("ascii"),
            "min": lo, "max": hi,
        },
        "info": {
            "method": "stochastic-sequential",
            "delta": float(_stochastic.delta(b1, b2)),
            # Echo the seed the CALLER gave, not the normalised one, so the
            # page shows what to type to get this sample again.
            "b1": b1, "b2": b2, "seed": data.get("seed", seed),
        },
    })


@app.route("/api/config", methods=["GET"])
def api_config():
    """Authoritative client-facing constants, so the browser never has to
    keep its own copy of a rule the server enforces."""
    return jsonify({"ok": True, "max_exact_n": MAX_EXACT_N})


@app.route("/api/init", methods=["POST"])
def api_init():
    try:
        data = request.get_json(force=True, silent=True)
        n, w, seed = _parse_request(data)
    except BadRequest as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    with _lock:
        s = SixVertexSampler(
            n=n, c_up=w["c1"], c_down=w["c2"],
            a1=w["a1"], a2=w["a2"], b1=w["b1"], b2=w["b2"], seed=seed
        )
        session_id = uuid.uuid4().hex
        _sessions[session_id] = {"sampler": s, "touched": time.time()}
        _prune_sessions()
        # Report the ACTUAL condition governing exact sampling, and the
        # limit itself, so the client never has to duplicate the rule.
        # (The old "is_symmetric_regime" flag tested a1=a2=b1=b2=1, which
        # was the superseded CFTP criterion: it ignored c1,c2 entirely and
        # no longer corresponded to any decision the server makes.)
        _w = {"a1": w["a1"], "a2": w["a2"], "b1": w["b1"],
              "b2": w["b2"], "c1": w["c1"], "c2": w["c2"]}
        cftp_ok = in_monotone_region(_w)
        info = {
            "n": n,
            "device": str(s.device),
            # `use_torch` only means torch imported successfully -- with torch
            # installed on a CPU-only host it is True while the device is
            # "cpu", so reporting it as "using_gpu" was false. Report the
            # device actually in use, and a flag derived from it.
            "using_torch": s.use_torch,
            "using_gpu": "cuda" in str(s.device).lower(),
            "max_exact_n": MAX_EXACT_N,
            "exact_available": (n <= MAX_EXACT_N) or cftp_ok,
            "session_id": session_id,
        }
    return jsonify({"ok": True, **info, "frame": s.to_binary_frame()})


@app.route("/api/step", methods=["POST"])
def api_step():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
    try:
        sweeps = int(data.get("sweeps", 1))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "sweeps must be an integer"}), 400
    # Reject rather than clamp, matching how `n` is handled.
    #
    # This used to be `sweeps = max(1, min(sweeps, 500))`, so a request for
    # 9999 sweeps quietly ran 500 and returned success. The caller then
    # believes the chain is far more equilibrated than it is -- the same
    # silent-wrong-data failure as a picture labelled with parameters it was
    # not drawn from. Say no instead of quietly doing something else.
    _MAX_SWEEPS = 500
    if sweeps < 1 or sweeps > _MAX_SWEEPS:
        return jsonify({"ok": False, "error":
                        f"sweeps must be between 1 and {_MAX_SWEEPS}, got "
                        f"{sweeps}"}), 400

    # Bound the WORK, not just the sweep count.
    #
    # Cost scales like n^2 * sweeps. The per-parameter caps alone allowed
    # n=400 with sweeps=500, measured at ~44 s -- served synchronously, under
    # a lock, on a single gunicorn worker (see Procfile). One such request
    # stalls every other request on the site, including the status polls of
    # in-flight exact-sampling jobs. This endpoint is public, so that is a
    # trivially reachable denial of service, not just a slow path.
    #
    # ~1.8e6 work-units/second measured, so this budget keeps any single
    # request near two seconds.
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"ok": False, "error":
                        "session_id is required; call /api/init first and pass "
                        "the session_id it returns"}), 400
    with _lock:
        entry = _touch_session(session_id)
        if entry is None:
            return jsonify({"ok": False, "error":
                            "unknown or expired session_id; call /api/init "
                            "again"}), 400
        s = entry["sampler"]
        _MAX_STEP_WORK = 4_000_000          # n^2 * sweeps, ~2 s
        work = (s.n ** 2) * sweeps
        if work > _MAX_STEP_WORK:
            max_sweeps = max(1, _MAX_STEP_WORK // (s.n ** 2))
            return jsonify({"ok": False, "error":
                            f"request too large: n={s.n} with sweeps={sweeps} "
                            f"would run synchronously for too long. Use "
                            f"sweeps <= {max_sweeps} at this n, or call "
                            f"repeatedly."}), 400
        s.step(sweeps=sweeps)
        frame = s.to_binary_frame()
    return jsonify({"ok": True, "frame": frame})


@app.route("/api/exact/start", methods=["POST"])
def api_exact_start():
    # Now accepts ALL SIX weights: the exact sequential sampler (exact.py)
    # is correct for arbitrary weights at small n. _run_exact_job decides
    # which method is applicable and refuses rather than returning an
    # unjustified result.
    try:
        data = request.get_json(force=True, silent=True)
        n, w, seed = _parse_request(data, n_min=4, n_max=250)
    except BadRequest as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    a1, a2 = w["a1"], w["a2"]
    b1, b2 = w["b1"], w["b2"]
    c1, c2 = w["c1"], w["c2"]

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _cleanup_old_jobs()
        # Cap concurrent jobs.
        #
        # Each running job builds its own transfer-matrix tables in a
        # background thread; the CACHE is memory-bounded but these transient
        # builds are not. Measured: 6 concurrent n=13 jobs took the server
        # from 44 MB to 151 MB, and an n=14 build is ~3x larger again, so a
        # few dozen requests would exhaust a 512 MB instance. The browser only
        # ever runs one at a time, but /api/exact/start is public.
        running = sum(1 for j in _jobs.values() if j.get("status") == "running")
        if running >= _MAX_CONCURRENT_JOBS:
            return jsonify({"ok": False, "error":
                            f"too many exact-sampling jobs running "
                            f"({running}); try again shortly"}), 429
        _jobs[job_id] = {
            "status": "running",
            "created_at": time.time(),
            "last_T": None,
            "attempts": 0,
            "n": n,
        }

    thread = threading.Thread(
        target=_run_exact_job,
        args=(job_id, n, a1, a2, b1, b2, c1, c2, seed),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "job_id": job_id})


# ----------------------------------------------------------- general domains
#
# Gorin-Kenyon domains (arXiv:2408.14446, Section 2.1): the DWBC square with
# parts removed, encoded by cut sequences X, Y and a k x l binary matrix I.
# Fixed at k = l = 3 here, which is the size Gorin suggested starting from --
# the interface for general k and l is the awkward part, not the algorithm.
#
# Server-side only, by CFTP. The browser's live Markov chain works on a square
# grid; teaching it arbitrary domains is a much larger change than teaching
# the exact sampler, and the exact path is where a new domain belongs anyway.

DOMAIN_K = 3


def _parse_domain(data, n):
    """Cuts and I from the request, or a message saying what is wrong.

    Every rejection here is a message the user can act on. In particular an
    EMPTY domain is caught up front with domains.admits_configuration, which
    answers in one relaxation pass -- without it the sampler would simply
    never coalesce and the user would watch a progress bar forever.
    """
    def cuts(name):
        raw = data.get(name)
        if not isinstance(raw, list) or len(raw) != DOMAIN_K - 1:
            raise ValueError(f"{name} must be a list of {DOMAIN_K - 1} cut positions")
        vals = []
        for v in raw:
            try:
                vals.append(int(v))
            except (TypeError, ValueError):
                raise ValueError(f"{name} entries must be integers")
        out = [0] + vals + [n]
        if any(b <= a for a, b in zip(out, out[1:])):
            raise ValueError(f"{name} must be strictly increasing and inside 0..{n}"
                             f" (got {out})")
        return out

    X, Y = cuts("x_cuts"), cuts("y_cuts")
    raw = data.get("I")
    if (not isinstance(raw, list) or len(raw) != DOMAIN_K
            or any(not isinstance(r, list) or len(r) != DOMAIN_K for r in raw)):
        raise ValueError(f"I must be a {DOMAIN_K}x{DOMAIN_K} array of 0/1")
    I = [[1 if v else 0 for v in row] for row in raw]

    try:
        mask = _domains.domain_mask(X, Y, I)
    except _domains.DomainError as e:
        raise ValueError(str(e))
    if not _domains.admits_configuration(mask):
        raise ValueError(
            "this domain carries no valid configuration -- about half of the "
            "convex patterns do not, and whether one does depends on the cut "
            "positions as well as on I. Try moving a cut or filling in a block.")
    try:
        _domains.check_monotone_precondition(mask)
    except _domains.DomainError as e:
        raise ValueError(str(e))
    return X, Y, I, mask


def _run_domain_job(job_id, mask, w, seed):
    try:
        H, info = _domains.cftp_sample(mask, w, master_seed=seed,
                                       max_T=1 << 20)
        n = mask.shape[0]
        Hn = np.asarray(H, dtype=np.int16)
        cm = _domains.corner_mask(mask)
        frame = {
            "n": int(n),
            "height_b64": base64.b64encode(Hn.tobytes()).decode("ascii"),
            "active_b64": base64.b64encode(
                np.zeros((n + 1, n + 1), dtype=np.uint8).tobytes()).decode("ascii"),
            "domain_b64": base64.b64encode(
                mask.astype(np.uint8).tobytes()).decode("ascii"),
            "min": float(Hn[cm].min()),
            "max": float(Hn[cm].max()),
        }
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(status="done", finished_at=time.time(),
                                     frame=frame, info=info)
    except Exception as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(status="error", finished_at=time.time(),
                                     error=str(e))


@app.route("/api/domain/start", methods=["POST"])
def api_domain_start():
    try:
        data = request.get_json(force=True, silent=True) or {}
        n, w, seed = _parse_request(data, n_min=6, n_max=160)
        X, Y, I, mask = _parse_domain(data, n)
    except (BadRequest, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    if not in_monotone_region(w):
        A, B, C = w["a1"]*w["a2"], w["b1"]*w["b2"], w["c1"]*w["c2"]
        return jsonify({"ok": False, "error":
                        f"CFTP needs c1c2 >= a1a2 and c1c2 >= b1b2 "
                        f"(here A={A:.4g}, B={B:.4g}, C={C:.4g}); outside that "
                        f"region no monotone coupling of the update exists, so "
                        f"there is no exact method to fall back on."}), 400

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _cleanup_old_jobs()
        running = sum(1 for j in _jobs.values() if j.get("status") == "running")
        if running >= _MAX_CONCURRENT_JOBS:
            return jsonify({"ok": False, "error":
                            f"too many jobs running ({running}); try again "
                            f"shortly"}), 429
        _jobs[job_id] = {"status": "running", "created_at": time.time(),
                         "last_T": None, "attempts": 0, "n": n}
    threading.Thread(target=_run_domain_job,
                     args=(job_id, mask, w, seed), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/exact/status/<job_id>", methods=["GET"])
def api_exact_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "unknown or expired job"}), 404
        status = job["status"]
        response = {
            "ok": True,
            "status": status,
            "last_T": job["last_T"],
            "attempts": job["attempts"],
        }
        if status == "done":
            response["frame"] = job["frame"]
            response["info"] = job["info"]
        elif status == "error":
            response["error"] = job["error"]
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Six-vertex sampler running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
