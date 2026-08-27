from flask import Flask, request, jsonify, send_from_directory
import math
from collections import OrderedDict
import os
import threading
import time
import uuid

from sixvertex.sampler import SixVertexSampler
import numpy as np

from sixvertex.cftp import cftp_sample
from sixvertex.exact import exact_sample, MAX_EXACT_N

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


def _run_exact_job(job_id, n, a1, a2, b1, b2, c1, c2, seed):
    """Run exact sampling in the background.

    Routing:
      - n <= MAX_EXACT_N: use the exact SEQUENTIAL sampler (exact.py). This
        is correct for ARBITRARY weights and has no mixing-time or
        monotonicity requirement, so it is always preferred when affordable.
      - larger n, but weights at the uniform point: fall back to CFTP, which
        is only valid there (see cftp.py and ALGORITHM.md).
      - larger n with non-uniform weights: refuse, rather than return a
        result we cannot justify. CFTP's coupling provably does not target
        the right measure there.
    """
    def progress_cb(T, attempts, coalesced):
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["last_T"] = T
                _jobs[job_id]["attempts"] = attempts

    try:
        is_uniform = (a1 == 1.0 and a2 == 1.0 and b1 == 1.0 and b2 == 1.0
                      and c1 == 1.0 and c2 == 1.0)

        if n <= MAX_EXACT_N:
            H, info = exact_sample(n, a1=a1, a2=a2, b1=b1, b2=b2,
                                   c1=c1, c2=c2, seed=seed)
            H = H.astype(np.float32)
        elif is_uniform:
            H, info = cftp_sample(n=n, c_up=c1, c_down=c2,
                                   master_seed=seed, max_T=1 << 21,
                                   progress_cb=progress_cb)
            info = dict(info)
            info["method"] = "cftp"
        else:
            raise RuntimeError(
                f"Exact sampling for non-uniform weights is only available for "
                f"n <= {MAX_EXACT_N} (exact sequential method; cost grows "
                f"exponentially with n). CFTP is not valid away from the "
                f"uniform point, so no exact result can be given here -- "
                f"reduce n to {MAX_EXACT_N} or below, or set all weights to 1."
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
        n = int(n_raw)
    except (TypeError, ValueError):
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
    return n, weights, data.get("seed")


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
        is_uniform = all(
            v == 1.0 for v in (w["a1"], w["a2"], w["b1"], w["b2"], w["c1"], w["c2"])
        )
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
            "exact_available": (n <= MAX_EXACT_N) or is_uniform,
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
