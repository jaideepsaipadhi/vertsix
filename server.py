from flask import Flask, request, jsonify, send_from_directory
import math
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
_state = {"sampler": None}

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
    cutoff = time.time() - _JOB_TTL_SECONDS
    stale = [jid for jid, j in _jobs.items() if j["created_at"] < cutoff]
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
                _jobs[job_id]["frame"] = frame
                _jobs[job_id]["info"] = info
    except (RuntimeError, ValueError) as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(e)
    except Exception as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
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
        _state["sampler"] = s
        info = {
            "n": n,
            "device": str(s.device),
            "using_gpu": s.use_torch,
            "is_symmetric_regime": s.is_symmetric_regime(),
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
    sweeps = max(1, min(sweeps, 500))
    with _lock:
        s = _state["sampler"]
        if s is None:
            return jsonify({"ok": False, "error": "not initialized"}), 400
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
            response["is_symmetric_regime"] = True
        elif status == "error":
            response["error"] = job["error"]
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Six-vertex sampler running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
