from flask import Flask, request, jsonify, send_from_directory
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


@app.route("/api/init", methods=["POST"])
def api_init():
    data = request.get_json(force=True)
    n = int(data.get("n", 40))
    c_up = float(data.get("c_up", 1.0))
    c_down = float(data.get("c_down", 1.0))
    a1 = float(data.get("a1", 1.0))
    a2 = float(data.get("a2", 1.0))
    b1 = float(data.get("b1", 1.0))
    b2 = float(data.get("b2", 1.0))
    seed = data.get("seed")
    n = max(4, min(n, 400))
    with _lock:
        s = SixVertexSampler(
            n=n, c_up=c_up, c_down=c_down, a1=a1, a2=a2, b1=b1, b2=b2, seed=seed
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
    data = request.get_json(force=True)
    sweeps = int(data.get("sweeps", 1))
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
    data = request.get_json(force=True)
    n = int(data.get("n", 40))
    n = max(4, min(n, 250))
    c1 = float(data.get("c_up", 1.0))
    c2 = float(data.get("c_down", 1.0))
    a1 = float(data.get("a1", 1.0))
    a2 = float(data.get("a2", 1.0))
    b1 = float(data.get("b1", 1.0))
    b2 = float(data.get("b2", 1.0))
    seed = data.get("seed")

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
