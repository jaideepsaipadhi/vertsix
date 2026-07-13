from flask import Flask, request, jsonify, send_from_directory
import gzip
import os
import threading

from sixvertex.sampler import SixVertexSampler
from sixvertex.cftp import cftp_sample

app = Flask(__name__, static_folder="static", static_url_path="")


@app.after_request
def compress_response(response):
    accept_encoding = request.headers.get("Accept-Encoding", "")
    if "gzip" not in accept_encoding.lower():
        return response
    if response.direct_passthrough or response.content_length is not None and response.content_length < 500:
        return response
    response.data = gzip.compress(response.get_data())
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Content-Length"] = len(response.data)
    return response


_lock = threading.Lock()
_state = {"sampler": None}


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


@app.route("/api/exact", methods=["POST"])
def api_exact():
    # Deliberately does NOT accept a1/a2/b1/b2 -- CFTP here is only proven
    # monotone for the a1=a2=b1=b2=1 (c-bias only) regime. See cftp.py.
    data = request.get_json(force=True)
    n = int(data.get("n", 40))
    n = max(4, min(n, 250))
    c_up = float(data.get("c_up", 1.0))
    c_down = float(data.get("c_down", 1.0))
    seed = data.get("seed")
    try:
        with _lock:
            H, info = cftp_sample(n=n, c_up=c_up, c_down=c_down,
                                   master_seed=seed, max_T=1 << 18)
            s = SixVertexSampler(n=n, c_up=c_up, c_down=c_down)
            if s.use_torch:
                import torch
                s.H = torch.from_numpy(H).to(s.device)
            else:
                s.H = H
            _state["sampler"] = s
            frame = s.to_binary_frame()
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "frame": frame, "info": info, "is_symmetric_regime": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Six-vertex sampler running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
