from flask import Flask, request, jsonify, send_from_directory
import os
import threading

from sixvertex.sampler import SixVertexSampler
from sixvertex.cftp import cftp_sample

app = Flask(__name__, static_folder="static", static_url_path="")

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
    seed = data.get("seed")
    n = max(4, min(n, 400))
    with _lock:
        _state["sampler"] = SixVertexSampler(
            n=n, c_up=c_up, c_down=c_down, seed=seed
        )
        info = {
            "n": n,
            "device": str(_state["sampler"].device),
            "using_gpu": _state["sampler"].use_torch,
        }
    return jsonify({"ok": True, **info, "frame": _state["sampler"].to_json()})


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
        frame = s.to_json()
    return jsonify({"ok": True, "frame": frame})


@app.route("/api/exact", methods=["POST"])
def api_exact():
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
            frame = s.to_json()
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "frame": frame, "info": info})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Six-vertex sampler running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
