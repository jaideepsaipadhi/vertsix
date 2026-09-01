#!/usr/bin/env python3
"""Exact CFTP sample at Gorin's Figure 17 (top) parameters, N = 256.

    Delta = 1/4,  a1 = a2 = 1,  b1 = b2 = c1 = c2 = 2

These weights give A = 1, B = 4, C = 4, so B >= A and B >= C (the second with
equality). They sit exactly on the boundary of the monotone region, which is
why CFTP is valid here at all.

Run from the repository root:

    python3 run_n256.py

Expect roughly 40-60 minutes. Cost scales like n^4, so it is dominated by the
last doubling. The raw sample is written to disk before plotting, so if you
want to re-draw it later you do not have to re-run the sampler:

    python3 run_n256.py --render-only

Output:
    n256_delta_quarter.npy   the height function
    n256_delta_quarter.png   the figure
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NPY = "n256_delta_quarter.npy"
PNG = "n256_delta_quarter.png"

N = 256
WEIGHTS = dict(a1=1.0, a2=1.0, b1=2.0, b2=2.0, c1=2.0, c2=2.0)
SEED = 20260819


def generate():
    from sixvertex.cftp_exact import cftp_sample, in_monotone_region

    assert in_monotone_region(WEIGHTS), "weights are outside the monotone region"
    A = WEIGHTS["a1"] * WEIGHTS["a2"]
    B = WEIGHTS["b1"] * WEIGHTS["b2"]
    C = WEIGHTS["c1"] * WEIGHTS["c2"]
    print(f"N = {N},  A = {A:g}, B = {B:g}, C = {C:g},  "
          f"Delta = {(A + B - C) / (2 * np.sqrt(A * B)):.4f}")
    print("starting CFTP; each line is one doubling of the look-back window\n")

    t0 = time.time()

    def progress(T, attempts, coalesced):
        print(f"  attempt {attempts:2d}:  T = {T:7d} sweeps   "
              f"elapsed {time.time() - t0:7.0f}s   "
              f"{'COALESCED' if coalesced else 'not yet'}", flush=True)

    H, info = cftp_sample(N, WEIGHTS, master_seed=SEED,
                          max_T=1 << 21, progress_cb=progress)
    dt = time.time() - t0
    print(f"\ndone in {dt / 60:.1f} min: {info['sweeps']} sweeps, "
          f"{info['attempts']} doublings")
    np.save(NPY, np.asarray(H, dtype=np.int16))
    print(f"saved {NPY}")
    return np.asarray(H, dtype=float)


def render(H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = H.shape[0] - 1
    levels = np.arange(H.min() + 0.5, H.max() + 0.5, 1.0)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    ax.contour(np.arange(n + 1), np.arange(n + 1), H,
               levels=levels, colors="k", linewidths=0.25)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(PNG, bbox_inches="tight", facecolor="white")
    print(f"saved {PNG}")


def sanity_check(H):
    """The sample must be a legal height function with DWBC."""
    n = H.shape[0] - 1
    Hi = np.asarray(H, dtype=np.int64)
    ok_h = np.all(np.abs(np.diff(Hi, axis=1)) == 1)
    ok_v = np.all(np.abs(np.diff(Hi, axis=0)) == 1)
    j = np.arange(n + 1)
    ok_b = (np.array_equal(Hi[0, :], j) and np.array_equal(Hi[n, :], n - j)
            and np.array_equal(Hi[:, 0], j) and np.array_equal(Hi[:, n], n - j))
    print(f"valid height function: edges {bool(ok_h and ok_v)}, boundary {ok_b}")
    return bool(ok_h and ok_v and ok_b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true",
                    help="re-draw from the saved .npy without resampling")
    args = ap.parse_args()

    if args.render_only:
        if not os.path.exists(NPY):
            sys.exit(f"{NPY} not found; run without --render-only first")
        H = np.load(NPY).astype(float)
        print(f"loaded {NPY}")
    else:
        H = generate()

    sanity_check(H)
    render(H)
