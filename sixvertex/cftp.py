from __future__ import annotations
import numpy as np

from .sampler import SixVertexSampler

# CFTP here deliberately implements ONLY the restricted regime (a1=a2=b1=b2=1,
# varying c1/c2 via c_up/c_down) -- this is the one case proven monotone (the
# shared-random-target coupling below preserves pointwise ordering between the
# lo/hi chains, verified by direct simulation with zero violations over
# thousands of coupled half-sweeps).
#
# The general (a1,a2,b1,b2,c1,c2) weighted dynamics implemented in
# sampler.py's live MCMC path is NOT used here, because monotonicity was
# empirically checked and found to BREAK under it (a shared-seed coupling
# violated pointwise ordering within ~50 sweeps in testing). Exact sampling
# via CFTP for the fully general model is an open problem, not just an
# engineering gap -- do not extend this function to accept a1/a2/b1/b2
# without a fresh monotonicity proof/verification first.


def _rand_field(master_seed: int, k: int, shape):
    ss = np.random.SeedSequence([master_seed, k])
    rng = np.random.default_rng(ss)
    return rng.random(shape).astype(np.float32)


def _half_sweep(H, mask, rnd, p_up):
    N = np.zeros_like(H); S = np.zeros_like(H)
    E = np.zeros_like(H); W = np.zeros_like(H)
    N[1:, :] = H[:-1, :]
    S[:-1, :] = H[1:, :]
    E[:, :-1] = H[:, 1:]
    W[:, 1:] = H[:, :-1]
    same = (N == S) & (S == E) & (E == W)
    is_extremum = mask & same
    target = np.where(rnd < p_up, N + 1, N - 1)
    H = np.where(is_extremum, target, H)
    return H


def cftp_sample(n: int, c_up: float = 1.0, c_down: float = 1.0,
                 master_seed: int | None = None,
                 initial_T: int = 16, max_T: int = 1 << 20,
                 progress_cb=None):
    if master_seed is None:
        master_seed = np.random.randint(0, 2**31 - 1)

    p_up = c_up / (c_up + c_down)

    i_idx, j_idx = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    interior = (i_idx > 0) & (i_idx < n) & (j_idx > 0) & (j_idx < n)
    parity = (i_idx + j_idx) % 2
    masks = [interior & (parity == 0), interior & (parity == 1)]

    T = initial_T
    attempts = 0
    while T <= max_T:
        attempts += 1
        Hlo = SixVertexSampler.extremal_height(n, "lo")
        Hhi = SixVertexSampler.extremal_height(n, "hi")

        for k in range(T, 0, -1):
            mask = masks[k % 2]
            rnd = _rand_field(master_seed, k, Hlo.shape)
            Hlo = _half_sweep(Hlo, mask, rnd, p_up)
            Hhi = _half_sweep(Hhi, mask, rnd, p_up)

        coalesced = np.array_equal(Hlo, Hhi)
        if progress_cb:
            progress_cb(T, attempts, coalesced)
        if coalesced:
            return Hlo, {"half_sweeps": T, "attempts": attempts, "master_seed": int(master_seed)}
        T *= 2

    raise RuntimeError(
        f"CFTP did not coalesce within {max_T} half-sweeps ({attempts} attempts). "
        "Try a smaller n, or a smaller |bias|."
    )
