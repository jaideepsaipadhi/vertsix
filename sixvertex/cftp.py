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
# violated pointwise ordering within ~50 sweeps in testing).
#
# That last sentence is still true as an observation and no longer true as a
# conclusion. The ordering breaks OUTSIDE the monotone region, not for general
# weights as such: sixvertex/cftp_exact.py does CFTP for general
# (a1,a2,b1,b2,c1,c2) wherever c1c2 >= a1a2 and c1c2 >= b1b2, and
# experiments/monotone_check.py checks monotonicity exhaustively there -- every
# ordered pair of configurations, every site, every threshold of the driving
# uniform, on five domains at five values of Delta, zero violations -- and also
# shows the region is sharp, the coupling failing already at a = 1.01 with
# b = c = 1. So the old warning not to extend this without "a fresh
# monotonicity proof/verification" has been met; it is recorded here as
# history, not as an open problem.
#
# THE NOTE THAT USED TO SIT HERE WAS FALSE, and dangerously so. It claimed
# this regime "allows a=b=1 with ARBITRARY c1, c2 -- including deep
# ferroelectric/antiferroelectric weights (|Delta|>1). CFTP remains exactly
# correct there regardless of mixing time", citing as verification that at
# c1=c2=sqrt(8) it "coalesces correctly". Coalescing is not sampling the right
# measure, and that is the whole of the error: the chain does coalesce, to a
# draw from the WRONG distribution.
#
# The reason is visible in one line of this file. p_up = c_up/(c_up+c_down) is
# the only place the weights enter, so at c1=c2 it is 1/2 for EVERY value of
# c -- the chain cannot tell c=1 from c=sqrt(8) and returns the uniform
# measure in both cases. Measured at n=4 against the exact distribution,
# 20000 samples (sampling noise is about 0.017 in total variation):
#
#     c1     c2    p_up    TV to the measure asked for    TV to uniform
#     1.00  1.00   0.500            0.0166                   0.0166
#     2.00  2.00   0.500            0.3700                   0.0166
#     0.50  0.50   0.500            0.2833                   0.0166
#     1.00  2.00   0.333            0.6139                   0.5589
#     2.00  1.00   0.667            0.6118                   0.5566
#     1.00  4.00   0.200            0.8461                   0.7830
#
# So: correct at c1 = c2 = 1 and NOWHERE ELSE. At c1 = c2 != 1 it silently
# returns the uniform measure. At c1 != c2 it returns neither the requested
# measure nor the uniform one -- p_up is no longer 1/2 and the result is not a
# six-vertex DWBC measure at all. tests/test_sampler.py has carried a test
# named test_cftp_only_correct_at_uniform_point since the rewrite; this
# comment was simply never brought into line with it.
#
# Use sixvertex.cftp_exact.cftp_sample for anything else. Nothing in the
# product calls this module (server.py's import of it was unused and has been
# removed, since it shadowed the general cftp_sample by name); it is kept only
# because the tests above pin the defect it documents.


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
                 initial_T: int = 16, max_T: int = 1 << 21,
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
