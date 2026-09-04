"""Exact sampling by Coupling From The Past, using the CORRECT transition rule.

The older `cftp.py` hardcodes `p_up = c1/(c1+c2)`, which targets the right
measure only at the uniform point. That is why the server historically
restricted CFTP to all-weights-equal-1.

The correct single-site rule is the four-face heat-bath probability (the same
one `sampler.py` uses), and the coupling of the two extremal chains driven by
shared uniforms is monotone exactly on

    B >= A   and   B >= C,        A = a1a2, B = b1b2, C = c1c2

so CFTP is valid on that whole region, not merely at the point. Outside it no
coupling is monotone at all -- the marginals are not ordered there -- so this
module refuses rather than returning something it cannot justify.
"""
from __future__ import annotations

import numpy as np

FACE = {(0, 0, 0, 0): "a1", (1, 1, 1, 1): "a2",
        (1, 1, 0, 0): "c1", (0, 0, 1, 1): "c2",
        (0, 1, 1, 0): "b1", (1, 0, 0, 1): "b2"}


def in_monotone_region(w, tol=1e-12):
    """Monotone iff c1c2 >= a1a2 and c1c2 >= b1b2.

    The hand calculation was done in this code's original labelling, where
    it read b1b2 >= a1a2 and b1b2 >= c1c2. Those labels had b and c swapped
    relative to the standard six-vertex convention (confirmed against the
    Izergin-Korepin determinant, and by the fact that the single N=1
    DWBC configuration is a c-vertex). With the labels corrected, the same
    theorem reads with c dominant.

    This is the antiferroelectric-favouring direction, so the monotone
    region *contains* Delta < -1 rather than excluding it: Gorin's Figure 17
    bottom (a=b=1, c=sqrt(8), Delta=-3) lies inside it and can be sampled
    exactly.
    """
    A = w["a1"] * w["a2"]
    B = w["b1"] * w["b2"]
    C = w["c1"] * w["c2"]
    return (C >= A - tol) and (C >= B - tol)


def _weight_table(w):
    """16-entry lookup indexed by the packed bits (l,t,b,r) -> l*8+t*4+b*2+r.

    Replaces a chain of six full-array np.where calls with a single take.
    Profiling showed the where-chain was ~79% of sweep time: it is invoked 32
    times per sweep, and each call walked the whole n^2 array six times
    regardless of which few entries mattered.
    """
    tbl = np.zeros(16, dtype=np.float64)
    tbl[0 * 8 + 0 * 4 + 0 * 2 + 0] = w["a1"]   # (l,t,b,r) = 0,0,0,0
    tbl[1 * 8 + 1 * 4 + 1 * 2 + 1] = w["a2"]   # 1,1,1,1
    tbl[1 * 8 + 1 * 4 + 0 * 2 + 0] = w["c1"]   # 1,1,0,0
    tbl[0 * 8 + 0 * 4 + 1 * 2 + 1] = w["c2"]   # 0,0,1,1
    tbl[0 * 8 + 1 * 4 + 1 * 2 + 0] = w["b1"]   # 0,1,1,0
    tbl[1 * 8 + 0 * 4 + 0 * 2 + 1] = w["b2"]   # 1,0,0,1
    return tbl


def _face_weights(tl, tr, bl, br, tbl):
    """Face weights for arrays of corner heights, via a packed-bit lookup."""
    idx = (((bl - tl) == 1).astype(np.int8) << 3)
    idx |= (((tr - tl) == 1).astype(np.int8) << 2)
    idx |= (((br - bl) == 1).astype(np.int8) << 1)
    idx |= (((br - tr) == 1).astype(np.int8))
    return tbl[idx]


def extremal_height(n, kind):
    i = np.arange(n + 1).reshape(-1, 1).astype(np.int64)
    j = np.arange(n + 1).reshape(1, -1).astype(np.int64)
    corners = [(0, 0, 0), (n, 0, n), (0, n, n), (n, n, 0)]
    if kind == "lo":
        H = np.full((n + 1, n + 1), -(10 ** 9), dtype=np.int64)
        for a, b, hv in corners:
            H = np.maximum(H, hv - (np.abs(i - a) + np.abs(j - b)))
    else:
        H = np.full((n + 1, n + 1), 10 ** 9, dtype=np.int64)
        for a, b, hv in corners:
            H = np.minimum(H, hv + (np.abs(i - a) + np.abs(j - b)))
    return H.astype(np.float64)


def _colour_indices(n):
    """Flat indices of the interior sites of each colour class, with the flat
    offsets of their neighbours.

    Working on the ~n^2/4 active sites of a colour class rather than the whole
    (n+1)^2 grid is worth roughly a factor of four: profiling showed the face
    weights dominated the sweep, and three quarters of every array they
    touched was discarded by the colour mask immediately afterwards.
    """
    size = n + 1
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    interior = (i > 0) & (i < n) & (j > 0) & (j < n)
    colour = (i % 2) * 2 + (j % 2)
    out = []
    for c in range(4):
        idx = np.flatnonzero((interior & (colour == c)).ravel())
        out.append(idx)
    return out, size


def _sweep(H, tbl, rnd, colour_idx, size):
    """One full sweep over the 4-colouring, driven by the field `rnd`.

    Operates on flat views restricted to each colour class.
    """
    Hf = H.ravel()
    rf = rnd.ravel()
    for idx in colour_idx:
        N = Hf[idx - size]; S = Hf[idx + size]
        W = Hf[idx - 1];    E = Hf[idx + 1]
        flip = (N == S) & (S == E) & (E == W)
        if not flip.any():
            continue
        sel = idx[flip]
        N = Hf[sel - size]; S = Hf[sel + size]
        W = Hf[sel - 1];    E = Hf[sel + 1]
        NW = Hf[sel - size - 1]; NE = Hf[sel - size + 1]
        SW = Hf[sel + size - 1]; SE = Hf[sel + size + 1]

        v = N
        up, dn = v + 1, v - 1
        Wup = (_face_weights(NW, N, W, up, tbl) * _face_weights(N, NE, up, E, tbl) *
               _face_weights(W, up, SW, S, tbl) * _face_weights(up, E, S, SE, tbl))
        Wdn = (_face_weights(NW, N, W, dn, tbl) * _face_weights(N, NE, dn, E, tbl) *
               _face_weights(W, dn, SW, S, tbl) * _face_weights(dn, E, S, SE, tbl))
        tot = Wup + Wdn
        p_up = np.where(tot > 0, Wup / np.maximum(tot, 1e-300), 0.5)
        Hf[sel] = np.where(rf[sel] < p_up, up, dn)
    return H


# ---------------------------------------------------------------------------
# Optional compiled sweep.
#
# The numpy sweep already restricts work to the active colour class, but still
# allocates a dozen temporary arrays per class. A compiled explicit loop avoids
# them: measured 76x at n=40, 30x at n=80, 16x at n=128, with bit-for-bit
# identical output -- it consumes the same random values at the same positions,
# so it is the same computation, not an approximation.
#
# numba is optional; without it everything falls back to the numpy path, which
# remains the reference implementation.
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:                                     # pragma: no cover
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @_njit(cache=True)
    def _sweep_compiled(Hf, tbl, rf, idx_tuple, size):
        for ci in range(len(idx_tuple)):
            idx = idx_tuple[ci]
            for k in range(idx.shape[0]):
                p = idx[k]
                N = Hf[p - size]; S = Hf[p + size]
                W = Hf[p - 1];    E = Hf[p + 1]
                if not (N == S and S == E and E == W):
                    continue
                NW = Hf[p - size - 1]; NE = Hf[p - size + 1]
                SW = Hf[p + size - 1]; SE = Hf[p + size + 1]
                v = N
                up = v + 1.0; dn = v - 1.0
                wu = 1.0; wd = 1.0
                for which in range(2):
                    cen = up if which == 0 else dn
                    acc = 1.0
                    for f in range(4):
                        if f == 0:
                            tl = NW;  tr = N;   bl = W;   br = cen
                        elif f == 1:
                            tl = N;   tr = NE;  bl = cen; br = E
                        elif f == 2:
                            tl = W;   tr = cen; bl = SW;  br = S
                        else:
                            tl = cen; tr = E;   bl = S;   br = SE
                        i = 0
                        if bl - tl == 1.0: i += 8
                        if tr - tl == 1.0: i += 4
                        if br - bl == 1.0: i += 2
                        if br - tr == 1.0: i += 1
                        acc *= tbl[i]
                    if which == 0: wu = acc
                    else:          wd = acc
                tot = wu + wd
                pu = 0.5 if tot <= 0.0 else wu / tot
                Hf[p] = up if rf[p] < pu else dn
        return Hf


def _rand_field(master_seed, k, shape):
    """Randomness for virtual time -k. Reused across doublings so that a
    longer run reproduces the tail of a shorter one, as CFTP requires."""
    ss = np.random.SeedSequence([int(master_seed), int(k)])
    return np.random.default_rng(ss).random(shape)


def cftp_sample(n, weights, master_seed=None, initial_T=8, max_T=1 << 20,
                progress_cb=None, check_monotone=False):
    """Draw one exact sample. Raises ValueError outside the monotone region."""
    if not in_monotone_region(weights):
        A = weights["a1"] * weights["a2"]
        B = weights["b1"] * weights["b2"]
        C = weights["c1"] * weights["c2"]
        raise ValueError(
            f"weights outside the monotone region (A={A:.4g}, B={B:.4g}, "
            f"C={C:.4g}): CFTP requires c1c2 >= a1a2 and c1c2 >= b1b2. "
            f"Outside it no coupling of the update is monotone, so no CFTP "
            f"scheme is valid here."
        )
    if master_seed is None:
        master_seed = int(np.random.randint(0, 2 ** 31 - 1))

    colour_idx, size = _colour_indices(n)
    tbl = _weight_table(weights)
    T, attempts, violations = initial_T, 0, 0
    while T <= max_T:
        attempts += 1
        Hlo = extremal_height(n, "lo")
        Hhi = extremal_height(n, "hi")
        idx_tuple = tuple(colour_idx) if _HAS_NUMBA else None
        for k in range(T, 0, -1):
            rnd = _rand_field(master_seed, k, Hlo.shape)
            if _HAS_NUMBA:
                _sweep_compiled(Hlo.ravel(), tbl, rnd.ravel(), idx_tuple, size)
                _sweep_compiled(Hhi.ravel(), tbl, rnd.ravel(), idx_tuple, size)
            else:
                Hlo = _sweep(Hlo, tbl, rnd, colour_idx, size)
                Hhi = _sweep(Hhi, tbl, rnd, colour_idx, size)
            if check_monotone and not np.all(Hlo <= Hhi):
                violations += 1
        if progress_cb is not None:
            progress_cb(T, attempts, bool(np.array_equal(Hlo, Hhi)))
        if np.array_equal(Hlo, Hhi):
            info = {"sweeps": T, "attempts": attempts,
                    "master_seed": int(master_seed), "method": "cftp"}
            if check_monotone:
                info["monotonicity_violations"] = violations
            return Hlo.astype(np.int16), info
        T *= 2
    raise RuntimeError(f"did not coalesce within {max_T} sweeps")
