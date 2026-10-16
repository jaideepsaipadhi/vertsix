"""Coalescence-time measurement for the six-vertex model under DWBC.

Gorin's question: how does the time CFTP takes depend on Delta?

Two quantities are available and they are not the same:

* the time CFTP *reports*, which is the first window of its doubling schedule
  at which the coupled chains agree. That is what "CFTP stops" literally
  means, but it is quantised to powers of two, so a plot against Delta is a
  staircase that hides the dependence being measured.

* the **forward coupling time**: start the two extremal configurations at time
  0, drive them with one shared random stream, and record the first sweep at
  which they agree. Unquantised, one forward pass instead of a restarting
  schedule, and it is the quantity the doubling schedule is bracketing.

This module measures the second and reports the first alongside it.

The measurement is only meaningful where the coupling is monotone, which is
exactly the region CFTP is valid on (c1c2 >= a1a2 and c1c2 >= b1b2). Outside
it the two chains can cross and "coalescence" means nothing, so the caller is
refused rather than given a number.
"""
from __future__ import annotations

import numpy as np

from sixvertex.cftp_exact import (
    _colour_indices, _rand_field, _sweep, _weight_table, extremal_height,
    in_monotone_region, _HAS_NUMBA,
)

if _HAS_NUMBA:
    from sixvertex.cftp_exact import _sweep_compiled


def forward_coalescence_time(n, weights, seed, max_sweeps=1 << 22,
                             check_monotone=False):
    """Sweeps until the two extremal chains, sharing randomness, agree.

    Returns None if they have not agreed within `max_sweeps` -- the caller must
    treat that as censored data, not as `max_sweeps`, or the mean is silently
    biased downwards wherever the chain is slow.
    """
    if not in_monotone_region(weights):
        raise ValueError(
            "coalescence time is only defined where the coupling is monotone "
            "(c1c2 >= a1a2 and c1c2 >= b1b2); outside it the chains can cross")

    tbl = _weight_table(weights)
    colour_idx, size = _colour_indices(n)
    lo = extremal_height(n, "lo")
    hi = extremal_height(n, "hi")
    idx_tuple = tuple(colour_idx) if _HAS_NUMBA else None

    for t in range(1, max_sweeps + 1):
        rnd = _rand_field(seed, t, lo.shape)
        if _HAS_NUMBA:
            _sweep_compiled(lo.ravel(), tbl, rnd.ravel(), idx_tuple, size)
            _sweep_compiled(hi.ravel(), tbl, rnd.ravel(), idx_tuple, size)
        else:
            lo = _sweep(lo, tbl, rnd, colour_idx, size)
            hi = _sweep(hi, tbl, rnd, colour_idx, size)
        if check_monotone and not np.all(lo <= hi):
            raise AssertionError(
                f"ordering broken at sweep {t}: the coupling is not monotone "
                f"at these weights, so this measurement is meaningless")
        if np.array_equal(lo, hi):
            return t
    return None


def weights_for_delta(delta):
    """a1=a2=b1=b2=1, c1=c2=c with Delta = 1 - c^2/2.

    Sweeping c upward from 1 walks Delta down from +1/2, staying inside the
    monotone region the whole way (which needs only c >= 1) and crossing the
    antiferroelectric transition at c = 2. Delta > 1/2 is unreachable: 1/2 is
    the largest anisotropy the region contains, attained at the uniform point.
    """
    if delta > 0.5 + 1e-12:
        raise ValueError(
            f"Delta={delta} is outside the monotone region; the region is "
            f"contained in Delta <= 1/2, with equality at uniform weights")
    c = float(np.sqrt(2.0 * (1.0 - delta)))
    return dict(a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=c, c2=c)


def delta_of(weights):
    A = weights["a1"] * weights["a2"]
    B = weights["b1"] * weights["b2"]
    C = weights["c1"] * weights["c2"]
    return (A + B - C) / (2.0 * np.sqrt(A * B))
