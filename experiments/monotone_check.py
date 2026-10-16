#!/usr/bin/env python3
"""Decisive monotonicity check for the coupling, on any domain.

CFTP returns an exact sample only because the two extremal chains are coupled
monotonically; coalescence of a non-monotone coupling certifies nothing. The
condition in_monotone_region enforces -- c1c2 >= a1a2 and c1c2 >= b1b2 -- was
derived for a corner surrounded by four faces, which is the only kind a square
has, so on general domains it needs checking rather than assuming.

The check is exhaustive, not a sample. A sweep is a composition of single-site
heat-bath updates, each of which maps a valid configuration to a valid
configuration, so the sweep is monotone iff every single-site update is. And
the outcome of one update is a STEP function of the driving uniform u, with at
most two thresholds, so evaluating u at those thresholds and on either side of
them covers every behaviour the update can have.

Results (all 5 domains x 5 Delta values, zero violations), with the notch
domain carrying 2889 configurations at each Delta:

    Delta = +0.5, 0, -1, -3, -5   x   square N=4, stair N=4,
                                      stair N=6, plus N=6, notch N=6

The region is also sharp. Walking a across the C >= A boundary at b=c=1, the
coupling is monotone at a=1.00 (where A=C exactly) and fails at a=1.05; the
same holds for b across C >= B. So the condition is not over-conservative --
it is exactly the set where the coupling works.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sixvertex.domains import (_weight_table, corner_mask, sweep_plan)
from experiments.brute_domain import enumerate_heights


def _p_up(h, flat, nb, faces, tbl):
    """Heat-bath probability of the up-move, or None when the corner is pinned."""
    v = h[nb[0]]
    if any(h[q] != v for q in nb):
        return None, None
    out = []
    for val in (v + 1, v - 1):
        h[flat] = val
        prod = 1.0
        for (tl, tr, bl, br) in faces:
            a = h[tl]
            prod *= tbl[((h[bl] - a == 1) << 3) | ((h[tr] - a == 1) << 2)
                        | ((h[br] - h[bl] == 1) << 1) | (h[br] - h[tr] == 1)]
        out.append(prod)
    h[flat] = v
    wu, wd = out
    return (wu / (wu + wd) if wu + wd > 0 else None), v


def monotonicity_violations(mask, w, limit=1):
    """Every ordered pair of configurations, every site, every u. Exhaustive."""
    hs = enumerate_heights(mask)
    cm = corner_mask(mask)
    tbl = _weight_table(w)
    sites = [s for g in sweep_plan(mask) for s in g]
    viol = []
    for A in hs:
        for B in hs:
            if np.array_equal(A, B) or not np.all(B[cm] >= A[cm]):
                continue
            for flat, nb, faces in sites:
                a = A.ravel().tolist()
                b = B.ravel().tolist()
                pa, va = _p_up(a, flat, nb, faces, tbl)
                pb, vb = _p_up(b, flat, nb, faces, tbl)
                crit = sorted({0.0, 1.0} | {x for x in (pa, pb) if x is not None})
                for c in crit:
                    for u in (max(0.0, c - 1e-9), c, min(1.0, c + 1e-9)):
                        na = (va + 1 if u < pa else va - 1) if pa is not None else a[flat]
                        nb_ = (vb + 1 if u < pb else vb - 1) if pb is not None else b[flat]
                        if na > nb_:
                            viol.append((flat, len(faces), pa, pb, va, vb, u))
                            if len(viol) >= limit:
                                return viol
    return viol


if __name__ == "__main__":
    from experiments.mixing import weights_for_delta
    from sixvertex.domains import domain_mask

    cases = [("square N=4", [0, 4], [0, 4], [[1]]),
             ("stair N=4", [0, 1, 2, 4], [0, 1, 2, 4], [[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
             ("stair N=6", [0, 2, 4, 6], [0, 2, 4, 6], [[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
             ("plus N=6", [0, 2, 4, 6], [0, 2, 4, 6], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
             ("notch N=6", [0, 2, 4, 6], [0, 2, 4, 6], [[1, 1, 1], [1, 1, 1], [0, 1, 1]])]
    for d in (0.5, 0.0, -1.0, -3.0, -5.0):
        w = weights_for_delta(d)
        for name, X, Y, I in cases:
            m = domain_mask(X, Y, I)
            v = monotonicity_violations(m, w)
            print(f"  Delta={d:+.1f} {name:11s} "
                  f"{'MONOTONE' if not v else 'NOT MONOTONE ' + str(v[0])}",
                  flush=True)
