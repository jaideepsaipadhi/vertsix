"""Samplers for the six-vertex model with domain-wall boundary conditions.

Three exact methods, covering complementary regimes:

``exact_sample``
    Sequential transfer-matrix sampling. Exact for *arbitrary* positive
    weights, but its cost grows like ``C(n, n/2)``, so it is limited to
    roughly ``n <= 14``.

``cftp_sample``
    Coupling from the past with the four-face heat-bath rule. Exact wherever
    the coupling is monotone, which is precisely

        c1*c2 >= a1*a2   and   c1*c2 >= b1*b2

    Outside that region no monotone coupling of the single-site update exists
    at all, so the function refuses rather than returning something it cannot
    justify. Cost scales like ``n^4``.

``stochastic.sample``
    The stochastic six-vertex model with step initial condition and free exit.
    There the vertex weights are conditional probabilities, so a configuration
    is generated in one sweep of the lattice: exact, ``O(n^2)``, no Markov
    chain. Covers ``Delta >= 1``.

``SixVertexSampler`` is the general-weight single-site heat-bath chain. It is
correct for any weights, but it is a Markov chain: in the ordered phases it
mixes slowly, and what you draw from it is not guaranteed to be a sample from
the measure.

Conventions follow the standard six-vertex labelling, checked against the
Izergin-Korepin determinant for DWBC (itself validated by reproducing the
alternating-sign-matrix numbers 1, 2, 7, 42, 429 at the ice point). With
``A = a1*a2``, ``B = b1*b2``, ``C = c1*c2`` the anisotropy is

    Delta = (A + B - C) / (2 * sqrt(A * B))
"""
from __future__ import annotations

import math as _math

from .sampler import SixVertexSampler
from .exact import exact_sample, ExactSampler, MAX_EXACT_N
from .cftp_exact import cftp_sample, in_monotone_region
from . import stochastic

__version__ = "0.1.0"

__all__ = [
    "SixVertexSampler", "exact_sample", "ExactSampler", "cftp_sample",
    "in_monotone_region", "stochastic", "delta", "sample", "MAX_EXACT_N",
    "__version__",
]


def delta(weights):
    """Anisotropy parameter. < -1 antiferroelectric, (-1,1) disordered,
    > 1 ferroelectric."""
    A = weights["a1"] * weights["a2"]
    B = weights["b1"] * weights["b2"]
    C = weights["c1"] * weights["c2"]
    return (A + B - C) / (2.0 * _math.sqrt(A * B))


def sample(n, weights, seed=None, method="auto"):
    """Draw one exact sample, choosing a method valid for the input.

    ``method="auto"`` uses the sequential sampler when ``n`` is small enough,
    and CFTP otherwise when the weights admit a monotone coupling. It raises
    rather than falling back on an approximate method: a configuration that is
    not a sample from the measure is worse than no configuration.

    Returns ``(height_function, info)``; ``info["method"]`` records what ran.
    """
    if method not in ("auto", "sequential", "cftp"):
        raise ValueError(f"unknown method {method!r}")

    if method == "sequential" or (method == "auto" and n <= MAX_EXACT_N):
        if n > MAX_EXACT_N:
            raise ValueError(
                f"the sequential sampler is limited to n <= {MAX_EXACT_N}; "
                f"its cost grows like C(n, n/2)")
        return exact_sample(n, seed=seed, **weights)

    if in_monotone_region(weights):
        return cftp_sample(n, weights, master_seed=seed)

    A = weights["a1"] * weights["a2"]
    B = weights["b1"] * weights["b2"]
    C = weights["c1"] * weights["c2"]
    raise ValueError(
        f"no exact method available for n={n} at these weights "
        f"(A={A:.4g}, B={B:.4g}, C={C:.4g}, Delta={delta(weights):+.4g}). "
        f"The sequential sampler is limited to n <= {MAX_EXACT_N}, and CFTP "
        f"requires c1*c2 >= a1*a2 and c1*c2 >= b1*b2 -- outside that region "
        f"no monotone coupling of the update exists."
    )
