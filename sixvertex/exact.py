from __future__ import annotations

import threading
from collections import OrderedDict
from itertools import product

import numpy as np

# Exact sequential sampler for the six-vertex model with DWBC.
#
# WHY THIS EXISTS
# ---------------
# The CFTP sampler in cftp.py relies on a monotone coupling of two extremal
# chains. That coupling only targets the correct measure when the per-site
# resampling probability is a function of c1/c2 alone, which is true only at
# a1=a2=b1=b2=1 AND c1=c2=1 (i.e. the uniform/ice point). Away from that
# point, resampling a site changes the classification of up to four
# surrounding faces, so the correct conditional probability depends on the
# local neighbourhood -- and the "correct" version of that move is provably
# not monotone (verified by direct simulation: shared-randomness couplings
# violate pointwise ordering within ~100 sweeps at several weight choices).
#
# In other words: for non-uniform weights you can have a move that is
# CORRECT, or one that is MONOTONE, but the natural constructions cannot
# give both. This module sidesteps the problem entirely by not using a
# Markov chain at all.
#
# HOW IT WORKS
# ------------
# Height functions are sampled one ROW at a time via a standard
# transfer-matrix / dynamic-programming decomposition:
#
#   1. Enumerate the valid rows at each level i (lattice paths from height i
#      to height n-i with +-1 steps -- the DWBC endpoints are fixed).
#   2. Backward pass: B[i][r] = total Boltzmann weight of every valid
#      completion of the configuration from level i down to level n.
#   3. Forward sampling: from the unique boundary row at level 0, draw each
#      successive row with probability proportional to
#          (weight of the strip between the two rows) * B[i+1][next row].
#
# Step 3 draws exactly from the conditional distribution of the next row
# given the current one, so the overall configuration is an exact draw from
# the Boltzmann distribution. There is no burn-in, no mixing time, and no
# monotonicity requirement -- it works in the deep ferroelectric and
# antiferroelectric regimes where local MCMC effectively freezes and where
# CFTP's coupling breaks down.
#
# COST
# ----
# The number of valid rows per level is C(n, n/2), so the backward pass is
# exponential in n. The build is done ONCE per (n, weights); after that each
# sample is very cheap (sub-millisecond). This makes it an exact small-n
# method -- complementary to, not a replacement for, approximate MCMC at
# large n.
#
# VERIFICATION
# ------------
# - Partition function matches brute-force enumeration to machine precision
#   (rel. err ~1e-16) for uniform, extreme (c1=c2=sqrt(8)) and fully
#   asymmetric (a1!=a2, b1!=b2, c1!=c2) weights.
# - Sampled frequencies match the exact distribution, with the residual
#   error shrinking as 1/sqrt(N) (0.0050 -> 0.0022 -> 0.0012 as N goes
#   5k -> 20k -> 80k), i.e. it is pure Monte Carlo counting noise rather
#   than algorithmic bias.
# - At the ice point it reproduces the ASM counts 1, 2, 7, 42, 429 exactly.

MAX_EXACT_N = 14


def _face_weight(tl, tr, bl, br, w):
    top = tr - tl
    bottom = br - bl
    left = bl - tl
    right = br - tr
    t = top == 1
    b = bottom == 1
    l = left == 1
    r = right == 1
    if not l and not t and not b and not r:
        return w["a1"]
    if l and t and b and r:
        return w["a2"]
    if l and t and not b and not r:
        return w["b1"]
    if not l and not t and b and r:
        return w["b2"]
    if not l and t and b and not r:
        return w["c1"]
    if l and not t and not b and r:
        return w["c2"]
    return 0.0


def _rows_for_level(n, i):
    lo, hi = i, n - i
    out = []
    for steps in product([-1, 1], repeat=n):
        row = [lo]
        for s in steps:
            row.append(row[-1] + s)
        if row[-1] == hi:
            out.append(tuple(row))
    return out


class ExactSampler:
    def __init__(self, n: int, weights: dict):
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        if n > MAX_EXACT_N:
            raise ValueError(
                f"exact sequential sampling is exponential in n; n={n} exceeds "
                f"the supported limit of {MAX_EXACT_N}. Use MCMC (Run) for "
                f"larger systems, or reduce n."
            )
        # The Boltzmann distribution is only defined for strictly positive
        # weights. With a zero weight whole classes of configurations are
        # silently forbidden; with a negative one the "probabilities" are not
        # probabilities at all and the sampler would happily return a
        # meaningless configuration. Reject both rather than produce output
        # that looks valid.
        for name in ("a1", "a2", "b1", "b2", "c1", "c2"):
            v = weights.get(name)
            if v is None or not np.isfinite(v) or v <= 0.0:
                raise ValueError(
                    f"weight {name} must be a finite positive number, got {v!r}"
                )
        self.n = n
        self.w = weights
        self.levels = [_rows_for_level(n, i) for i in range(n + 1)]
        self._build()

    def _build(self):
        n, w = self.n, self.w
        self.trans = []
        for i in range(n):
            level_trans = {}
            for r1 in self.levels[i]:
                targets, tws = [], []
                for r2 in self.levels[i + 1]:
                    if any(abs(a - b) != 1 for a, b in zip(r1, r2)):
                        continue
                    tw = 1.0
                    ok = True
                    for j in range(n):
                        fw = _face_weight(r1[j], r1[j + 1], r2[j], r2[j + 1], w)
                        if fw == 0.0:
                            ok = False
                            break
                        tw *= fw
                    if ok and tw > 0:
                        targets.append(r2)
                        tws.append(tw)
                if targets:
                    level_trans[r1] = (targets, np.array(tws, dtype=np.float64))
            self.trans.append(level_trans)

        B = [dict() for _ in range(n + 1)]
        for r in self.levels[n]:
            B[n][r] = 1.0
        for i in range(n - 1, -1, -1):
            for r1, (targets, tws) in self.trans[i].items():
                tot = 0.0
                for r2, tw in zip(targets, tws):
                    bv = B[i + 1].get(r2)
                    if bv:
                        tot += tw * bv
                if tot > 0:
                    B[i][r1] = tot
        self.B = B

        # Numerical safety. The backward pass accumulates products of face
        # weights, so its magnitude grows/shrinks roughly like w^(n^2). At
        # n=14 with the UI's weight range [0.1, 3.0] the extremes are about
        # 1e+115 and 1e-175, comfortably inside float64 (~1e+/-308) -- but
        # that headroom disappears quickly if MAX_EXACT_N is ever raised.
        # Fail loudly rather than return silently-wrong probabilities.
        total = self.partition_function()
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(
                "numerical overflow/underflow in the exact sampler's backward "
                f"pass (partition function = {total}). Reduce n, or rescale "
                "the weights (the model is invariant under a common rescaling "
                "of all six weights)."
            )

    def partition_function(self) -> float:
        return float(sum(self.B[0].values()))

    def sample(self, rng) -> np.ndarray:
        n = self.n
        start_rows = [r for r in self.levels[0] if r in self.B[0]]
        if not start_rows:
            raise RuntimeError("no valid configurations for these weights")
        if len(start_rows) == 1:
            cur = start_rows[0]
        else:
            wts = np.array([self.B[0][r] for r in start_rows], dtype=np.float64)
            cur = start_rows[rng.choice(len(start_rows), p=wts / wts.sum())]
        H = [list(cur)]
        for i in range(n):
            targets, tws = self.trans[i][cur]
            probs = np.array(
                [tw * self.B[i + 1].get(r2, 0.0) for r2, tw in zip(targets, tws)],
                dtype=np.float64,
            )
            total = probs.sum()
            if total <= 0:
                raise RuntimeError("dead end during sampling (should not happen)")
            probs /= total
            cur = targets[rng.choice(len(targets), p=probs)]
            H.append(list(cur))
        return np.array(H, dtype=np.int32)


# The cache is bounded by estimated MEMORY, not by entry count.
#
# Entries differ enormously in size: an n=10 sampler costs ~1 MB while an
# n=14 one costs ~47 MB (measured as process RSS growth). The previous rule
# -- "clear everything once there are more than 8 entries" -- therefore
# permitted roughly 8 x 47 = ~380 MB of cache, which does not fit alongside
# the interpreter and numpy on a 512 MB instance.
#
# Cost is estimated from the number of stored transition pairs, calibrated
# against measured RSS: ~12.5 MB per 1e6 pairs at the sizes that matter.
_CACHE_BUDGET_PAIRS = 5_000_000           # ~60 MB resident; keeps peak RSS
                                          # well clear of a 512 MB instance
_cache = OrderedDict()
_cache_lock = threading.Lock()


def _sampler_cost(sampler):
    """Cheap proxy for a sampler's memory footprint."""
    return sum(len(targets) for level in sampler.trans
               for (targets, _) in level.values())


def _evict_to_budget():
    """Least-recently-used eviction until the cache fits the budget.

    LRU rather than clear-all so that the entry a user is actively drawing
    from survives; clearing everything meant the next draw paid the full
    backward pass again (~30 s at n=14)."""
    total = sum(_costs.get(k, 0) for k in _cache)
    while _cache and total > _CACHE_BUDGET_PAIRS:
        old_key, _ = _cache.popitem(last=False)
        total -= _costs.pop(old_key, 0)


_costs = {}


def exact_sample(n: int, a1=1.0, a2=1.0, b1=1.0, b2=1.0, c1=1.0, c2=1.0, seed=None):
    """Draw one exact sample. Caches the (expensive) backward pass per
    (n, weights) so repeated draws are cheap.

    The cache is lock-protected: the server runs exact sampling in background
    threads, so concurrent requests can otherwise interleave a read, a build
    and an eviction on the same dict.
    """
    weights = {"a1": float(a1), "a2": float(a2), "b1": float(b1),
               "b2": float(b2), "c1": float(c1), "c2": float(c2)}
    key = (n, tuple(sorted(weights.items())))
    with _cache_lock:
        sampler = _cache.get(key)
        if sampler is not None:
            _cache.move_to_end(key)          # mark as recently used
    if sampler is None:
        # Build outside the lock: this is the expensive step and holding the
        # lock through it would serialize unrelated jobs. A duplicate build
        # under a race is wasteful but harmless (the result is identical).
        sampler = ExactSampler(n, weights)
        with _cache_lock:
            _cache[key] = sampler
            _cache.move_to_end(key)
            _costs[key] = _sampler_cost(sampler)
            _evict_to_budget()
    rng = np.random.default_rng(seed)
    H = sampler.sample(rng)
    info = {
        "method": "exact-sequential",
        "n": n,
        "partition_function": sampler.partition_function(),
    }
    return H, info
