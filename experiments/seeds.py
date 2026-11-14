"""One seed allocator for the measurement campaigns.

Three times now a campaign script has drawn the SAME random stream for cells
it then compared against each other, and each time it was found by measuring
rather than by reading:

  * run_mixing.py reused one block across the grid; the Delta axis came out
    correlated at rho = +0.248 (p = 0.026).
  * run_asymptotics.py had the same shape; there the measured correlation was
    not significant (Spearman 0.14, p = 0.29), so the data survived, but the
    blocks were separated anyway.
  * run_island.py -- Experiment 2's main campaign -- passed `master_seed=s`
    with s the bare loop index, so EVERY (n, Delta) cell drew seeds
    0 .. samples-1. Measured on k at n=24, 250 samples:

        Delta -3 vs -4, shared seeds      r = +0.399
        Delta -2 vs -3, shared seeds      r = +0.180
        Delta -2 vs -4, shared seeds      r = +0.126
        Delta -2 vs -3, disjoint seeds    r = -0.075   (SE ~ 0.063)

    r = +0.399 on 250 points is not a subtlety. Every cross-cell claim in
    Experiment 2 -- the Delta dependence of var(k), the island-ratio curve,
    the theta exponent across sizes, the crossing estimates -- was made on
    coupled samples.

And a fourth, of a different kind: cftp_stopping.py built its base as
7_000_000 + n*1000 + s, which omits Delta, so every Delta at fixed n shared
seeds; and that base collided with run_asymptotics.py at n = 16, |Delta| = 4,
where both scripts land on 7_016_000. Both call forward_coalescence_time
there, so the two campaigns published the same numbers:

    run_asymptotics (n=16, Delta=-4)   [155, 165, 307, 336, 294]
    cftp_stopping   (n=16, any Delta)  [155, 165, 307, 336, 294]

run_asymptotics' own comment said the blocks "are made disjoint anyway" --
true of its own cells, and not true across scripts, which is the hole an
in-script fix cannot see.

A fifth, found by the global test as soon as it existed: run_asymptotics
keyed on int(abs(d) * 1000), so Delta = +0.5 and -0.5 landed on ONE block,
and so did +-0.25. That is the identical mistake its own comment describes
having been fixed in run_mixing ("abs(delta) would collide +0.5 with -0.5",
which there overlapped 8 of 1540 cell pairs). Its default grid is
[-4.0, -1.0], both negative, so no published data was affected -- the defect
was reachable, not reached. It is routed through this module now, which
changes its seeds, so asym_sample.csv has to be regenerated to match.

So allocation lives here instead of being open-coded per script. Each
experiment gets a PREFIX, each cell an index inside it, and blocks cannot
overlap by construction as long as the count fits the stride:

    seed = PREFIX[experiment] + cell_index * STRIDE + s,   0 <= s < count

STRIDE is 10**4, so a cell may take up to 10,000 seeds. Prefixes are 10**10
apart and sit above every legacy footprint, so nothing allocated here can
collide with the blocks the three older scripts still compute themselves.
Those are left alone on purpose: changing their seeds would invalidate the
CSVs already measured with them, and they are internally disjoint and tested.
What covers them is the global disjointness test in the suite, which
enumerates every script's real blocks over its default grid rather than
reasoning about spans -- the spans interleave even though the blocks do not,
so a span-based registry would report collisions that are not there.

Seeds reach numpy's default_rng, which takes arbitrary integers; values here
stay below 2**53 so they are exact as floats too, and both samplers were
checked at 9e9, 8e9 and 2**40.
"""
from __future__ import annotations

STRIDE = 10 ** 4

# Prefixes for experiments that allocate through this module. Above every
# legacy footprint (the largest of those is run_mixing's, which reaches
# roughly 2e8) and 10**10 apart from each other.
PREFIX = {
    "run_island": 90 * 10 ** 10,
    "cftp_stopping": 80 * 10 ** 10,
    "run_asymptotics": 70 * 10 ** 10,
}


def cell_index(n: int, delta: float) -> int:
    """A distinct index per (n, Delta) cell.

    Delta is SIGNED through the +10 offset. abs(delta) would map +0.5 and
    -0.5 to one cell, which is exactly how the first attempt at the
    run_mixing fix still overlapped 8 of its 1540 cell pairs, so it is worth
    not repeating.
    """
    if not -10.0 < delta < 10.0:
        raise ValueError(
            f"delta={delta!r} is outside the (-10, 10) the offset assumes; "
            f"widen the offset rather than letting cells wrap onto each other")
    d = int(round((delta + 10.0) * 1000))          # 0 .. 20000
    return n * 100_000 + d


def seed_block(experiment: str, n: int, delta: float, count: int) -> range:
    """The seeds cell (n, Delta) of `experiment` may use.

    Disjoint from every other cell of every registered experiment, by
    construction, provided count <= STRIDE.
    """
    if experiment not in PREFIX:
        raise ValueError(
            f"unknown experiment {experiment!r}; register a prefix in "
            f"experiments/seeds.PREFIX rather than inventing a base inline, "
            f"which is how four seed collisions got into this project")
    if not 0 < count <= STRIDE:
        raise ValueError(f"count must be in 1..{STRIDE}, got {count}")
    base = PREFIX[experiment] + cell_index(n, delta) * STRIDE
    return range(base, base + count)
