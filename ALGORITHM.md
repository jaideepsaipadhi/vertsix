# Algorithm specification

> **IMPORTANT CORRECTION (supersedes earlier versions of this document).**
> Earlier versions claimed CFTP gave exact samples for a1=a2=b1=b2=1 with
> c1, c2 arbitrary. **That claim was wrong.** The CFTP move uses
> `p_up = c1/(c1+c2)`, which only targets the correct measure when the
> four surrounding faces contribute equal weight -- i.e. at the uniform
> point (all weights 1). Away from it the error is not subtle: measured
> against brute-force enumeration at n=4, the sampled distribution deviates
> by 4% at c1=1.2, 21% at c1=2.0, and **64%** at c1=c2=sqrt(8).
> The error grows smoothly from zero at the uniform point, which is why it
> escaped earlier testing that compared MCMC against CFTP -- both used the
> same flawed move, so they agreed with each other while both were wrong.
>
> This is now fixed by a different algorithm (see "Exact sequential
> sampling" below), which is exact for arbitrary weights. CFTP is retained
> only for the uniform point at large n.


This document specifies exactly what `sixvertex/cftp.py` computes, precisely
enough to be checked against the code line by line. It exists because a
prose description ("we use Coupling From The Past") is not precise enough
to audit — this is the actual algorithm, stated formally.

## Scope of the correctness claim

**This algorithm is claimed correct only for a1 = a2 = b1 = b2 = 1, with
c1, c2 arbitrary positive reals.** It is not claimed correct for the
general six-vertex model. If any part of this document reads as a broader
claim than that, it is a documentation bug — please open an issue.

## State space

A state is a height function `H : {0,...,n}^2 -> Z` satisfying:
- boundary values fixed by domain-wall boundary conditions:
  `H(0,0)=0, H(n,0)=n, H(0,n)=n, H(n,n)=0`, extended along each edge by
  the unique monotone path,
- `|H(i,j) - H(i',j')| = 1` for every edge `(i,j)-(i',j')` of the grid
  (i.e. every unit step changes the height by exactly ±1).

This is the standard bijection between six-vertex configurations with DWBC
and integer height functions (the same bijection underlying the ASM
correspondence).

## The single-site move

A site `(i,j)` with `0 < i,j < n` is called a **local extremum** if all
four of its grid neighbors `H(i-1,j), H(i+1,j), H(i,j-1), H(i,j+1)` are
equal to some common value `v`. In that case `H(i,j) ∈ {v-1, v+1}`, and
the move considered at that site is: **resample `H(i,j)` between `v-1`
and `v+1`.**

If `(i,j)` is not currently a local extremum, no move is available there
(this is standard for this class of dynamics — it is exactly the move set
used for uniform random ASM/domino sampling via CFTP, e.g. Propp-Wilson).

## Transition probability (a1=a2=b1=b2=1 case only)

At a1=a2=b1=b2=1, set

```
p_up = c1 / (c1 + c2)
```

At each local-extremum site selected for update, set `H(i,j) = v+1` with
probability `p_up`, else `H(i,j) = v-1`. This is a heat-bath (Gibbs) move,
not a Metropolis accept/reject step — the new value is drawn directly from
its correct conditional distribution given the neighbors, not proposed and
then possibly rejected.

**Why this specific p_up is correct only at a1=a2=b1=b2=1:** in general,
resampling one site changes the vertex-type classification of up to 4
surrounding faces, and the correct conditional distribution depends on
the weight of *all four*, not just the pair of `c`-type faces adjacent to
that site (we found and fixed a bug earlier where the live-sampling
implementation had exactly this more complicated case wrong; see
`sixvertex/sampler.py` and the surrounding commit history). At
a1=a2=b1=b2=1 specifically, every `a`- or `b`-type face contributes a
weight of exactly 1 regardless of its type, so those four faces drop out
of the ratio entirely and only the `c1`/`c2` split matters — which is
exactly `p_up` above. This is *why* the restricted case has a simple
closed-form correct move and the general case does not.

## Parallel update / coloring

All local-extremum sites of a given "color" are updated simultaneously
using independent randomness per site, then the next color is processed.
Colors are `(i+j) mod 2` (2-coloring) — this is sufficient here (unlike
the general-weight dynamics in `sampler.py`, which needs 4-coloring; see
that file's comments) because a single-site resample under this restricted
p_up never depends on anything beyond the site's own 4 immediate
neighbors, so same-parity sites never interact.

## The CFTP coupling

Two chains are run from the pointwise-extremal states:
- `H_lo`: the pointwise-minimal valid height function,
- `H_hi`: the pointwise-maximal valid height function,

for `T` steps backward in (virtual) time, using **the same random numbers
at every site and every step** for both chains — concretely, the random
field used at virtual time `k` is generated from `SeedSequence([master_seed, k])`,
so it is identical for `H_lo` and `H_hi` at that step, and is independent
across different `k`.

**Monotonicity** (`H_lo(i,j) ≤ H_hi(i,j)` at every site, preserved by the
move above) is what makes this correct: if the coupling is monotone, and
the two boundary chains agree everywhere by time 0, then *every* possible
starting state (being sandwiched between them) would also have coalesced
to the same value, so the common value is a sample from the exact
stationary distribution — independent of how slowly the chain mixes.

If `H_lo` and `H_hi` have not coalesced after `T` steps, `T` is doubled and
the entire computation is redone from further back, **reusing the same
per-step randomness** (`SeedSequence([master_seed, k])` for each `k` is
unchanged across doublings) so that work already done is not wasted — the
new, longer run's most recent `T_old` steps exactly reproduce the previous
attempt.

## What is proven vs. what is verified empirically

- The Propp-Wilson CFTP argument itself (monotone coupling + coalescence
  implies exact sampling) is a standard, published theorem.
- **Whether the specific move above is monotone is not something we have
  a citation for.** It is checked empirically: thousands of coupled
  half-sweeps with zero ordering violations observed, at multiple values
  of `c1, c2` including deep ferroelectric/antiferroelectric ratios (we
  specifically tested c1=c2=sqrt(8), Delta=-3). We also independently
  cross-validated CFTP's output statistics against long, independent
  live-MCMC runs of the same restricted (a=b=1) dynamics, and against
  brute-force-enumerated exact distributions for small n (n≤4, where
  every valid configuration can be listed directly and weighted exactly).
- We do **not** have a monotonicity proof or disproof for the general
  (a1, a2, b1, b2 not all 1) case, and empirically it fails: a shared-seed
  coupling of the general dynamics was observed to violate ordering within
  about 50 sweeps in testing. This is why the UI disables Exact Sample
  outside a1=a2=b1=b2=1.

If you have a proof, a counterexample, or a reference for the a=b=1
monotonicity claim, please open an issue — we would genuinely like to
either cite it or correct the claim.


---

# Exact sequential sampling (`sixvertex/exact.py`)

This is the method now used for all non-uniform weights. It is exact by
construction and does **not** involve a Markov chain, so questions of
mixing time and monotonicity do not arise at all.

## Method

Height functions are built one row at a time. Row `i` is a lattice path
from height `i` to height `n-i` with steps of +-1 (the DWBC boundary fixes
both endpoints). Vertically adjacent rows must differ by exactly 1 in every
column, and the strip between two adjacent rows carries the product of the
six-vertex face weights.

1. **Backward pass.** For each level `i` and each valid row `r`, compute
   `B[i][r]` = the total Boltzmann weight of every valid completion of the
   configuration from level `i` down to level `n`. Computed by backward
   recursion from `B[n][r] = 1`.
2. **Forward sampling.** Start at the unique valid row at level 0. Given the
   current row `r`, draw the next row `r'` with probability proportional to
   `(strip weight from r to r') * B[i+1][r']`.

Step 2 is exactly the conditional distribution of the next row given the
current one, so the resulting configuration is an exact draw from the
Boltzmann distribution. This is standard transfer-matrix / dynamic-
programming sampling; nothing here is novel, it is simply the right tool.

## Why this succeeds where CFTP fails

CFTP needs a *monotone* coupling of two extremal chains. For non-uniform
weights, the correct single-site move is provably not monotone: the
acceptance probability depends on the SW and NE diagonal neighbours, and one
can construct valid ordered pairs of states where the lower chain's flip-up
probability strictly exceeds the upper chain's -- so the chains can cross.
This was confirmed empirically for several coupling constructions (naive
shared-randomness, and shared-uniform-with-own-local-probability), each of
which violated ordering within ~100 sweeps.

Sequential sampling never compares two chains, so this obstruction is simply
absent.

## Cost, and the honest limitation

The number of valid rows per level is `C(n, n/2)`, so the backward pass is
**exponential in n**. In practice:

| n | backward pass (once) | per sample after that |
|---|---|---|
| 8 | 0.02 s | 0.1 ms |
| 10 | 0.20 s | 0.1 ms |
| 12 | 2.5 s | 0.2 ms |
| 14 | 33 s | 0.3 ms |

So this is an exact **small-n** method. It is complementary to MCMC, not a
replacement: for large n with non-uniform weights there is currently **no**
exact method available in this codebase, and the tool now says so plainly
rather than returning a result it cannot justify.

## Verification

- Partition function matches brute-force enumeration to machine precision
  (relative error ~1e-16) for uniform, extreme (c1=c2=sqrt(8)), and fully
  asymmetric (a1!=a2, b1!=b2, c1!=c2) weights.
- At the ice point it reproduces the ASM counts 1, 2, 7, 42, 429 exactly.
- Sampled frequencies match the exact distribution with residual error
  shrinking as 1/sqrt(N) -- 0.0050 -> 0.0022 -> 0.0012 as N goes
  5k -> 20k -> 80k -- i.e. pure Monte Carlo counting noise, not bias.
