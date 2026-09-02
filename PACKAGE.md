# vertsix

Exact samplers for the six-vertex model with domain-wall boundary conditions.

```bash
pip install vertsix
```

```python
import math, sixvertex as sv   # installed as `vertsix`

# a = b = 1, c = sqrt(8)  ->  Delta = -3, antiferroelectric
w = dict(a1=1., a2=1., b1=1., b2=1., c1=math.sqrt(8), c2=math.sqrt(8))

H, info = sv.sample(80, w, seed=1)
print(info["method"])          # 'cftp'
print(sv.delta(w))             # -3.0
```

`H` is the height function as an `(n+1) x (n+1)` integer array.

## What is exact, and where

The point of the package is that it will not hand you a configuration it
cannot justify. Three methods cover complementary regimes:

| method | valid for | cost |
|---|---|---|
| `exact_sample` | **any** positive weights | `C(n, n/2)`, so `n <= 14` |
| `cftp_sample` | `c1*c2 >= a1*a2` and `c1*c2 >= b1*b2` | `~n^4` |
| `stochastic.sample` | stochastic weights, free exit, `Delta >= 1` | `O(n^2)` |

`sv.sample(...)` picks between the first two and **raises** if neither
applies, rather than silently falling back on a Markov chain.

The CFTP condition is not a convenience bound. The single-site update has two
outcomes, so a monotone coupling exists only if the two chains' flip-up
marginals are ordered; outside that region they are not, and therefore *no*
coupling of the update is monotone, however it is constructed. That region is
the antiferroelectric-favouring one, so `Delta < -1` is inside it and exactly
samplable.

## The Markov chain, if you want it

```python
s = sv.SixVertexSampler(n=64, a1=1, a2=1, b1=1, b2=1, c_up=1, c_down=1)
s.step(sweeps=1000)
H = s.height_array()
```

Correct for arbitrary weights, but it is a Markov chain and in the ordered
phases it mixes slowly. A configuration drawn from it is not guaranteed to be
a sample from the measure. If you need that guarantee, use the exact methods
above.

## The stochastic model

```python
h, v = sv.stochastic.sample(n=300, b1=0.3, b2=0.8, seed=1)
```

Vertex weights are conditional probabilities, so one sweep of the lattice
produces an exact sample -- no Markov chain, no mixing time. Its anisotropy is
`(b1 + b2) / (2 sqrt(b1 b2)) >= 1` by AM-GM, so it covers the ferroelectric
regime.

## Conventions

Standard six-vertex labelling. With `A = a1*a2`, `B = b1*b2`, `C = c1*c2`,

    Delta = (A + B - C) / (2 sqrt(A B))

Only the products matter: `a1` and `a2` never appear separately in the
measure, which is a conservation law of the boundary conditions rather than a
modelling choice.

The labelling is pinned to the Izergin-Korepin determinant for DWBC, itself
validated by reproducing the alternating-sign-matrix numbers 1, 2, 7, 42, 429,
7436, 218348 at the ice point.

## Optional extras

```bash
pip install vertsix[torch]   # GPU/torch backend for the MCMC engine
pip install vertsix[web]     # the Flask demo app
```

## Links

- Interactive version: <https://vertsix.com>
- Source and full notes: <https://github.com/jaideepsaipadhi/vertsix>

MIT licensed.
