# six-vertex sampler

A sampler and viewer for the six-vertex model with domain-wall boundary
conditions (DWBC) — in the spirit of Leonid Petrov's
[domino-draw](https://lpetrov.cc/domino-draw/) and
[lozenge-draw](https://lpetrov.cc/lozenge-draw/), for the six-vertex / ASM
case that (per Vadim Gorin) didn't yet have an easily accessible
implementation.

## What it does

The six-vertex model with DWBC is in exact bijection with integer **height
functions** on an (n+1)x(n+1) grid (the same bijection used for alternating
sign matrices). Live sampling runs a **corner-flip / heat-bath Markov
chain** on this height function, correctly weighted for the full general
model (all six vertex weights a1, a2, b1, b2, c1, c2):

- every interior grid point that is a strict local max or local min of its
  four neighbours can flip to the other value,
- flipping a site changes the classification of up to 4 surrounding
  vertices at once, so the acceptance ratio is computed from the actual
  local neighbourhood at flip time, using all six weights correctly,
- parallel updates use a 4-coloring of the grid (`(i%2, j%2)`), not the
  simpler 2-coloring `(i+j)%2` — under this dynamics, diagonal neighbours
  of the same `(i+j)%2` parity can share a face, so 2-coloring introduces a
  small but real, measurable sampling bias. This was caught and fixed by
  comparing against exact brute-force distributions for small n.

The real anisotropy parameter is (matching the standard literature
convention, e.g. Gorin & Nicoletti's *Six-vertex model and random matrix
distributions*):

```
Delta = (a1*a2 + b1*b2 - c1*c2) / (2 * sqrt(a1*a2*b1*b2))
```

The UI exposes all six weights individually (a1, a2, b1, b2, c1, c2), with
a **symmetric** checkbox that links a1=a2, b1=b2, c1=c2 for the common
case. Delta is computed and displayed live. Note: uniform weights (all = 1)
give Delta = 0.5, *not* 0 — Delta = 0 is a different, specific point (the
free-fermion locus, where a1a2 + b1b2 = c1c2).

**Live sampling runs entirely client-side**, in `static/sixvertex.js` —
there is no server round-trip per step. The Python/Flask backend
(`sixvertex/sampler.py`, with PyTorch + NumPy fallback) exists for
**exact sampling (CFTP) only** — see below.

The viewer supports three view modes: a smoothed height-field color map, a
liquid/frozen (arctic boundary) view, and a **6 vertex types (colored)**
view that colors every face by its actual vertex type, with a legend —
useful for visually checking the simulation against theory independent of
the smoothed color gradient.

## Correction: the MCMC is NOT stuck at |Delta| > 1

Earlier versions of this file, and an email to a collaborator, claimed that
local sampling fails to equilibrate in the ordered phases -- citing that at
`a=b=1, c=sqrt(8)` (`Delta=-3`) the flippable-site fraction sat near 3% while
"exact" samples showed ~25%.

**That comparison was invalid.** The ~25% came from the old `cftp.py`, which
used `p_up = c1/(c1+c2)` and therefore sampled the wrong measure entirely at
these weights (it is off by 64% at `c=sqrt(8)`; see the correction notice in
ALGORITHM.md). The MCMC was being measured against a broken reference.

With a correct reference -- the sequential transfer-matrix sampler, which is
exact for arbitrary weights -- the true equilibrium flippable fraction at
`Delta=-3` is

| `n` | exact flippable fraction | `n x fraction` |
|---|---|---|
| 8 | 0.1456 | 1.165 |
| 10 | 0.1154 | 1.154 |
| 12 | 0.0935 | 1.122 |
| 14 | 0.0795 | 1.113 |

i.e. it scales as about `1.14/n` and *decreases* with system size. Predicted
at `n = 64`: `0.0178`. Measured by MCMC at `n = 64`: `0.0164`. At `n = 128`:
predicted `0.0089`, measured `0.0081`.

So the chain equilibrates. A low flippable fraction in the antiferroelectric
phase is the *correct* equilibrium behaviour, not evidence of being stuck: the
zig-zag structure that dominates there leaves very few local extrema, which is
exactly what a "flippable site" is. The c-vertex fraction, a better diagnostic,
reaches 0.98 within 2000 sweeps at `n = 64` and 0.988 at `n = 128`.

The visual impression of a "frozen" picture is likewise correct physics rather
than a failure -- compare Figure 17 of arXiv:2309.12495, whose caption notes
that in the gaseous region "the paths form a regular zig-zag pattern, with
only occasional defects".

## Setup

```bash
cd six-vertex-sampler
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is CPU-only (flask, numpy, gunicorn) since the deployed
server only needs to run CFTP, which is fast enough on CPU in typical
regimes (capped at n<=250 specifically for this reason). If you want to
experiment with the Python/torch MCMC engine directly (e.g. from a
notebook, not through the web UI), `pip install torch` separately.

## Run

```bash
python server.py
```

Then open **http://127.0.0.1:5000** in a browser.

Controls:
- **size n** — grid dimension (n x n vertices). Reduced automatically to 40
  on narrow/touch-only devices for performance.
- **symmetric** checkbox — link a1=a2, b1=b2, c1=c2 (the common case).
  Uncheck to set all six weights independently.
- **a1, a2, b1, b2, c1, c2 weight sliders** — the six vertex weights.
- **Delta** — computed live from the current weights, with regime label
  (ferroelectric / disordered / antiferroelectric). A note appears when
  |Delta|>1 explaining the mixing tradeoffs above.
- **sweeps per frame** — how many full 4-color sweeps run between each
  redraw while playing.
- **reset** — reinitialize with current n/weights.
- **run / pause** — continuously sample, client-side.
- **step** — advance one batch of sweeps manually.
- **exact sample (CFTP)** — see below. Disabled (with an explanation)
  unless a1=a2=b1=b2=1 exactly. Shows live elapsed time while running.
- **save PNG / save SVG** — export. PNG is a screenshot and honours the
  current zoom and pan, so a zoomed-in view exports a crop; SVG exports the
  complete configuration regardless of zoom or pan. See Pan / zoom / export
  below.

## Files

```
sixvertex/sampler.py                General-weight MCMC engine (torch + numpy fallback), CFTP-only in production
sixvertex/cftp.py                    Coupling From The Past exact sampler (restricted regime, see below)
server.py                            Flask app: /api/exact/start + /api/exact/status (CFTP), static hosting
static/sixvertex.js                  Client-side live sampling engine (mirrors sampler.py's verified dynamics)
static/index.html, style.css, draw.js   Browser UI
requirements.txt
```

## Exact sampling

**Correction, August 2026:** this tool previously claimed CFTP produced
exact samples whenever a1=a2=b1=b2=1, for any c1, c2. That was wrong. CFTP's
move uses `p_up = c1/(c1+c2)`, which only targets the correct measure at the
uniform point (all weights 1); away from it the sampled distribution
deviates from the truth by 4% at c1=1.2, 21% at c1=2.0, and 64% at
c1=c2=sqrt(8) (measured against brute-force enumeration at n=4). The bug
went unnoticed because the error vanishes smoothly at the uniform point, and
because MCMC and CFTP were checked against each other while sharing the same
flawed move.

Exact sampling now works as follows:

- **n <= 14, any weights:** exact *sequential* sampling via a
  transfer-matrix decomposition (`sixvertex/exact.py`). Rows are drawn one
  at a time from their exact conditional distributions. No Markov chain, so
  no mixing time and no monotonicity requirement -- it is exact even deep in
  the ferroelectric/antiferroelectric regimes where local MCMC freezes.
  Verified against brute force to machine precision.
- **n > 14, with `b1b2 >= a1a2` and `b1b2 >= c1c2`:** CFTP
  (`sixvertex/cftp_exact.py`), using the correct four-face heat-bath rule.
  The shared-uniform coupling of the extremal chains is monotone exactly on
  this region, so CFTP is valid throughout it — not merely at the uniform
  point, which lies on its boundary.
- **n > 14, outside that region:** *no exact method is available.* This is not
  a gap in the implementation: outside the region no monotone coupling of the
  single-site update exists at all, since a monotone coupling forces the two
  chains' flip-up marginals to be ordered and they are not. Sequential
  sampling costs `C(n, n/2)`, so it cannot cover large n either. The tool
  refuses rather than returning a number it cannot justify.

See [ALGORITHM.md](ALGORITHM.md) for the full specification of both methods.

## CFTP details

For a precise, line-by-line-checkable specification of exactly what this
computes (state space, move set, transition probability, coupling,
doubling scheme, and exactly what's proven vs. empirically verified), see
[ALGORITHM.md](ALGORITHM.md).

The **exact sample (CFTP)** button runs Coupling From The Past
(Propp-Wilson): two coupled copies of the corner-flip chain start from the
pointwise-minimal and pointwise-maximal valid height functions and run
backward from time -T to 0 with *identical* randomness. If they coalesce
(agree everywhere) by time 0, the common result is a mathematically exact
sample from the true stationary distribution — not an MCMC approximation,
no burn-in, no mixing-time guesswork. If they haven't coalesced, T doubles
and the whole run repeats from further back (reusing the same near-present
randomness, so work isn't wasted).

**This is only implemented, and only proven safe, for a1=a2=b1=b2=1**
(c1, c2 free to differ, including deep ferroelectric/antiferroelectric
values). The general-weight dynamics used for live MCMC sampling does
*not* preserve the monotone coupling this relies on — this was checked
directly by simulation (a shared-seed coupling of the general dynamics
broke pointwise ordering within ~50 sweeps), not just assumed. The UI
enforces this by disabling the button whenever a1, a2, b1, or b2 move
away from 1.

To be precise about how strong a claim this actually is: monotonicity at
a1=a2=b1=b2=1 is based on extensive empirical verification (thousands of
coupled half-sweeps with zero ordering violations, plus cross-checking
CFTP's center-height statistics against long-run MCMC), not a cited
theorem. If you know of a formal proof (or counterexample) for this
specific case, please open an issue.

`/api/exact/start` limits n to 250 for interactive use, with
up to 2^21 half-sweeps allowed per coalescence attempt (raised from an
earlier, smaller cap after finding it could cut off large-n runs in the
antiferroelectric regime before they finished). At n=80, uniform weights,
this typically takes 15-25 seconds; in the deep antiferroelectric regime
it can take substantially longer at large n. The UI shows live elapsed
time and an expectation-setting note past 5 seconds so this isn't mistaken
for the tool being broken.

## Seeing the arctic boundary

At large n, the "height field" color view can look smooth even though the
liquid region is really there — the overall boundary range (0 to n) dwarfs
the local fluctuations, so they wash out visually. Switch **view mode** to
**liquid / frozen (arctic boundary)** to see it directly, or **6 vertex
types (colored)** to see the actual underlying vertex configuration.

## Pan / zoom / export

- **scroll** on the canvas to zoom, **drag** to pan.
- **save PNG** is a screenshot: it captures *exactly what is on screen*,
  including the current zoom and pan. If you are zoomed in, the PNG contains
  only the visible crop — verified: at 1083% zoom on an n=40 lattice the PNG
  held a fraction of the configuration. Nothing about the file looks wrong,
  so this is easy to miss. Reset the view (double-click) before exporting if
  you want the whole lattice.
- **save SVG** exports the complete configuration regardless of zoom or pan —
  the same n=40 case produced all 40 cells. This is the one to use for a
  figure. It writes one `<rect>` per lattice face, so file size grows with
  n^2: fine through a few hundred n, unwieldy much beyond that.

## Extending

- `SixVertexSampler.to_binary_frame()` / `.height_array()` can be used
  directly in a notebook if you just want the raw samples without the web
  UI.
- Exact sampling for the fully general (a1, a2, b1, b2, c1, c2) model is an
  open problem in this codebase, not just an engineering gap — see the
  CFTP section above before attempting to extend it.
- A genuinely fast algorithm for the ordered (|Delta|>1) phase would need
  non-local moves — see "Known limitation" above before attempting this;
  it's real research, not a quick fix.

## Deployment: the server must run as a single worker

`Procfile` pins `--workers 1`. That is a correctness requirement, not a
performance choice.

Exact sampling keeps state in process memory: the background job store, the
per-client session store, and the transfer-matrix cache. None of it is shared
across processes, so with two or more workers a load balancer can start a job
on one worker and route the client's status poll to another. Verified: the
poll returns `404 unknown or expired job`, and a session created on one worker
is rejected by the other.

Raising the worker count will silently break exact sampling for every user.
The correct way to get concurrency is to move that state into Redis or a
database.

## HTTP API

The browser UI does live sampling entirely client-side and only calls
`/api/config` and the exact-sampling endpoints. The rest are here for
scripting.

**`GET /api/config`** — authoritative client-facing constants.
```json
{"ok": true, "max_exact_n": 14}
```

**`POST /api/init`** — build a sampler and open a session.
Body: `n`, and any of `a1 a2 b1 b2 c_up c_down` (default 1.0), optional `seed`.
Returns `session_id`, `max_exact_n`, `exact_available`, `device`,
`using_torch`, `using_gpu`, and a `frame`.

`using_gpu` is derived from the device actually in use. `using_torch` only
reports that the optional torch backend imported; torch on a CPU-only host
is `using_torch: true` with `using_gpu: false`. The deployed instance runs
neither — `requirements.txt` is CPU-only by design, and live sampling
happens in the browser.

**`POST /api/step`** — advance a session's chain.
Body: `session_id` (**required**), `sweeps`.
Rejects with `400` if `session_id` is missing/expired, or if
`n^2 * sweeps` exceeds the per-request work budget — the endpoint runs
synchronously on a single worker, so an unbounded request would stall the
whole server. The error names a sweep count that will work.

**`POST /api/exact/start`** — begin exact sampling; returns `job_id`
immediately. Long computations must not sit on an open HTTP request.
Returns `429` if `_MAX_CONCURRENT_JOBS` are already running.

**`GET /api/exact/status/<job_id>`** — poll. `status` is
`running` | `done` | `error`. On `done` the body carries `frame` and `info`
(`info.method` is `exact-sequential` or `cftp`). Jobs stuck in `running`
past a watchdog threshold are reaped and reported as `error`.

### Out-of-range values are rejected, not clamped

`n` and `sweeps` outside their supported ranges return `400` rather than
being silently adjusted. Quietly running 500 sweeps for a request that asked
for 9999 -- as this did previously -- leaves the caller believing the chain
is far more equilibrated than it is, which is the same silent-wrong-data
failure as a picture labelled with parameters it was not drawn from.

### Breaking changes

Two changes to `/api/init` and `/api/step` will break older scripts:

* **`/api/step` now requires `session_id`.** State used to be a single
  module-level sampler shared by every caller, so two clients clobbered each
  other and `/api/step` could return the *other* client's model. Sessions are
  scoped, TTL-bounded, and capped.
* **`/api/init` no longer returns `is_symmetric_regime`.** It tested
  `a1=a2=b1=b2=1` — the superseded CFTP criterion, which ignored `c1,c2` and
  matched no decision the server actually makes. `/api/exact/status` also
  hardcoded it to `true`, so the two endpoints contradicted each other for
  identical weights. Use `exact_available` instead.

## Why large n outside the monotone region is not an engineering gap

Exact sampling is unavailable for large `n` when `b1b2 < a1a2` or
`b1b2 < c1c2`. Three separate facts explain why, and none of them is a
missing feature.

**The region never reaches the antiferroelectric phase.** With
`A = a1a2, B = b1b2, C = c1c2`, the anisotropy is
`Delta = (A + B - C) / (2 sqrt(AB))`. Inside the region `C <= B`, so
`A + B - C >= A > 0` and hence

    Delta >= sqrt(A/B) / 2 > 0.

So the monotone region is contained in `{Delta > 0}`, and the whole
antiferroelectric phase `Delta < -1` lies outside it. This is exactly the
regime where the live MCMC visibly freezes, so the place a user most wants
exact sampling is the place the region cannot reach.

**Outside the region, CFTP is impossible rather than unimplemented.** A
monotone coupling of a two-outcome update exists only if the marginals are
ordered, and outside the region they are not — so no coupling of the
single-site update is monotone, however it is constructed.

**Non-monotone exact methods exist but inherit a different obstruction.**
Bounding chains (Huber, *Perfect sampling using bounding chains*, Ann. Appl.
Probab. 14 (2004); Häggström–Nelander; Kendall–Møller) give perfect sampling
without monotonicity, so the argument above does not close them off. But
they are still driven by the underlying local chain, and local chains for
this model are provably torpidly mixing in both ordered phases — Liu,
*Torpid Mixing of Markov Chains for the Six-vertex Model on Z^2*
(APPROX/RANDOM 2018), covering Glauber dynamics and the directed-loop
algorithm, strengthened for the ferroelectric case by Fahrbach–Randall
(APPROX/RANDOM 2019).

**DWBC does not escape this.** An earlier version of this section reported no
degradation, but that test used `Delta = 1.5` — barely past the phase
boundary — and read coalescence off the doubling schedule, which only reports
powers of two. Measuring forward coalescence directly (the first sweep at
which the two extremal chains agree) and going deeper into the phase shows
the blow-up clearly. At `Delta = 5`:

| `n` | sweeps to coalesce |
|---|---|
| 6 | 78 |
| 8 | 1956 |
| 10 | 32506 |

That is roughly 20x per step of two in `n`, where `n^2` growth would be about
1.6x — exponential, matching the torpid-mixing predictions.

The onset is near `Delta = 2` rather than at the phase boundary `Delta = 1`.
At `n = 12`: about 100 sweeps for `Delta <= 1.5`, 293 at `Delta = 2`, 1647 at
`Delta = 2.5`, 4564 at `Delta = 3`.

**Consequence: inside the region means correct, not fast.** The monotone
region guarantees CFTP is *valid*; it says nothing about how long
coalescence takes. Deep in the ferroelectric phase the tool would accept a
large-`n` request and then run until the watchdog reaped it. The UI now warns
when `Delta > 2` and `n` is above the sequential-sampler limit.

Separately, computing the partition function itself is `#P`-hard for generic
weights even on planar graphs (Cai–Fu–Shao trichotomy), which rules out the
transfer-matrix route scaling to large `n`.

## Performance of exact sampling

CFTP cost scales as `n^4`: the number of sweeps to coalescence grows like
`n^2`, and each sweep touches `O(n^2)` sites. Measured, weights `B=4, A=1, C=1`:

| `n` | sweeps | wall clock |
|---|---|---|
| 32 | 2048 | 4.2 s |
| 64 | 4096 | 10.6 s |
| 100 | 16384 | 59 s |
| 150 | — | ~5 min (projected) |
| 200 | — | ~16 min (projected) |

Two optimisations account for most of the current speed, both found by
profiling rather than guesswork:

1. **Packed-bit weight lookup.** The face-type classifier was a chain of six
   full-array `np.where` calls, invoked 32 times per sweep — 79% of runtime.
   Packing `(l,t,b,r)` into a 4-bit index and taking from a 16-entry table
   replaces it with one gather.

2. **Restricting to the active colour class.** Each sweep updates one colour
   at a time, so three quarters of every array the classifier touched was
   discarded immediately afterwards. Operating on flat indices of the active
   sites cut per-sweep cost at `n=200` from 14.4 ms to 0.51 ms.

Together these took the per-sweep cost at `n=200` from 18.8 ms to 0.51 ms,
about 37x.

**GPU.** `sixvertex/sampler.py` carries an optional torch path for the MCMC
engine, but the CFTP sampler is numpy-only and the deployed instance is
CPU-only by design (`requirements.txt` has no torch, and the free tier has no
GPU). Porting the four-face rule to torch is the obvious next step for `n` in
the high hundreds; it has not been done here because this environment has
neither torch nor a GPU, and shipping unverified numerical code is exactly
how the original CFTP weight bug happened.

## Tests

```bash
python3 tests/test_sampler.py        # no pytest needed
python3 -m pytest tests/ -v          # or with pytest
```

Every correctness test compares against **brute-force enumeration**, not
against the other sampler. That matters: this project has shipped two subtle
correctness bugs that survived for months precisely because the MCMC and
CFTP paths were validated against each other while sharing the same flawed
assumption. Cross-checking two implementations of the same mistake proves
nothing.

The suite also pins down a *non*-bug: under DWBC the differences
N(a1)-N(a2), N(b1)-N(b2), N(c1)-N(c2) are conserved, so swapping a1 and a2
leaves the distribution exactly unchanged. Naive "boost one weight and see
which vertex type grows" tests flag this as a wiring error. It isn't.

## Contributing

Issues and pull requests are welcome. Please include your `n`, weights,
and device (desktop/mobile) when filing a bug.

## License

MIT — see [LICENSE](LICENSE).
