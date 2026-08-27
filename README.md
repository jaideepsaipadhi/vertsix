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

## Known limitation: slow mixing when |Delta| > 1

When |Delta| > 1 (ferroelectric or antiferroelectric regime), **live
sampling ("Run") will look visually stuck** — this is expected, verified
physics, not a bug in the underlying algorithm. We confirmed this two
separate ways:

1. Comparing mixing behavior across Delta = 0.25, 0.5 (disordered, both
   mix in a healthy, improving way) versus Delta = -1.5, -3
   (antiferroelectric, both get stuck almost immediately regardless of how
   extreme), the transition happens precisely at the known theoretical
   phase boundary |Delta|=1.
2. At a1=a2=b1=b2=1, c1=c2=sqrt(8) (Delta=-3), we brute-force-verified the
   *algorithm itself* is exactly correct at small n (n=4, max deviation
   0.16% from the true distribution) — so there is no bug in the
   acceptance-ratio computation. But at n=30, running the same dynamics
   for 30,000 sweeps across 3 independent seeds, the fraction of
   locally-flippable sites stayed locked around 3-4%, nowhere near the
   true equilibrium value of ~25% (confirmed against 8 independent exact
   CFTP samples at the same n and weights). **This means the mixing
   problem is already severe well before "large" n — it is not something
   that only shows up once n gets big.** An earlier version of this
   document said the opposite; that was wrong, and this is the corrected
   version.

This is critical slowing down for local Monte Carlo dynamics in ordered
phases, a well-documented phenomenon in the literature, not something
specific to this tool.

**Exact Sample (CFTP) still gives a mathematically correct result in this
regime** — CFTP's correctness never depends on mixing time, only its
runtime does. Verified directly: at a1=a2=b1=b2=1, c1=c2=sqrt(8)
(Delta=-3, deep antiferroelectric), CFTP coalesces correctly at n=40, 60,
80 (1.8s, 8s, 27s respectively) — just needing substantially more
half-sweeps than in the disordered regime. Runtime grows quickly with n in
this regime, though, so for large n and extreme Delta, consider reducing n
first. The UI shows a note when |Delta|>1 explaining this tradeoff.

**CFTP runs as a background job, not a single HTTP request.** On
constrained/free hosting tiers, a CFTP run that takes minutes can exceed
the platform's own request timeout, killing the connection before the
(correct, still-running) computation can respond — this happened in
practice at n=140 in the deep antiferroelectric regime on Render's free
tier. `/api/exact/start` returns almost immediately with a job id; the
actual computation runs in a background thread; the frontend polls
`/api/exact/status/<job_id>` roughly once a second. Verified directly:
individual poll requests stay under 15ms even while the underlying job
runs for 10+ seconds in the background — the fix works regardless of how
long the actual computation takes, not just for the specific cases we
happened to test.

We looked into making *live* sampling (or CFTP at large n) fast in this
regime too. That would require fundamentally different, non-local moves
(cluster/worm-type algorithms). We checked the literature directly:
transfer matrices for domain-wall boundary conditions are explicitly known
to be non-diagonalizable in general — a real, documented structural
obstruction, not something nobody has looked for. This would be new
research if attempted, not a scoped engineering fix, so we haven't
attempted it.

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
- **n > 14, uniform weights only:** CFTP (`sixvertex/cftp.py`), which is
  valid at that point.
- **n > 14, non-uniform weights:** *no exact method is available.* The tool
  refuses and says so, rather than returning a number it cannot justify.
  Sequential sampling costs `C(n, n/2)` and CFTP is invalid there. Closing
  this gap is an open problem, not an engineering task.

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
