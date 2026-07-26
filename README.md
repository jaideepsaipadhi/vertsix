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
physics, not a bug. We confirmed this directly: comparing mixing behavior
across Delta = 0.25, 0.5 (disordered, both mix in a healthy, improving
way) versus Delta = -1.5, -3 (antiferroelectric, both get stuck almost
immediately regardless of how extreme), the transition happens precisely
at the known theoretical phase boundary |Delta|=1. This is critical
slowing down for local Monte Carlo dynamics in ordered phases, a
well-documented phenomenon in the literature, not something specific to
this tool.

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
- **save PNG / save SVG** — export the current view.

## Files

```
sixvertex/sampler.py                General-weight MCMC engine (torch + numpy fallback), CFTP-only in production
sixvertex/cftp.py                    Coupling From The Past exact sampler (restricted regime, see below)
server.py                            Flask app: /api/exact/start + /api/exact/status (CFTP), static hosting
static/sixvertex.js                  Client-side live sampling engine (mirrors sampler.py's verified dynamics)
static/index.html, style.css, draw.js   Browser UI
requirements.txt
```

## Exact sampling (CFTP)

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

`/api/exact/start` is capped at n<=250 in the server for interactive use, with
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
- **save PNG** exports the current view as a raster image.
- **save SVG** exports the full grid as a vector `<rect>`-per-face SVG.
  Note: this is one rect per lattice face, so file size grows with n^2 —
  fine through a few hundred n, unwieldy much beyond that.

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

## Contributing

Issues and pull requests are welcome. Please include your `n`, weights,
and device (desktop/mobile) when filing a bug.

## License

MIT — see [LICENSE](LICENSE).
