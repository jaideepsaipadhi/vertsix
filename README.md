# six-vertex sampler

A local, GPU-accelerated sampler and viewer for the six-vertex model with
domain-wall boundary conditions (DWBC) — in the spirit of Leonid Petrov's
[domino-draw](https://lpetrov.cc/domino-draw/) and
[lozenge-draw](https://lpetrov.cc/lozenge-draw/), for the six-vertex / ASM
case that (per Vadim Gorin) doesn't yet have an easily accessible
implementation.

## What it does

The six-vertex model with DWBC is in exact bijection with integer **height
functions** on an (n+1)x(n+1) grid (the same bijection used for alternating
sign matrices). The sampler runs the standard **corner-flip / heat-bath
Markov chain** on this height function:

- every interior grid point that is a strict local max or local min of its
  four neighbours can flip to the other value,
- the flip is accepted with probability set by the ratio of the two
  "turning" vertex weights `c_up` and `c_down`,
- setting `c_up == c_down` recovers the uniform (ice-point) measure on
  height functions / ASMs.

Because a flip at site `(i,j)` only ever looks at neighbours of the
*opposite* checkerboard colour, every site of one colour can be updated
**in parallel** as a single tensor operation — this is what makes it GPU
friendly. The engine is written in PyTorch; if `torch` isn't installed (or
no CUDA GPU is available) it transparently falls back to a NumPy CPU
version, just slower.

The viewer renders the height function as a colored field (deep blue -> ice
cyan -> amber), which shows the "arctic circle" phenomenon the same way
Petrov's dimer/lozenge sites do.

**Caveat:** this single-flip local chain is the standard tool used for
sampling ASMs/six-vertex configurations, but like any local MCMC its mixing
time grows with system size and can be slow far from the ice point — this
is a limitation of the sampling method itself, not the implementation.

## Setup

```bash
cd six-vertex-sampler
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# CPU-only (simplest, works everywhere):
pip install flask numpy

# For GPU acceleration, install torch matching your system instead, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
# (see https://pytorch.org/get-started/locally/ for the right command
#  for your OS / CUDA version / Apple Silicon (mps))
```

## Run

```bash
python server.py
```

Then open **http://127.0.0.1:5000** in a browser.

Controls:
- **size n** — grid dimension (n x n vertices).
- **Δ = ln(c_up / c_down)** — weight bias between the two turning vertex
  types. Δ=0 is the uniform/ice point; large |Δ| pushes the frozen
  ("arctic") corners further in and slows mixing.
- **sweeps per frame** — how many full checkerboard sweeps run between
  each redraw while playing.
- **reset** — reinitialize with current n/Δ.
- **run / pause** — continuously sample.
- **step** — advance one batch of sweeps manually.
- **save PNG** — export the current frame.

## Files

```
sampler.py / sixvertex/sampler.py   Core MCMC engine (torch + numpy fallback)
server.py                           Flask app: /api/init, /api/step, static hosting
static/index.html, style.css, draw.js   Browser UI
requirements.txt
```

## Exact sampling (CFTP)

The **exact sample (CFTP)** button runs Coupling From The Past (Propp-Wilson):
two coupled copies of the corner-flip chain start from the pointwise-minimal
and pointwise-maximal valid height functions and run backward from time -T to
0 with *identical* randomness. If they coalesce (agree everywhere) by time 0,
the common result is a mathematically exact sample from the true stationary
distribution — not an MCMC approximation, no burn-in, no mixing-time
guesswork. If they haven't coalesced, T doubles and the whole run repeats
from further back (reusing the same near-present randomness, so work isn't
wasted).

This relies on the chain being *monotone*: order between two height
functions is preserved when both are updated with the same random field.
That took one real bug fix to get right during development — an early
version used separate independent thresholds for "flip a local max down" vs
"flip a local min up," which can let two coupled chains swap order at a site
where one has a max and the other a min. The fix: compute a single shared
"target value" from the random number first, then apply it to whichever
chain sits at that extremum, so both chains agree at that site whenever
their neighbourhoods already agree. Verified two ways: by direct simulation
(order preserved across thousands of coupled half-sweeps with no
violations) and by cross-checking CFTP's center-height statistics against a
long-run, burned-in MCMC chain (both landed at the same mean, well within
sampling noise).

`/api/exact` is capped at n<=250 in the server for interactive use — CFTP
cost grows with the chain's mixing time, which itself grows with n and
shrinks as the weight bias Delta moves away from 0. For unbiased (Delta=0)
large n this can be slow; that's a property of the sampling method itself,
not a bug.

## Seeing the arctic boundary

At large n, the "height field" color view can look smooth even though the
liquid region is really there — the overall boundary range (0 to n) dwarfs
the ±1 local fluctuations, so they wash out visually. Switch **view mode**
to **liquid / frozen (arctic boundary)** to see it directly: this colors
each face by whether any of its corners is currently an active local
max/min (i.e. eligible to flip right now). Frozen corners have zero active
sites; the liquid region in the middle lights up, and the boundary between
them is the arctic circle/curve itself.

## Pan / zoom / export

- **scroll** on the canvas to zoom, **drag** to pan.
- **save PNG** exports the current view as a raster image.
- **save SVG** exports the full grid as a vector `<rect>`-per-face SVG
  (usable directly, or as a starting point for a TikZ conversion). Note:
  this is one rect per lattice face, so file size grows with n^2 — fine
  through a few hundred n, unwieldy much beyond that.

## Extending

- To use the *general* asymmetric six-vertex model (a1≠a2, b1≠b2), the
  corner-flip chain alone isn't sufficient — you'd need additional moves
  (e.g. loop updates) or a different exact-sampling approach. This
  implementation covers the symmetric case (a1=a2=a, b1=b2=b, general
  c1=c2 bias), which is the standard case in the height-function
  literature and already reproduces the arctic-circle phenomenon.
- `SixVertexSampler.to_json()` / `.height_array()` can be used directly in
  a notebook if you just want the raw samples without the web UI.
