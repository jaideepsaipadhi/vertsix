// Minimal ("lo") or maximal ("hi") DWBC height function.
//
// kind is validated, matching the two Python copies. The ternary below used
// to treat every value that was not "lo" as "hi", so a misspelling returned
// the wrong extremal state silently -- and draw.js's makeShadow now depends
// on this function for the shadow chain's start, where getting "hi" when you
// asked for "lo" means both chains start together and the equilibration
// badge can never report anything but equilibrium.
function extremalHeight(n, kind) {
  if (kind !== "lo" && kind !== "hi") {
    throw new Error(`extremalHeight: kind must be "lo" or "hi", got ${JSON.stringify(kind)}`);
  }
  const size = n + 1;
  const H = new Float64Array(size * size);
  const corners = [[0, 0, 0], [n, 0, n], [0, n, n], [n, n, 0]];
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      let v = kind === "lo" ? -1e9 : 1e9;
      for (const [a, b, hv] of corners) {
        const dist = Math.abs(i - a) + Math.abs(j - b);
        v = kind === "lo" ? Math.max(v, hv - dist) : Math.min(v, hv + dist);
      }
      H[i * size + j] = v;
    }
  }
  return H;
}

function classifyFace(tl, tr, bl, br, w) {
  const top = tr - tl, bottom = br - bl, left = bl - tl, right = br - tr;
  const t = top === 1, b = bottom === 1, l = left === 1, r = right === 1;
  if (!l && !t && !b && !r) return w.a1;
  if (l && t && b && r) return w.a2;
  if (l && t && !b && !r) return w.c1;
  if (!l && !t && b && r) return w.c2;
  // Gorin-Kenyon Types 3 and 4: b1 is the HORIZONTAL through-vertex and b2
  // the vertical one. The b1/b2 correction reached exact.py, cftp_exact.py,
  // domains.py and draw.js but missed this file, so the local chain weighted
  // the vertical face as b1 while the legend beside it said b2.
  //
  // This does not change the measure, and provably cannot: under DWBC every
  // configuration has #b1 = #b2, so exchanging the two labels multiplies
  // each weight by (b1/b2)^0 = 1. It is a naming inconsistency, not a
  // sampling error -- but it is the same inconsistency Gorin flagged, and
  // leaving one copy of it behind defeats the point of fixing the others.
  if (!l && t && b && !r) return w.b2;   // vertical through
  if (l && !t && !b && r) return w.b1;   // horizontal through
  return 1;
}

class SixVertexJS {
  constructor(n, weights, rngSeed) {
    this.n = n;
    this.size = n + 1;
    this.w = Object.assign({ a1: 1, a2: 1, b1: 1, b2: 1, c1: 1, c2: 1 }, weights);
    this.H = extremalHeight(n, "lo");
    this._rngState = (rngSeed >>> 0) || 12345;
  }

  _rand() {
    let x = this._rngState;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this._rngState = x >>> 0;
    return (this._rngState >>> 0) / 4294967296;
  }

  idx(i, j) { return i * this.size + j; }

  step(sweeps) {
    const n = this.n, H = this.H, w = this.w;
    for (let s = 0; s < sweeps; s++) {
      for (let colorI = 0; colorI < 2; colorI++) {
        for (let colorJ = 0; colorJ < 2; colorJ++) {
          for (let i = 1; i < n; i++) {
            if (i % 2 !== colorI) continue;
            for (let j = 1; j < n; j++) {
              if (j % 2 !== colorJ) continue;
              const c = H[this.idx(i, j)];
              const N = H[this.idx(i - 1, j)];
              const S = H[this.idx(i + 1, j)];
              const E = H[this.idx(i, j + 1)];
              const W = H[this.idx(i, j - 1)];
              if (!(N === S && S === E && E === W)) continue;
              const v = N;
              const NW = H[this.idx(i - 1, j - 1)];
              const NE = H[this.idx(i - 1, j + 1)];
              const SW = H[this.idx(i + 1, j - 1)];
              const SE = H[this.idx(i + 1, j + 1)];
              const hAfter = (c === v + 1) ? v - 1 : v + 1;

              const before =
                classifyFace(NW, N, W, c, w) *
                classifyFace(N, NE, c, E, w) *
                classifyFace(W, c, SW, S, w) *
                classifyFace(c, E, S, SE, w);
              const after =
                classifyFace(NW, N, W, hAfter, w) *
                classifyFace(N, NE, hAfter, E, w) *
                classifyFace(W, hAfter, SW, S, w) *
                classifyFace(hAfter, E, S, SE, w);

              const ratio = after / Math.max(before, 1e-300);
              const pAccept = ratio / (1 + ratio);
              if (this._rand() < pAccept) {
                H[this.idx(i, j)] = hAfter;
              }
            }
          }
        }
      }
    }
  }

  activeMask() {
    const n = this.n, size = this.size, H = this.H;
    const mask = new Uint8Array(size * size);
    for (let i = 1; i < n; i++) {
      for (let j = 1; j < n; j++) {
        const c = H[i * size + j];
        const N = H[(i - 1) * size + j];
        const S = H[(i + 1) * size + j];
        const E = H[i * size + j + 1];
        const W = H[i * size + j - 1];
        if (N === S && S === E && E === W) mask[i * size + j] = 1;
      }
    }
    return mask;
  }

  minMax() {
    let min = Infinity, max = -Infinity;
    for (let k = 0; k < this.H.length; k++) {
      if (this.H[k] < min) min = this.H[k];
      if (this.H[k] > max) max = this.H[k];
    }
    return { min, max };
  }
}

