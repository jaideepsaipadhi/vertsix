function extremalHeight(n, kind) {
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
  if (l && t && !b && !r) return w.b1;
  if (!l && !t && b && r) return w.b2;
  if (!l && t && b && !r) return w.c1;
  if (l && !t && !b && r) return w.c2;
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

