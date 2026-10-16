#!/usr/bin/env python3
"""The c-vertex island on a general Gorin-Kenyon domain.

    python3 experiments/island_domain.py --delta -3 --n 48

experiments/island.py locates the island on the square. Two of its steps are
DWBC-specific and are replaced here:

* the c-face mask is taken over DOMAIN faces only, so cells outside the region
  cannot join a component;
* the centred height k = (level - n/2)/2 uses the square's boundary. A general
  domain has no such natural centre, so no centring is attempted. The
  quantities compared across domains are the NEIGHBOUR RATIOS

      R_lo = P(level = mode - 2) / P(level = mode)
      R_hi = P(level = mode + 2) / P(level = mode)

  which are translation invariant and therefore mean the same thing on every
  domain. (Levels differ by 2, as on the square, because the parity-corrected
  height changes by 2 between adjacent plateaus.)

The parity correction H - ((i+j) mod 2) carries over unchanged: it is a
property of the height function, not of the boundary.

Sampling uses the stationary-chain estimator -- one exact CFTP sample per
chain, then plain sweeps -- with chain-to-chain batch means for the errors.
"""
from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sixvertex.domains as sd
from sixvertex.domains import (_rand_for, _weight_table, cftp_sample,
                               corner_mask, domain_mask, face_bits,
                               sweep, sweep_fast, sweep_plan, sweep_plan_arrays)
from experiments.mixing import weights_for_delta

DOMAINS = {
    "square":    [[1]],
    "staircase": [[1, 1, 1], [0, 1, 1], [0, 0, 1]],
    "plus":      [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    "notch":     [[1, 1, 1], [1, 1, 1], [0, 1, 1]],
}


def c_faces(H, mask):
    """Domain faces that are c1 or c2."""
    N = mask.shape[0]
    H = np.asarray(H, dtype=np.int64)
    tl = H[:N, :N]; tr = H[:N, 1:]; bl = H[1:, :N]; br = H[1:, 1:]
    l = (bl - tl) == 1
    t = (tr - tl) == 1
    b = (br - bl) == 1
    r = (br - tr) == 1
    return ((l & t & ~b & ~r) | (~l & ~t & b & r)) & mask


def island_level(H, mask):
    """Modal parity-corrected height on the largest 8-connected c-component.

    Returns (level, size, runner_up) or (None, 0, 0) if there is no c-face.
    """
    N = mask.shape[0]
    H = np.asarray(H, dtype=np.int64)
    i, j = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
    G = H - ((i + j) & 1)
    isc = c_faces(H, mask)
    lab, nlab = ndimage.label(isc, structure=np.ones((3, 3)))
    if nlab == 0:
        return None, 0, 0
    sizes = ndimage.sum(isc, lab, range(1, nlab + 1))
    big = int(np.argmax(sizes))
    size = int(sizes[big])
    runner = int(np.sort(sizes)[-2]) if nlab > 1 else 0
    ii, jj = np.nonzero(lab == big + 1)
    corners = np.concatenate([G[ii, jj], G[ii, jj + 1],
                              G[ii + 1, jj], G[ii + 1, jj + 1]])
    vals, counts = np.unique(corners, return_counts=True)
    return int(vals[np.argmax(counts)]), size, runner


def chain_levels(mask, w, nmeas, thin, seed, start_seed):
    """Island levels along one stationary chain started from an exact sample."""
    H, _ = cftp_sample(mask, w, master_seed=start_seed)
    tbl = _weight_table(w)
    if sd._HAS_NUMBA:
        pa = sweep_plan_arrays(mask)
        tarr = np.array(tbl, dtype=np.float64)
        step = lambda X, r: sweep_fast(X, pa, tarr, r)
    else:
        plan = sweep_plan(mask)
        step = lambda X, r: sweep(X, mask, w, r, plan, tbl)
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(H, dtype=np.float64)
    out, sizes, seps = [], [], []
    for t in range(nmeas):
        for _ in range(thin):
            step(X, rng.random(X.shape))
        lev, sz, run = island_level(X.astype(np.int64), mask)
        out.append(np.nan if lev is None else lev)
        sizes.append(sz)
        seps.append(sz / max(1, run))
    return np.array(out), float(np.mean(sizes)), float(np.median(seps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--delta", type=float, default=-3.0)
    ap.add_argument("--chains", type=int, default=24)
    ap.add_argument("--meas", type=int, default=1500)
    ap.add_argument("--thin", type=int, default=48)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--out", default="island_domain.csv")
    args = ap.parse_args()

    w = weights_for_delta(args.delta)
    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["domain", "n", "delta", "cells", "R_lo", "se_lo",
                     "R_hi", "se_hi", "hits", "area_frac", "separation"])
        for name in args.domains:
            I = DOMAINS[name]
            k = len(I)
            cuts = [0, args.n] if k == 1 else \
                [0, args.n // 3, 2 * args.n // 3, args.n]
            m = domain_mask(cuts, cuts, I)
            per, areas, seps = [], [], []
            for c in range(args.chains):
                lv, area, sep = chain_levels(m, w, args.meas, args.thin,
                                             seed=60_000 + c,
                                             start_seed=61_000 + c)
                good = lv[~np.isnan(lv)]
                if good.size == 0:
                    continue
                vals, cnts = np.unique(good, return_counts=True)
                mode = vals[np.argmax(cnts)]
                per.append(((good == mode - 2).sum(), (good == mode).sum(),
                            (good == mode + 2).sum()))
                areas.append(area / m.sum())
                seps.append(sep)
            P = np.array(per, float)
            R_lo = P[:, 0].sum() / P[:, 1].sum()
            R_hi = P[:, 2].sum() / P[:, 1].sum()
            se_lo = (P[:, 0] / P[:, 1]).std(ddof=1) / np.sqrt(len(P))
            se_hi = (P[:, 2] / P[:, 1]).std(ddof=1) / np.sqrt(len(P))
            wr.writerow([name, args.n, args.delta, int(m.sum()),
                         f"{R_lo:.5f}", f"{se_lo:.5f}", f"{R_hi:.6f}",
                         f"{se_hi:.6f}", int(P[:, 2].sum()),
                         f"{np.mean(areas):.4f}", f"{np.median(seps):.1f}"])
            fh.flush()
            print(f"  {name:10s} cells={int(m.sum()):5d}  "
                  f"R_lo={R_lo:.4f}+-{se_lo:.4f}  R_hi={R_hi:.5f}+-{se_hi:.5f}"
                  f" ({int(P[:,2].sum()):4d} hits)  island area "
                  f"{np.mean(areas):5.1%}  separation {np.median(seps):6.1f}x"
                  f"   [{time.time()-t0:.0f}s]", flush=True)
    print(f"\nwrote {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
