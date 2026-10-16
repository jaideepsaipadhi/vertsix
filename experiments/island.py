"""The c-vertex island in the antiferroelectric phase, and its height.

Gorin's description: at Delta < -1 a region forms in the middle of the domain
made predominantly of c1 and c2 vertices, and the height of that region is
conjectured to be a discrete Gaussian after a deterministic shift. He warned
that defining it is delicate, because there are small defect components around
and one must pick the large one.

What the data says, measured on exact CFTP samples:

* In the island the raw height function alternates between L and L+1 in a
  perfect checkerboard. So the right object is the **parity-corrected height**
  G = H - ((i+j) mod 2), which is genuinely constant on the island. Levels of
  neighbouring plateaus differ by 2, not 1, so the natural lattice variable is
  k = (G - n/2) / 2, which comes out an integer.

* Connectivity matters more than it looks. Defining the island as the largest
  4-connected component of c-faces gives a median separation from the runner-up
  of 3.6x and a biased mean, and defining it as the largest constant plateau of
  G leaves the runner-up within a factor 2 in 43% of samples -- an ambiguity
  that does not shrink with n. The largest **8-connected** component of c-faces
  separates by a median factor of 50 and was unambiguous in 100% of samples
  measured. That is the definition used here, and it is also the one Gorin
  described.

* The component is NOT a pure plateau, and the number here was wrong for a
  while. An earlier version of this file claimed 92-96% of its corners carry
  the modal level; measured on 40 exact samples at n=32, Delta=-3 the purity
  is 0.43 to 0.72 with median 0.58. That figure predated the switch to
  8-connectivity and was never re-measured. Around 50% is the expected
  geometry -- the 8-connected component reaches a little past the flat
  plateau -- not an error, which is why experiments/validate.py
  cross-validates the island LEVEL against an independent plateau definition
  instead of thresholding purity.

Caveat worth carrying into any conclusion: the fluctuation is small. The
measured variance of k is about 0.13-0.22 across Delta from -1.2 to -5 and does
not grow with n (0.14-0.18 for n = 24..64, which is the bounded variance a
discrete Gaussian predicts). With sd ~ 0.42 only two or three lattice points
carry appreciable mass, so testing the *shape* needs large sample counts rather
than large n -- the tails are where the Gaussian claim actually bites.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def c_face_mask(H, n):
    """Faces that are c1 or c2, in the standard convention.

    c1 = (l,t) occupied, c2 = (b,r) occupied.
    """
    tl = H[:n, :n]; tr = H[:n, 1:]
    bl = H[1:, :n]; br = H[1:, 1:]
    l = (bl - tl) == 1
    t = (tr - tl) == 1
    b = (br - bl) == 1
    r = (br - tr) == 1
    return (l & t & ~b & ~r) | (~l & ~t & b & r)


def corrected_height(H, n):
    """H minus the checkerboard, which is constant on an island."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    return H - ((i + j) & 1)


def island(H, n, connectivity=8):
    """Locate the island and measure its height.

    Returns a dict with:
      k              (level - n/2)/2, the centred integer height, or None
      level          the modal parity-corrected height on the island
      size           faces in the island
      runner_up      faces in the next largest component
      separation     size / runner_up -- a sample is ambiguous if this is small
      area_fraction  size / n^2
      purity         fraction of the island's corners carrying the modal level
    """
    H = np.asarray(H, dtype=np.int64)
    G = corrected_height(H, n)
    isc = c_face_mask(H, n)

    structure = np.ones((3, 3)) if connectivity == 8 else None
    lab, nlab = ndimage.label(isc, structure=structure)
    if nlab == 0:
        return dict(k=None, level=None, size=0, runner_up=0,
                    separation=0.0, area_fraction=0.0, purity=0.0)

    sizes = ndimage.sum(isc, lab, range(1, nlab + 1))
    biggest = int(np.argmax(sizes))
    size = int(sizes[biggest])
    runner_up = int(np.sort(sizes)[-2]) if nlab > 1 else 0

    ii, jj = np.nonzero(lab == biggest + 1)
    corners = np.concatenate([G[ii, jj], G[ii, jj + 1],
                              G[ii + 1, jj], G[ii + 1, jj + 1]])
    vals, counts = np.unique(corners, return_counts=True)
    level = int(vals[np.argmax(counts)])

    return dict(
        k=(level - n / 2) / 2,
        level=level,
        size=size,
        runner_up=runner_up,
        separation=size / max(1, runner_up),
        area_fraction=size / (n * n),
        purity=float(counts.max() / counts.sum()),
    )


def discrete_gaussian_fit(ks):
    """Fit P(k) ∝ exp(-(k-mu)^2 / (2 sigma^2)) on the integers by maximum
    likelihood, and return the fit plus a chi-square goodness-of-fit test.

    The normalisation is the sum over the lattice, not the Gaussian integral --
    that is what makes it a *discrete* Gaussian, and for small sigma the two
    differ substantially.
    """
    from scipy.optimize import minimize
    from scipy.stats import chisquare

    ks = np.asarray(ks, dtype=float)
    support = np.arange(ks.min() - 4, ks.max() + 5)

    def nll(p):
        mu, log_s = p
        s = np.exp(log_s)
        logw = -(support - mu) ** 2 / (2 * s * s)
        logZ = np.log(np.exp(logw).sum())
        return -np.sum(-(ks - mu) ** 2 / (2 * s * s) - logZ)

    fit = minimize(nll, [ks.mean(), np.log(max(ks.std(), 0.2))],
                   method="Nelder-Mead")
    mu, sigma = fit.x[0], float(np.exp(fit.x[1]))

    w = np.exp(-(support - mu) ** 2 / (2 * sigma * sigma))
    p = w / w.sum()
    obs = np.array([(ks == v).sum() for v in support], dtype=float)
    exp = p * len(ks)

    # pool cells with expected count < 5, or the chi-square is not valid
    keep = exp >= 5
    if keep.sum() >= 2:
        o = np.append(obs[keep], obs[~keep].sum())
        e = np.append(exp[keep], exp[~keep].sum())
        e = e * (o.sum() / e.sum())
        stat, pval = chisquare(o, e, ddof=2)
    else:
        stat, pval = float("nan"), float("nan")

    return dict(mu=float(mu), sigma=sigma, chi2=float(stat), pvalue=float(pval),
                support=support.tolist(), observed=obs.tolist(),
                expected=exp.tolist(),
                n_cells_used=int(keep.sum()))
