#!/usr/bin/env python3
"""Exhaustive enumeration of height functions on a general domain.

Slow by construction and only usable for tiny N, which is the point: it is the
ground truth the fast code is checked against, the same role the ASM numbers
played for DWBC.

It answers two questions that the fast path cannot tell apart on its own:
whether a domain admits any valid configuration at all, and whether the
cone-bound extremes in sixvertex.domains are the true pointwise extremes. A
domain can be perfectly legal while the cone bound is merely not tight, and
treating that as "no configuration" would silently discard good domains.
"""
from __future__ import annotations

import sys, os
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sixvertex.domains import (FACE, boundary_heights, corner_mask, face_bits,
                               _corner_graph)


def _nbrs_of(i, j, N, vert, horiz):
    out = []
    if i < N and vert[i, j]:
        out.append((i + 1, j))
    if i > 0 and vert[i - 1, j]:
        out.append((i - 1, j))
    if j < N and horiz[i, j]:
        out.append((i, j + 1))
    if j > 0 and horiz[i, j - 1]:
        out.append((i, j - 1))
    return out


def enumerate_heights(mask, cap=2_000_000):
    """Every valid height function on the domain, as int64 arrays."""
    N = mask.shape[0]
    val, known = boundary_heights(mask)
    cm = corner_mask(mask)
    vert, horiz = _corner_graph(mask)

    # BFS order from the known boundary, so every point has a FIXED neighbour
    # by the time it is assigned. In plain row-major order a staircase domain
    # reaches corners with no fixed neighbour yet; the old code then guessed an
    # absolute value there and over-counted enormously (2604 configurations
    # where the independent edge enumeration finds 5). Squares hid the bug
    # because row-major order never leaves a corner unconstrained there.
    from collections import deque
    seen = known.copy()
    dq = deque([(i, j) for i in range(N + 1) for j in range(N + 1) if known[i, j]])
    pts = []
    while dq:
        i, j = dq.popleft()
        for (a, b) in _nbrs_of(i, j, N, *_corner_graph(mask)):
            if not seen[a, b]:
                seen[a, b] = True
                pts.append((a, b))
                dq.append((a, b))
    missing = [(i, j) for i in range(N + 1) for j in range(N + 1)
               if cm[i, j] and not seen[i, j]]
    if missing:
        raise RuntimeError(f"corner graph is disconnected at {missing[0]}")

    def nbrs(i, j):
        out = []
        if i < N and vert[i, j]:
            out.append((i + 1, j))
        if i > 0 and vert[i - 1, j]:
            out.append((i - 1, j))
        if j < N and horiz[i, j]:
            out.append((i, j + 1))
        if j > 0 and horiz[i, j - 1]:
            out.append((i, j - 1))
        return out

    H = val.copy()
    fixed = known.copy()
    out = []

    all_faces = [(a, b) for a in range(N) for b in range(N) if mask[a, b]]

    # Faces whose FOUR corners are all boundary-fixed are never visited by the
    # incremental check below, because that only looks at faces adjacent to a
    # free corner as it is assigned. On domains that have such faces -- 150 of
    # 250 in the 3x3 family -- an illegal one was therefore never noticed, and
    # this enumerator reported 1 or 2 configurations for domains that have
    # NONE: 23 of those 250 disagreed with enumerate_edge_configs, always in
    # that direction. Those faces are decided entirely by the boundary, so
    # they are checked once, up front.
    for (a, b) in all_faces:
        if all(known[a + da, b + db] for da in (0, 1) for db in (0, 1)):
            if face_bits(H, a, b) not in FACE:
                return []

    def faces_ready(i, j):
        """Domain faces all of whose corners are now fixed."""
        for (a, b) in ((i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)):
            if 0 <= a < N and 0 <= b < N and mask[a, b]:
                if all(fixed[a + da, b + db] for da in (0, 1) for db in (0, 1)):
                    yield (a, b)

    def rec(k):
        if len(out) >= cap:
            raise RuntimeError("enumeration cap hit; use a smaller domain")
        if k == len(pts):
            # Belt as well as braces: an oracle should certify what it
            # returns, so every domain face is checked once more here rather
            # than trusting that the incremental check covered them all.
            #
            # Honestly labelled: this is defensive and NOT what repaired the
            # counts. Removing it while keeping the up-front check above
            # leaves the 874-domain cross-check green (and about 30% faster),
            # so no domain in that sweep reaches here holding an illegal face.
            # It is kept because this function is the ground truth the rest of
            # the repository is checked against, and an oracle that certifies
            # its own output is worth more than a fast one.
            if all(face_bits(H, a, b) in FACE for (a, b) in all_faces):
                out.append(H.copy())
            return
        i, j = pts[k]
        anchor = next((H[a, b] for (a, b) in nbrs(i, j) if fixed[a, b]), None)
        if anchor is None:                      # cannot happen in BFS order
            raise RuntimeError(f"corner {(i, j)} has no fixed neighbour")
        for v in (-1, +1):
            cand = anchor + v
            if any(fixed[a, b] and abs(cand - H[a, b]) != 1
                   for (a, b) in nbrs(i, j)):
                continue
            H[i, j] = cand
            fixed[i, j] = True
            good = all(face_bits(H, a, b) in FACE for (a, b) in faces_ready(i, j))
            if good:
                rec(k + 1)
            fixed[i, j] = False
        return

    sys.setrecursionlimit(10000)
    rec(0)
    return out


def report(mask, name=""):
    hs = enumerate_heights(mask)
    if not hs:
        return dict(name=name, count=0, hmax=None, hmin=None)
    A = np.stack(hs)
    return dict(name=name, count=len(hs), hmax=A.max(0), hmin=A.min(0),
                all=hs)


# --------------------------------------------------------------- independent
def enumerate_edge_configs(mask, cap=5_000_000):
    """Count configurations by assigning 0/1 to EDGES, never touching a height
    function.

    Deliberately shares no code with enumerate_heights or sixvertex.domains
    beyond the mask itself. The height route and the boundary propagation are
    exactly where a convention error would hide, and a domain reported as
    admitting nothing is too strong a claim to rest on one implementation of
    them.

    Vertical edges V[i, j] sit between corners (i,j) and (i+1,j) -- they carry
    horizontal path segments. Horizontal edges Hh[i, j] sit between (i,j) and
    (i,j+1) and carry vertical segments. Cell (i,j) has edges
    l=V[i,j], r=V[i,j+1], t=Hh[i,j], b=Hh[i+1,j], and must be one of the six.
    """
    N = mask.shape[0]
    rows, cols = [], []
    # The row/column spans below assume each row and column of the domain is
    # CONTIGUOUS: a path enters at the first cell and leaves after the last,
    # with nothing in between. domain_mask enforces convexity of I, and that
    # holds across all 1078 masks of the 3x3 family (checked), but a
    # hand-built mask would break it silently -- the entry and exit edges
    # would be set across a gap and the count would be wrong rather than
    # refused. Stated where it is relied on.
    for i in range(N):
        w = np.flatnonzero(mask[i])
        if w.size:
            if w.size != int(w[-1]) - int(w[0]) + 1:
                raise ValueError(
                    f"row {i} of the domain is not contiguous ({w.tolist()}); "
                    f"this enumerator's entry/exit edges assume it is")
            rows.append((int(w[0]), int(w[-1])))
        else:
            rows.append((0, -1))
    for j in range(N):
        w = np.flatnonzero(mask[:, j])
        if w.size:
            if w.size != int(w[-1]) - int(w[0]) + 1:
                raise ValueError(
                    f"column {j} of the domain is not contiguous "
                    f"({w.tolist()}); this enumerator's entry/exit edges "
                    f"assume it is")
            cols.append((int(w[0]), int(w[-1])))
        else:
            cols.append((0, -1))

    V = np.full((N, N + 1), -1, dtype=np.int8)      # -1 = free
    Hh = np.full((N + 1, N), -1, dtype=np.int8)
    for i, (j0, j1) in enumerate(rows):
        V[i, j0] = 1                                 # path enters row i
        V[i, j1 + 1] = 0                             # nothing leaves right
    for j, (i0, i1) in enumerate(cols):
        Hh[i0, j] = 1                                # path leaves column j
        Hh[i1 + 1, j] = 0                            # nothing leaves below

    cells = [(i, j) for i in range(N) for j in range(N) if mask[i, j]]
    count = 0

    def rec(k):
        nonlocal count
        if count > cap:
            raise RuntimeError("edge enumeration cap hit")
        if k == len(cells):
            count += 1
            return
        i, j = cells[k]
        slots = [(V, i, j), (Hh, i, j), (Hh, i + 1, j), (V, i, j + 1)]  # l,t,b,r
        for bits in FACE:                     # only the six legal vertices
            old = []
            ok = True
            for (arr, a, b), want in zip(slots, bits):
                if arr[a, b] != -1 and arr[a, b] != want:
                    ok = False
                    break
                old.append((arr, a, b, arr[a, b]))
                arr[a, b] = want
            if ok:
                rec(k + 1)
            for arr, a, b, v in old:
                arr[a, b] = v
        return

    rec(0)
    return count
