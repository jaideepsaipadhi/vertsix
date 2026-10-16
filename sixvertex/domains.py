"""General domains for the six-vertex model, following Gorin-Kenyon
"Six-Vertex Model with Rare Corners and Random Restricted Permutations"
(arXiv:2408.14446), Section 2.1.

The domain is built from two strictly increasing cut sequences

    0 = X_0 < X_1 < ... < X_k = N        (column cuts)
    0 = Y_0 < Y_1 < ... < Y_l = N        (row cuts)

and a k x l binary array I whose every row and every column is a CONTIGUOUS
block of ones. A cell belongs to the domain when the block containing it is
marked. DWBC on the full square is k = l = 1 with I = [[1]], and this module
is written so that case reproduces sixvertex.cftp_exact exactly -- that
equality is a test, not a remark, because the conventions here are the same
ones that were silently wrong once before.

CONVENTION (read off the existing DWBC code rather than re-derived).
H lives on cell corners, index (i, j) = (row, column), i increasing downward.
For the face with top-left corner (i, j):

    l = [H(i+1, j  ) - H(i,   j  ) == 1]      left edge
    t = [H(i,   j+1) - H(i,   j  ) == 1]      top edge
    b = [H(i+1, j+1) - H(i+1, j  ) == 1]      bottom edge
    r = [H(i+1, j+1) - H(i,   j+1) == 1]      right edge

Checked against DWBC: H(i,0) = i gives l = 1 down the whole left column, which
is "a horizontal path enters each row"; H(0,j) = j gives t = 1 along the whole
top row, which is "a vertical path leaves each column"; the right and bottom
boundaries come out unoccupied. Those are exactly the paper's boundary
conditions, so the same four rules carry over unchanged to a general domain:

    left edge of the leftmost cell of each row      occupied
    top edge of the topmost cell of each column     occupied
    right edge of the rightmost cell of each row    empty
    bottom edge of the bottommost cell of each col  empty

Everything else is forced by the height function being +-1 along every edge.
"""
from __future__ import annotations

import numpy as np


class DomainError(ValueError):
    pass


def domain_mask(X, Y, I):
    """Boolean (N, N) cell mask, indexed [row, column].

    X cuts columns, Y cuts rows, matching the paper. Raises rather than
    silently producing a domain the sampler cannot handle.
    """
    X = list(map(int, X)); Y = list(map(int, Y))
    I = np.asarray(I, dtype=bool)
    if X[0] != 0 or Y[0] != 0:
        raise DomainError("X and Y must start at 0")
    if any(b <= a for a, b in zip(X, X[1:])) or any(b <= a for a, b in zip(Y, Y[1:])):
        raise DomainError("X and Y must be strictly increasing")
    if X[-1] != Y[-1]:
        raise DomainError(f"X and Y must end at the same N ({X[-1]} vs {Y[-1]})")
    if I.shape != (len(X) - 1, len(Y) - 1):
        raise DomainError(f"I must be {len(X)-1} x {len(Y)-1}, got {I.shape}")

    # convexity: every row and every column of I is a contiguous run of ones
    for name, arr in (("row", I), ("column", I.T)):
        for idx, line in enumerate(arr):
            w = np.flatnonzero(line)
            if w.size and w[-1] - w[0] + 1 != w.size:
                raise DomainError(
                    f"I is not convex: {name} {idx} = {line.astype(int).tolist()} "
                    "is not a contiguous block of ones")

    N = X[-1]
    mask = np.zeros((N, N), dtype=bool)
    for u in range(len(X) - 1):
        for v in range(len(Y) - 1):
            if I[u, v]:
                # I is indexed [column block, row block]; the mask is [row, col]
                mask[Y[v]:Y[v + 1], X[u]:X[u + 1]] = True

    if not mask.any(axis=1).all():
        raise DomainError("every row must contain at least one cell "
                          "(a path enters each row)")
    if not mask.any(axis=0).all():
        raise DomainError("every column must contain at least one cell "
                          "(a path leaves each column)")
    return mask


def row_spans(mask):
    """(first, last) column of each row. Rows are intervals by convexity."""
    out = []
    for i in range(mask.shape[0]):
        w = np.flatnonzero(mask[i])
        out.append((int(w[0]), int(w[-1])))
    return out


def col_spans(mask):
    out = []
    for j in range(mask.shape[1]):
        w = np.flatnonzero(mask[:, j])
        out.append((int(w[0]), int(w[-1])))
    return out


def corner_mask(mask):
    """Corners touching at least one domain cell: where H is defined."""
    N = mask.shape[0]
    c = np.zeros((N + 1, N + 1), dtype=bool)
    c[:N, :N] |= mask
    c[:N, 1:] |= mask
    c[1:, :N] |= mask
    c[1:, 1:] |= mask
    return c


def _corner_graph(mask):
    """Adjacency between corners along edges of domain cells.

    A step between two corners is allowed only along an edge that bounds at
    least one cell of the domain -- otherwise the Lipschitz constraint does
    not apply across it and using it would over-constrain the extremes.
    """
    N = mask.shape[0]
    # vertical steps (i,j)-(i+1,j): the edge is the left/right side of a cell
    vert = np.zeros((N, N + 1), dtype=bool)
    vert[:, :N] |= mask                     # left side of cell (i,j)
    vert[:, 1:] |= mask                     # right side of cell (i,j)
    # horizontal steps (i,j)-(i,j+1): top/bottom side of a cell
    horiz = np.zeros((N + 1, N), dtype=bool)
    horiz[:N, :] |= mask
    horiz[1:, :] |= mask
    return vert, horiz


def boundary_heights(mask):
    """H on every corner where the boundary conditions force it.

    Returns (values, known) with values int64 and known boolean, both
    (N+1, N+1). The four rules in the module docstring fix one edge increment
    at each boundary cell; walking those increments fixes H along the whole
    outer boundary of the domain once H is pinned at a single corner.
    """
    N = mask.shape[0]
    rows, cols = row_spans(mask), col_spans(mask)

    # forced increments, as (corner_a, corner_b, H(b) - H(a))
    forced = []
    for i, (j0, j1) in enumerate(rows):
        forced.append(((i, j0), (i + 1, j0), +1))        # path enters row i
        forced.append(((i, j1 + 1), (i + 1, j1 + 1), -1))  # nothing exits right
    for j, (i0, i1) in enumerate(cols):
        forced.append(((i0, j), (i0, j + 1), +1))        # path leaves column j
        forced.append(((i1 + 1, j), (i1 + 1, j + 1), -1))  # nothing exits below

    # propagate from the top-left corner of row 0
    val = np.zeros((N + 1, N + 1), dtype=np.int64)
    known = np.zeros((N + 1, N + 1), dtype=bool)
    start = (0, rows[0][0])
    known[start] = True
    adj = {}
    for a, b, d in forced:
        adj.setdefault(a, []).append((b, d))
        adj.setdefault(b, []).append((a, -d))
    stack = [start]
    while stack:
        p = stack.pop()
        for q, d in adj.get(p, ()):
            if not known[q]:
                known[q] = True
                val[q] = val[p] + d
                stack.append(q)
            elif val[q] != val[p] + d:
                raise DomainError(
                    "boundary conditions are inconsistent at "
                    f"{q}: {val[q]} vs {val[p] + d}. The domain does not admit "
                    "a height function with these paths.")
    return val, known


def extremal_heights(mask):
    """The pointwise largest and smallest valid height functions.

    Height functions on a fixed boundary form a distributive lattice, so the
    pointwise extremes are themselves valid configurations, and each is the
    tightest bound the boundary implies:

        H_max(p) = min over boundary b of  H(b) + dist(p, b)
        H_min(p) = max over boundary b of  H(b) - dist(p, b)

    with dist the graph distance through domain edges. On the full square this
    collapses to the four-corner L1 cone formula the DWBC code already uses.

    The result is VERIFIED to be a valid configuration before being returned.
    That check is not ceremony: the cone bound is tight only when the boundary
    values are geodesically consistent, and a domain where it is not would
    otherwise hand the sampler a starting state that is not a configuration at
    all, which CFTP cannot detect for itself.
    """
    val, known = boundary_heights(mask)
    N = mask.shape[0]
    vert, horiz = _corner_graph(mask)
    cm = corner_mask(mask)
    INF = 10 ** 9

    def relax(init, sign):
        """Multi-source shortest path with unit edges (BFS by rounds)."""
        H = np.where(known, init, sign * INF).astype(np.int64)
        H[~cm] = sign * INF
        for _ in range(4 * N + 4):
            prev = H.copy()
            cand = [H]
            # vert[i, j] is the edge (i,j)-(i+1,j); horiz[i, j] is (i,j)-(i,j+1)
            up = np.full_like(H, sign * INF)
            up[:N, :] = np.where(vert, H[1:, :] + sign, sign * INF)
            cand.append(up)
            dn = np.full_like(H, sign * INF)
            dn[1:, :] = np.where(vert, H[:N, :] + sign, sign * INF)
            cand.append(dn)
            le = np.full_like(H, sign * INF)
            le[:, :N] = np.where(horiz, H[:, 1:] + sign, sign * INF)
            cand.append(le)
            ri = np.full_like(H, sign * INF)
            ri[:, 1:] = np.where(horiz, H[:, :N] + sign, sign * INF)
            cand.append(ri)
            H = np.minimum.reduce(cand) if sign > 0 else np.maximum.reduce(cand)
            H = np.where(known, init, H)
            H[~cm] = sign * INF
            if np.array_equal(H, prev):
                break
        return H

    hi = relax(val, +1)
    lo = relax(val, -1)
    for name, H in (("hi", hi), ("lo", lo)):
        bad = classify_invalid(H, mask)
        if bad:
            raise DomainError(
                f"extremal height '{name}' is not a valid configuration; "
                f"{len(bad)} bad face(s), first at {bad[0]}")
        if np.any(H[cm] >= 10 ** 8) or np.any(H[cm] <= -10 ** 8):
            raise DomainError(f"extremal height '{name}' left corners unreached")
    if np.any(hi[cm] < lo[cm]):
        raise DomainError("hi < lo somewhere: the bounds are inconsistent")
    # Zero the corners outside the domain. They carried +-INF sentinels from
    # the relaxation, and leaving them in made cftp_sample's array comparison
    # compare -1e9 against +1e9 forever: the chains coalesced after a single
    # sweep and the sampler still reported "no coalescence by T=2^20", which
    # reads exactly like a mixing failure and is not one.
    hi = np.where(cm, hi, 0)
    lo = np.where(cm, lo, 0)
    return lo, hi


FACE = {(0, 0, 0, 0): "a1", (1, 1, 1, 1): "a2",
        (1, 1, 0, 0): "c1", (0, 0, 1, 1): "c2",
        (0, 1, 1, 0): "b1", (1, 0, 0, 1): "b2"}


def face_bits(H, i, j):
    return (int(H[i + 1, j] - H[i, j] == 1),
            int(H[i, j + 1] - H[i, j] == 1),
            int(H[i + 1, j + 1] - H[i + 1, j] == 1),
            int(H[i + 1, j + 1] - H[i, j + 1] == 1))


def classify_invalid(H, mask):
    """Domain faces whose four edges do not form one of the six vertices."""
    bad = []
    N = mask.shape[0]
    for i in range(N):
        for j in range(N):
            if not mask[i, j]:
                continue
            d = (abs(H[i + 1, j] - H[i, j]), abs(H[i, j + 1] - H[i, j]),
                 abs(H[i + 1, j + 1] - H[i + 1, j]),
                 abs(H[i + 1, j + 1] - H[i, j + 1]))
            if any(x != 1 for x in d) or face_bits(H, i, j) not in FACE:
                bad.append((i, j))
    return bad


def classify_counts(H, mask):
    from collections import Counter
    c = Counter()
    N = mask.shape[0]
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                c[FACE[face_bits(H, i, j)]] += 1
    return dict(c)


# ------------------------------------------------------------------ sampling
def free_corners(mask):
    """Corners the chain may move: inside the domain and not fixed by the
    boundary conditions."""
    _, known = boundary_heights(mask)
    return corner_mask(mask) & ~known


def _faces_of_corner(mask, i, j):
    N = mask.shape[0]
    out = []
    for (a, b) in ((i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)):
        if 0 <= a < N and 0 <= b < N and mask[a, b]:
            out.append((a, b))
    return out


def _neighbours(mask, i, j):
    N = mask.shape[0]
    vert, horiz = _corner_graph(mask)
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


def sweep_plan(mask):
    """Precomputed per-corner update data, grouped into independent colours.

    Two corners may be updated in the same pass only when no face touches
    both; the (i%2, j%2) four-colouring guarantees that, and is the same
    colouring the DWBC sweep uses.

    Everything is stored as FLAT integer indices. Scalar indexing into a numpy
    array costs roughly a hundred nanoseconds a time and this loop does it
    millions of times -- profiling put 43% of the sweep inside face_bits
    alone. Flat Python ints on a flat list cut that out; the sweep is
    otherwise unchanged and produces identical states.
    """
    N = mask.shape[0]
    size = N + 1
    free = free_corners(mask)
    groups = [[] for _ in range(4)]
    for i in range(N + 1):
        for j in range(N + 1):
            if not free[i, j]:
                continue
            flat = i * size + j
            nb = [a * size + b for (a, b) in _neighbours(mask, i, j)]
            faces = []
            for (a, b) in _faces_of_corner(mask, i, j):
                tl = a * size + b
                faces.append((tl, tl + 1, tl + size, tl + size + 1))
            groups[(i % 2) * 2 + (j % 2)].append((flat, tuple(nb), tuple(faces)))
    return groups


def _weight_table(w):
    """16 entries indexed by l*8 + t*4 + b*2 + r; illegal patterns weigh 0."""
    tbl = [0.0] * 16
    for bits, name in FACE.items():
        l, t, b, r = bits
        tbl[l * 8 + t * 4 + b * 2 + r] = float(w[name])
    return tbl


def _face_weight(H, i, j, w):
    b = face_bits(H, i, j)
    t = FACE.get(b)
    return 0.0 if t is None else w[t]


def sweep(H, mask, w, rand, plan=None, table=None):
    """One heat-bath sweep in place, driven by the array `rand`.

    A corner is free to move only when every one of its neighbours carries the
    same value v, in which case it may sit at v-1 or v+1; otherwise it is
    pinned. Driving both extremal chains with the SAME `rand` is what makes
    the coupling monotone, so `rand` is indexed by corner, never drawn inside
    the loop.
    """
    plan = plan if plan is not None else sweep_plan(mask)
    tbl = table if table is not None else _weight_table(w)
    h = H.ravel().tolist()
    u = rand.ravel().tolist()
    for group in plan:
        for flat, nb, faces in group:
            if not nb:                       # cannot happen: a free corner
                continue                     # always touches a domain cell
            v = h[nb[0]]
            same = True
            for q in nb:
                if h[q] != v:
                    same = False
                    break
            if not same:
                continue
            wu = wd = 1.0
            for up, val in ((True, v + 1), (False, v - 1)):
                h[flat] = val
                prod = 1.0
                for (tl, tr, bl, br) in faces:
                    a = h[tl]
                    prod *= tbl[((h[bl] - a == 1) << 3) | ((h[tr] - a == 1) << 2)
                                | ((h[br] - h[bl] == 1) << 1)
                                | (h[br] - h[tr] == 1)]
                    if prod == 0.0:
                        break
                if up:
                    wu = prod
                else:
                    wd = prod
            tot = wu + wd
            if tot <= 0.0:
                # Both candidates illegal is impossible from a legal state --
                # the corner currently sits at v+1 or v-1, and that one is
                # legal by assumption. Raise rather than pick, because
                # silently choosing here would corrupt the chain invisibly.
                raise DomainError(
                    f"no legal value at corner {divmod(flat, H.shape[1])}; "
                    "the state was already invalid")
            h[flat] = (v + 1) if u[flat] < wu / tot else (v - 1)
    H[...] = np.asarray(h, dtype=H.dtype).reshape(H.shape)
    return H


def cftp_sample(mask, w, master_seed=None, initial_T=16, max_T=1 << 20):
    """Exact sample on a general domain by Coupling From The Past.

    Same doubling schedule and same monotonicity requirement as the DWBC
    sampler: the two extremal height functions are run forward from -T under
    one shared random stream, and T doubles until they meet.
    """
    if master_seed is None:
        # The DWBC sampler already did this; this one did not, so a request
        # that simply omitted the seed reached numpy as SeedSequence([None])
        # and surfaced to the user as "int() argument must be a string...".
        master_seed = int(np.random.randint(0, 2 ** 31 - 1))
    if not in_monotone_region(w):
        raise DomainError(
            "weights are outside the monotone region c1c2 >= a1a2 and "
            "c1c2 >= b1b2, where no monotone coupling exists")
    lo0, hi0 = extremal_heights(mask)
    cm = corner_mask(mask)
    check_monotone_precondition(mask)
    tbl = _weight_table(w)
    if _HAS_NUMBA:
        plan = sweep_plan_arrays(mask)
        tbl_arr = np.array(tbl, dtype=np.float64)
        step = lambda H, r: sweep_fast(H, plan, tbl_arr, r)
    else:
        plan = sweep_plan(mask)
        step = lambda H, r: sweep(H, mask, w, r, plan, tbl)
    N = mask.shape[0]
    T = initial_T
    while T <= max_T:
        lo, hi = lo0.astype(np.float64), hi0.astype(np.float64)
        for k in range(T, 0, -1):
            r = _rand_for(master_seed, k, (N + 1, N + 1))
            step(lo, r)
            step(hi, r)
        if np.array_equal(lo[cm], hi[cm]):   # only where H is defined
            out = lo.astype(np.int64)
            bad = classify_invalid(out, mask)
            if bad:
                raise DomainError(f"coalesced state is invalid at {bad[0]}")
            return out, {"T": T, "master_seed": int(master_seed)}
        T *= 2
    raise DomainError(f"no coalescence by T={max_T}")


def _rand_for(master_seed, k, shape):
    ss = np.random.SeedSequence([int(master_seed), int(k)])
    return np.random.default_rng(ss).random(shape)


def in_monotone_region(w, tol=1e-12):
    from sixvertex.cftp_exact import in_monotone_region as _f
    return _f(w, tol)


# -------------------------------------------------- existence characterisation
def _components(mask):
    """Connected components of the corner graph, as lists of (i, j)."""
    from collections import deque
    N = mask.shape[0]
    cm = corner_mask(mask)
    vert, horiz = _corner_graph(mask)

    def nb(i, j):
        out = []
        if i < N and vert[i, j] and cm[i + 1, j]:
            out.append((i + 1, j))
        if i > 0 and vert[i - 1, j] and cm[i - 1, j]:
            out.append((i - 1, j))
        if j < N and horiz[i, j] and cm[i, j + 1]:
            out.append((i, j + 1))
        if j > 0 and horiz[i, j - 1] and cm[i, j - 1]:
            out.append((i, j - 1))
        return out

    seen = set()
    comps = []
    for p in map(tuple, np.argwhere(cm)):
        p = (int(p[0]), int(p[1]))
        if p in seen:
            continue
        comp, dq = [], deque([p])
        seen.add(p)
        while dq:
            q = dq.popleft()
            comp.append(q)
            for r in nb(*q):
                if r not in seen:
                    seen.add(r)
                    dq.append(r)
        comps.append(comp)
    return comps, nb


def admits_configuration(mask):
    """Whether the domain carries any valid configuration, in polynomial time.

    The criterion, verified against exhaustive enumeration on 980 domains:

        the domain is nonempty  <=>  in EVERY connected component of the
        corner graph, the forced boundary heights are consistent and
        1-Lipschitz with respect to graph distance inside that component,
        parity included.

    Why that is the right statement rather than a guess. Around any face the
    four signed height increments traverse a cycle and so sum to zero; with
    every increment +-1 that forces exactly two of each sign, which is exactly
    the six legal vertices out of sixteen patterns. So legality of every face
    is automatic for any exact-+-1 height function, and existence reduces to
    extending the boundary values -- a Lipschitz extension problem, whose
    classical obstruction is precisely a boundary pair closer in value-gap
    than in distance.

    Two earlier versions of this were wrong in opposite directions and both
    looked fine on most inputs. Checking Lipschitz alone accepted 7 empty
    domains, because boundary propagation from one corner never reaches a
    disconnected piece and the unreached piece was silently not checked.
    Adding "the corner graph must be connected" then rejected 34 nonempty
    ones, because a disconnected domain is perfectly samplable -- its
    components just carry independent additive constants. Per component is
    the version that agrees everywhere.

    NOTE the exponential alternative: enumerating configurations to answer
    this is infeasible past about N=6, while this runs in O(|boundary| * |V|).
    """
    comps, nb = _components(mask)
    N = mask.shape[0]
    rows, cols = row_spans(mask), col_spans(mask)

    forced = []
    for i, (j0, j1) in enumerate(rows):
        forced.append(((i, j0), (i + 1, j0), +1))
        forced.append(((i, j1 + 1), (i + 1, j1 + 1), -1))
    for j, (i0, i1) in enumerate(cols):
        forced.append(((i0, j), (i0, j + 1), +1))
        forced.append(((i1 + 1, j), (i1 + 1, j + 1), -1))
    adj = {}
    for a, b, d in forced:
        adj.setdefault(a, []).append((b, d))
        adj.setdefault(b, []).append((a, -d))

    for comp in comps:
        member = set(comp)
        # propagate the forced increments inside this component
        val, known = {}, set()
        for seed in comp:
            if seed in known or seed not in adj:
                continue
            val[seed] = 0
            known.add(seed)
            stack = [seed]
            while stack:
                p = stack.pop()
                for q, d in adj.get(p, ()):
                    if q not in member:
                        continue
                    if q not in known:
                        known.add(q)
                        val[q] = val[p] + d
                        stack.append(q)
                    elif val[q] != val[p] + d:
                        return False           # boundary inconsistent here
        if not known:
            continue                            # no forced edge in this piece
        # 1-Lipschitz against graph distance, within this component.
        #
        # Checking every boundary PAIR is O(|boundary| * |V|) -- 68 seconds at
        # N=240. The same condition is one multi-source relaxation: build
        #     U(p) = min over boundary b of  val(b) + dist(p, b)
        # and the boundary is Lipschitz exactly when no boundary point is
        # pushed below its own value, U(b) == val(b) for all b. One pass,
        # O(|V|). Verified to agree with the all-pairs version on every domain
        # in the 1364-domain sweep.
        INF = 10 ** 9
        U = {q: (val[q] if q in known else INF) for q in member}
        frontier = set(known)
        while frontier:
            nxt = set()
            for p in frontier:
                for q in nb(*p):
                    if q in member and U[p] + 1 < U[q]:
                        U[q] = U[p] + 1
                        nxt.add(q)
            frontier = nxt
        for b in known:
            if U[b] != val[b]:
                return False                    # a closer boundary point
        # every corner must be reachable, and parity must be global
        par = None
        for q in member:
            if U[q] >= INF:
                return False
            r = (U[q] + q[0] + q[1]) % 2
            if par is None:
                par = r
            elif r != par:
                return False
    return True


def check_monotone_precondition(mask):
    """Refuse domains where the monotonicity argument has not been established.

    CFTP is exact only because the coupling is monotone, and the condition
    c1c2 >= a1a2, c1c2 >= b1b2 was derived for a corner surrounded by FOUR
    faces -- the only kind a square has. On a general domain a free corner can
    touch one, two or three faces instead, and at such a corner the heat-bath
    ratio is a product over fewer factors, so that derivation simply does not
    apply. Coalescence would still happen and the sampler would still return
    an answer; it just would not be a proof of exactness.

    Measured over 213 samplable domains of the Gorin-Kenyon family, every
    domain this sampler accepts has only degree-4 free corners, and the 11
    with lower-degree ones were already refused for having a disconnected
    corner graph. The mechanism is that a corner of degree < 4 lies on the
    domain boundary, and on a connected domain the boundary conditions pin it.

    That is an observation across a finite scan, not a theorem, so it is
    enforced here rather than assumed. A domain that violates it gets a
    refusal naming the corner, not a silently unjustified "exact" sample.
    """
    bad = []
    for (i, j) in map(tuple, np.argwhere(free_corners(mask))):
        d = len(_faces_of_corner(mask, int(i), int(j)))
        if d != 4:
            bad.append((int(i), int(j), d))
    if bad:
        i, j, d = bad[0]
        raise DomainError(
            f"free corner ({i}, {j}) touches {d} domain face(s), not 4; the "
            "monotone-coupling condition is only established for degree-4 "
            f"corners, so exactness is not justified here ({len(bad)} such "
            "corner(s))")
    return True


# ------------------------------------------------------------ compiled sweep
def sweep_plan_arrays(mask):
    """The sweep plan as regular numpy arrays, for the compiled sweep.

    This is only possible because check_monotone_precondition refuses domains
    with free corners of degree other than 4: every corner the sampler accepts
    touches exactly four faces, and therefore has exactly four neighbours, so
    the plan is rectangular rather than ragged. A domain that slipped past
    that check would not fit these shapes, which is why the shapes are
    asserted here rather than trusted.

    Returns a list of four (flat, nb, faces) tuples, one per colour class:
        flat  (m,)        corner index
        nb    (m, 4)      its four neighbours
        faces (m, 4, 4)   the four corners (tl, tr, bl, br) of each of its
                          four faces
    """
    N = mask.shape[0]
    size = N + 1
    free = free_corners(mask)
    out = []
    for colour in range(4):
        flat, nbs, fcs = [], [], []
        for (i, j) in map(tuple, np.argwhere(free)):
            i, j = int(i), int(j)
            if (i % 2) * 2 + (j % 2) != colour:
                continue
            nb = [a * size + b for (a, b) in _neighbours(mask, i, j)]
            fs = []
            for (a, b) in _faces_of_corner(mask, i, j):
                tl = a * size + b
                fs.append([tl, tl + 1, tl + size, tl + size + 1])
            if len(nb) != 4 or len(fs) != 4:
                raise DomainError(
                    f"corner {(i, j)} has {len(nb)} neighbours and {len(fs)} "
                    "faces; the compiled sweep requires exactly 4 of each")
            flat.append(i * size + j)
            nbs.append(nb)
            fcs.append(fs)
        out.append((np.array(flat, dtype=np.int64).reshape(-1),
                    np.array(nbs, dtype=np.int64).reshape(-1, 4),
                    np.array(fcs, dtype=np.int64).reshape(-1, 4, 4)))
    return out


try:
    from numba import njit

    @njit(cache=True)
    def _sweep_group_compiled(h, u, flat, nb, faces, tbl):
        for t in range(flat.shape[0]):
            p = flat[t]
            v = h[nb[t, 0]]
            same = True
            for q in range(1, 4):
                if h[nb[t, q]] != v:
                    same = False
                    break
            if not same:
                continue
            wu = 1.0
            wd = 1.0
            for which in range(2):
                h[p] = v + 1 if which == 0 else v - 1
                prod = 1.0
                for f in range(4):
                    tl = faces[t, f, 0]; tr = faces[t, f, 1]
                    bl = faces[t, f, 2]; br = faces[t, f, 3]
                    a = h[tl]
                    idx = 0
                    if h[bl] - a == 1:
                        idx += 8
                    if h[tr] - a == 1:
                        idx += 4
                    if h[br] - h[bl] == 1:
                        idx += 2
                    if h[br] - h[tr] == 1:
                        idx += 1
                    prod *= tbl[idx]
                    if prod == 0.0:
                        break
                if which == 0:
                    wu = prod
                else:
                    wd = prod
            tot = wu + wd
            if tot <= 0.0:
                h[p] = v + 1          # caller validates; cannot happen
                continue
            h[p] = v + 1 if u[p] < wu / tot else v - 1

    _HAS_NUMBA = True
except Exception:                      # pragma: no cover
    _HAS_NUMBA = False


def sweep_fast(H, plan_arrays, tbl_arr, rand):
    """Compiled equivalent of `sweep`, producing bit-identical states.

    Identity with the reference implementation is a test, not an aspiration:
    the two differ only in how the same arithmetic is scheduled, so any
    divergence is a bug in one of them.
    """
    h = H.ravel()
    u = rand.ravel()
    for flat, nb, faces in plan_arrays:
        if flat.shape[0]:
            _sweep_group_compiled(h, u, flat, nb, faces, tbl_arr)
    return H
