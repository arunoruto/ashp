"""
Clustering from the Delaunay alpha filtration.

Building an alpha shape at radius ``r = 1/alpha`` keeps the simplices whose
circumradius is below ``r``; the connected components of those simplices form a
clustering (the single-linkage / DBSCAN-like view of the alpha complex).  As
``alpha`` sweeps from 0 to infinity the components split from one (the convex
hull) down to isolated points, and :func:`cluster_persistence` records that
sweep so a stable scale can be read off like a k-means elbow plot.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from .alphashape import _delaunay_circumradii

__all__ = ['cluster', 'cluster_persistence', 'ClusterPersistence',
           'alpha_sweep', 'AlphaSweep', 'homogenisation_rate', 'usable_band',
           'alpha_band']


class _UnionFind:
    """Union-find with union-by-size and path halving."""

    def __init__(self, n: int):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(self, a: int, b: int, min_size: int) -> int:
        """Merge ``a`` and ``b``; return the change in the number of components
        whose size is at least ``min_size``."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return 0
        a_big = self.size[ra] >= min_size
        b_big = self.size[rb] >= min_size
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        new_big = self.size[ra] >= min_size
        return int(new_big) - int(a_big) - int(b_big)


def _connectivity(simplices: np.ndarray, radii: np.ndarray):
    """Drop non-finite circumradii and sort simplices by circumradius."""
    finite = np.isfinite(radii)
    simplices, radii = simplices[finite], radii[finite]
    order = np.argsort(radii, kind="mergesort")
    return simplices[order], radii[order]


def cluster(points: Union[List[Tuple[float]], np.ndarray],
            alpha: Union[None, float] = None,
            min_size: int = 2) -> np.ndarray:
    """
    Cluster points by the connected components of the alpha complex.

    Two points share a cluster when they are joined through simplices whose
    circumradius is below ``1 / alpha`` (i.e. through the alpha shape at that
    scale).  Components smaller than ``min_size`` are treated as noise.

    Args:
        points: an iterable container of points (2-D or 3-D).
        alpha: alpha value.  If ``None``, the most persistent scale found by
            :func:`cluster_persistence` is used.
        min_size: components with fewer points than this are labelled noise.

    Returns:
        An integer label per point: clusters are ``0, 1, 2, ...`` ordered by
        descending size, and noise points are ``-1``.
    """
    coords, simplices, radii = _delaunay_circumradii(points)
    n = coords.shape[0]

    if alpha is None:
        alpha = cluster_persistence(points, min_size=min_size).best_alpha

    threshold = np.inf if alpha <= 0 else 1.0 / alpha
    keep = simplices[radii < threshold]

    uf = _UnionFind(n)
    for simplex in keep:
        first = int(simplex[0])
        for v in simplex[1:]:
            uf.union(first, int(v), min_size)

    roots = np.array([uf.find(i) for i in range(n)])
    uniq, counts = np.unique(roots, return_counts=True)
    size_of = dict(zip(uniq.tolist(), counts.tolist()))
    big_roots = sorted((r for r in uniq.tolist() if size_of[r] >= min_size),
                       key=lambda r: -size_of[r])
    remap = {r: i for i, r in enumerate(big_roots)}

    labels = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        labels[i] = remap.get(int(roots[i]), -1)
    return labels


@dataclass
class ClusterPersistence:
    """Result of :func:`cluster_persistence`.

    Attributes:
        alpha: ascending alpha values (the filtration scale).
        n_clusters: number of clusters (size >= ``min_size``) at each alpha; a
            step function, the value holding until the next alpha.
        best_alpha: alpha at the centre of the most persistent plateau.
        best_k: number of clusters on that plateau.
    """
    alpha: np.ndarray
    n_clusters: np.ndarray
    best_alpha: float
    best_k: int


def cluster_persistence(points: Union[List[Tuple[float]], np.ndarray],
                        min_size: int = 2) -> ClusterPersistence:
    """
    Sweep the alpha filtration and record how many clusters survive at each
    scale, then pick the most persistent one.

    The component count is constant over ranges of alpha (plateaus); the widest
    plateau measured in ``alpha`` (the natural "lambda" scale, as in HDBSCAN's
    stability) is the most robust clustering.  ``best_alpha`` sits at its centre.

    Args:
        points: an iterable container of points (2-D or 3-D).
        min_size: minimum size for a component to count as a cluster.

    Returns:
        A :class:`ClusterPersistence` with the curve and the suggested scale.
    """
    coords, simplices, radii = _delaunay_circumradii(points)
    n = coords.shape[0]
    simplices, radii = _connectivity(simplices, radii)

    uf = _UnionFind(int(n))
    big = n if min_size <= 1 else 0
    # Step function as (radius, k) change points; start before any simplex.
    r_change: List[float] = [0.0]
    k_change: List[int] = [int(big)]

    for simplex, r in zip(simplices, radii):
        first = int(simplex[0])
        for v in simplex[1:]:
            big += uf.union(first, int(v), min_size)
        if big != k_change[-1]:
            r_change.append(float(r))
            k_change.append(int(big))

    r_arr = np.asarray(r_change)
    k_arr = np.asarray(k_change)

    # Widest plateau among segments with >= 2 clusters, measured in log-radius
    # so the metric is scale-invariant (measuring in alpha = 1/r would bias the
    # choice toward the noisy small-radius / high-alpha fragmentation regime).
    r_max = float(radii.max()) if radii.size else 1.0
    best = None  # (persistence, k, r_lo, r_hi)
    for i in range(len(r_arr)):
        k = int(k_arr[i])
        r_lo = float(r_arr[i])
        if k < 2 or r_lo <= 0.0:
            continue
        r_hi = float(r_arr[i + 1]) if i + 1 < len(r_arr) else r_max
        persistence = np.log(r_hi) - np.log(r_lo)
        if best is None or persistence > best[0]:
            best = (persistence, k, r_lo, r_hi)

    if best is None:
        best_k, best_alpha = 1, 0.0
    else:
        _, best_k, r_lo, r_hi = best
        best_alpha = 1.0 / np.sqrt(r_lo * r_hi)  # geometric centre of plateau

    # Return the curve as ascending alpha (drop the r = 0 start point).
    positive = r_arr > 0.0
    alpha_curve = (1.0 / r_arr[positive])[::-1]
    k_curve = k_arr[positive][::-1]
    return ClusterPersistence(alpha=alpha_curve, n_clusters=k_curve,
                              best_alpha=best_alpha, best_k=int(best_k))


@dataclass
class AlphaSweep:
    """Result of :func:`alpha_sweep` (all arrays aligned and ascending in alpha).

    Attributes:
        alpha: the sweep of alpha values (``1 / radius`` threshold).
        edge_std: standard deviation of the kept Delaunay edge lengths.
        edge_cv: their coefficient of variation (``std / mean``) — dimensionless,
            so it is comparable across point counts and scales.
        n_components: number of connected components of the kept edges.
        n_edges: number of kept edges.
    """
    alpha: np.ndarray
    edge_std: np.ndarray
    edge_cv: np.ndarray
    n_components: np.ndarray
    n_edges: np.ndarray


def alpha_sweep(points: Union[List[Tuple[float]], np.ndarray],
                n_steps: int = 80) -> AlphaSweep:
    """
    Track edge-length spread and connectivity as alpha sweeps.

    The Delaunay graph is fixed; alpha only chooses which edges survive (an edge
    is kept while its birth radius — the smallest circumradius of an incident
    simplex — is below ``1/alpha``).  So the whole sweep is computed once from
    the sorted edges via prefix sums (for the length statistics) and a single
    incremental union-find pass (for connectivity), with no per-alpha alpha-shape
    recomputation.

    Args:
        points: an iterable container of points (2-D or 3-D).
        n_steps: number of alpha samples.

    Returns:
        An :class:`AlphaSweep` with the metrics as functions of alpha.
    """
    coords, simplices, radii = _delaunay_circumradii(points)
    n = coords.shape[0]
    finite = np.isfinite(radii)
    simplices, radii = simplices[finite], radii[finite]

    # Every simplex facet down to its edges, tagged with the simplex circumradius.
    pairs = list(itertools.combinations(range(simplices.shape[1]), 2))
    raw_edges = np.sort(np.vstack([simplices[:, p] for p in pairs]), axis=1)
    raw_birth = np.tile(radii, len(pairs))
    edges, inverse = np.unique(raw_edges, axis=0, return_inverse=True)
    birth = np.full(edges.shape[0], np.inf)
    np.minimum.at(birth, inverse, raw_birth)      # edge birth = min incident radius
    length = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)

    order = np.argsort(birth, kind="mergesort")
    edges, birth, length = edges[order], birth[order], length[order]
    m = length.size
    empty = np.zeros(0)
    if m == 0:
        return AlphaSweep(empty, empty, empty, empty.astype(int),
                          empty.astype(int))

    # Prefix sums of length (ordered by birth) -> O(1) stats for any threshold.
    csum = np.concatenate(([0.0], np.cumsum(length)))
    csum2 = np.concatenate(([0.0], np.cumsum(length * length)))

    # Components after adding the first k edges (increasing birth).
    uf = _UnionFind(n)
    comp_after = np.empty(m + 1, dtype=np.int64)
    comp_after[0] = n
    comp = n
    for i in range(m):
        comp += uf.union(int(edges[i, 0]), int(edges[i, 1]), 1)
        comp_after[i + 1] = comp

    alpha = np.geomspace(1.0 / np.percentile(birth, 99.5),
                         1.0 / np.percentile(birth, 2.0), n_steps)
    k = np.clip(np.searchsorted(birth, 1.0 / alpha, side="right"), 0, m)

    counts = np.maximum(k, 1)
    mean = csum[k] / counts
    var = np.clip(csum2[k] / counts - mean ** 2, 0.0, None)
    std = np.where(k > 0, np.sqrt(var), 0.0)
    cv = np.where(mean > 0, std / mean, 0.0)
    return AlphaSweep(alpha=alpha, edge_std=std, edge_cv=cv,
                      n_components=comp_after[k], n_edges=k)


def homogenisation_rate(sweep: AlphaSweep) -> np.ndarray:
    """``-d(CV)/d(log alpha)`` over a :class:`AlphaSweep`.

    The CV curve is a smooth descent; its negated derivative turns the gradual
    elbow into a sharp peak at the blob -> structure transition (the scale at
    which the long bridges are being cut), which is far more stable across point
    count than the circumradius knee.
    """
    if sweep.alpha.size < 3:
        return np.zeros_like(sweep.edge_cv)
    rate = -np.gradient(sweep.edge_cv, np.log(sweep.alpha))
    return np.convolve(rate, np.ones(3) / 3.0, mode="same")  # light smoothing


def _rise_knee(values: np.ndarray, alpha: np.ndarray) -> float:
    """Alpha where a flat-then-rising curve starts climbing (Kneedle)."""
    if alpha.size < 3:
        return float(alpha[-1]) if alpha.size else 0.0
    y = np.log(np.maximum(values, 1.0))
    la = np.log(alpha)
    x = (la - la[0]) / (la[-1] - la[0])
    yy = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    return float(alpha[int(np.argmax(x - yy))])


def usable_band(sweep: AlphaSweep):
    """
    The alpha range worth picking from, or ``None`` if there isn't one.

    Lower edge: the homogenisation-rate peak (the long bridges have been cut, so
    below it the shape is still a blob).  Upper edge: where the component count
    starts climbing (above it the shape fragments).  When the lower edge is above
    the upper one there is no clean structural scale (e.g. a featureless uniform
    cloud) and ``None`` is returned.

    Returns:
        ``(lo, hi, centre)`` with ``centre`` the geometric mean, or ``None``.
    """
    rate = homogenisation_rate(sweep)
    if rate.size == 0:
        return None
    lo = float(sweep.alpha[int(np.argmax(rate))])
    hi = _rise_knee(sweep.n_components, sweep.alpha)
    if lo >= hi:
        return None
    return lo, hi, float(np.sqrt(lo * hi))


def alpha_band(points: Union[List[Tuple[float]], np.ndarray],
               n_steps: int = 80):
    """
    Suggested alpha at the centre of the usable band (see :func:`usable_band`).

    Returns ``None`` when there is no clean structural scale, so callers can fall
    back to another selector.
    """
    band = usable_band(alpha_sweep(points, n_steps))
    return None if band is None else band[2]
