"""
Tools for working with alpha shapes.
"""
__all__ = ['alphashape']

import warnings
import shapely
from shapely.geometry import MultiPoint
from scipy.spatial import Delaunay
import numpy as np
from typing import Union, Tuple, List

try:
    import geopandas
    USE_GP = True
except ImportError:
    USE_GP = False

# Optional numba acceleration.  The per-simplex circumradius computation is the
# hot path: one small linear solve for every Delaunay simplex.  JIT compiling
# that kernel (cached to disk) and running the simplices in parallel gives a
# large speed-up.  If numba is not installed the same functions run as plain
# numpy via the shims below, so the package keeps working without it.
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    USE_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in for ``numba.njit`` when numba is unavailable."""
        if args and callable(args[0]):
            return args[0]

        def wrap(func):
            return func
        return wrap

    prange = range


@njit(cache=True)
def _circumradius_simplex(points: np.ndarray) -> float:
    """
    Circumradius of a single simplex.

    Parameters
    ----------
    points : numpy.ndarray
        An ``N``-by-``K`` array defining one ``(N-1)``-simplex.

    Returns
    -------
    float
        The circumradius, or ``inf`` for a degenerate (singular) simplex so
        that it is harmlessly filtered out downstream.
    """
    n = points.shape[0]
    gram = points @ points.T
    a = np.ones((n + 1, n + 1))
    a[:n, :n] = 2.0 * gram
    a[n, n] = 0.0
    b = np.ones(n + 1)
    for i in range(n):
        b[i] = gram[i, i]
    if abs(np.linalg.det(a)) < 1e-12:
        return np.inf
    bary = np.linalg.solve(a, b)[:n]
    center = bary @ points
    diff = points[0] - center
    return np.sqrt(np.sum(diff * diff))


# Spawning numba's thread pool costs a fixed ~10 ms per call, so the parallel
# kernel only pays off for large meshes.  Below this many simplices the serial
# kernel is faster (and avoids penalising the many small calls that
# ``optimizealpha`` makes during bisection).
_PARALLEL_THRESHOLD = 15000


@njit(cache=True)
def _circumradii_serial(coords: np.ndarray,
                        simplices: np.ndarray) -> np.ndarray:
    """Circumradius of every simplex (serial)."""
    out = np.empty(simplices.shape[0])
    for i in range(simplices.shape[0]):
        out[i] = _circumradius_simplex(coords[simplices[i]])
    return out


@njit(cache=True, parallel=True)
def _circumradii_parallel(coords: np.ndarray,
                          simplices: np.ndarray) -> np.ndarray:
    """Circumradius of every simplex, computed in parallel."""
    out = np.empty(simplices.shape[0])
    for i in prange(simplices.shape[0]):
        out[i] = _circumradius_simplex(coords[simplices[i]])
    return out


def _circumradii(coords: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Circumradius of every simplex, parallelising only large meshes."""
    if USE_NUMBA and simplices.shape[0] >= _PARALLEL_THRESHOLD:
        return _circumradii_parallel(coords, simplices)
    return _circumradii_serial(coords, simplices)


def circumcenter(points: Union[List[Tuple[float]], np.ndarray]) -> np.ndarray:
    """
    Circumcenter of a set of points, in barycentric coordinates.

    Parameters
    ----------
    points : list of tuple of float or numpy.ndarray
        An ``N``-by-``K`` array of points defining an ``(N-1)``-simplex in
        ``K``-dimensional space, with ``1 <= N <= K`` and ``K >= 1``.

    Returns
    -------
    numpy.ndarray
        The circumcenter of the points, in barycentric coordinates.

    References
    ----------
    :cite:p:`edelsbrunner1983`
    """
    points = np.asarray(points, dtype=np.float64)
    num_rows, num_columns = points.shape
    A = np.ones((num_rows + 1, num_rows + 1))
    A[:num_rows, :num_rows] = 2 * np.dot(points, points.T)
    A[num_rows, num_rows] = 0.0
    b = np.ones(num_rows + 1)
    b[:num_rows] = np.sum(points * points, axis=1)
    return np.linalg.solve(A, b)[:-1]


def circumradius(points: Union[List[Tuple[float]], np.ndarray]) -> float:
    """
    Circumradius of a set of points.

    Parameters
    ----------
    points : list of tuple of float or numpy.ndarray
        An ``N``-by-``K`` array of points defining an ``(N-1)``-simplex in
        ``K``-dimensional space, with ``1 <= N <= K`` and ``K >= 1``.

    Returns
    -------
    float
        The circumradius of the points.

    References
    ----------
    :cite:p:`edelsbrunner1983`
    """
    points = np.asarray(points)
    return np.linalg.norm(points[0, :] - np.dot(circumcenter(points), points))


def _delaunay_circumradii(points: Union[List[Tuple[float]], np.ndarray]):
    """Delaunay-triangulate ``points``; return ``(coords, simplices, radii)``."""
    coords = np.ascontiguousarray(points, dtype=np.float64)
    simplices = Delaunay(coords).simplices
    radii = _circumradii(coords, simplices)
    if not np.all(np.isfinite(radii)):
        warnings.warn('Singular matrix. Likely caused by all points '
                      'lying in an N-1 space.')
    return coords, simplices, radii


def alphasimplices(points: Union[List[Tuple[float]], np.ndarray]) -> \
        Union[List[Tuple[float]], np.ndarray]:
    """
    Iterate over the Delaunay simplices and their circumradii.

    Parameters
    ----------
    points : list of tuple of float or numpy.ndarray
        An ``N``-by-``M`` array of points.

    Yields
    ------
    tuple of (numpy.ndarray, float)
        Each simplex (as an array of vertex indices) paired with its
        circumradius.
    """
    _, simplices, radii = _delaunay_circumradii(points)
    for simplex, radius in zip(simplices, radii):
        yield simplex, radius


def _perimeter_facets(simplices: np.ndarray, radii: np.ndarray,
                      alpha) -> np.ndarray:
    """
    Vectorised boundary-facet extraction for any dimension.

    A facet (an edge in 2-D, a triangle in 3-D, ... — the ``dim``-vertex faces
    of each ``dim+1``-vertex simplex) lies on the alpha shape's boundary when it
    is incident to exactly one passing simplex.  Canonicalising every facet and
    keeping those with a count of one yields the perimeter directly, replacing
    the per-simplex Python loop with array operations.

    Each returned facet keeps the vertex order of its originating simplex, so
    the output is identical (not just equivalent) to the previous loop.

    Parameters
    ----------
    simplices : numpy.ndarray
        The Delaunay simplices, an ``M``-by-``(dim+1)`` array of vertex indices.
    radii : numpy.ndarray
        The circumradius of each simplex.
    alpha : float or callable
        The alpha value, or a callable ``alpha(simplex, circumradius)`` giving a
        per-simplex value.

    Returns
    -------
    numpy.ndarray
        A ``(B, dim)`` array of boundary-facet vertex indices.
    """
    dim = simplices.shape[1] - 1

    if callable(alpha):
        resolved = np.fromiter(
            (alpha(simplices[i], radii[i]) for i in range(simplices.shape[0])),
            dtype=np.float64, count=simplices.shape[0])
        mask = radii < 1.0 / resolved
    else:
        mask = radii < 1.0 / alpha

    passing = simplices[mask]
    if passing.shape[0] == 0:
        return np.empty((0, dim), dtype=simplices.dtype)

    # Every facet of every passing simplex, dropping one vertex at a time and
    # preserving the order of the remaining vertices.
    keep = [np.delete(np.arange(dim + 1), j) for j in range(dim + 1)]
    facets = np.concatenate([passing[:, k] for k in keep])

    # A canonical (sorted) copy is used only to match shared facets; the facet
    # itself is emitted in its original order.
    canonical = np.sort(facets, axis=1)
    _, index, counts = np.unique(
        canonical, axis=0, return_index=True, return_counts=True)
    return facets[index[counts == 1]]


def alphashape(points: Union[List[Tuple[float]], np.ndarray],
               alpha: Union[None, float] = None):
    """
    Compute the alpha shape (concave hull) of a set of points.

    Delaunay-triangulate the points and keep the simplices whose circumradius is
    below ``1 / alpha``; the boundary of the kept simplices is the alpha shape
    :cite:p:`edelsbrunner1983,edelsbrunner1994`. With three points or fewer, or
    ``alpha <= 0``, the convex hull is returned (a ``Polygon``, or a
    ``LineString`` for two points / a ``Point`` for one).

    Parameters
    ----------
    points : list of tuple of float, numpy.ndarray, shapely.geometry.MultiPoint or geopandas.GeoDataFrame
        An iterable container of 2-D or 3-D points.
    alpha : float or callable, optional
        The alpha value. ``0`` (or less) returns the convex hull; larger values
        give a tighter shape. A callable ``alpha(simplex, circumradius)`` may be
        passed for a per-simplex value. If ``None`` (the default), it is solved
        for with :func:`optimizealpha`.

    Returns
    -------
    shapely.geometry.base.BaseGeometry or trimesh.Trimesh or geopandas.GeoDataFrame
        For 2-D input, a shapely ``Polygon`` / ``LineString`` / ``Point`` (or a
        ``GeoDataFrame`` if one was given); for 3-D input, a
        ``trimesh.Trimesh`` surface mesh; for higher dimensions, the set of
        boundary-facet index tuples.

    See Also
    --------
    select_alpha : fast, data-driven alpha from the circumradius distribution.
    optimizealpha : solve for the tightest alpha that covers every point.

    References
    ----------
    :cite:p:`edelsbrunner1983,edelsbrunner1994`
    """
    # If given a geodataframe, extract the geometry
    if USE_GP and isinstance(points, geopandas.GeoDataFrame):
        crs = points.crs
        points = points['geometry']
    else:
        crs = None

    # If given a triangle for input, or an alpha value of zero or less,
    # return the convex hull.
    if len(points) < 4 or (alpha is not None and not callable(
            alpha) and alpha <= 0):
        if not isinstance(points, MultiPoint):
            points = MultiPoint(list(points))
        result = points.convex_hull
        if crs:
            gdf = geopandas.GeoDataFrame(geopandas.GeoSeries(result)).rename(
                columns={0: 'geometry'}).set_geometry('geometry')
            gdf.crs = crs
            return gdf
        else:
            return result

    # Determine alpha parameter if one is not given
    if alpha is None:
        from .optimizealpha import optimizealpha
        alpha = optimizealpha(points)

    # Convert the points to a numpy array
    if USE_GP and isinstance(points, geopandas.geoseries.GeoSeries):
        coords = np.array([point.coords[0] for point in points])
    else:
        coords = np.array(points)

    # Triangulate once, compute every circumradius, and extract the boundary
    # facets in one vectorised pass (works for any dimension).
    coords, simplices, radii = _delaunay_circumradii(coords)
    dim = coords.shape[-1]
    boundary = _perimeter_facets(simplices, radii, alpha)

    # N-D (> 3): return the raw set of boundary facets.
    if dim > 3:
        return set(map(tuple, boundary))

    # 3-D: build a surface mesh and fix the face windings.
    if dim == 3:
        import trimesh
        faces = list(set(map(tuple, boundary)))
        result = trimesh.Trimesh(vertices=coords, faces=faces)
        # fix_normals inspects mesh edges and fails on an empty face set, which
        # happens when no simplex passes the radius filter (e.g. alpha too high).
        if faces:
            trimesh.repair.fix_normals(result)
        return result

    # 2-D: bulk-build the boundary edges and polygonise.
    if boundary.shape[0] == 0:
        result = shapely.geometry.GeometryCollection()
    else:
        lines = shapely.linestrings(coords[boundary])
        result = shapely.unary_union(shapely.polygonize(lines))

    # Convert to a geopandas geodataframe if that is what was given as input.
    if crs:
        gdf = geopandas.GeoDataFrame(geopandas.GeoSeries(result)).rename(
            columns={0: 'geometry'}).set_geometry('geometry')
        gdf.crs = crs
        return gdf
    return result
