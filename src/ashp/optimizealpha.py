__all__ = ['optimizealpha', 'select_alpha', 'alpha_knee', 'AlphaKnee']
import sys
import warnings
from dataclasses import dataclass
import shapely
from shapely.geometry import MultiPoint
import trimesh
from typing import Union, Tuple, List
import rtree  # noqa: F401  # runtime dependency of trimesh's 3-D proximity
import numpy as np
try:
    import geopandas
    USE_GP = True
except ImportError:
    USE_GP = False


def _testalpha(points: Union[List[Tuple[float]], np.ndarray], alpha: float):
    """
    Evaluates an alpha parameter.

    This helper function creates an alpha shape with the given points and alpha
    parameter.  It then checks that the produced shape is a Polygon and that it
    intersects all the input points.

    Args:
        points: data points
        alpha: alpha value

    Returns:
        bool: True if the resulting alpha shape is a single polygon that
            intersects all the input data points.
    """
    from .alphashape import alphashape
    polygon = alphashape(points, alpha)
    if isinstance(polygon, shapely.geometry.polygon.Polygon):
        if not isinstance(points, MultiPoint):
            points = MultiPoint(list(points)).geoms
        return all([polygon.intersects(point) for point in points])
    elif isinstance(polygon, trimesh.base.Trimesh):
        return len(polygon.faces) > 0 and all(
            trimesh.proximity.signed_distance(polygon, list(points)) >= 0)
    else:
        return False


def _as_point_array(points):
    """Extract a plain coordinate array, unwrapping geopandas inputs."""
    if USE_GP and isinstance(points, geopandas.GeoDataFrame):
        points = points['geometry']
    if USE_GP and isinstance(points, geopandas.geoseries.GeoSeries):
        points = np.array([point.coords[0] for point in points])
    return points


@dataclass
class AlphaKnee:
    """Result of :func:`alpha_knee`.

    Attributes:
        radii_sorted: the Delaunay circumradii, sorted ascending (the curve).
        knee_index: index into ``radii_sorted`` of the detected knee.
        cut: circumradius at the knee (the cutoff; simplices above it are
            dropped).
        alpha: ``1 / cut`` — the selected alpha.
    """
    radii_sorted: np.ndarray
    knee_index: int
    cut: float
    alpha: float


def _knee_index(values: np.ndarray) -> int:
    """Index of the knee of a sorted, convex-increasing curve (Kneedle): the
    point lying farthest below the chord from the first to the last sample."""
    n = len(values)
    if n < 3:
        return n - 1
    span = values[-1] - values[0]
    if span <= 0:
        return n - 1
    x = np.arange(n) / (n - 1)
    y = (values - values[0]) / span
    return int(np.argmax(x - y))


def alpha_knee(points: Union[List[Tuple[float]], np.ndarray]) -> AlphaKnee:
    """
    Pick alpha at the knee of the sorted circumradius curve.

    The Delaunay circumradii split into a homogeneous bulk (the local
    neighbour connections) and a long tail (the slivers that bridge gaps or
    holes).  The knee of the sorted curve — the classic heuristic also used to
    choose DBSCAN's ``eps`` — separates the two, so every simplex above it (the
    unwanted long connections) is dropped.  Parameter-free, and works the same
    in 2-D and 3-D.

    The knee is found in log-radius, which is robust to the handful of
    near-degenerate slivers (huge circumradius) that 3-D triangulations in
    particular produce; on a linear scale those outliers dominate and push the
    knee to the very end of the curve.

    Args:
        points: an iterable container of points (2-D or 3-D).

    Returns:
        An :class:`AlphaKnee` with the chosen alpha and the diagnostic curve.
    """
    from .alphashape import _delaunay_circumradii
    _, _, radii = _delaunay_circumradii(_as_point_array(points))
    radii = np.sort(radii[np.isfinite(radii)])
    radii = radii[radii > 0.0]
    if radii.size == 0:
        return AlphaKnee(radii, 0, 0.0, 0.0)
    knee = _knee_index(np.log(radii))
    cut = float(radii[knee])
    return AlphaKnee(radii, knee, cut, 1.0 / cut if cut > 0.0 else 0.0)


def select_alpha(points: Union[List[Tuple[float]], np.ndarray],
                 q: float = 0.9, method: str = "quantile") -> float:
    """
    Pick an alpha from the Delaunay circumradius distribution.

    A fast, outlier-robust alternative to :func:`optimizealpha` (a single
    triangulation, no bisection; works in 2-D and 3-D).  Two strategies:

    * ``method="quantile"`` keeps the fraction ``q`` of simplices with the
      smallest circumradius and drops the largest ``1 - q`` (the slivers that
      bridge gaps).  Because ``q`` is a rank statistic the resulting *shape*
      stays consistent as the point count changes.  ``q = 1`` returns ``0.0``
      (the convex hull); smaller ``q`` carves tighter.
    * ``method="knee"`` finds the cutoff automatically at the knee of the
      sorted circumradius curve (see :func:`alpha_knee`) — no ``q`` to choose.
    * ``method="band"`` picks the centre of the usable alpha band (see
      :func:`ashp.alpha_band`): between where the long bridges are cut and where
      the shape starts to fragment.  Falls back to the knee when there is no
      clean structural scale.

    Args:
        points: an iterable container of points (2-D or 3-D).
        q: fraction of simplices to keep, in ``[0, 1]`` (quantile method).
        method: ``"quantile"``, ``"knee"`` or ``"band"``.

    Returns:
        float: an alpha value, or ``0.0`` (convex hull) if it cannot be found.
    """
    if method == "knee":
        return alpha_knee(points).alpha
    if method == "band":
        from .cluster import alpha_band
        centre = alpha_band(points)
        return centre if centre is not None else alpha_knee(points).alpha
    if method != "quantile":
        raise ValueError("method must be 'quantile', 'knee' or 'band'")

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if q >= 1.0:
        return 0.0  # keep every simplex -> convex hull

    from .alphashape import _delaunay_circumradii
    _, _, radii = _delaunay_circumradii(_as_point_array(points))
    radii = radii[np.isfinite(radii)]
    if radii.size == 0:
        return 0.0

    threshold = float(np.quantile(radii, q))
    return 1.0 / threshold if threshold > 0.0 else 0.0


def optimizealpha(points: Union[List[Tuple[float]], np.ndarray],
                  max_iterations: int = 10000, lower: float = 0.,
                  upper: float = sys.float_info.max, silent: bool = False,
                  rel_tol: float = 1e-6):
    """
    Solve for the alpha parameter.

    Attempt to determine the alpha parameter that best wraps the given set of
    points in one polygon without dropping any points.

    Note:  If the solver fails to find a solution, a value of zero will be
    returned, which when used with the alphashape function will safely return a
    convex hull around the points.

    Args:

        points: an iterable container of points
        max_iterations (int): maximum number of iterations while finding the
            solution
        lower: lower limit for optimization
        upper: upper limit for optimization
        silent: silence warnings
        rel_tol: relative tolerance for the bisection.  The search stops once
            the bracket width is below ``rel_tol`` times the current upper
            bound.  A purely absolute tolerance would be finer than the
            floating-point resolution for any alpha above ~1, which would make
            the bisection run until ``max_iterations`` for every realistic
            input.

    Returns:

        float: The optimized alpha parameter

    """
    # Convert to a shapely multipoint object if not one already
    if USE_GP and isinstance(points, geopandas.GeoDataFrame):
        points = points['geometry']

    # Set the bounds
    assert lower >= 0, "The lower bounds must be at least 0"
    # Ensure the upper limit bounds the solution
    assert upper <= sys.float_info.max, (
        f'The upper bounds must be less than or equal to {sys.float_info.max} '
        'on your system')

    if _testalpha(points, upper):
        if not silent:
            warnings.warn('the max float value does not bound the alpha '
                          'parameter solution')
        return 0.

    # Begin the bisection loop.  Converge on a relative tolerance scaled to the
    # magnitude of the bound (with an absolute floor at the float resolution),
    # otherwise the loop can never satisfy the stopping criterion for alphas
    # above ~1 and grinds through every allowed iteration.
    counter = 0
    while (upper - lower) > max(np.finfo(float).eps * 2, rel_tol * upper):
        # Bisect the current bounds
        test_alpha = (upper + lower) * .5

        # Update the bounds to include the solution space
        if _testalpha(points, test_alpha):
            lower = test_alpha
        else:
            upper = test_alpha

        # Handle exceeding maximum allowed number of iterations
        counter += 1
        if counter > max_iterations:
            if not silent:
                warnings.warn('maximum allowed iterations reached while '
                              'optimizing the alpha parameter')
            lower = 0.
            break
    return lower
