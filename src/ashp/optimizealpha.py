__all__ = ['optimizealpha', 'select_alpha']
import sys
import warnings
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


def select_alpha(points: Union[List[Tuple[float]], np.ndarray],
                 q: float = 0.9) -> float:
    """
    Pick an alpha from a quantile of the Delaunay circumradii.

    A fast, outlier-robust alternative to :func:`optimizealpha`.  It keeps the
    fraction ``q`` of simplices with the smallest circumradius and drops the
    largest ``1 - q`` (the slivers that bridge gaps), returning the matching
    ``alpha = 1 / circumradius`` threshold.

    Because ``q`` is a rank statistic, the *shape* it produces stays consistent
    as the number of points changes, even though the alpha value itself scales
    with point density.  ``q = 1`` keeps every simplex and returns ``0.0`` (the
    convex hull); smaller ``q`` carves the shape more tightly.

    Unlike :func:`optimizealpha` this needs a single triangulation and no
    bisection, and it works the same way in 2-D and 3-D.

    Args:
        points: an iterable container of points (2-D or 3-D).
        q: fraction of simplices to keep, in ``[0, 1]``.

    Returns:
        float: an alpha value, or ``0.0`` (convex hull) if it cannot be found.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if q >= 1.0:
        return 0.0  # keep every simplex -> convex hull

    if USE_GP and isinstance(points, geopandas.GeoDataFrame):
        points = points['geometry']
    if USE_GP and isinstance(points, geopandas.geoseries.GeoSeries):
        points = np.array([point.coords[0] for point in points])

    from .alphashape import _delaunay_circumradii
    _, _, radii = _delaunay_circumradii(points)
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
