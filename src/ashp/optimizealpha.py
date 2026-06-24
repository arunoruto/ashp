__all__ = ['optimizealpha']
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
