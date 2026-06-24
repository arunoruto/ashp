#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ashp` package."""

import numpy as np
import pytest
import shapely

from ashp import alphashape, alpha_knee, optimizealpha, select_alpha
from ashp.alphashape import _delaunay_circumradii


def _annulus(seed=0, n=400, r_in=0.55):
    """Ring with an empty centre — its Delaunay has long hole-crossing slivers."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(rng.uniform(r_in ** 2, 1.0, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def test_given_a_point_return_a_point():
    """The solver should find an alpha in the expected range."""
    alpha = optimizealpha(
        [(0., 0.), (0., 1.), (1., 1.), (1., 0.),
         (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)])
    assert alpha > 3. and alpha < 3.5


def test_reach_max_iterations():
    """
    Given a non-trivial set of points, a non-trivial interval of possible
    alpha values and allowing only 2 iterations, the optimizealpha function
    should reach max_iterations and return 0.
    """
    with pytest.warns(Warning):
        alpha = optimizealpha(
            [(0., 0.), (0., 1.), (1., 1.), (1., 0.),
             (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)],
            max_iterations=2, lower=0.0, upper=1000.0)
    assert alpha == 0.0


def test_select_alpha_monotonic_and_hull():
    """Smaller q carves tighter (larger alpha); q = 1 gives the convex hull."""
    pts = np.random.default_rng(0).random((300, 2))
    a_tight = select_alpha(pts, q=0.7)
    a_loose = select_alpha(pts, q=0.97)
    assert a_tight > a_loose > 0.0
    hull = shapely.geometry.MultiPoint([tuple(p) for p in pts]).convex_hull
    shape = alphashape(pts, select_alpha(pts, q=1.0))
    assert shape.symmetric_difference(hull).area < 1e-9


def test_select_alpha_rejects_bad_q():
    with pytest.raises(ValueError):
        select_alpha([(0., 0.), (1., 0.), (0., 1.), (1., 1.)], q=1.5)


def test_alpha_knee_cuts_the_outlier_tail():
    pts = _annulus()
    ak = alpha_knee(pts)
    assert ak.alpha > 0.0
    assert ak.cut == ak.radii_sorted[ak.knee_index]
    # there is a long tail, and the knee sits in the upper part of the curve,
    # cutting only a small fraction of (the longest) simplices.
    assert ak.radii_sorted[-1] > 5.0 * ak.cut
    assert ak.knee_index > 0.5 * len(ak.radii_sorted)
    _, _, radii = _delaunay_circumradii(pts)
    radii = radii[np.isfinite(radii)]
    assert (radii > ak.cut).sum() < 0.25 * len(radii)


def test_select_alpha_knee_matches_alpha_knee():
    pts = _annulus(seed=1)
    assert select_alpha(pts, method="knee") == alpha_knee(pts).alpha


def test_select_alpha_rejects_bad_method():
    with pytest.raises(ValueError):
        select_alpha([(0., 0.), (1., 0.), (0., 1.), (1., 1.)], method="nope")
