#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ashp` package."""

import numpy as np
import pytest
import shapely

from ashp import alphashape, optimizealpha, select_alpha


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
