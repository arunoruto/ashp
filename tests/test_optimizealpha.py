#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ashp` package."""

import pytest

from ashp import optimizealpha


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
