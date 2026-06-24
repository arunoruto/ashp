#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ashp` package."""

import itertools

import numpy as np
import shapely
from click.testing import CliRunner

from ashp.alphashape import alphashape
from ashp import cli


def test_given_a_point_return_a_point():
    """Given a point, alphashape should return the same point."""
    assert shapely.geometry.Point([0., 0.]) == alphashape([(0., 0.)], 0)
    assert shapely.geometry.Point([1., 0.]) == alphashape([(1., 0.)], 0)
    assert shapely.geometry.Point([0., 1.]) == alphashape([(0., 1.)], 0)
    assert shapely.geometry.Point([0., 0.]) == alphashape([(0., 0.)], 99)
    assert shapely.geometry.Point([1., 0.]) == alphashape([(1., 0.)], 99)
    assert shapely.geometry.Point([0., 1.]) == alphashape([(0., 1.)], 99)


def test_given_a_line_with_dupicate_points_return_a_point():
    """Given a line with duplicate points, alphashape should return a point."""
    assert shapely.geometry.Point([0., 1.]) == alphashape(
        [(0., 1.), (0., 1.)], 0)


def test_given_a_line_with_unique_points_return_a_line():
    """Given a line with unique points, alphashape should return the line."""
    assert shapely.geometry.LineString([(0., 0.), (0., 1.)]) == alphashape(
        [(0., 0.), (0., 1.)], 0)
    assert shapely.geometry.LineString([(1., 0.), (0., 1.)]) == alphashape(
        [(1., 0.), (0., 1.)], 0)


def test_given_a_triangle_with_duplicate_points_returns_a_point():
    """Given a triangle with one unique point, return that point."""
    assert shapely.geometry.Point((0., 1.)) == alphashape(
        [(0., 1.), (0., 1.), (0., 1.)], 0)


def test_given_a_triangle_with_two_duplicate_points_returns_a_line():
    """Given a triangle with two unique points, return a line."""
    assert shapely.geometry.LineString([(1., 0.), (0., 1.)]) == alphashape(
        [(1., 0.), (0., 1.), (0., 1.)], 0)


def test_given_a_four_point_polygon_with_small_alpha_return_input():
    """Given four points and a tiny alpha, return the input polygon."""
    assert shapely.geometry.Polygon([
        (0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)]).equals(
            alphashape([(0., 0.), (0., 1.), (1., 1.), (1., 0.)], 1.e-9))


def test_given_a_four_point_polygon_with_no_alpha_return_input():
    """Given four points and no alpha, return the input polygon."""
    assert shapely.geometry.Polygon([
        (0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)]).equals(
            alphashape([(0., 0.), (0., 1.), (1., 1.), (1., 0.)]))


def test_2d_large_cloud_returns_valid_concave_polygon():
    """
    A non-trivial 2-D cloud (exercising the vectorised boundary path) yields a
    valid polygon no larger than the convex hull.
    """
    rng = np.random.default_rng(0)
    pts = rng.random((300, 2))
    shape = alphashape(pts, 8.0)
    hull = shapely.geometry.MultiPoint([tuple(p) for p in pts]).convex_hull
    assert shape.geom_type in ('Polygon', 'MultiPolygon')
    assert shape.is_valid
    assert 0.0 < shape.area <= hull.area


def test_2d_tiny_alpha_fills_to_convex_hull():
    """
    With a tiny (positive) alpha every Delaunay triangle passes the radius
    filter, so the 2-D alpha shape equals the convex hull of the points.
    """
    rng = np.random.default_rng(1)
    pts = rng.random((200, 2))
    hull = shapely.geometry.MultiPoint([tuple(p) for p in pts]).convex_hull
    shape = alphashape(pts, 1e-9)
    assert shape.symmetric_difference(hull).area < 1e-9


POINTS_3D = [
    (0., 0., 0.), (0., 0., 1.), (0., 1., 0.),
    (1., 0., 0.), (1., 1., 0.), (1., 0., 1.),
    (0., 1., 1.), (1., 1., 1.), (.25, .5, .5),
    (.5, .25, .5), (.5, .5, .25), (.75, .5, .5),
    (.5, .75, .5), (.5, .5, .75),
]

EXPECTED_VERTICES_3D = [
    [0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
    [0., 1., 1.], [1., 0., 0.], [1., 0., 1.],
    [1., 1., 0.], [1., 1., 1.], [0.25, 0.5, 0.5],
    [0.5, 0.25, 0.5], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75],
    [0.5, 0.75, 0.5], [0.75, 0.5, 0.5]]

EXPECTED_FACES_3D = [
    (13, 10, 6), (13, 9, 4), (6, 12, 13),
    (13, 12, 7), (5, 11, 9), (8, 10, 0),
    (3, 12, 8), (0, 10, 9), (5, 9, 13),
    (12, 11, 7), (9, 10, 4), (8, 9, 1),
    (12, 10, 2), (13, 11, 5), (1, 11, 8),
    (4, 10, 13), (9, 11, 1), (2, 10, 8),
    (8, 12, 2), (3, 11, 12), (0, 9, 8),
    (7, 11, 13), (6, 10, 12), (8, 11, 3)]


def _check_3d_result(results):
    assert len(results.vertices) == len(EXPECTED_VERTICES_3D)
    assert len(POINTS_3D) == len(EXPECTED_VERTICES_3D)
    assert len(results.faces) == len(EXPECTED_FACES_3D)
    vertex_map = {i: EXPECTED_VERTICES_3D.index(list(vertex))
                  for i, vertex in enumerate(results.vertices)}
    for edge in list(results.faces):
        assert any([(
            vertex_map[e[0]],
            vertex_map[e[1]],
            vertex_map[e[2]]) in EXPECTED_FACES_3D
            for e in itertools.combinations(edge, r=len(edge))])


def test_3_dimensional_regression_with_dynamic_alpha():
    """Given a 3-dimensional data set, return an expected set of edges."""
    _check_3d_result(alphashape(POINTS_3D, lambda a, b: 2.1))


def test_3_dimensional_regression():
    """Given a 3-dimensional data set, return an expected set of edges."""
    _check_3d_result(alphashape(POINTS_3D, 2.1))


def test_4_dimensional_regression():
    """Given a 4-dimensional data set, return an expected set of edges."""
    points_4d = [
        (0., 0., 0., 0.), (0., 0., 0., 1.), (0., 0., 1., 0.),
        (0., 1., 0., 0.), (0., 1., 1., 0.), (0., 1., 0., 1.),
        (0., 0., 1., 1.), (0., 1., 1., 1.), (1., 0., 0., 0.),
        (1., 0., 0., 1.), (1., 0., 1., 0.), (1., 1., 0., 0.),
        (1., 1., 1., 0.), (1., 1., 0., 1.), (1., 0., 1., 1.),
        (1., 1., 1., 1.), (.25, .5, .5, .5), (.5, .25, .5, .5),
        (.5, .5, .25, .5), (.5, .5, .5, .25), (.75, .5, .5, .5),
        (.5, .75, .5, .5), (.5, .5, .75, .5), (.5, .5, .5, .75),
    ]
    expected = {
        (16, 1, 2, 0), (16, 1, 3, 0), (16, 2, 3, 0),
        (16, 4, 2, 3), (16, 4, 7, 2), (16, 4, 7, 3),
        (16, 5, 1, 3), (16, 5, 7, 1), (16, 5, 7, 3),
        (16, 6, 1, 2), (16, 6, 7, 1), (16, 6, 7, 2),
        (17, 1, 2, 0), (17, 1, 8, 0), (17, 2, 8, 0),
        (17, 6, 1, 2), (17, 6, 14, 1), (17, 6, 14, 2),
        (17, 9, 1, 8), (17, 9, 14, 1), (17, 9, 14, 8),
        (17, 10, 2, 8), (17, 10, 14, 2), (17, 10, 14, 8),
        (18, 1, 3, 0), (18, 1, 8, 0), (18, 3, 8, 0),
        (18, 5, 1, 3), (18, 5, 13, 1), (18, 5, 13, 3),
        (18, 9, 1, 8), (18, 9, 13, 1), (18, 9, 13, 8),
        (18, 11, 3, 8), (18, 11, 13, 3), (18, 11, 13, 8),
        (19, 2, 3, 0), (19, 2, 8, 0), (19, 3, 8, 0),
        (19, 4, 2, 3), (19, 4, 12, 2), (19, 4, 12, 3),
        (19, 10, 2, 8), (19, 10, 12, 2), (19, 10, 12, 8),
        (19, 11, 3, 8), (19, 11, 12, 3), (19, 11, 12, 8),
        (20, 9, 13, 8), (20, 9, 14, 8), (20, 9, 14, 13),
        (20, 10, 12, 8), (20, 10, 14, 8), (20, 10, 14, 12),
        (20, 11, 12, 8), (20, 11, 13, 8), (20, 11, 13, 12),
        (20, 13, 12, 15), (20, 14, 12, 15), (20, 14, 13, 15),
        (21, 4, 7, 3), (21, 4, 7, 12), (21, 4, 12, 3),
        (21, 5, 7, 3), (21, 5, 7, 13), (21, 5, 13, 3),
        (21, 7, 12, 15), (21, 7, 13, 15), (21, 11, 12, 3),
        (21, 11, 13, 3), (21, 11, 13, 12), (21, 13, 12, 15),
        (22, 4, 7, 2), (22, 4, 7, 12), (22, 4, 12, 2),
        (22, 6, 7, 2), (22, 6, 7, 14), (22, 6, 14, 2),
        (22, 7, 12, 15), (22, 7, 14, 15), (22, 10, 12, 2),
        (22, 10, 14, 2), (22, 10, 14, 12), (22, 14, 12, 15),
        (23, 5, 7, 1), (23, 5, 7, 13), (23, 5, 13, 1),
        (23, 6, 7, 1), (23, 6, 7, 14), (23, 6, 14, 1),
        (23, 7, 13, 15), (23, 7, 14, 15), (23, 9, 13, 1),
        (23, 9, 14, 1), (23, 9, 14, 13), (23, 14, 13, 15)}
    results = alphashape(points_4d, 1.0)
    assert len(results) == len(expected)
    for edge in list(results):
        assert any([e in expected for e in itertools.combinations(
            edge, r=len(edge))])


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert 'Show this message and exit.' in help_result.output
