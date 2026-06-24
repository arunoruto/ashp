#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ashp` clustering."""

import numpy as np

from ashp import (alpha_band, alpha_sweep, cluster, cluster_persistence,
                  select_alpha, usable_band)


def _two_blobs(seed=0, n=120, gap=6.0):
    rng = np.random.default_rng(seed)
    a = rng.normal([0.0, 0.0], 0.3, (n, 2))
    b = rng.normal([gap, 0.0], 0.3, (n, 2))
    return np.vstack([a, b]), n


def test_persistence_finds_two_blobs():
    pts, _ = _two_blobs()
    cp = cluster_persistence(pts, min_size=10)
    assert cp.best_k == 2
    assert cp.best_alpha > 0.0
    # the curve is a 1-D ascending alpha grid aligned with the counts
    assert cp.alpha.shape == cp.n_clusters.shape
    assert np.all(np.diff(cp.alpha) >= 0)


def test_cluster_labels_two_blobs():
    pts, n = _two_blobs()
    labels = cluster(pts, min_size=10)  # alpha auto-selected
    assert set(labels[labels >= 0].tolist()) == {0, 1}
    # each blob is dominated by a single, distinct cluster
    first = np.bincount(labels[:n][labels[:n] >= 0]).argmax()
    second = np.bincount(labels[n:][labels[n:] >= 0]).argmax()
    assert first != second


def test_cluster_marks_isolated_point_as_noise():
    rng = np.random.default_rng(1)
    pts = np.vstack([rng.normal([0.0, 0.0], 0.3, (60, 2)), [[100.0, 100.0]]])
    labels = cluster(pts, alpha=5.0, min_size=5)
    assert labels[-1] == -1            # the far point is noise
    assert (labels >= 0).any()          # the blob is still a cluster


def test_alpha_sweep_metrics_are_monotonic():
    pts, _ = _two_blobs()
    sw = alpha_sweep(pts, n_steps=60)
    assert sw.alpha.shape == sw.edge_std.shape == sw.n_components.shape
    assert np.all(np.diff(sw.alpha) > 0)              # ascending alpha
    assert np.all(np.diff(sw.n_components) >= 0)       # only fragments as alpha grows
    assert np.all(np.diff(sw.n_edges) <= 0)            # fewer edges as alpha grows
    assert np.all(sw.edge_std >= 0) and np.all(sw.edge_cv >= 0)


def test_usable_band_present_for_structure_absent_for_uniform():
    pts, _ = _two_blobs()                       # clear structure -> a band
    band = usable_band(alpha_sweep(pts))
    assert band is not None
    lo, hi, centre = band
    assert lo < centre < hi

    rng = np.random.default_rng(0)               # no structure -> no band
    assert usable_band(alpha_sweep(rng.random((400, 2)))) is None


def test_select_alpha_band_falls_back_when_no_band():
    pts, _ = _two_blobs()
    assert select_alpha(pts, method="band") == alpha_band(pts)
    # uniform has no band -> select_alpha falls back to a positive (knee) alpha
    rng = np.random.default_rng(0)
    uni = rng.random((400, 2))
    assert alpha_band(uni) is None
    assert select_alpha(uni, method="band") > 0.0
