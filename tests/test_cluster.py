#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ashp` clustering."""

import numpy as np

from ashp import cluster, cluster_persistence


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
