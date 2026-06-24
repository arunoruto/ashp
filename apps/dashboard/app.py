"""Interactive Streamlit dashboard for exploring alpha shapes with ashp.

Run it with::

    uv run --group dashboard streamlit run apps/dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Make the app importable no matter how it is launched: ensure this directory
# (for ``plotting``) and the project's ``src`` (for ``ashp``, when it is not
# installed in the running interpreter) are both on the path.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parents[1] / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from ashp import (alphashape, cluster, cluster_persistence,  # noqa: E402
                  optimizealpha, select_alpha)
from plotting import (cluster_figure, make_figure, make_figure_3d,  # noqa: E402
                      persistence_figure, sample_points)

DATASETS_2D = ["two moons", "spiral", "annulus", "uniform"]
DATASETS_3D = ["ball (3D)", "torus (3D)", "blobs (3D)"]

st.set_page_config(page_title="ashp · alpha shapes", layout="wide")
st.title("ashp — interactive alpha shapes")
st.caption(
    "Concave hulls of a point cloud, in 2-D and 3-D. Drop alpha to 0 for the "
    "convex hull; raise it to wrap the points more tightly.")

with st.sidebar:
    st.header("Data")
    dataset = st.selectbox("Dataset", DATASETS_2D + DATASETS_3D)
    is_3d = "(3D)" in dataset
    n = st.slider("Number of points", 20, 1000, 400 if is_3d else 200, step=10)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=0)

    st.header("Alpha")
    modes = ["Quantile (robust)", "Manual"]
    if not is_3d:
        modes.append("Optimize (cover all)")
    mode = st.radio("Selection", modes, index=0)

    alpha_input, q_input = 4.0, 0.90
    if mode == "Manual":
        alpha_input = st.slider(
            "alpha", 0.0, 30.0, 6.0 if is_3d else 4.0, step=0.1,
            help="Larger alpha = tighter shape. Note the useful range shifts "
                 "with the point count — see the 'Quantile' mode.")
    elif mode == "Quantile (robust)":
        q_input = st.slider(
            "q — fraction of simplices kept", 0.50, 1.0, 0.90, step=0.01,
            help="select_alpha: a rank statistic, so the shape stays stable as "
                 "you change the point count. 1.0 = convex hull; lower carves "
                 "tighter.")


@st.cache_data(show_spinner=False)
def compute(dataset: str, n: int, seed: int, mode: str, alpha: float, q: float):
    """Sample points and build the alpha shape, cached on the inputs so that
    Streamlit reruns (e.g. unrelated widget changes) don't recompute it.

    Returns ``(points, payload, used_alpha)`` where ``payload`` is either
    ``("2d", geometry)`` or ``("3d", vertices, faces)`` (arrays, so the result
    stays picklable for the cache)."""
    points = sample_points(dataset, n, seed)
    if mode == "Quantile (robust)":
        used = select_alpha(points, q)
    elif mode == "Optimize (cover all)":
        used = optimizealpha(points, upper=200.0, silent=True)
    else:
        used = alpha
    if points.shape[1] == 3:
        mesh = alphashape(points, used)
        faces = np.asarray(mesh.faces)
        if faces.ndim != 2:
            faces = faces.reshape(0, 3)
        return points, ("3d", np.asarray(mesh.vertices), faces), used
    return points, ("2d", alphashape(points, used)), used


@st.cache_data(show_spinner=False)
def cluster_data(dataset: str, n: int, seed: int, min_size: int):
    """Cluster the sampled points and the persistence curve, cached together."""
    points = sample_points(dataset, n, seed)
    cp = cluster_persistence(points, min_size=min_size)
    labels = cluster(points, cp.best_alpha, min_size=min_size)
    return points, labels, cp.best_alpha, cp.best_k, cp.alpha, cp.n_clusters


tab_shape, tab_cluster = st.tabs(["Alpha shape", "Clustering"])

with tab_shape:
    start = time.perf_counter()
    spinner = ("Optimizing alpha…" if mode == "Optimize (cover all)"
               else "Computing alpha shape…")
    with st.spinner(spinner):
        points, payload, used_alpha = compute(
            dataset, n, int(seed), mode, alpha_input, q_input)
    elapsed = time.perf_counter() - start

    if payload[0] == "3d":
        _, vertices, faces = payload
        st.plotly_chart(make_figure_3d(points, vertices, faces),
                        width="stretch")
    else:
        _, geom = payload
        st.plotly_chart(make_figure(points, geom), width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points", len(points))
    c2.metric("Alpha used", f"{used_alpha:.2f}")
    if payload[0] == "3d":
        c3.metric("Faces", len(faces))
        c4.metric("Compute", f"{elapsed * 1e3:.0f} ms")
        if len(faces) == 0:
            st.info("No surface at this alpha — it is too high. Lower the "
                    "slider.")
    else:
        c3.metric("Geometry", geom.geom_type)
        c4.metric("Compute", f"{elapsed * 1e3:.0f} ms")
        st.caption(f"Resulting area: {getattr(geom, 'area', 0.0):.3f}")
        if mode == "Optimize (cover all)" and used_alpha == 0.0:
            st.info(
                "The optimizer returned alpha = 0 (convex hull). That happens "
                "when no single alpha wraps every point into one polygon — "
                "common for uniform clouds. Try the *Quantile* mode, or the "
                "*two moons* / *spiral* datasets.")

with tab_cluster:
    st.caption(
        "Clusters are the connected components of the alpha complex. The "
        "persistence curve picks the most stable scale automatically — read it "
        "like a k-means elbow plot.")
    min_size = st.slider("Minimum cluster size", 2, 60, 10,
                         help="Components smaller than this are treated as "
                              "noise (grey).")
    points_c, labels, best_alpha, best_k, curve_a, curve_k = cluster_data(
        dataset, n, int(seed), min_size)

    left, right = st.columns(2)
    left.plotly_chart(
        persistence_figure(curve_a, curve_k, best_alpha, best_k),
        width="stretch")
    right.plotly_chart(cluster_figure(points_c, labels), width="stretch")

    found = len(set(labels[labels >= 0].tolist()))
    d1, d2, d3 = st.columns(3)
    d1.metric("Suggested k", best_k)
    d2.metric("Clusters shown", found)
    d3.metric("Noise points", int((labels == -1).sum()))
