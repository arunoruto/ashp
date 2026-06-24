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

from ashp import alphashape, optimizealpha  # noqa: E402
from plotting import make_figure, make_figure_3d, sample_points  # noqa: E402

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
    auto = st.toggle(
        "Optimize alpha automatically", value=False, disabled=is_3d,
        help="2-D only." if is_3d else "Solve for the tightest covering alpha.")
    alpha_input = st.slider(
        "alpha", 0.0, 30.0, 6.0 if is_3d else 4.0, step=0.1, disabled=auto,
        help="Larger alpha = tighter (more concave) shape.")


@st.cache_data(show_spinner=False)
def compute(dataset: str, n: int, seed: int, auto: bool, alpha: float):
    """Sample points and build the alpha shape, cached on the inputs so that
    Streamlit reruns (e.g. unrelated widget changes) don't recompute it.

    Returns ``(points, payload, used_alpha)`` where ``payload`` is either
    ``("2d", geometry)`` or ``("3d", vertices, faces)`` (arrays, so the result
    stays picklable for the cache)."""
    points = sample_points(dataset, n, seed)
    if points.shape[1] == 3:
        mesh = alphashape(points, alpha)
        faces = np.asarray(mesh.faces)
        if faces.ndim != 2:
            faces = faces.reshape(0, 3)
        return points, ("3d", np.asarray(mesh.vertices), faces), alpha
    used = optimizealpha(points, upper=200.0, silent=True) if auto else alpha
    return points, ("2d", alphashape(points, used)), used


start = time.perf_counter()
spinner = "Optimizing alpha…" if (auto and not is_3d) else "Computing alpha shape…"
with st.spinner(spinner):
    points, payload, used_alpha = compute(
        dataset, n, int(seed), auto and not is_3d, alpha_input)
elapsed = time.perf_counter() - start

if payload[0] == "3d":
    _, vertices, faces = payload
    st.plotly_chart(make_figure_3d(points, vertices, faces), width="stretch")
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
        st.info("No surface at this alpha — it is too high. Lower the slider.")
else:
    c3.metric("Geometry", geom.geom_type)
    c4.metric("Compute", f"{elapsed * 1e3:.0f} ms")
    st.caption(f"Resulting area: {getattr(geom, 'area', 0.0):.3f}")
    if auto and used_alpha == 0.0:
        st.info(
            "The optimizer returned alpha = 0 (convex hull). That happens when "
            "no single alpha wraps every point into one polygon — common for "
            "uniform clouds. Try the *two moons* or *spiral* datasets.")
