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

from ashp import alphashape, optimizealpha  # noqa: E402
from plotting import make_figure, sample_points  # noqa: E402

st.set_page_config(page_title="ashp · alpha shapes", layout="wide")
st.title("ashp — interactive alpha shapes")
st.caption(
    "Concave hulls of a point cloud. Drop alpha to 0 for the convex hull; "
    "raise it to wrap the points more tightly.")

with st.sidebar:
    st.header("Data")
    dataset = st.selectbox(
        "Dataset", ["two moons", "spiral", "annulus", "uniform"])
    n = st.slider("Number of points", 20, 1000, 200, step=10)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=0)

    st.header("Alpha")
    auto = st.toggle("Optimize alpha automatically", value=False)
    alpha_input = st.slider(
        "alpha", 0.0, 30.0, 4.0, step=0.1, disabled=auto,
        help="Larger alpha = tighter (more concave) shape.")

@st.cache_data(show_spinner=False)
def compute(dataset: str, n: int, seed: int, auto: bool, alpha: float):
    """Sample points and build the alpha shape, cached on the inputs so that
    Streamlit reruns (e.g. unrelated widget changes) don't recompute it."""
    points = sample_points(dataset, n, seed)
    if auto:
        # Bound the bisection so it stays responsive in the UI.
        used = optimizealpha(points, upper=200.0, silent=True)
    else:
        used = alpha
    return points, alphashape(points, used), used


start = time.perf_counter()
spinner = "Optimizing alpha…" if auto else "Computing alpha shape…"
with st.spinner(spinner):
    points, geom, used_alpha = compute(dataset, n, int(seed), auto, alpha_input)
elapsed = time.perf_counter() - start

st.plotly_chart(make_figure(points, geom), width="stretch")

area = getattr(geom, "area", 0.0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Points", len(points))
c2.metric("Alpha used", f"{used_alpha:.2f}")
c3.metric("Geometry", geom.geom_type)
c4.metric("Compute", f"{elapsed * 1e3:.0f} ms")
st.caption(f"Resulting area: {area:.3f}")

if auto and used_alpha == 0.0:
    st.info(
        "The optimizer returned alpha = 0 (convex hull). That happens when no "
        "single alpha wraps every point into one polygon — common for uniform "
        "clouds. Try the *two moons* or *annulus* datasets.")
