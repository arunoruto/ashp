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

from ashp import (alphashape, alpha_knee, optimizealpha,  # noqa: E402
                  select_alpha)
from plotting import (knee_figure, make_figure, make_figure_3d,  # noqa: E402
                      sample_points)

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
    modes = ["Knee (auto)", "Quantile (robust)", "Manual"]
    if not is_3d:
        modes.append("Optimize (cover all)")
    mode = st.radio("Selection", modes, index=0)

    alpha_input, q_input = 4.0, 0.90
    if mode == "Manual":
        alpha_input = st.slider(
            "alpha", 0.0, 30.0, 6.0 if is_3d else 4.0, step=0.1,
            help="Larger alpha = tighter shape. Note the useful range shifts "
                 "with the point count — see the 'Knee' / 'Quantile' modes.")
    elif mode == "Quantile (robust)":
        q_input = st.slider(
            "q — fraction of simplices kept", 0.50, 1.0, 0.90, step=0.01,
            help="select_alpha: a rank statistic, so the shape stays stable as "
                 "you change the point count. 1.0 = convex hull; lower carves "
                 "tighter.")
    elif mode == "Knee (auto)":
        st.caption("alpha_knee: cuts the long 'bridge' edges automatically at "
                   "the knee of the circumradius distribution. See the "
                   "diagnostic below the shape.")


@st.cache_data(show_spinner=False)
def compute(dataset: str, n: int, seed: int, mode: str, alpha: float, q: float):
    """Sample points and build the alpha shape, cached on the inputs so that
    Streamlit reruns (e.g. unrelated widget changes) don't recompute it.

    Returns ``(points, payload, used_alpha, knee)`` where ``payload`` is either
    ``("2d", geometry)`` or ``("3d", vertices, faces)`` (arrays, so the result
    stays picklable for the cache), and ``knee`` is the AlphaKnee diagnostic
    when the knee mode is active, else ``None``."""
    points = sample_points(dataset, n, seed)
    knee = None
    if mode == "Knee (auto)":
        knee = alpha_knee(points)
        used = knee.alpha
    elif mode == "Quantile (robust)":
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
        return points, ("3d", np.asarray(mesh.vertices), faces), used, knee
    return points, ("2d", alphashape(points, used)), used, knee


start = time.perf_counter()
spinner = ("Optimizing alpha…" if mode == "Optimize (cover all)"
           else "Computing alpha shape…")
with st.spinner(spinner):
    points, payload, used_alpha, knee = compute(
        dataset, n, int(seed), mode, alpha_input, q_input)
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
    if mode == "Optimize (cover all)" and used_alpha == 0.0:
        st.info(
            "The optimizer returned alpha = 0 (convex hull). That happens when "
            "no single alpha wraps every point into one polygon — common for "
            "uniform clouds. Try the *Knee* mode, or the *two moons* / "
            "*spiral* datasets.")

if knee is not None:
    st.subheader("Why this alpha?")
    st.caption(
        "Delaunay simplices sorted by circumradius. The bulk on the left are "
        "homogeneous local connections; the steep tail on the right is the long "
        "edges that bridge gaps. The knee (dashed) is the cutoff — everything "
        "above it is dropped.")
    st.plotly_chart(knee_figure(knee), width="stretch")
