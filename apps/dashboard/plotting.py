"""Shared plotting helpers for the ashp dashboard and the docs image generator.

Everything here turns a point cloud plus the ``shapely`` geometry returned by
:func:`ashp.alphashape` into Plotly traces, so the interactive app and the
static documentation images stay visually consistent.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

POINT_COLOR = "rgb(55, 65, 80)"
SHAPE_LINE = "rgb(31, 119, 180)"
SHAPE_FILL = "rgba(31, 119, 180, 0.22)"


def sample_points(kind: str, n: int, seed: int = 0) -> np.ndarray:
    """Generate a 2-D demo point cloud.

    Args:
      kind: one of ``"uniform"``, ``"annulus"`` or ``"two moons"``.
      n: number of points.
      seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.random((n, 2))
    if kind == "annulus":
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        r = np.sqrt(rng.uniform(0.55 ** 2, 1.0, n))
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    if kind == "two moons":
        n1 = n // 2
        n2 = n - n1
        t1 = np.linspace(0.0, np.pi, n1)
        t2 = np.linspace(0.0, np.pi, n2)
        moon1 = np.column_stack([np.cos(t1), np.sin(t1)])
        moon2 = np.column_stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)])
        pts = np.vstack([moon1, moon2])
        return pts + rng.normal(0.0, 0.06, pts.shape)
    if kind == "spiral":
        t = np.linspace(0.0, 3.5 * np.pi, n)
        r = 0.15 + t / (3.5 * np.pi)
        pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
        return pts + rng.normal(0.0, 0.025, pts.shape)
    if kind == "ball (3D)":
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        r = rng.uniform(0.0, 1.0, n) ** (1.0 / 3.0)   # uniform in the volume
        return v * r[:, None]
    if kind == "torus (3D)":
        big, small = 1.0, 0.4
        u = rng.uniform(0.0, 2.0 * np.pi, n)
        w = rng.uniform(0.0, 2.0 * np.pi, n)
        pts = np.column_stack([(big + small * np.cos(w)) * np.cos(u),
                               (big + small * np.cos(w)) * np.sin(u),
                               small * np.sin(w)])
        return pts + rng.normal(0.0, 0.02, pts.shape)
    if kind == "blobs (3D)":
        n1 = n // 2
        a = rng.normal([-0.6, 0.0, 0.0], 0.25, (n1, 3))
        b = rng.normal([0.6, 0.0, 0.0], 0.25, (n - n1, 3))
        return np.vstack([a, b])
    raise ValueError(f"unknown dataset: {kind!r}")


def _iter_polygons(geom):
    """Yield every Polygon contained in an arbitrary shapely geometry."""
    gt = geom.geom_type
    if gt == "Polygon":
        yield geom
    elif gt in ("MultiPolygon", "GeometryCollection"):
        for sub in geom.geoms:
            yield from _iter_polygons(sub)


def _rings(geom):
    """Yield (x, y) arrays for every boundary ring / line in ``geom``."""
    gt = geom.geom_type
    if gt == "Polygon":
        yield np.asarray(geom.exterior.coords).T
        for interior in geom.interiors:
            yield np.asarray(interior.coords).T
    elif gt in ("LineString", "LinearRing"):
        yield np.asarray(geom.coords).T
    elif gt in ("MultiPolygon", "MultiLineString", "GeometryCollection"):
        for sub in geom.geoms:
            yield from _rings(sub)


def geom_traces(geom) -> list[go.Scatter]:
    """Build translucent fill + boundary traces for an alpha-shape geometry."""
    fills: list[go.Scatter] = []
    holes: list[go.Scatter] = []
    lines: list[go.Scatter] = []

    for poly in _iter_polygons(geom):
        x, y = np.asarray(poly.exterior.coords).T
        fills.append(go.Scatter(
            x=x, y=y, fill="toself", mode="none",
            fillcolor=SHAPE_FILL, hoverinfo="skip", showlegend=False))
        # Plotly can't subtract holes from a fill, so paint interiors with the
        # (white) background to carve them back out.  Drawn after every
        # exterior fill so it is never covered up again.
        for interior in poly.interiors:
            hx, hy = np.asarray(interior.coords).T
            holes.append(go.Scatter(
                x=hx, y=hy, fill="toself", mode="none",
                fillcolor="white", hoverinfo="skip", showlegend=False))

    for x, y in _rings(geom):
        lines.append(go.Scatter(
            x=x, y=y, mode="lines", line=dict(color=SHAPE_LINE, width=2),
            hoverinfo="skip", showlegend=False))

    traces = fills + holes + lines

    # A degenerate alpha shape can collapse to a single Point.
    if geom.geom_type == "Point":
        traces.append(go.Scatter(
            x=[geom.x], y=[geom.y], mode="markers",
            marker=dict(color=SHAPE_LINE, size=8), showlegend=False))

    return traces


def point_trace(points: np.ndarray) -> go.Scatter:
    """Marker trace for the input point cloud."""
    points = np.asarray(points)
    return go.Scatter(
        x=points[:, 0], y=points[:, 1], mode="markers",
        marker=dict(color=POINT_COLOR, size=5, opacity=0.85),
        name="points", hoverinfo="x+y", showlegend=False)


def style_axes(fig: go.Figure, row: int | None = None, col: int | None = None,
               xref: str = "x") -> None:
    """Apply a clean, equal-aspect style to one subplot (or the whole fig)."""
    hidden = dict(showgrid=False, zeroline=False, showticklabels=False,
                  ticks="", showline=False)
    if row is None:
        fig.update_xaxes(hidden)
        fig.update_yaxes(hidden, scaleanchor="x")
    else:
        fig.update_xaxes(hidden, row=row, col=col)
        fig.update_yaxes(hidden, row=row, col=col, scaleanchor=xref)


def make_figure(points: np.ndarray, geom, title: str = "") -> go.Figure:
    """Single-panel figure: input points overlaid with their alpha shape."""
    fig = go.Figure()
    for trace in geom_traces(geom):
        fig.add_trace(trace)
    fig.add_trace(point_trace(points))
    fig.update_layout(
        title=title, template="simple_white",
        margin=dict(l=10, r=10, t=40 if title else 10, b=10))
    style_axes(fig)
    return fig


def make_figure_3d(points: np.ndarray, vertices: np.ndarray,
                   faces: np.ndarray) -> go.Figure:
    """
    3-D figure: the input point cloud plus the alpha-shape surface mesh.

    ``vertices``/``faces`` are the ``trimesh.Trimesh`` arrays returned by
    :func:`ashp.alphashape` for 3-D input (``faces`` may be empty when alpha is
    too high to retain any simplex).
    """
    points = np.asarray(points)
    fig = go.Figure()
    if faces.ndim == 2 and faces.shape[0] > 0:
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=SHAPE_LINE, opacity=0.45, flatshading=True,
            hoverinfo="skip", name="alpha shape"))
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
        marker=dict(size=2.5, color=POINT_COLOR, opacity=0.85),
        hoverinfo="skip", name="points"))
    fig.update_layout(
        template="simple_white", showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(aspectmode="data",
                   xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)))
    return fig


# --------------------------------------------------------------------------- #
# Alpha-selection diagnostic
# --------------------------------------------------------------------------- #
def knee_figure(knee) -> go.Figure:
    """
    Plot the sorted Delaunay circumradii with the chosen knee marked.

    ``knee`` is an :class:`ashp.AlphaKnee`.  Simplices left of the dashed line
    (the homogeneous local connections) are kept; the steep tail to its right
    (the long edges bridging gaps) is dropped.
    """
    radii = np.asarray(knee.radii_sorted)
    fig = go.Figure()
    if radii.size == 0:
        return fig

    pct = np.linspace(0.0, 100.0, radii.size)
    knee_pct = 100.0 * knee.knee_index / max(radii.size - 1, 1)
    fig.add_trace(go.Scatter(
        x=pct, y=radii, mode="lines", line=dict(color=SHAPE_LINE, width=2),
        hoverinfo="x+y", name="circumradius"))
    fig.add_vline(x=knee_pct, line=dict(color="#d62728", dash="dash"),
                  annotation_text=f"knee → α ≈ {knee.alpha:.2f}",
                  annotation_position="top left")
    fig.add_hline(y=knee.cut, line=dict(color="#d62728", dash="dot"))
    fig.update_layout(
        template="simple_white", showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Delaunay simplices (sorted, percentile)"),
        yaxis=dict(title="circumradius", type="log"))
    return fig


def homogenisation_rate(sweep) -> np.ndarray:
    """``-d(CV)/d(log alpha)`` — the rate the kept edges become homogeneous.

    The CV curve is a smooth descent; its (negated) derivative turns the gradual
    elbow into a sharp peak at the blob -> structure transition, which is far
    more stable across point count than the circumradius knee.
    """
    if sweep.alpha.size < 3:
        return np.zeros_like(sweep.edge_cv)
    rate = -np.gradient(sweep.edge_cv, np.log(sweep.alpha))
    return np.convolve(rate, np.ones(3) / 3.0, mode="same")  # light smoothing


def sweep_figure(sweep, markers=()) -> go.Figure:
    """
    Three stacked panels vs alpha (``sweep`` is an :class:`ashp.AlphaSweep`):
    edge-length spread (std + CV), the homogenisation rate ``-d(CV)/d(log a)``
    (sharp peak at the transition), and the connected-component count.

    ``markers`` is an iterable of ``(alpha, label, color)`` drawn as vertical
    lines across all panels (e.g. the alpha in use and the candidate selectors).
    """
    rate = homogenisation_rate(sweep)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}],
               [{"secondary_y": False}]],
        subplot_titles=("edge-length spread",
                        "homogenisation rate  −d(CV)/d(log α)",
                        "connectivity"))

    fig.add_trace(go.Scatter(
        x=sweep.alpha, y=sweep.edge_std, name="std",
        line=dict(color=SHAPE_LINE, width=2)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=sweep.alpha, y=sweep.edge_cv, name="CV = std/mean",
        line=dict(color="#ff7f0e", width=2)), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=sweep.alpha, y=rate, name="−dCV/dlogα",
        line=dict(color="#9467bd", width=2)), row=2, col=1)
    if rate.size:
        pk = int(np.argmax(rate))
        fig.add_trace(go.Scatter(
            x=[sweep.alpha[pk]], y=[rate[pk]], mode="markers", showlegend=False,
            marker=dict(color="#9467bd", size=9)), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=sweep.alpha, y=sweep.n_components, name="# components",
        line=dict(color="#2ca02c", width=2)), row=3, col=1)

    positions = ["top left", "top", "top right"]
    for i, (alpha, label, color) in enumerate(markers):
        if not alpha or alpha <= 0:
            continue
        dash = "dash" if label == "in use" else "dot"
        for r in (1, 2, 3):
            fig.add_vline(x=alpha, line=dict(color=color, dash=dash, width=1.5),
                          row=r, col=1)
        fig.add_vline(x=alpha, line=dict(color=color, dash=dash, width=1.5),
                      row=1, col=1, annotation_text=label,
                      annotation_position=positions[i % len(positions)],
                      annotation_font=dict(color=color, size=11))

    for r in (1, 2, 3):
        fig.update_xaxes(type="log", row=r, col=1)
    fig.update_xaxes(title="alpha (log scale)", row=3, col=1)
    fig.update_yaxes(title="std", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title="CV", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title="rate", row=2, col=1)
    fig.update_yaxes(title="# components", type="log", row=3, col=1)
    fig.update_layout(
        template="simple_white", height=620,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0))
    return fig
