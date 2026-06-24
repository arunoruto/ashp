"""Shared plotting helpers for the ashp dashboard and the docs image generator.

Everything here turns a point cloud plus the ``shapely`` geometry returned by
:func:`ashp.alphashape` into Plotly traces, so the interactive app and the
static documentation images stay visually consistent.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

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
