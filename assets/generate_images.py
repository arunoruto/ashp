"""Generate the static gallery images used in the README.

Usage::

    uv run --group dashboard python assets/generate_images.py

Writes PNGs into ``assets/img/`` (override with ``--out``).  Everything is
rendered with matplotlib so the 2-D and 3-D panels share a consistent look (and
3-D meshes export headless, which Plotly's WebGL cannot).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

# Reuse the dashboard's point sampler and palette (single source of truth), and
# make ``ashp`` importable from ``src`` even if it isn't installed.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "apps" / "dashboard", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ashp import alphashape  # noqa: E402
from plotting import POINT_COLOR, SHAPE_LINE, sample_points  # noqa: E402

DPI = 200


def _rgb(css: str):
    """Convert a ``"rgb(r, g, b)"`` string to a matplotlib RGB tuple."""
    r, g, b = (int(v) for v in css[css.index("(") + 1:css.index(")")].split(","))
    return (r / 255, g / 255, b / 255)


FACE = _rgb(SHAPE_LINE)
LINE = _rgb(SHAPE_LINE)
POINT = _rgb(POINT_COLOR)


# --------------------------------------------------------------------------- #
# 2-D rendering
# --------------------------------------------------------------------------- #
def _iter_polygons(geom):
    gt = geom.geom_type
    if gt == "Polygon":
        yield geom
    elif gt in ("MultiPolygon", "GeometryCollection"):
        for sub in geom.geoms:
            yield from _iter_polygons(sub)


def _polygon_path(poly) -> MplPath:
    """Compound path for a polygon's exterior and holes (even-odd fill)."""
    verts, codes = [], []
    for ring in (poly.exterior, *poly.interiors):
        coords = np.asarray(ring.coords)
        n = len(coords)
        verts.append(coords)
        codes.append([MplPath.MOVETO] + [MplPath.LINETO] * (n - 2)
                     + [MplPath.CLOSEPOLY])
    return MplPath(np.concatenate(verts), np.concatenate(codes))


def _draw_2d(ax, points: np.ndarray, geom) -> None:
    for poly in _iter_polygons(geom):
        ax.add_patch(PathPatch(_polygon_path(poly), facecolor=FACE,
                               edgecolor="none", alpha=0.3))
        ax.plot(*poly.exterior.xy, color=LINE, linewidth=1.5)
        for ring in poly.interiors:
            ax.plot(*ring.xy, color=LINE, linewidth=1.5)
    if geom.geom_type == "LineString":
        ax.plot(*geom.xy, color=LINE, linewidth=1.5)
    elif geom.geom_type == "Point":
        ax.plot([geom.x], [geom.y], marker="o", color=LINE, markersize=5)

    ax.scatter(points[:, 0], points[:, 1], s=7, color=POINT, alpha=0.85,
               linewidths=0)
    ax.set_aspect("equal")
    ax.set_axis_off()


def alpha_sweep(out: Path) -> None:
    """A 1x4 panel showing the same cloud at increasing alpha."""
    points = sample_points("two moons", 220, seed=1)
    specs = [(0.0, "alpha = 0 (convex hull)"), (3.0, "alpha = 3"),
             (6.0, "alpha = 6"), (12.0, "alpha = 12")]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (alpha, title) in zip(axes, specs):
        _draw_2d(ax, points, alphashape(points, alpha))
        ax.set_title(title, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.05)
    fig.savefig(out / "alpha_sweep.png", dpi=DPI)
    plt.close(fig)


def single(out: Path, dataset: str, alpha: float, name: str) -> None:
    points = sample_points(dataset, 260, seed=2)
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_2d(ax, points, alphashape(points, alpha))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3-D rendering
# --------------------------------------------------------------------------- #
def gallery_3d(out: Path) -> None:
    """A 1x3 panel of 3-D alpha-shape surface meshes."""
    datasets = [("ball (3D)", 6.0), ("torus (3D)", 6.0), ("blobs (3D)", 7.0)]

    fig = plt.figure(figsize=(12, 4.3))
    for i, (dataset, alpha) in enumerate(datasets, start=1):
        points = sample_points(dataset, 600, seed=1)
        mesh = alphashape(points, alpha)
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)

        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.add_collection3d(Poly3DCollection(
            v[f], alpha=0.45, facecolor=FACE, edgecolor=FACE, linewidths=0.1))
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=2, color=POINT, alpha=0.35, depthshade=True)
        ax.set_title(dataset.replace(" (3D)", ""), fontsize=13)
        ax.set_box_aspect(np.ptp(v, axis=0))
        ax.view_init(elev=22, azim=45)
        ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.96, wspace=0.0)
    fig.savefig(out / "gallery_3d.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "img")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    alpha_sweep(args.out)
    single(args.out, "spiral", 9.0, "spiral.png")
    single(args.out, "two moons", 6.0, "two_moons.png")
    gallery_3d(args.out)

    print(f"Wrote gallery images to {args.out}")
    for png in sorted(args.out.glob("*.png")):
        print(f"  {png.name}  ({png.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
