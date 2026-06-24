"""Generate the static gallery images used in the README.

Usage::

    uv run --group dashboard python assets/generate_images.py

Writes PNGs into ``assets/img/`` (override with ``--out``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the dashboard's sampling + Plotly helpers (single source of truth), and
# make ``ashp`` importable from ``src`` even if it isn't installed.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "apps" / "dashboard", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # headless, software 3-D rendering (no WebGL needed)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from ashp import alphashape  # noqa: E402
from plotting import (POINT_COLOR, SHAPE_LINE, geom_traces,  # noqa: E402
                      make_figure, point_trace, sample_points, style_axes)

SCALE = 2  # render at 2x for crisp images


def _rgb(css: str):
    """Convert a ``"rgb(r, g, b)"`` string to a matplotlib RGB tuple."""
    r, g, b = (int(v) for v in css[css.index("(") + 1:css.index(")")].split(","))
    return (r / 255, g / 255, b / 255)


def alpha_sweep(out: Path) -> None:
    """A 1x4 panel showing the same cloud at increasing alpha."""
    points = sample_points("two moons", 220, seed=1)
    alphas = [0.0, 3.0, 6.0, 12.0]
    titles = ["alpha = 0 (convex hull)", "alpha = 3", "alpha = 6", "alpha = 12"]

    fig = make_subplots(rows=1, cols=4, subplot_titles=titles,
                        horizontal_spacing=0.02)
    for i, alpha in enumerate(alphas, start=1):
        geom = alphashape(points, alpha)
        for trace in geom_traces(geom):
            fig.add_trace(trace, row=1, col=i)
        fig.add_trace(point_trace(points), row=1, col=i)
        style_axes(fig, row=1, col=i, xref="x" if i == 1 else f"x{i}")

    fig.update_layout(template="simple_white", showlegend=False,
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.update_annotations(font_size=13)
    fig.write_image(str(out / "alpha_sweep.png"), width=1200, height=340,
                    scale=SCALE)


def single(out: Path, dataset: str, alpha: float, name: str) -> None:
    points = sample_points(dataset, 260, seed=2)
    geom = alphashape(points, alpha)
    fig = make_figure(points, geom)
    fig.update_layout(width=520, height=520)
    fig.write_image(str(out / name), width=520, height=520, scale=SCALE)


def gallery_3d(out: Path) -> None:
    """A 1x3 panel of 3-D alpha-shape surface meshes (rendered with matplotlib,
    since Plotly's WebGL meshes cannot be exported headless)."""
    datasets = [("ball (3D)", 6.0), ("torus (3D)", 6.0), ("blobs (3D)", 7.0)]
    face_rgb, point_rgb = _rgb(SHAPE_LINE), _rgb(POINT_COLOR)

    fig = plt.figure(figsize=(12, 4.3))
    for i, (dataset, alpha) in enumerate(datasets, start=1):
        points = sample_points(dataset, 600, seed=1)
        mesh = alphashape(points, alpha)
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)

        ax = fig.add_subplot(1, 3, i, projection="3d")
        tris = Poly3DCollection(v[f], alpha=0.45, facecolor=face_rgb,
                                edgecolor=face_rgb, linewidths=0.1)
        ax.add_collection3d(tris)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=2, color=point_rgb, alpha=0.35, depthshade=True)
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
