# Usage

## Installation

```bash
uv add ashp            # or: pip install ashp
uv add "ashp[geo]"     # optional geopandas support
```

## Computing an alpha shape

```python
from ashp import alphashape

points = [(0., 0.), (0., 1.), (1., 1.), (1., 0.),
          (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)]

shape = alphashape(points, alpha=2.0)   # -> shapely geometry
hull  = alphashape(points, alpha=0.0)   # alpha = 0 gives the convex hull
```

`alphashape` accepts 2-D points (returns a shapely `Polygon` / `LineString` /
`Point`), 3-D points (returns a `trimesh.Trimesh`), a `MultiPoint`, or a
`geopandas.GeoDataFrame`.

## Choosing `alpha`

`alpha` has units of inverse length, so the "right" value scales with point
density. ashp provides several data-driven selectors — see the
{doc}`API reference <autoapi/index>` for details:

```python
from ashp import select_alpha, optimizealpha

# Quantile of the circumradius distribution (q in [0, 1]; 1 = convex hull):
shape = alphashape(points, select_alpha(points, q=0.9))

# Automatic cutoff at the knee of the sorted circumradius curve:
shape = alphashape(points, select_alpha(points, method="knee"))

# Centre of the usable band (past the blob, before fragmentation):
shape = alphashape(points, select_alpha(points, method="band"))

# Tightest single polygon that still covers every point:
shape = alphashape(points, optimizealpha(points))
```

## Inspecting the scale

`alpha_sweep` reports, as `alpha` varies, the spread of the kept Delaunay edge
lengths and the number of connected components — the metrics behind the band
selector and the interactive dashboard:

```python
from ashp import alpha_sweep, usable_band

sweep = alpha_sweep(points)
band = usable_band(sweep)   # (lo, hi, centre) or None when no clean scale
```

## Command-line interface

```bash
ashp input_points.geojson output_shape.geojson --alpha 2.0
```
