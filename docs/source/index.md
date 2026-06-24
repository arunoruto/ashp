# ashp

**ashp** is a maintained fork of
[`alphashape`](https://github.com/bellockk/alphashape) for computing alpha
shapes (concave hulls) of point clouds in 2-D and 3-D, with a
numba-accelerated, vectorised core and a set of data-driven tools for choosing
the `alpha` parameter.

An alpha shape generalises the convex hull: keeping only the Delaunay simplices
whose circumradius is below ``1 / alpha`` carves the hull inward to follow the
shape of the data {cite:p}`edelsbrunner1983,edelsbrunner1994`.

```{toctree}
:maxdepth: 2
:caption: Contents

usage
references
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
