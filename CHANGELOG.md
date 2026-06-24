# CHANGELOG


## v1.4.0 (2026-06-24)

### Bug Fixes

- **dashboard**: Pin sweep-panel x-axis so markers don't stretch it
  ([`395e607`](https://github.com/arunoruto/ashp/commit/395e60761eda6fb384856bf842c50d3239dd639f))

The add_vline annotations pushed plotly's autorange out to ~1e14, squashing all the data into the
  left edge. Drop the annotations (use legend entries for the marker labels instead), pin the x-axis
  to the sweep's alpha range, and only draw markers that fall inside it. Also widen the subplot
  spacing so the panel titles no longer overlap the plot above.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

### Build System

- **deps**: Drop unused kaleido from the dashboard group
  ([`ad491a2`](https://github.com/arunoruto/ashp/commit/ad491a224a818b1770e05d1634662d494c9c62f1))

The gallery is now rendered entirely with matplotlib and the dashboard renders Plotly in-browser, so
  kaleido (Plotly's static export engine) is no longer used.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

### Continuous Integration

- Add workflow to publish to PyPI with uv
  ([`09fdaf1`](https://github.com/arunoruto/ashp/commit/09fdaf1d23d466783bf413228a77d6460473becf))

Runs the test suite, then builds (uv build) and publishes (uv publish) on a published GitHub Release
  or manual dispatch, using PyPI Trusted Publishing (OIDC) so no tokens or secrets are required.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Drive releases with python-semantic-release
  ([`df444b9`](https://github.com/arunoruto/ashp/commit/df444b9c576cefacbab3eb372b464db84998f42e))

Replace the release-triggered publish workflow with a push-to-main flow: python-semantic-release
  computes the next version from the Conventional Commits, bumps it, tags, builds with uv, and
  creates the GitHub release; uv then uploads to PyPI via Trusted Publishing. Update the
  semantic-release config for v9 (branches table, uv build_command, [skip ci] release commit).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Sync uv.lock version in the semantic-release build command
  ([`46e8ac4`](https://github.com/arunoruto/ashp/commit/46e8ac4e50981c068895eef04f2d4da9e6949a34))

Per python-semantic-release's uv guide, the build command must refresh the project's own version in
  uv.lock after the pyproject bump (uv lock --upgrade-package + git add) so the lockfile doesn't
  drift; then uv build.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

### Documentation

- Add 3-D gallery image
  ([`5dd106a`](https://github.com/arunoruto/ashp/commit/5dd106a62794ad78a272a7cc70746694f9dfbfb0))

Render the ball/torus/blobs alpha-shape surface meshes to a static panel via matplotlib (Plotly's
  WebGL meshes cannot be exported headless with kaleido) and show it in the README. Adds matplotlib
  to the dashboard dependency group.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Add project README
  ([`e02cb5a`](https://github.com/arunoruto/ashp/commit/e02cb5a691512dd9b4e5774a74e9c2b0a2b7e09d))

README based on the Best-README-Template with usage, performance notes, the generated gallery,
  dashboard instructions, and credit to the original alphashape project by Kenneth E. Bellock.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Render 2-D gallery images with matplotlib
  ([`3c0495a`](https://github.com/arunoruto/ashp/commit/3c0495af126c7f76c4f792f7d21db2b4fe2ccda0))

Move the 2-D docs images (alpha sweep + single examples) off Plotly/kaleido onto matplotlib so the
  whole gallery — 2-D and 3-D — shares one renderer and a consistent look. Adds a small matplotlib
  renderer for shapely geometries.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

### Features

- Add alpha-filtration clustering and persistence curve
  ([`aef5680`](https://github.com/arunoruto/ashp/commit/aef5680f1efb7db5604182e9b9a134e9be358056))

cluster(points, alpha, min_size) labels points by the connected components of the alpha complex (the
  single-linkage / DBSCAN-like view): components smaller than min_size are noise (-1), the rest are
  numbered by descending size.

cluster_persistence(points) sweeps the filtration with union-find over the Delaunay simplices
  ordered by circumradius, records the cluster count at every scale, and selects the most persistent
  plateau measured in log-radius (scale invariant; measuring in alpha=1/r would bias toward
  small-scale fragmentation). Returns the curve plus a suggested best_alpha/best_k, readable like an
  elbow plot. Works in 2-D and 3-D.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Add alpha_sweep edge/connectivity metrics and dashboard panel
  ([`9c3b64e`](https://github.com/arunoruto/ashp/commit/9c3b64ed9dcd36a17676c3baea10c195181ac1cd))

alpha_sweep(points) tracks how the kept Delaunay edge lengths (std, coefficient of variation) and
  the connected-component count change as alpha sweeps. The triangulation is fixed and alpha only
  drops edges, so it is computed once from the sorted edges via prefix sums plus a single
  incremental union-find pass — no per-alpha alpha-shape recomputation.

Dashboard: a "Metrics vs alpha" panel under the shape plots CV/std and component count against
  alpha, with the current alpha marked, so the density behaviour is visible at a glance.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Add band-based alpha selector
  ([`531ba31`](https://github.com/arunoruto/ashp/commit/531ba31e0143890856d35e1180e0c78ee829af25))

Move the band logic into the package (homogenisation_rate, usable_band, alpha_band) and add
  select_alpha(points, method="band"), which returns the centre of the usable alpha range and falls
  back to the knee when there is no clean structural scale. Add a "Band (auto)" dashboard mode (the
  new default) and have the dashboard reuse the package versions instead of duplicating them.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Add knee-based alpha selection and dashboard diagnostic
  ([`4381179`](https://github.com/arunoruto/ashp/commit/43811793b5545cd4b61d2f21699ea3633b08eec3))

alpha_knee(points) finds the cutoff at the knee of the sorted Delaunay circumradius curve (the
  DBSCAN-eps heuristic): it separates the homogeneous bulk of local connections from the long tail
  of slivers that bridge gaps/holes, and drops the latter. The knee is found in log-radius so the
  near-degenerate slivers common in 3-D don't dominate the scale. Also exposed as
  select_alpha(points, method="knee"); returns the diagnostic curve.

Dashboard: add a "Knee (auto)" alpha mode (default) that plots the circumradius curve with the
  chosen cutoff marked, so the selection is visible. Drop the clustering tab (the clustering API
  stays in the package).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Add quantile-based select_alpha and dashboard mode
  ([`ab4a08c`](https://github.com/arunoruto/ashp/commit/ab4a08c23d92d02c43ebd75c2026ab1032468ae2))

optimizealpha returns the largest alpha keeping one polygon that covers every point — a bottleneck
  statistic that is outlier-sensitive and swings with point density. Add select_alpha(points, q):
  alpha from a quantile of the Delaunay circumradii, which is fast (one triangulation, no
  bisection), outlier-robust, and produces a shape that stays stable as the point count changes (q
  is a rank statistic; q=1 is the convex hull). Works in 2-D and 3-D.

Wire it into the dashboard as the default "Quantile (robust)" alpha mode alongside Manual and
  Optimize, with tests.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Fork alphashape into src/ashp with accelerated core
  ([`c421a6d`](https://github.com/arunoruto/ashp/commit/c421a6db5c36e96631a13f6d65da2bb485489ab2))

Vendor the alphashape toolbox as a maintained fork under src/ashp, packaged with uv_build and modern
  uv tooling. Functional changes over upstream:

- numba-JIT the per-simplex circumradius (cached, parallel above a threshold), with a transparent
  numpy fallback when numba is absent; drop the deprecated np.matrix circumcenter construction. -
  Vectorize boundary-facet extraction for all dimensions (numpy set logic in place of the
  per-simplex Python loop) and bulk-build 2-D geometry via shapely's array API. Output is
  byte-identical; 2-D ~14x faster end to end. - Fix optimizealpha: the absolute convergence
  tolerance was finer than float resolution for alpha > ~1, so the bisection ran to max_iterations
  and returned 0. Use a relative tolerance (new rel_tol parameter). - Drop the unused packaging
  dependency and its obsolete shapely<2.0 check.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- **dashboard**: Add 3-D datasets and surface-mesh visualization
  ([`4b9c31d`](https://github.com/arunoruto/ashp/commit/4b9c31d82a146a97d147c1213462f3945f31d85c))

- Add ball/torus/blobs 3-D datasets and a Plotly Mesh3d figure to the dashboard, dispatching 2-D vs
  3-D by input dimensionality (optimize stays 2-D only). - Guard the 3-D alphashape path:
  fix_normals fails on an empty face set, which occurs when no simplex passes the radius filter
  (alpha too high). Now returns an empty mesh instead of raising, with a regression test.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- **dashboard**: Add clustering tab
  ([`c49d8c7`](https://github.com/arunoruto/ashp/commit/c49d8c72c38c6ee38bf0ae16e97a56b5848c8fd3))

A second tab clusters the points via the alpha filtration and shows the persistence curve (cluster
  count vs alpha, with the auto-selected scale marked) beside the points coloured by cluster. Works
  for the 2-D and 3-D datasets; a min-cluster-size control sets the noise threshold.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- **dashboard**: Add homogenisation-rate metric and selector overlay
  ([`47cacb1`](https://github.com/arunoruto/ashp/commit/47cacb159ea98c9f7be8d8c7fe711b0d7b662cfd))

The integral metrics (CV, components) show the transition only as a gradual elbow. Add -d(CV)/d(log
  alpha): its peak sharply marks the blob->structure transition and is markedly more stable across
  point count than the circumradius knee (e.g. spiral: ~4.1-4.2 at n=400/800). Overlay the alpha in
  use plus the knee and persistence selectors as vertical lines across all three panels so they can
  be compared against the curves.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- **dashboard**: Add streamlit dashboard and docs gallery generator
  ([`8fca14a`](https://github.com/arunoruto/ashp/commit/8fca14a638e6efe04c8e955804bda115f190215f))

Interactive Streamlit + Plotly dashboard (apps/dashboard) for exploring alpha shapes over several
  point distributions, behind an optional `dashboard` dependency group. Shared plotting helpers are
  reused by assets/generate_images.py, which renders the README gallery images into assets/img.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- **dashboard**: Shade a usable alpha band in the sweep panel
  ([`0a364f0`](https://github.com/arunoruto/ashp/commit/0a364f04afc7aaedcc382b73e5b244c009859d3c))

The homogenisation-rate peak alone only marks the lower bound (where the long bridges get cut),
  which wasn't actionable on its own. Pair it with the upper bound (where the component count starts
  climbing = fragmentation) to shade the usable alpha band, with its centre marked as a suggested
  pick. When the lower bound exceeds the upper one there is no clean structural scale (e.g. a
  uniform cloud) and no band is drawn — which is itself the useful signal.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

### Testing

- Add pytest suite
  ([`f999bba`](https://github.com/arunoruto/ashp/commit/f999bba6567d95bd7fb6ff08b8e4b9c99cd88200))

Port the upstream unittest cases to pytest and add 2-D regression tests covering the vectorized
  boundary reconstruction (concave polygon validity and the tiny-alpha == convex-hull property).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>


## v1.3.1 (2026-06-24)
