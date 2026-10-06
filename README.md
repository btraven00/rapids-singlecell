# rapids-singlecell

A omnibenchmark module to use rapids-singlecell

## Tests

```
pixi run tests
```

Tests live in `tests/`. The GPU is initialized once per session; individual tests do not reinitialize it.

### Fixtures (`tests/data/`)

| File | Description |
|------|-------------|
| `datasets_normalized_selected.h5` | Normalized, feature-selected expression matrix (input to PCA) |
| `datasets_pcas.tsv` | PCA embedding produced from the above (input to kNN) |
| `datasets_knn.h5` | kNN graph produced from the above (input to clustering) |

The three files form a chain: normalized → PCA → kNN → cluster. Adding a new stage fixture means running the previous stage's entrypoint and committing the output to `tests/data/`.

## Solver controls and how they compare across tools

The scanpy, rapids-singlecell and seurat modules expose the same knobs, with
defaults equal to each module's historical behaviour (so existing results are
unchanged). Checked 2026-10-05 on tm-facs.

**Leiden iterations (`--n_iterations`, `cluster` entrypoint)**

| module | backend | default | meaning | until convergence |
|---|---|---|---|---|
| scanpy | igraph (`flavor igraph`) / leidenalg | 2 | iterations of the Leiden algorithm | **yes**: any negative value runs until an iteration no longer improves quality |
| rapids-singlecell | cuGraph `leiden(max_iter=)` | 100 | a *cap* on aggregation **levels**; stops early at convergence | effectively: a large cap (the default 100). **Never set it low**: at 2, tm-facs gave ~3,200 clusters at every resolution 0.05-2.65 (levels truncated before communities merge) |
| seurat | leidenbase `num_iter` (`FindClusters n.iter`) | 10 | runs *exactly* this many iterations, no early stop | **no**: values < 1 are rejected; use a large value (cost grows linearly: 2 = 4.3 s, 10 = 11.9 s, 50 = 68.5 s per resolution on tm-facs) |

The units are not identical (igraph iterations vs cuGraph levels vs leidenbase
iterations). For comparisons across tools set the same small value in all
three (2 is the current choice): on tm-facs, Seurat modularity is 0.96261 at 2
vs 0.96293 at 10 vs 0.96292 at 50, i.e. 2 is near-converged.

**Randomized / iterative PCA (`pca` entrypoint)**

| module | solver | knobs | default |
|---|---|---|---|
| scanpy | `randomized` with `--dense true` (sklearn) | `--n_iter`, `--n_oversamples` | `auto` (= 7 power iterations here), 10 |
| rapids-singlecell | `randomized-halko` | `--n_iter`, `--n_oversamples` | 7, 10 (matches scanpy) |
| seurat | `approximate` (irlba, Krylov) | `--irlba_work`, `--irlba_maxit`, `--irlba_tol` | nv + 7, 1000, 1e-5 (runs to tolerance) |

scanpy and rapids are matched (7 / 10). irlba is a different algorithm run to
convergence; its knobs do not map onto power iterations / oversamples. Note:
rapids' halko PCA is not bit-reproducible between identical runs (GPU; max
|diff| ~2e-3 on tm-facs, the size of scanpy's whole seed effect).

