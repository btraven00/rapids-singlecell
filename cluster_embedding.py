#!/usr/bin/env python3
"""Embedding-based clustering module (rapids-singlecell-backed) for omnibenchmark.

Input
-----
File: ``--embedding_tsv`` from any producer of the shared embedding_tsv output
id (cell-id-indexed embedding coordinates).

Output
------
File: {output_dir}/{name}_clusters.tsv

  Header row:    cluster
  Each data row: cell_barcode <TAB> cluster_label

Implementation notes
--------------------
- ``rapids-kmeans`` uses ``rsc.tl.kmeans`` on obsm["X_pca"]. Requires
  ``--n_clusters``. Labels are integers 0..k-1, deterministic given
  ``--random_seed``.
- ``rapids-hdbscan`` uses ``cuml.cluster.HDBSCAN`` directly on obsm["X_pca"]
  (rsc does not expose HDBSCAN as a tl function). Requires ``--min_samples``
  and ``--min_cluster_size``. Noise points receive label -1 and are preserved
  as-is in the output; downstream metric stages must handle them.
  HDBSCAN has no seed parameter; passing ``--random_seed`` is rejected to
  avoid the false impression that the run is seed-controlled.
- ``rapids-dbscan`` uses ``cuml.cluster.DBSCAN`` directly on obsm["X_pca"].
  Requires ``--eps`` (neighborhood radius) and ``--min_samples`` (min points
  per core neighborhood). Noise points receive label -1, same convention as
  HDBSCAN. DBSCAN has no seed parameter; passing ``--random_seed`` is
  rejected.
"""

import argparse
import sys
from pathlib import Path

import cuml.cluster
import cupy as cp
import numpy as np
import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local helpers
from common import cli  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import embedding_to_adata  # noqa: E402
from options import ClusterEmbeddingOptions, build_cluster_embedding_opts  # noqa: E402
from phases import phase  # noqa: E402
from writers import Labels, read_embeddings, write_labels  # noqa: E402


def parse_args():
    # No plan stage covers embedding-based clustering yet, but its input is the
    # same embedding_tsv the NNG stage consumes, so borrow that arg contract.
    p = argparse.ArgumentParser(description="OmniBenchmark cluster-embedding module (rapids-singlecell)")
    cli.add_base_args(p)            # --output_dir, --name
    cli.add_stage_args(p, "NNG")    # --embedding_tsv
    p.add_argument("--method", type=str, required=True,
                   choices=["rapids-kmeans", "rapids-hdbscan", "rapids-dbscan"],
                   help="Clustering method token (see module docstring)")
    p.add_argument("--n_clusters", type=int, default=None,
                   help="Number of clusters; required for rapids-kmeans")
    p.add_argument("--min_samples", type=int, default=None,
                   help="Min samples per core point; required for rapids-hdbscan and rapids-dbscan")
    p.add_argument("--min_cluster_size", type=int, default=None,
                   help="Min cluster size; required for rapids-hdbscan")
    # TODO: eps is an absolute distance in PCA space, so a value is only
    # meaningful for the dataset and n_components it was chosen against --
    # too small and every cell is noise, too large and everything is one
    # cluster. It should be DERIVED per run (k-distance knee at min_samples,
    # or a quantile of the kNN distances) rather than hard-coded in a plan.
    p.add_argument("--eps", type=float, default=None,
                   help="Neighborhood radius; required for rapids-dbscan. "
                        "Dataset-dependent: see the TODO above")
    p.add_argument("--random_seed", type=int, default=None,
                   help="Random seed (required for rapids-kmeans; rejected for "
                        "rapids-hdbscan and rapids-dbscan)")
    return p.parse_args()


def run_cluster(adata, opts: ClusterEmbeddingOptions):
    """GPU-only clustering on obsm["X_pca"]. Mutates adata.obs["cluster"] in place."""
    if opts.method == "rapids-kmeans":
        rsc.tl.kmeans(
            adata,
            n_clusters=opts.n_clusters,
            use_rep="X_pca",
            n_pcs=adata.obsm["X_pca"].shape[1],
            random_state=opts.random_seed,
            key_added="cluster",
        )
    elif opts.method == "rapids-hdbscan":
        model = cuml.cluster.HDBSCAN(
            min_samples=opts.min_samples,
            min_cluster_size=opts.min_cluster_size,
        )
        labels = model.fit_predict(adata.obsm["X_pca"])
        adata.obs["cluster"] = cp.asnumpy(labels).astype(str)
    elif opts.method == "rapids-dbscan":
        model = cuml.cluster.DBSCAN(
            eps=opts.eps,
            min_samples=opts.min_samples,
        )
        labels = model.fit_predict(adata.obsm["X_pca"])
        adata.obs["cluster"] = cp.asnumpy(labels).astype(str)
    else:
        raise ValueError(f"unknown method: {opts.method!r}")


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "embedding_tsv", "method",
              "n_clusters", "min_samples", "min_cluster_size", "eps", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    opts = build_cluster_embedding_opts(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output_dir))

    setup_gpu()

    with phase("load") as attrs:
        embedding = read_embeddings(args.embedding_tsv)
        adata = embedding_to_adata(embedding)
        attrs["n_cells"], attrs["n_dims"] = adata.obsm["X_pca"].shape
        print(f"  embedding: {adata.n_obs} cells x {adata.obsm['X_pca'].shape[1]} dims")

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)

    with phase("compute") as attrs:
        run_cluster(adata, opts)
        attrs["method"] = opts.method

    with phase("gpu_download"):
        rsc.get.anndata_to_CPU(adata, convert_all=True)

    with phase("write") as attrs:
        labels_arr = adata.obs["cluster"].astype(str).to_numpy()
        out = Path(args.output_dir) / f"{args.name}_clusters.tsv"
        write_labels(
            Labels(values=labels_arr, row_ids=list(embedding.row_ids)),
            out,
        )
        n_noise = int(np.sum(labels_arr == "-1"))
        n_clusters = len(set(labels_arr) - {"-1"})
        attrs["n_clusters"] = n_clusters
        attrs["n_noise"] = n_noise
        attrs["path"] = str(out)
        print(f"  n_clusters: {n_clusters}")
        if n_noise:
            print(f"  n_noise:    {n_noise}")
        print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
