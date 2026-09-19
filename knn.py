#!/usr/bin/env python3
"""kNN graph module (rapids-singlecell-backed) for omnibenchmark.

Input
-----
File: ``--embedding_tsv`` from any producer of the shared embedding_tsv output id.
Header = cell_id + dimension names (PC* or dim_*), each data row prefixed by cell barcode.

Output
------
File: {output_dir}/{name}_neighbors.h5  (NNG stage output: neighbors_h5)

  /distances/{data,indices,indptr,shape}        CSR sparse, n_cells x n_cells
  /connectivities/{data,indices,indptr,shape}   CSR sparse, n_cells x n_cells
  /cell_ids                                     1D string array of length n_cells

cell_ids is preserved on the output so downstream stages can join on
identity, not on row order. Both sparse matrices share that ordering.

Implementation notes
--------------------
- ``--flavor`` selects the ANN backend / precision. ``rapids`` leaves
  rsc.pp.neighbors at its cuVS default (currently CAGRA, FP32). Add new
  tokens (rapids-ivf-flat, rapids-brute, ...) to the ``--flavor`` choices
  to expose other backends.
- The synthetic ``X = zeros((n_cells, 1))`` is just a stand-in to give
  AnnData a well-formed obs axis; the actual neighbors computation runs
  on ``obsm["X_pca"]`` (use_rep="X_pca").
- ``random_seed`` is best-effort: only some ANN backends consult it
  (IVF training, for instance). For a fully deterministic graph, prefer
  brute-force once that token is added.
"""

import argparse
import sys
from pathlib import Path

import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local helpers
from common import cli  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import embedding_to_adata  # noqa: E402
from phases import phase  # noqa: E402
from writers import NeighborGraph, read_embeddings, write_graph  # noqa: E402


def parse_args():
    # common/cli injects the synced contract; method params are hand-rolled.
    p = argparse.ArgumentParser(description="OmniBenchmark kNN module (rapids-singlecell)")
    cli.add_base_args(p)            # --output_dir, --name
    cli.add_stage_args(p, "NNG")    # --embedding_tsv
    p.add_argument("--n_neighbors", type=int, required=True,
                   help="Number of nearest neighbors")
    p.add_argument("--flavor", type=str, required=True,
                   choices=["rapids"],
                   help="kNN flavor token (see module docstring)")
    p.add_argument("--random_seed", type=int, required=True, help="Random seed")
    return p.parse_args()


def run_knn(adata, args):
    """GPU-only neighbors. Pre/post: adata stays on GPU. Mutates in place."""
    rsc.pp.neighbors(
        adata,
        n_neighbors=args.n_neighbors,
        use_rep="X_pca",
        random_state=args.random_seed,
    )


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "embedding_tsv", "n_neighbors", "flavor", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output_dir))

    setup_gpu()

    with phase("load") as attrs:
        emb = read_embeddings(args.embedding_tsv)
        adata = embedding_to_adata(emb)
        attrs["n_cells"], attrs["n_components"] = emb.matrix.shape
        print(f"  embedding: {emb.matrix.shape}")

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)

    with phase("compute") as attrs:
        run_knn(adata, args)
        attrs["n_neighbors"] = args.n_neighbors

    with phase("gpu_download"):
        rsc.get.anndata_to_CPU(adata, convert_all=True)

    with phase("write") as attrs:
        out = Path(args.output_dir) / f"{args.name}_neighbors.h5"
        graph = NeighborGraph(
            distances=adata.obsp["distances"],
            connectivities=adata.obsp["connectivities"],
            row_ids=list(emb.row_ids),
        )
        write_graph(graph, out)
        attrs["distances_nnz"] = int(graph.distances.nnz)
        attrs["connectivities_nnz"] = int(graph.connectivities.nnz)
        attrs["path"] = str(out)
        print(f"  distances nnz:      {attrs['distances_nnz']}")
        print(f"  connectivities nnz: {attrs['connectivities_nnz']}")
        print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
