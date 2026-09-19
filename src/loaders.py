"""Loaders for omnibenchmark rapids-singlecell modules.

The HDF5 schema is the omni-scrna intermediate layout (CSC, genes x cells):

    /matrix/data        (1D)  CSC values
    /matrix/indices     (1D)  CSC row indices (gene index per nonzero)
    /matrix/indptr      (1D)  CSC column pointers (one per cell + 1)
    /matrix/shape       (1D)  [n_genes, n_cells]
    /matrix/genes       (1D)  gene IDs (bytes)
    /matrix/barcodes    (1D)  cell barcodes (bytes)

Note this is *not* 10X-conformant (no /matrix/gene_names for v2, no
/matrix/features for v3), so scanpy.read_10x_h5 cannot ingest it. We read
the bare datasets directly. Transposed to cells x genes (CSR) on the way
out to match AnnData conventions.
"""

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp


def load_matrix(h5_path, dense=False):
    with h5py.File(h5_path, "r") as h5:
        g = h5["matrix"]
        data = g["data"][:]
        # Cast during the read. On disk these are uint32/int64 and scipy accepts
        # neither as an index dtype, so reading them raw costs a second full-size
        # buffer while scipy recasts (~150MB on full pbmc); h5py converts
        # in-flight instead. scipy picks ONE dtype for both arrays, so they must
        # agree -- and indptr tops out at nnz, so int32 is only safe under 2**31.
        idx_dtype = np.int32 if g["data"].shape[0] < 2**31 else np.int64
        indices = g["indices"].astype(idx_dtype)[:]
        indptr = g["indptr"].astype(idx_dtype)[:]
        shape = tuple(g["shape"][:])
        gene_ids = g["genes"][:].astype(str)
        cell_ids = g["barcodes"][:].astype(str)

    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()  # cells x genes
    if dense:
        # Drop the sparse copy before AnnData is built so peak reflects one
        # representation. Required for the dense-only solvers (full, jacobi).
        X = X.toarray()
    adata = ad.AnnData(X=X)
    adata.obs_names = cell_ids
    adata.var_names = gene_ids
    return adata


def embedding_to_adata(embedding):
    """Wrap an Embedding in a host AnnData with the matrix in obsm["X_pca"].

    ascontiguousarray is load-bearing: read_embeddings' matrix comes from pandas
    .to_numpy() and is F-contiguous, but the RAPIDS consumers read the raw buffer
    as C-ordered (cuML 26.06 HDBSCAN silently labels every cell noise). This is
    the one place every GPU consumer of an embedding routes through.
    """
    n = len(embedding.row_ids)
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obs_names = embedding.row_ids
    adata.obsm["X_pca"] = np.ascontiguousarray(embedding.matrix, dtype=np.float32)
    return adata


def graph_to_adata(graph):
    """Wrap a NeighborGraph in a host AnnData ready for upload to the GPU."""
    n = graph.connectivities.shape[0]
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obs_names = graph.row_ids
    adata.obsp["connectivities"] = graph.connectivities
    adata.obsp["distances"]      = graph.distances
    return adata
