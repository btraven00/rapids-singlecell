"""Reusable writers for objects produced by this module, conforming to the omni-scrna benchmark spec."""

from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

TSV = "tsv"
H5  = "h5"


@dataclass
class Embedding:
    matrix: np.ndarray        # shape (n_cells, n_dims)
    row_ids: list             # cell barcodes, length n_cells
    col_names: list = field(default_factory=list)  # dim labels; auto-generated if empty


def _col_names(embedding):
    if embedding.col_names:
        return embedding.col_names
    return [f"dim_{i + 1}" for i in range(embedding.matrix.shape[1])]


def _write_tsv(path, obj, row_label):
    # Same code as the scanpy module's _write_tsv, deliberately: pandas'
    # to_csv writes small magnitudes as 3.3e-05 where polars writes
    # 0.000033, so two modules on different writers emit the same numbers as
    # different text. Identical writers keep the TSVs diffable across arms.
    # It is also ~15x faster (0.99s -> 0.06s for 19696 x 50), which is why
    # this module's `write` phase used to dwarf the CPU module's.
    #
    # Keep the first column named: left unnamed (pandas' default),
    # R's read.table(header=TRUE) calls that column "X".
    cols = _col_names(obj)
    df = pl.from_numpy(obj.matrix, schema=cols).insert_column(
        0, pl.Series("", obj.row_ids))
    with open(path, "w") as f:
        f.write(row_label + "\t" + "\t".join(cols) + "\n")
        df.write_csv(f, separator="\t", include_header=False)


def write_embeddings(obj, path, format=TSV):
    if format != TSV:
        raise ValueError(f"unsupported format: {format!r}")
    _write_tsv(path, obj, "cell_id")


def write_loadings(obj, path, format=TSV):
    """Embedding layout, but rows are genes: first column is gene_id."""
    if format != TSV:
        raise ValueError(f"unsupported format: {format!r}")
    _write_tsv(path, obj, "gene_id")


def read_embeddings(path, format=TSV):
    """Inverse of write_embeddings. Round-trip-stable for the TSV format."""
    if format != TSV:
        raise ValueError(f"unsupported format: {format!r}")
    # float_precision="round_trip": the default parser is off by 1-2 ULP on
    # ~0.6% of values, and knn.py reads its embedding through here while the
    # scanpy module reads the same file with polars (which is exact). Two arms
    # disagreeing on the last bit of the input is precisely the size of effect
    # this benchmark measures.
    df = pd.read_csv(path, sep="\t", index_col=0, float_precision="round_trip")
    return Embedding(
        # to_numpy() on a single-dtype frame hands back the internal block
        # transposed, i.e. F-contiguous. cuml.cluster.HDBSCAN does not copy and
        # does not complain -- it just labels every point noise. Force C order
        # here so no GPU consumer downstream has to know that.
        matrix=np.ascontiguousarray(df.to_numpy(dtype=np.float64)),
        row_ids=list(df.index),
        col_names=list(df.columns),
    )


@dataclass
class NeighborGraph:
    distances: sp.csr_matrix       # n_cells x n_cells, sparse
    connectivities: sp.csr_matrix  # n_cells x n_cells, sparse
    row_ids: list                  # cell barcodes, length n_cells


def _write_h5_sparse(grp, m):
    m = m.tocsr()
    grp.create_dataset("data",    data=m.data)
    grp.create_dataset("indices", data=m.indices)
    grp.create_dataset("indptr",  data=m.indptr)


def _write_h5_graph(path, graph):
    # NNG spec layout, byte-for-byte the shape the scanpy module writes: the
    # distance CSR flat at the file root -- that is what the R metrics reader
    # consumes -- and only connectivities nested. No `shape`: both graphs are
    # square over the cell axis, so cell_ids is the authority.
    # dtype="S": h5py cannot write numpy unicode ('<U'); bytes give the
    # portable fixed-length HDF5 strings the R reader expects.
    with h5py.File(path, "w") as h5:
        h5.create_dataset("cell_ids", data=np.array(graph.row_ids, dtype="S"))
        _write_h5_sparse(h5, graph.distances)
        _write_h5_sparse(h5.create_group("connectivities"), graph.connectivities)


def write_graph(obj, path, format=H5):
    if format == H5:
        _write_h5_graph(path, obj)
    else:
        raise ValueError(f"unsupported format: {format!r}")


def _read_h5_sparse(grp, n_cells):
    # No `shape` in the spec: both graphs are square over the cell axis, so
    # cell_ids is the authority.
    return sp.csr_matrix(
        (grp["data"][:], grp["indices"][:], grp["indptr"][:]),
        shape=(n_cells, n_cells),
    )


def read_graph(path, format=H5):
    """Inverse of write_graph, and reader for the NNG stage output spec:
    the distance CSR flat at the file root, only connectivities nested.
    """
    if format != H5:
        raise ValueError(f"unsupported format: {format!r}")
    with h5py.File(path, "r") as h5:
        row_ids = list(h5["cell_ids"][:].astype(str))
        return NeighborGraph(
            distances=_read_h5_sparse(h5, len(row_ids)),
            connectivities=_read_h5_sparse(h5["connectivities"], len(row_ids)),
            row_ids=row_ids,
        )


@dataclass
class Labels:
    values: np.ndarray   # shape (n_cells,)
    row_ids: list        # length n_cells
    col_name: str = "cluster"


def write_labels(obj, path, format=TSV):
    if format != TSV:
        raise ValueError(f"unsupported format: {format!r}")
    pd.Series(obj.values, index=obj.row_ids, name=obj.col_name).rename_axis(
        "cell_id"
    ).to_csv(path, sep="\t")


def read_labels(path, format=TSV):
    """Inverse of write_labels. Round-trip-stable for the TSV format."""
    if format != TSV:
        raise ValueError(f"unsupported format: {format!r}")
    # dtype=str or pandas infers int64 for all-numeric labels (kmeans 0..k-1,
    # hdbscan's -1 noise marker) and the round trip stops being type-stable.
    s = pd.read_csv(path, sep="\t", index_col=0, header=0, dtype=str).squeeze()
    return Labels(
        values=s.to_numpy(dtype=object),
        row_ids=list(s.index),
        col_name=s.name,
    )
