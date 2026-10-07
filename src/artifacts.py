"""Artifact types for the fused runner: host types (this module's writers), device twins, transfers.

Host types and their files are src/writers.py. They write the omni-scrna stage
contract, byte-compatible with the scanpy module (same TSV writer, same flat-root
graph layout). Device types hold cupy / cupyx arrays. TRANSFER maps
(from_type, to_type) to the function that moves a value across the PCIe boundary.
The runner calls these transfers, never the steps, and times each one as its own phase.
"""

from dataclasses import dataclass, field

import anndata as ad
import cupy as cp
import cupyx.scipy.sparse as cpsp
import rapids_singlecell as rsc
import scanpy as sc

from writers import (Embedding, Labels, NeighborGraph, read_embeddings, read_graph,
                     read_labels, write_embeddings, write_graph, write_labels)


# --- host: load/save per type -------------------------------------------------

class MatrixIO:
    @staticmethod
    def load(path):
        return sc.read_h5ad(path)


class EmbeddingIO:
    suffix = "_embedding.tsv"
    load, save = staticmethod(read_embeddings), staticmethod(write_embeddings)


class GraphIO:
    suffix = "_neighbors.h5"
    load, save = staticmethod(read_graph), staticmethod(write_graph)


class LabelsIO:
    suffix = "_clusters.tsv"
    load, save = staticmethod(read_labels), staticmethod(write_labels)


IO = {ad.AnnData: MatrixIO, Embedding: EmbeddingIO, NeighborGraph: GraphIO, Labels: LabelsIO}


# --- device ------------------------------------------------------------------

@dataclass
class DevMatrix:
    adata: ad.AnnData  # X is a cupyx sparse matrix


@dataclass
class DevEmbedding:
    matrix: cp.ndarray
    row_ids: list
    col_names: list = field(default_factory=list)


@dataclass
class DevGraph:
    distances: cpsp.csr_matrix
    connectivities: cpsp.csr_matrix
    row_ids: list = field(default_factory=list)


def h2d(x):
    if isinstance(x, ad.AnnData):
        # New object sharing x.X: the caller's AnnData is left as it was. Whether the host
        # matrix then survives is the runner's call (fuse._as(move=True) drops it).
        a = ad.AnnData(X=x.X, obs=x.obs[[]], var=x.var[[]])
        rsc.get.anndata_to_GPU(a)
        return DevMatrix(a)
    if isinstance(x, Embedding):
        return DevEmbedding(cp.asarray(x.matrix), x.row_ids, x.col_names)
    if isinstance(x, NeighborGraph):
        return DevGraph(cpsp.csr_matrix(x.distances), cpsp.csr_matrix(x.connectivities), x.row_ids)
    raise TypeError(type(x))


def d2h(x):
    if isinstance(x, DevEmbedding):
        return Embedding(cp.asnumpy(x.matrix), x.row_ids, x.col_names)
    if isinstance(x, DevGraph):
        return NeighborGraph(x.distances.get(), x.connectivities.get(), x.row_ids)
    raise TypeError(type(x))


TRANSFER = {
    (ad.AnnData, DevMatrix): h2d,
    (Embedding, DevEmbedding): h2d,
    (NeighborGraph, DevGraph): h2d,
    (DevEmbedding, Embedding): d2h,
    (DevGraph, NeighborGraph): d2h,
}


def sync():
    """Block until the GPU is idle. Every phase ends with this; without it, async kernel
    launches would bill compute time to whichever later phase waits on the device."""
    cp.cuda.Device().synchronize()
