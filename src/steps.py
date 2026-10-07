"""Typed, pure GPU steps (rapids-singlecell public API) for the fused runner (prototype).

Each step's contract is its signature run(ins: <In>, p: <Params>) -> <Out>, and the
types say *where* the data lives. The runner inserts and times every host<->device
transfer a chain needs (src/artifacts.py:TRANSFER) as an obkit phase h2d:<id> / d2h:<id>.

What the public API allows (rapids-singlecell 0.16, measured): only X stays on the
device. pp.pca stores X_pca as numpy, pp.neighbors stores scipy graphs, and tl.leiden
uploads the graph again. The step types say so: every output is a host type. So in a
fused chain the runner's only crossing is h2d:data_h5ad. The small results (embedding,
graph) cross *inside* the rsc calls and are billed to the step phases, where the
runner can't separate them. Residency therefore covers the big object, the matrix,
and nothing downstream of PCA.
"""

from dataclasses import dataclass
from typing import Any, Callable, TypedDict, get_type_hints

import anndata as ad
import cupy as cp
import numpy as np
import rapids_singlecell as rsc

from artifacts import IO, TRANSFER, DevMatrix, Embedding, Labels, NeighborGraph, sync  # noqa: F401


class PCAIn(TypedDict):
    data_h5ad: DevMatrix

class PCAOut(TypedDict):
    embedding_tsv: Embedding          # host: rsc.pp.pca downloads X_pca itself

class PCAParams(TypedDict):
    n_components: int
    dtype: str          # compute precision: X is cast before PCA (rsc's own dtype= only casts the result)
    random_seed: int

def pca_step(ins: PCAIn, p: PCAParams) -> PCAOut:
    src = ins["data_h5ad"].adata
    X = src.X if src.X.dtype == p["dtype"] else src.X.astype(p["dtype"])  # device-side cast (a copy)
    a = ad.AnnData(X=X, obs=src.obs[[]], var=src.var[[]])  # results land on `a`, not the caller's object
    # rsc's sparse PCA refuses all-zero genes, which small subsamples of the HVG set have
    # (9 of 2000 at 10k). They have zero variance, so masking them leaves the centred PCs
    # unchanged. Counted on the device, inside the timed phase.
    a.var["expressed"] = cp.asnumpy(cp.bincount(a.X.tocsr().indices, minlength=a.n_vars) > 0)
    # centred, no variance scaling (protocol); sparse input stays sparse
    rsc.pp.pca(a, n_comps=p["n_components"], zero_center=True, random_state=p["random_seed"],
               mask_var="expressed")
    emb = np.asarray(a.obsm["X_pca"])
    return {"embedding_tsv": Embedding(emb, list(a.obs_names), [f"PC{i + 1}" for i in range(emb.shape[1])])}


class NNGIn(TypedDict):
    embedding_tsv: Embedding          # host: rsc.pp.neighbors uploads it itself

class NNGOut(TypedDict):
    neighbors_h5: NeighborGraph       # host: rsc.pp.neighbors downloads the graphs

class NNGParams(TypedDict):
    n_neighbors: int
    knn_algorithm: str  # brute = exact (cuVS brute force); cagra / ivfflat = approximate (tier 3a)
    random_seed: int

def nng_step(ins: NNGIn, p: NNGParams) -> NNGOut:
    e = ins["embedding_tsv"]
    a = ad.AnnData(X=np.zeros((len(e.row_ids), 1), dtype=np.float32))
    a.obs_names = e.row_ids
    a.obsm["X_pca"] = e.matrix
    rsc.pp.neighbors(a, n_neighbors=p["n_neighbors"], use_rep="X_pca",
                     algorithm=p["knn_algorithm"], random_state=p["random_seed"])
    return {"neighbors_h5": NeighborGraph(a.obsp["distances"], a.obsp["connectivities"], list(a.obs_names))}


class CLUSTIn(TypedDict):
    neighbors_h5: NeighborGraph       # host: rsc.tl.leiden uploads it itself

class CLUSTOut(TypedDict):
    clusters_tsv: Labels

class CLUSTParams(TypedDict):
    resolution: float
    random_seed: int

def clust_step(ins: CLUSTIn, p: CLUSTParams) -> CLUSTOut:
    g = ins["neighbors_h5"]
    a = ad.AnnData(X=np.zeros((len(g.row_ids), 1), dtype=np.float32))
    a.obs_names = g.row_ids
    a.obsp["connectivities"], a.obsp["distances"] = g.connectivities, g.distances
    a.uns["neighbors"] = {"distances_key": "distances", "connectivities_key": "connectivities"}
    rsc.tl.leiden(a, resolution=p["resolution"], random_state=p["random_seed"])
    return {"clusters_tsv": Labels(a.obs["leiden"].to_numpy(dtype=object), list(a.obs_names))}


@dataclass
class Step:
    stage: str
    run: Callable[..., Any]

    def _hints(self, arg):
        return get_type_hints(get_type_hints(self.run)[arg])

    @property
    def inputs(self):   # input id -> type the step wants (host or device)
        return self._hints("ins")

    @property
    def outputs(self):  # output id -> type the step returns
        return self._hints("return")

    @property
    def params(self):
        return self._hints("p")


STEPS = {s.stage: s for s in [Step("PCA", pca_step), Step("NNG", nng_step), Step("CLUST", clust_step)]}
