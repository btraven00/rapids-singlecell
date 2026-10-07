"""Fused runner on the GPU: the boundary crossings show up as their own phases, and
fused output matches split output. GPU PCA is not bit-reproducible run to run: the
reductions run in a varying order (measured: 1.2e-5 relative in float32, 2.9e-9 in
float64). So embeddings are compared to sqrt(eps) of the compute dtype, relative to
the largest magnitude and ignoring per-component sign, and clusters by ARI. The scanpy
module compares byte for byte instead."""

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

import fuse
from artifacts import EmbeddingIO, LabelsIO

P = {"n_components": "10", "dtype": "float64", "n_neighbors": "10", "knn_algorithm": "brute", "resolution": "1.0", "random_seed": "0"}


def _args(stages, **files):
    flat = [x for s in stages for k in fuse.STEPS[s].params for x in (f"--{s.lower()}_{k}", P[k])]
    return ["--steps", ",".join(stages)] + [x for k, v in files.items() for x in (f"--{k}", str(v))] + flat


def assert_close_pcs(a, b, dtype):
    """Equal to half the significant digits of `dtype` (sqrt(eps)), up to per-PC sign."""
    tol = np.sqrt(np.finfo(dtype).eps) * np.abs(b).max()
    np.testing.assert_allclose(np.abs(a), np.abs(b), rtol=0, atol=tol)


def _phases(d):
    return [json.loads(l)["event"] for l in open(d / "obkit-events.jsonl") if '"end"' in l]


@pytest.fixture
def h5ad(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.random(400, 60, density=0.2, format="csr", random_state=rng, dtype=np.float32)
    X[:, 0] = 0  # an all-zero gene: rsc's sparse PCA refuses these unless masked
    a = ad.AnnData(X=sp.csr_matrix(X))
    a.obs_names = [f"c{i}" for i in range(400)]
    a.write_h5ad(tmp_path / "in.h5ad")
    return tmp_path / "in.h5ad"


def test_fused_crosses_once_and_matches_split(tmp_path, h5ad):
    f, s = tmp_path / "fused", tmp_path / "split"
    fuse.main(["--output_dir", str(f), "--name", "x"] + _args(["PCA", "NNG", "CLUST"], data_h5ad=h5ad))
    assert _phases(f) == ["init", "load", "h2d:data_h5ad", "pca", "nng", "clust", "write"]

    fuse.main(["--output_dir", str(s / "1"), "--name", "x"] + _args(["PCA"], data_h5ad=h5ad))
    fuse.main(["--output_dir", str(s / "2"), "--name", "x"] + _args(["NNG"], embedding_tsv=s / "1" / "x_embedding.tsv"))
    fuse.main(["--output_dir", str(s / "3"), "--name", "x"] + _args(["CLUST"], neighbors_h5=s / "2" / "x_neighbors.h5"))

    ef, es = EmbeddingIO.load(f / "x_embedding.tsv"), EmbeddingIO.load(s / "1" / "x_embedding.tsv")
    assert ef.row_ids == es.row_ids
    assert_close_pcs(ef.matrix, es.matrix, P["dtype"])
    lf, ls = LabelsIO.load(f / "x_clusters.tsv"), LabelsIO.load(s / "3" / "x_clusters.tsv")
    assert adjusted_rand_score(lf.values, ls.values) > 0.95


def test_repeat_within_eps(tmp_path, h5ad):
    runs = []
    for i in range(2):
        fuse.main(["--output_dir", str(tmp_path / str(i)), "--name", "x"] + _args(["PCA"], data_h5ad=h5ad))
        runs.append(EmbeddingIO.load(tmp_path / str(i) / "x_embedding.tsv").matrix)
    assert_close_pcs(runs[0], runs[1], P["dtype"])


def test_chain_typecheck():
    assert set(fuse.plan(["PCA", "NNG", "CLUST"])) == {"data_h5ad"}
    assert set(fuse.plan(["NNG"])) == {"embedding_tsv"}
