"""--permutation_seed: reproducible, keyed by barcode, and the same permutation
the scanpy knn module applies for the same seed."""

import hashlib
import sys

import numpy as np

from writers import read_embeddings, read_graph


def _run(monkeypatch, tmp_path, pcas_tsv, perm):
    out = tmp_path / f"perm{perm}"
    monkeypatch.setattr(sys, "argv", [
        "knn.py", "--embedding_tsv", str(pcas_tsv), "--n_neighbors", "15",
        "--flavor", "rapids", "--random_seed", "42", "--permutation_seed", str(perm),
        "--output_dir", str(out), "--name", "t",
    ])
    from knn import main
    main()
    return out / "t_neighbors.h5"


def _by_barcode(m, ids):
    """CSR as {(barcode, barcode): weight}, comparable across permutations."""
    m = m.tocoo()
    return {(ids[r], ids[c]): w for r, c, w in zip(m.row, m.col, m.data)}


def test_same_seed_is_byte_identical(monkeypatch, tmp_path, pcas_tsv):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    a = sha(_run(monkeypatch, tmp_path / "a", pcas_tsv, 3))
    b = sha(_run(monkeypatch, tmp_path / "b", pcas_tsv, 3))
    assert a == b


def test_seed_0_is_identity(monkeypatch, tmp_path, pcas_tsv):
    g = read_graph(_run(monkeypatch, tmp_path, pcas_tsv, 0))
    assert g.row_ids == read_embeddings(pcas_tsv).row_ids


def test_on_disk_order_is_numpy_default_rng(monkeypatch, tmp_path, pcas_tsv):
    """Same RNG as the scanpy module, so seed N pairs the two arms."""
    src = read_embeddings(pcas_tsv).row_ids
    g = read_graph(_run(monkeypatch, tmp_path, pcas_tsv, 11))
    assert g.row_ids == [src[i] for i in np.random.default_rng(11).permutation(len(src))]
    assert g.row_ids != src


def test_brute_graph_is_order_invariant(monkeypatch, tmp_path, pcas_tsv):
    """The default backend is exact, so permuting must leave the graph unchanged
    by barcode, weights included. If this fails, the GPU search itself depends on
    row order (ties or reduction order), and permutation_seed perturbs the graph
    as well as the clusterer's walk."""
    base = read_graph(_run(monkeypatch, tmp_path / "a", pcas_tsv, 0))
    perm = read_graph(_run(monkeypatch, tmp_path / "b", pcas_tsv, 3))
    assert set(perm.row_ids) == set(base.row_ids)
    for attr in ("distances", "connectivities"):
        assert _by_barcode(getattr(perm, attr), perm.row_ids) == \
            _by_barcode(getattr(base, attr), base.row_ids), attr
