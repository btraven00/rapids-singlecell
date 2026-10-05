import sys

from writers import read_embeddings, read_graph


def _run(monkeypatch, tmp_path, pcas_tsv, n_neighbors=15, seed=42):
    monkeypatch.setattr(sys, "argv", [
        "knn.py",
        "--embedding_tsv", str(pcas_tsv),
        "--n_neighbors", str(n_neighbors),
        "--flavor", "rapids",
        "--random_seed", str(seed),
        "--output_dir", str(tmp_path),
        "--name", "test",
    ])
    from knn import main
    main()
    return tmp_path / "test_neighbors.h5"


def test_knn_graph_shape(monkeypatch, tmp_path, pcas_tsv):
    n_neighbors = 15
    out = _run(monkeypatch, tmp_path, pcas_tsv, n_neighbors=n_neighbors)
    assert out.exists()
    graph = read_graph(out)
    emb = read_embeddings(pcas_tsv)
    n_cells = len(emb.row_ids)
    assert graph.distances.shape == (n_cells, n_cells)
    assert graph.connectivities.shape == (n_cells, n_cells)


def test_knn_distances_nnz(monkeypatch, tmp_path, pcas_tsv):
    n_neighbors = 15
    out = _run(monkeypatch, tmp_path, pcas_tsv, n_neighbors=n_neighbors)
    graph = read_graph(out)
    n_cells = graph.distances.shape[0]
    assert graph.distances.nnz == n_neighbors * n_cells


def test_knn_row_ids_match_embedding(monkeypatch, tmp_path, pcas_tsv):
    out = _run(monkeypatch, tmp_path, pcas_tsv)
    graph = read_graph(out)
    emb = read_embeddings(pcas_tsv)
    assert graph.row_ids == emb.row_ids


def test_cagra_flavor_runs_with_k_neighbours(monkeypatch, tmp_path, pcas_tsv):
    # No determinism assertion: CAGRA's index build is nondeterministic (see knn.py).
    monkeypatch.setattr(sys, "argv", [
        "knn.py", "--embedding_tsv", str(pcas_tsv), "--n_neighbors", "15",
        "--flavor", "rapids-cagra", "--random_seed", "42",
        "--output_dir", str(tmp_path), "--name", "test",
    ])
    from knn import main
    main()
    graph = read_graph(tmp_path / "test_neighbors.h5")
    assert graph.distances.nnz == 15 * graph.distances.shape[0]
