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
