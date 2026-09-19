import sys

import pytest

from writers import read_graph, read_labels


def _run(monkeypatch, tmp_path, neighbors_h5, method, resolution=0.5, seed=42):
    argv = [
        "cluster_graph.py",
        "--neighbors_h5", str(neighbors_h5),
        "--method", method,
        "--resolution", str(resolution),
        "--output_dir", str(tmp_path),
        "--name", "test",
    ]
    if method == "rapids-leiden":
        argv += ["--random_seed", str(seed)]
    monkeypatch.setattr(sys, "argv", argv)
    from cluster_graph import main
    main()
    return tmp_path / "test_clusters.tsv"


def _check(out, neighbors_h5):
    assert out.exists()
    labels = read_labels(out)
    graph = read_graph(neighbors_h5)
    assert labels.col_name == "cluster"
    assert labels.row_ids == graph.row_ids
    assert len(set(labels.values)) > 1
    assert out.read_text().splitlines()[0].split("\t") == ["cell_id", "cluster"]


def test_leiden(monkeypatch, tmp_path, neighbors_h5):
    out = _run(monkeypatch, tmp_path, neighbors_h5, "rapids-leiden")
    _check(out, neighbors_h5)


def test_louvain(monkeypatch, tmp_path, neighbors_h5):
    out = _run(monkeypatch, tmp_path, neighbors_h5, "rapids-louvain")
    _check(out, neighbors_h5)


def test_louvain_rejects_seed(monkeypatch, tmp_path, neighbors_h5):
    monkeypatch.setattr(sys, "argv", [
        "cluster_graph.py",
        "--neighbors_h5", str(neighbors_h5),
        "--method", "rapids-louvain",
        "--resolution", "0.5",
        "--random_seed", "42",
        "--output_dir", str(tmp_path),
        "--name", "test",
    ])
    from cluster_graph import main
    with pytest.raises(ValueError, match="--random_seed is not accepted"):
        main()


def test_leiden_resolution_affects_n_clusters(monkeypatch, tmp_path, neighbors_h5):
    out_lo = _run(monkeypatch, tmp_path / "lo", neighbors_h5, "rapids-leiden", resolution=0.2)
    out_hi = _run(monkeypatch, tmp_path / "hi", neighbors_h5, "rapids-leiden", resolution=1.5)
    n_lo = len(set(read_labels(out_lo).values))
    n_hi = len(set(read_labels(out_hi).values))
    assert n_hi > n_lo
