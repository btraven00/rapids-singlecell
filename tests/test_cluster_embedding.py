import sys

from writers import read_embeddings, read_labels


def _run(monkeypatch, tmp_path, pcas_tsv, method, extra_args=(), seed=42):
    argv = [
        "cluster_embedding.py",
        "--embedding_tsv", str(pcas_tsv),
        "--method", method,
        "--output_dir", str(tmp_path),
        "--name", "test",
        *extra_args,
    ]
    if method == "rapids-kmeans":
        argv += ["--random_seed", str(seed)]
    monkeypatch.setattr(sys, "argv", argv)
    from cluster_embedding import main
    main()
    return tmp_path / "test_clusters.tsv"


def _check(out, pcas_tsv):
    assert out.exists()
    labels = read_labels(out)
    emb = read_embeddings(pcas_tsv)
    assert labels.col_name == "cluster"
    assert labels.row_ids == emb.row_ids


def test_kmeans(monkeypatch, tmp_path, pcas_tsv):
    out = _run(monkeypatch, tmp_path, pcas_tsv, "rapids-kmeans",
               extra_args=("--n_clusters", "10"))
    _check(out, pcas_tsv)
    labels = read_labels(out)
    assert len(set(labels.values)) == 10


def test_hdbscan(monkeypatch, tmp_path, pcas_tsv):
    out = _run(monkeypatch, tmp_path, pcas_tsv, "rapids-hdbscan",
               extra_args=("--min_samples", "5", "--min_cluster_size", "30"))
    _check(out, pcas_tsv)
    labels = read_labels(out)
    assert len(set(labels.values) - {"-1"}) > 1


def test_dbscan(monkeypatch, tmp_path, pcas_tsv):
    # TODO: same dataset-dependence as the --eps TODO in cluster_embedding.py --
    # this 15 is derived from THIS fixture and silently wrong for any other.
    # eps has to be picked against the data: 5th-NN distance here runs
    # 6.5-33.7, median 13.3 (3779 cells, 50 PCs). The old 0.5 was
    # below the closest pair, so every cell came back noise -- and the assert
    # below did not catch it, because read_labels used to infer int64 and
    # `- {"-1"}` never matched the int -1. Hence the string check too.
    out = _run(monkeypatch, tmp_path, pcas_tsv, "rapids-dbscan",
               extra_args=("--eps", "15", "--min_samples", "5"))
    _check(out, pcas_tsv)
    labels = read_labels(out)
    assert all(isinstance(v, str) for v in labels.values)
    assert len(set(labels.values) - {"-1"}) >= 1


def test_dbscan_missing_eps(monkeypatch, tmp_path, pcas_tsv):
    import pytest
    with pytest.raises(ValueError, match="--eps"):
        _run(monkeypatch, tmp_path, pcas_tsv, "rapids-dbscan",
             extra_args=("--min_samples", "5"))


def test_kmeans_missing_n_clusters(monkeypatch, tmp_path, pcas_tsv):
    import pytest
    with pytest.raises(ValueError, match="--n_clusters"):
        _run(monkeypatch, tmp_path, pcas_tsv, "rapids-kmeans")


def test_hdbscan_rejects_seed(monkeypatch, tmp_path, pcas_tsv):
    import pytest
    monkeypatch.setattr(sys, "argv", [
        "cluster_embedding.py",
        "--embedding_tsv", str(pcas_tsv),
        "--method", "rapids-hdbscan",
        "--min_samples", "5",
        "--min_cluster_size", "30",
        "--random_seed", "42",
        "--output_dir", str(tmp_path),
        "--name", "test",
    ])
    from cluster_embedding import main
    with pytest.raises(ValueError, match="--random_seed is not accepted"):
        main()
