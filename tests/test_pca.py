import sys

import h5py

from writers import read_embeddings


def _n_cells(normalized_h5):
    with h5py.File(normalized_h5) as h5:
        return int(h5["matrix"]["shape"][1])


def _run(monkeypatch, tmp_path, normalized_h5, n_components=20, seed=42):
    monkeypatch.setattr(sys, "argv", [
        "pca.py",
        "--normalized_selected_h5", str(normalized_h5),
        "--solver", "covariance-eigh",
        "--n_components", str(n_components),
        "--random_seed", str(seed),
        "--output_dir", str(tmp_path),
        "--name", "test",
    ])
    from pca import main
    main()
    return tmp_path / "test_embedding.tsv"


def test_pca_output_shape(monkeypatch, tmp_path, normalized_h5):
    n_components = 20
    out = _run(monkeypatch, tmp_path, normalized_h5, n_components=n_components)
    assert out.exists()
    emb = read_embeddings(out)
    assert emb.matrix.shape == (_n_cells(normalized_h5), n_components)


def test_pca_column_names(monkeypatch, tmp_path, normalized_h5):
    n_components = 15
    out = _run(monkeypatch, tmp_path, normalized_h5, n_components=n_components)
    emb = read_embeddings(out)
    assert emb.col_names == [f"PC{i + 1}" for i in range(n_components)]


def test_pca_row_ids_match_barcodes(monkeypatch, tmp_path, normalized_h5):
    out = _run(monkeypatch, tmp_path, normalized_h5)
    emb = read_embeddings(out)
    with h5py.File(normalized_h5) as h5:
        expected = list(h5["matrix"]["barcodes"][:].astype(str))
    assert emb.row_ids == expected


def test_pca_loadings_output(monkeypatch, tmp_path, normalized_h5):
    n_components = 20
    out = _run(monkeypatch, tmp_path, normalized_h5, n_components=n_components)
    loadings = out.parent / "test_loadings.tsv"
    assert loadings.exists()
    load = read_embeddings(loadings)
    with h5py.File(normalized_h5) as h5:
        n_genes = int(h5["matrix"]["shape"][0])
        gene_ids = list(h5["matrix"]["genes"][:].astype(str))
    assert load.matrix.shape == (n_genes, n_components)
    assert load.col_names == [f"PC{i + 1}" for i in range(n_components)]
    assert load.row_ids == gene_ids

    # Named first column (scanpy layout); unnamed, R's read.table calls it "X".
    pcs = [f"PC{i + 1}" for i in range(n_components)]
    assert out.read_text().splitlines()[0].split("\t") == ["cell_id"] + pcs
    assert loadings.read_text().splitlines()[0].split("\t") == ["gene_id"] + pcs
