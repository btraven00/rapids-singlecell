"""read_graph must accept the NNG stage output spec, not just our own writer."""

import h5py
import numpy as np

from writers import read_graph


def _write_csr(grp, m):
    grp.create_dataset("data", data=m.data)
    grp.create_dataset("indices", data=m.indices)
    grp.create_dataset("indptr", data=m.indptr)


def test_reads_spec_layout(tmp_path, neighbors_h5):
    """read_graph must not depend on our own writer.

    Same layout our writer emits, but built here by hand from the scanpy
    module's recipe, so the reader is pinned to the spec rather than to
    whatever write_graph happens to do.
    """
    ours = read_graph(neighbors_h5)

    spec = tmp_path / "spec_neighbors.h5"
    with h5py.File(spec, "w") as h5:
        h5.create_dataset("cell_ids", data=np.array(ours.row_ids, dtype="S"))
        _write_csr(h5, ours.distances)
        _write_csr(h5.create_group("connectivities"), ours.connectivities)

    got = read_graph(spec)
    assert got.row_ids == ours.row_ids
    assert got.distances.shape == ours.distances.shape
    assert (got.distances != ours.distances).nnz == 0
    assert (got.connectivities != ours.connectivities).nnz == 0


def test_writes_spec_layout(tmp_path, neighbors_h5):
    """write_graph must EMIT the spec layout, not just read it.

    The reader accepts both, so a nested-distances writer looks fine in a
    round-trip and only fails downstream, in the R metrics reader that reads
    /data directly.
    """
    from writers import write_graph

    out = tmp_path / "out_neighbors.h5"
    write_graph(read_graph(neighbors_h5), out)

    with h5py.File(out, "r") as h5:
        assert set(h5.keys()) == {"cell_ids", "connectivities",
                                  "data", "indices", "indptr"}
        assert "distances" not in h5          # flat at the root, not a group
        assert "shape" not in h5
        assert "shape" not in h5["connectivities"]

    got = read_graph(out)
    ours = read_graph(neighbors_h5)
    assert got.row_ids == ours.row_ids
    assert (got.distances != ours.distances).nnz == 0
    assert (got.connectivities != ours.connectivities).nnz == 0


def test_embedding_round_trip_is_exact(tmp_path):
    """read_embeddings(write_embeddings(x)) == x bit for bit, ids and names kept as strings."""
    from writers import Embedding, read_embeddings, write_embeddings

    rng = np.random.default_rng(0)
    m = rng.standard_normal((500, 7)) * 10.0 ** rng.integers(-12, 12, (500, 7))
    m[:, 0] = np.arange(500)                      # integer-looking column stays float64
    emb = Embedding(m, [str(i) for i in range(500)], [f"PC{i + 1}" for i in range(7)])  # numeric-looking ids
    write_embeddings(emb, tmp_path / "e.tsv")
    back = read_embeddings(tmp_path / "e.tsv")
    assert back.matrix.dtype == np.float64 and back.matrix.flags.c_contiguous
    assert back.matrix.tobytes() == m.tobytes()
    assert back.row_ids == emb.row_ids and back.col_names == emb.col_names


def test_reads_embedding_without_id_header(pcas_tsv):
    """R's write.table layout (N header names over N+1 fields), as in the fixture."""
    from writers import read_embeddings

    emb = read_embeddings(pcas_tsv)
    assert emb.col_names[0] == "PC1" and emb.matrix.shape == (len(emb.row_ids), len(emb.col_names))
