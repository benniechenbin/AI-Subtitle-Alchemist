import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src import db

MODULE_PATH = Path(__file__).parents[1] / "src" / "services" / "vector_index.py"
SPEC = importlib.util.spec_from_file_location("vector_index", MODULE_PATH)
vector_index = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["vector_index"] = vector_index
SPEC.loader.exec_module(vector_index)


class FakeModel:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, _query):
        return self.vector


class BatchFakeModel:
    def encode(self, texts, **_kwargs):
        lookup = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [0.5, 0.5],
        }
        return np.asarray([lookup[text] for text in texts], dtype=np.float32)


class FailingBackend:
    name = "failing"

    def ensure_index(self, db_path, embedding_model, embedding_dim):
        raise vector_index.VectorIndexUnavailable("boom")

    def upsert(self, db_path, items):
        raise vector_index.VectorIndexUnavailable("boom")

    def delete(self, db_path, subtitle_ids):
        raise vector_index.VectorIndexUnavailable("boom")

    def search(self, db_path, query_vector, embedding_model, embedding_dim, top_k, target_movie=None):
        raise vector_index.VectorIndexUnavailable("boom")

    def rebuild(self, db_path, items):
        raise vector_index.VectorIndexUnavailable("boom")

    def clear(self, db_path, embedding_model=None):
        raise vector_index.VectorIndexUnavailable("boom")

    def stats(self, db_path):
        return {}


class RecordingBackend:
    name = "recording"

    def __init__(self):
        self.upserted = []
        self.upsert_batch_sizes = []
        self.deleted = []
        self.cleared = []

    def ensure_index(self, db_path, embedding_model, embedding_dim):
        return None

    def upsert(self, db_path, items):
        self.upserted.extend(items)
        self.upsert_batch_sizes.append(len(items))

    def delete(self, db_path, subtitle_ids):
        self.deleted.extend(subtitle_ids)

    def search(self, db_path, query_vector, embedding_model, embedding_dim, top_k, target_movie=None):
        return []

    def rebuild(self, db_path, items):
        self.upserted.extend(items)

    def clear(self, db_path, embedding_model=None):
        self.cleared.append(embedding_model)

    def stats(self, db_path):
        return {}


class EmptyCountingBackend:
    name = "empty_counting"

    def __init__(self):
        self.search_calls = 0

    def ensure_index(self, db_path, embedding_model, embedding_dim):
        return None

    def upsert(self, db_path, items):
        return None

    def delete(self, db_path, subtitle_ids):
        return None

    def search(self, db_path, query_vector, embedding_model, embedding_dim, top_k, target_movie=None):
        self.search_calls += 1
        return []

    def rebuild(self, db_path, items):
        return None

    def clear(self, db_path, embedding_model=None):
        return None

    def stats(self, db_path):
        return {}


def _row(file_hash, path, movie, content, model_name="fake-model", dim=2):
    return (
        file_hash,
        path,
        movie,
        2026,
        0,
        0,
        0,
        "00:00:01,000",
        "00:00:02,000",
        content,
        model_name,
        dim,
    )


def _vector_row(embedding, model_name="fake-model"):
    vector = np.asarray(embedding, dtype=np.float32)
    return (vector.tobytes(), model_name, len(vector))


def _legacy_row(file_hash, path, movie, content, embedding, model_name="fake-model"):
    vector = np.asarray(embedding, dtype=np.float32)
    return (
        file_hash,
        path,
        movie,
        2026,
        0,
        0,
        0,
        "00:00:01,000",
        "00:00:02,000",
        content,
        vector.tobytes(),
        model_name,
        len(vector),
    )


def test_insert_returns_ids_without_embedding_blob(tmp_path):
    db_path = str(tmp_path / "vectors.db")
    db.init_db(db_path)
    rows = [
        _row("a", "/tmp/a.srt", "Movie A", "alpha"),
        _row("b", "/tmp/b.srt", "Movie B", "beta"),
    ]
    inserted_ids = db.insert_subtitles_batch(db_path, rows)

    conn = db.get_db_connection(db_path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(subtitles)").fetchall()]
        row = conn.execute(
            "SELECT content, embedding_model, embedding_dim FROM subtitles WHERE id = ?",
            (inserted_ids[0],),
        ).fetchone()
    finally:
        conn.close()

    assert inserted_ids == [1, 2]
    assert "embedding" not in columns
    assert row == ("alpha", "fake-model", 2)


def test_delete_records_by_path_returns_deleted_ids(tmp_path):
    db_path = str(tmp_path / "delete.db")
    db.init_db(db_path)
    inserted_ids = db.insert_subtitles_batch(
        db_path,
        [
            _row("a", "/tmp/a.srt", "Movie A", "alpha"),
            _row("b", "/tmp/b.srt", "Movie B", "beta"),
        ],
    )

    deleted_ids = db.delete_records_by_path(db_path, ["/tmp/a.srt"])
    remaining = db.fetch_subtitles_by_ids(db_path, inserted_ids)

    assert deleted_ids == [inserted_ids[0]]
    assert [row[0] for row in remaining] == [inserted_ids[1]]


def test_search_returns_empty_when_primary_unavailable_without_blob_fallback(tmp_path):
    db_path = str(tmp_path / "fallback.db")
    db.init_db(db_path)
    db.insert_subtitles_batch(
        db_path,
        [_row("a", "/tmp/a.srt", "Movie A", "alpha")],
    )

    service = vector_index.VectorIndexService(
        primary_backend=FailingBackend(),
        fallback_backend=vector_index.NumpyBlobBackend(),
    )
    results = service.search(
        query="alpha",
        model=FakeModel([1.0, 0.0]),
        db_path=db_path,
        embedding_model_name="fake-model",
        final_k=1,
        allow_duplicates=True,
    )

    assert results == []


def test_target_movie_fallback_is_not_called_twice(tmp_path):
    db_path = str(tmp_path / "fallback_once.db")
    db.init_db(db_path)
    fallback = EmptyCountingBackend()
    service = vector_index.VectorIndexService(
        primary_backend=FailingBackend(),
        fallback_backend=fallback,
    )

    results = service.search(
        query="missing",
        model=FakeModel([1.0, 0.0]),
        db_path=db_path,
        embedding_model_name="fake-model",
        final_k=1,
        allow_duplicates=True,
        target_movie="Movie A",
    )

    assert results == []
    assert fallback.search_calls == 1


def test_upsert_vector_rows_converts_vectors_to_items():
    backend = RecordingBackend()
    service = vector_index.VectorIndexService(primary_backend=backend)
    vector_row = _vector_row([1.0, 0.0])

    service.upsert_vector_rows("/tmp/unused.db", [42], [vector_row])

    assert len(backend.upserted) == 1
    assert backend.upserted[0].subtitle_id == 42
    assert backend.upserted[0].embedding_model == "fake-model"
    assert backend.upserted[0].embedding_dim == 2


def test_delete_delegates_to_backend():
    backend = RecordingBackend()
    service = vector_index.VectorIndexService(primary_backend=backend)

    service.delete("/tmp/unused.db", [1, 2])

    assert backend.deleted == [1, 2]


def test_rebuild_streams_vectors_in_batches(tmp_path):
    db_path = str(tmp_path / "rebuild.db")
    db.init_db(db_path)
    db.insert_subtitles_batch(
        db_path,
        [
            _row("a", "/tmp/a.srt", "Movie A", "alpha"),
            _row("b", "/tmp/b.srt", "Movie B", "beta"),
            _row("c", "/tmp/c.srt", "Movie C", "gamma"),
        ],
    )
    backend = RecordingBackend()
    service = vector_index.VectorIndexService(primary_backend=backend)

    service.rebuild(
        db_path,
        embedding_model="fake-model",
        model=BatchFakeModel(),
        batch_size=1,
    )

    assert backend.cleared == ["fake-model"]
    assert backend.upsert_batch_sizes == [1, 1, 1]
    assert [item.subtitle_id for item in backend.upserted] == [1, 2, 3]


def test_legacy_rows_drop_embedding_blob_on_insert(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    db.init_db(db_path)

    db.insert_subtitles_batch(
        db_path,
        [_legacy_row("a", "/tmp/a.srt", "Movie A", "alpha", [1.0, 0.0])],
    )

    conn = db.get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT content, embedding_model, embedding_dim FROM subtitles"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("alpha", "fake-model", 2)


def test_table_name_validation_and_score_clamp():
    table_name = vector_index._vec_table_name("fake-model", 384)

    assert table_name.startswith("vec_subtitles_384_")
    assert vector_index._score_from_distance(-0.25) == 1.0
    assert vector_index._score_from_distance(1.25) == 0.0
    with pytest.raises(ValueError):
        vector_index._validate_table_name("bad; DROP TABLE subtitles")


def test_get_vector_index_service_reuses_singleton():
    previous_service = vector_index._default_service
    try:
        vector_index._default_service = None
        first = vector_index.get_vector_index_service()
        second = vector_index.get_vector_index_service()
    finally:
        vector_index._default_service = previous_service

    assert first is second


def test_sqlite_vec_backend_creates_registry_when_available(tmp_path):
    pytest.importorskip("sqlite_vec")
    db_path = str(tmp_path / "sqlite_vec.db")
    db.init_db(db_path)
    backend = vector_index.SqliteVecBackend()

    try:
        backend.ensure_index(db_path, "fake-model", 2)
    except vector_index.VectorIndexUnavailable as exc:
        pytest.skip(str(exc))

    conn = db.get_db_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT backend, embedding_model, embedding_dim
            FROM vector_index_registry
            WHERE backend = 'sqlite_vec'
            """
        ).fetchone()
    finally:
        conn.close()

    assert row == ("sqlite_vec", "fake-model", 2)
