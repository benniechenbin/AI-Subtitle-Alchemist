import sqlite3

from app.tabs.clean_import import _commit_pending_import
from src import db


class FailingVectorService:
    def upsert_vector_rows(self, db_path, subtitle_ids, vector_rows):
        raise RuntimeError("vector failed")


def test_vector_failure_does_not_turn_successful_database_import_into_retry(tmp_path):
    db_path = tmp_path / "vector-failure.db"
    db.init_db(str(db_path))
    row = (
        "hash",
        "/tmp/example.srt",
        "Example",
        2025,
        0,
        0,
        0,
        "00:00:01,000",
        "00:00:02,000",
        "Hello",
        None,
        0,
    )

    inserted_ids, vector_error = _commit_pending_import(
        str(db_path),
        {
            "pending_rows": [row],
            "pending_vectors": [(None, None, 0)],
            "pending_meta": [{"movie_name": "Example", "year": 2025}],
        },
        vector_service=FailingVectorService(),
    )

    assert len(inserted_ids) == 1
    assert isinstance(vector_error, RuntimeError)
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM movies_meta").fetchone()[0] == 1
    finally:
        conn.close()
