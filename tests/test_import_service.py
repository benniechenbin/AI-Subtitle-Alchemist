import sqlite3

import pytest

from src import db
from src.models import UploadedFileInput
from src.services.import_service import process_uploaded_files
from src.utils import calculate_file_hash


def _srt(text="Hello") -> bytes:
    return f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n\n".encode()


def _metadata(title="规范片名", year=2025):
    return {
        "原始文件名": "Original.Movie.2025.srt",
        "识别片名": title,
        "年份": year,
        "season_num": 0,
        "episode_num": 0,
    }


def _tmdb_match(title="规范片名", year=2025):
    return {
        "expected_title": title,
        "expected_year": year,
        "metadata": {
            "title": title,
            "year": year,
            "tmdb_id": 42,
            "media_type": "movie",
            "raw": {
                "id": 42,
                "media_type": "movie",
                "title": title,
                "release_date": f"{year}-01-01",
                "poster_path": "/poster.jpg",
            },
        },
    }


def test_process_uploaded_file_safely_renames_matching_library_source(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    raw = _srt()
    original = library / "Original.Movie.2025.srt"
    original.write_bytes(raw)
    db_path = tmp_path / "library.db"
    db.init_db(str(db_path))

    result = process_uploaded_files(
        [UploadedFileInput(name=original.name, raw_bytes=raw)],
        [_metadata()],
        str(library),
        None,
        db_path=str(db_path),
    )

    stored_hash = calculate_file_hash(raw)
    renamed = library / f"规范片名 (2025).{stored_hash[:8]}.srt"
    assert not original.exists()
    assert renamed.read_bytes() == raw
    assert result[2]["success"] == 1
    assert result[3][0][1] == str(renamed)
    assert any("已安全改名" in log for log in result[0])


def test_manual_metadata_edit_invalidates_tmdb_binding(tmp_path):
    db_path = tmp_path / "manual.db"
    db.init_db(str(db_path))

    result = process_uploaded_files(
        [UploadedFileInput(name="Original.Movie.2025.srt", raw_bytes=_srt())],
        [_metadata(title="手工修改名")],
        str(tmp_path / "library"),
        None,
        db_path=str(db_path),
        tmdb_matches=[_tmdb_match()],
    )

    assert result[5] == []
    assert any("已取消 TMDB 绑定" in log for log in result[0])


def test_unchanged_tmdb_match_is_prepared_for_atomic_import(tmp_path):
    db_path = tmp_path / "tmdb.db"
    db.init_db(str(db_path))

    result = process_uploaded_files(
        [UploadedFileInput(name="Original.Movie.2025.srt", raw_bytes=_srt())],
        [_metadata()],
        str(tmp_path / "library"),
        None,
        db_path=str(db_path),
        tmdb_matches=[_tmdb_match()],
    )

    assert result[5][0]["movie_name"] == "规范片名"
    assert result[5][0]["tmdb_id"] == 42
    assert result[5][0]["tmdb_metadata"]["content"]["id"] == 42


def test_atomic_import_rolls_back_subtitles_when_metadata_fails(tmp_path, monkeypatch):
    from src.db.repos import subtitle_repo

    db_path = tmp_path / "rollback.db"
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

    def fail_metadata(_conn, _metadata):
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(
        subtitle_repo, "upsert_movie_metadata_on_connection", fail_metadata
    )

    with pytest.raises(RuntimeError, match="metadata failed"):
        db.insert_subtitles_with_metadata_batch(
            str(db_path), [row], [{"movie_name": "Example"}]
        )

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM movies_meta").fetchone()[0] == 0
    finally:
        conn.close()


def test_atomic_import_persists_tmdb_id_and_tv_original_name(tmp_path):
    db_path = tmp_path / "atomic.db"
    db.init_db(str(db_path))
    row = (
        "hash",
        "/tmp/example.srt",
        "中文剧名",
        2025,
        1,
        1,
        0,
        "00:00:01,000",
        "00:00:02,000",
        "Hello",
        None,
        0,
    )
    metadata = {
        "movie_name": "中文剧名",
        "year": 2025,
        "media_type": "tv",
        "tmdb_metadata": {
            "source": "tmdb",
            "schema_version": 1,
            "content": {
                "id": 42,
                "media_type": "tv",
                "original_name": "Original Series",
                "poster_path": "/poster.jpg",
            },
        },
    }

    db.insert_subtitles_with_metadata_batch(str(db_path), [row], [metadata])

    conn = sqlite3.connect(db_path)
    try:
        movie = conn.execute(
            "SELECT tmdb_id, original_title, poster_url FROM movies_meta"
        ).fetchone()
    finally:
        conn.close()

    assert movie == (
        42,
        "Original Series",
        "https://image.tmdb.org/t/p/w500/poster.jpg",
    )
