import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src import db
from src.services.scan_service import scan_library


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=16,
    ):
        return np.zeros((len(texts), 3), dtype=np.float32)


def _srt(text: str) -> str:
    return f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n\n"


def _write_srt(path: Path, text: str = "Hello world", encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_srt(text).encode(encoding))


def _write_manifest(root: Path, items: list[dict]) -> None:
    (root / "harvester_import_manifest.json").write_text(
        json.dumps({"items": items}, ensure_ascii=False),
        encoding="utf-8",
    )


def _scan_rows(root: Path, db_path: Path) -> list[sqlite3.Row]:
    db.init_db(str(db_path))
    logs = list(scan_library(str(root), FakeEmbeddingModel(), "fake-model", str(db_path)))
    assert logs[-1][0] == "DONE"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT movie_name, year, season, episode, content
            FROM subtitles
            ORDER BY movie_name, content
            """
        ).fetchall()
    finally:
        conn.close()


def test_scan_plain_directory_still_uses_guessit(tmp_path):
    root = tmp_path / "plain"
    _write_srt(root / "The.Matrix.1999.srt", "Wake up")

    rows = _scan_rows(root, tmp_path / "plain.db")

    assert len(rows) == 1
    assert rows[0]["movie_name"] == "The Matrix"
    assert rows[0]["year"] == 1999
    assert rows[0]["content"] == "Wake up"


def test_scan_staging_manifest_uses_json_title_and_year(tmp_path):
    root = tmp_path / "staging"
    _write_srt(root / "movie_980477" / "demo.zh.srt", "Manifest line")
    _write_manifest(
        root,
        [
            {
                "media_key": "movie_980477",
                "media_type": "movie",
                "tmdb_id": 980477,
                "imdb_id": "tt13539646",
                "title": "Manifest Movie",
                "year": 2023,
                "provider": "opensubtitles",
                "source_id": "demo",
                "subtitle_file": "movie_980477/demo.zh.srt",
                "original_manifest": {},
            }
        ],
    )

    rows = _scan_rows(root, tmp_path / "staging.db")

    assert len(rows) == 1
    assert rows[0]["movie_name"] == "Manifest Movie"
    assert rows[0]["year"] == 2023


def test_scan_staging_manifest_persists_tmdb_metadata(tmp_path):
    root = tmp_path / "staging"
    db_path = tmp_path / "tmdb.db"
    _write_srt(root / "movie_980477" / "demo.zh.srt", "Metadata line")
    tmdb_metadata = {
        "schema_version": 1,
        "source": "tmdb",
        "fetched_at": "2026-06-19T00:00:00+00:00",
        "language": "zh-CN",
        "region": "CN",
        "content": {
            "media_type": "movie",
            "tmdb_id": 980477,
            "imdb_id": "tt13539646",
            "original_title": "Demo Original",
            "aliases": ["Manifest Movie", "Demo Original"],
            "overview": "用于测试的剧情简介",
            "tagline": "测试标语",
            "genres": [{"id": 18, "name": "剧情"}],
            "keywords": [{"id": 42, "name": "成长"}],
            "certification": "PG-13",
            "certification_country": "US",
            "adult": False,
            "original_language": "en",
            "origin_countries": ["US"],
            "spoken_languages": [{"iso_639_1": "en", "name": "English"}],
            "release_date": "2023-01-01",
            "runtime_minutes": 121,
            "status": "Released",
            "poster_path": "/poster.jpg",
            "backdrop_path": "/backdrop.jpg",
            "homepage": "https://example.com/movie",
        },
        "raw": {"responses": {"zh-CN": {"id": 980477, "vote_average": 8.1}}},
        "errors": [],
    }
    _write_manifest(
        root,
        [
            {
                "media_key": "movie_980477",
                "media_type": "movie",
                "tmdb_id": 980477,
                "imdb_id": "tt13539646",
                "title": "Manifest Movie",
                "year": 2023,
                "subtitle_file": "movie_980477/demo.zh.srt",
                "tmdb_metadata": tmdb_metadata,
            }
        ],
    )

    rows = _scan_rows(root, db_path)
    assert rows[0]["movie_name"] == "Manifest Movie"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        movie = conn.execute(
            "SELECT * FROM movies_meta WHERE movie_name = ?",
            ("Manifest Movie",),
        ).fetchone()
    finally:
        conn.close()

    assert movie["media_key"] == "movie_980477"
    assert movie["tmdb_id"] == 980477
    assert movie["overview"] == "用于测试的剧情简介"
    assert movie["certification"] == "PG-13"
    assert movie["poster_url"] == "https://image.tmdb.org/t/p/w500/poster.jpg"
    assert json.loads(movie["genres_json"]) == [{"id": 18, "name": "剧情"}]
    assert json.loads(movie["keywords_json"]) == [{"id": 42, "name": "成长"}]
    assert json.loads(movie["tmdb_metadata_json"]) == tmdb_metadata
    assert json.loads(movie["extra_metadata_json"]) == {}


def test_movie_metadata_upsert_does_not_replace_values_with_null(tmp_path):
    db_path = tmp_path / "upsert.db"
    db.init_db(str(db_path))
    db.upsert_movie_metadata(
        str(db_path),
        {
            "movie_name": "Stable Movie",
            "year": 2024,
            "media_key": "movie_10",
            "media_type": "movie",
            "tmdb_id": 10,
            "tmdb_metadata": {
                "schema_version": 1,
                "content": {"overview": "保留这个简介", "tmdb_id": 10},
            },
        },
    )
    db.upsert_movie_metadata(
        str(db_path),
        {"movie_name": "Stable Movie", "year": None, "tmdb_metadata": None},
    )

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT release_year, overview, media_key FROM movies_meta WHERE movie_name = ?",
            ("Stable Movie",),
        ).fetchone()
    finally:
        conn.close()

    assert row == (2024, "保留这个简介", "movie_10")


def test_scan_weak_filename_uses_manifest_metadata(tmp_path):
    root = tmp_path / "staging"
    _write_srt(root / "movie_980477" / "Chs.srt", "Weak filename line")
    _write_manifest(
        root,
        [
            {
                "media_type": "movie",
                "tmdb_id": 980477,
                "imdb_id": "tt13539646",
                "title": "Correct Title From JSON",
                "year": 2024,
                "subtitle_file": "movie_980477/Chs.srt",
            }
        ],
    )

    rows = _scan_rows(root, tmp_path / "weak.db")

    assert len(rows) == 1
    assert rows[0]["movie_name"] == "Correct Title From JSON"
    assert rows[0]["year"] == 2024


def test_scan_manifest_falls_back_to_guessit_for_unlisted_file(tmp_path):
    root = tmp_path / "staging"
    _write_srt(root / "movie_980477" / "listed.srt", "Listed line")
    _write_srt(root / "Fallback.Movie.2022.srt", "Fallback line")
    _write_manifest(
        root,
        [
            {
                "media_type": "movie",
                "tmdb_id": 980477,
                "title": "Listed Manifest Movie",
                "year": 2021,
                "subtitle_file": "movie_980477/listed.srt",
            }
        ],
    )

    rows = _scan_rows(root, tmp_path / "fallback.db")
    by_content = {row["content"]: row for row in rows}

    assert by_content["Listed line"]["movie_name"] == "Listed Manifest Movie"
    assert by_content["Listed line"]["year"] == 2021
    assert by_content["Fallback line"]["movie_name"] == "Fallback Movie"
    assert by_content["Fallback line"]["year"] == 2022


def test_scan_manifest_accepts_windows_style_subtitle_paths(tmp_path):
    root = tmp_path / "staging"
    _write_srt(root / "movie_980477" / "Chinese.srt", "Windows path line")
    _write_manifest(
        root,
        [
            {
                "media_type": "movie",
                "tmdb_id": 980477,
                "title": "Windows Path Movie",
                "year": 2025,
                "subtitle_file": r"movie_980477\Chinese.srt",
            }
        ],
    )

    rows = _scan_rows(root, tmp_path / "windows.db")

    assert len(rows) == 1
    assert rows[0]["movie_name"] == "Windows Path Movie"
    assert rows[0]["year"] == 2025


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("gb2312", "你好，世界"),
        ("gb18030", "𠮷野家字幕"),
    ],
)
def test_scan_encoded_subtitles_are_still_decoded_and_parsed(
    tmp_path, encoding, text
):
    root = tmp_path / encoding
    _write_srt(root / f"Encoding.Movie.2026.{encoding}.srt", text, encoding=encoding)

    rows = _scan_rows(root, tmp_path / f"{encoding}.db")

    assert len(rows) == 1
    assert rows[0]["movie_name"] == "Encoding Movie"
    assert rows[0]["year"] == 2026
    assert rows[0]["content"] == text
