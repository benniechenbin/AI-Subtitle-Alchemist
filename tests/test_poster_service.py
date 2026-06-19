from types import SimpleNamespace

from src import db
from src.services import poster_service


def _subtitle_row(movie_name: str, year: int, suffix: str):
    return (
        f"hash-{suffix}",
        f"/library/{suffix}.srt",
        movie_name,
        year,
        0,
        0,
        0,
        "00:00:01,000",
        "00:00:02,000",
        f"{movie_name} line",
        "fake-model",
        0,
    )


def _poster_map(db_path):
    conn = db.get_db_connection(str(db_path))
    try:
        rows = conn.execute(
            "SELECT movie_name, poster_url FROM movies_meta ORDER BY movie_name"
        ).fetchall()
        return {movie_name: poster_url for movie_name, poster_url in rows}
    finally:
        conn.close()


def test_backfill_missing_posters_syncs_meta_and_preserves_existing(tmp_path):
    db_path = tmp_path / "posters.db"
    db.init_db(str(db_path))
    db.insert_subtitles_batch(
        str(db_path),
        [
            _subtitle_row("Existing Poster", 1999, "existing"),
            _subtitle_row("Missing Poster", 2020, "missing"),
        ],
    )
    db.sync_movies_to_meta(str(db_path))
    db.update_movie_poster(
        str(db_path),
        "Existing Poster",
        "https://example.test/existing.jpg",
    )

    db.insert_subtitles_batch(
        str(db_path),
        [
            _subtitle_row("Not Found Poster", 2021, "not-found"),
            _subtitle_row("Unsynced Poster", 2022, "unsynced"),
        ],
    )

    calls = []

    def fake_fetcher(movie_name, api_key=None, release_year=None):
        calls.append((movie_name, api_key, release_year))
        return {
            "Missing Poster": "https://example.test/missing.jpg",
            "Unsynced Poster": "https://example.test/unsynced.jpg",
        }.get(movie_name)

    result = poster_service.backfill_missing_posters(
        str(db_path),
        api_key="fake-key",
        fetcher=fake_fetcher,
    )

    assert result.pending_count == 3
    assert result.success_count == 2
    assert result.not_found_count == 1
    assert result.error_count == 0
    assert {call[0] for call in calls} == {
        "Missing Poster",
        "Not Found Poster",
        "Unsynced Poster",
    }
    assert ("Missing Poster", "fake-key", 2020) in calls
    assert ("Unsynced Poster", "fake-key", 2022) in calls

    posters = _poster_map(db_path)
    assert posters["Existing Poster"] == "https://example.test/existing.jpg"
    assert posters["Missing Poster"] == "https://example.test/missing.jpg"
    assert posters["Unsynced Poster"] == "https://example.test/unsynced.jpg"
    assert posters["Not Found Poster"] is None


def test_backfill_missing_posters_skips_without_tmdb_key(tmp_path, monkeypatch):
    db_path = tmp_path / "posters-no-key.db"
    db.init_db(str(db_path))
    db.insert_subtitles_batch(
        str(db_path),
        [_subtitle_row("Needs Key", 2024, "needs-key")],
    )
    monkeypatch.setattr(
        poster_service.settings,
        "env",
        SimpleNamespace(tmdb_api_key=""),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fetcher should not be called without a TMDB key")

    result = poster_service.backfill_missing_posters(
        str(db_path),
        fetcher=fail_if_called,
    )

    assert result.pending_count == 1
    assert result.skipped_count == 1
    assert result.skipped_reason == "missing_tmdb_api_key"
    assert _poster_map(db_path)["Needs Key"] is None
