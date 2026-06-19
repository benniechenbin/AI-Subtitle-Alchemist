import json
import time
from src.config.constants import TMDB_POSTER_BASE_URL
from src.db.base import get_db_connection
from src.observability.logger import logger


def get_all_movies(db_path=None):
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        c.execute("SELECT DISTINCT movie_name FROM subtitles ORDER BY movie_name")
        return [row[0] for row in c.fetchall()]
    finally:
        conn.close()

def get_library_stats(db_path=None):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(DISTINCT movie_name) FROM subtitles")
        m_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM subtitles")
        l_count = c.fetchone()[0]
        return {
            "movie_count": m_count,
            "line_count": l_count,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logger.error(f"⚠️ 统计数据获取失败: {e}")
        return {"movie_count": 0, "line_count": 0, "last_update": "从未"}
    finally:
        conn.close()

def sync_movies_to_meta(db_path=None):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT OR IGNORE INTO movies_meta (movie_name, release_year)
            SELECT movie_name, MAX(year) 
            FROM subtitles 
            WHERE movie_name IS NOT NULL
            GROUP BY movie_name
        """)
        conn.commit()
    finally:
        conn.close()

def get_movies_with_meta(db_path=None) -> list[dict]:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT 
                m.movie_name, 
                m.poster_url, 
                m.release_year, 
                m.highlight_status,
                COUNT(s.id) as line_count
            FROM movies_meta m
            LEFT JOIN subtitles s ON m.movie_name = s.movie_name
            GROUP BY m.movie_name
            ORDER BY line_count DESC
        """)
        columns = ['movie_name', 'poster_url', 'release_year', 'highlight_status', 'line_count']
        return [dict(zip(columns, row)) for row in c.fetchall()]
    finally:
        conn.close()

def get_movies_missing_posters(db_path=None) -> list[dict]:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT movie_name, release_year
            FROM movies_meta
            WHERE movie_name IS NOT NULL
              AND (poster_url IS NULL OR TRIM(poster_url) = '')
            ORDER BY movie_name
        """)
        columns = ["movie_name", "release_year"]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    finally:
        conn.close()

def update_movie_poster(db_path: str, movie_name: str, poster_url: str) -> None:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute(
            "UPDATE movies_meta SET poster_url = ? WHERE movie_name = ?",
            (poster_url, movie_name)
        )
        conn.commit()
    finally:
        conn.close()


def upsert_movie_metadata(db_path: str, metadata: dict) -> None:
    movie_name = _text_or_none(metadata.get("movie_name"))
    if not movie_name:
        return

    tmdb_metadata = metadata.get("tmdb_metadata")
    tmdb_payload = tmdb_metadata if isinstance(tmdb_metadata, dict) else {}
    content_value = tmdb_payload.get("content")
    content = content_value if isinstance(content_value, dict) else {}
    poster_path = _text_or_none(content.get("poster_path"))
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    values = {
        "movie_name": movie_name,
        "poster_url": f"{TMDB_POSTER_BASE_URL}{poster_path}" if poster_path else None,
        "release_year": _int_or_none(metadata.get("year")),
        "media_key": _text_or_none(metadata.get("media_key")),
        "media_type": _text_or_none(content.get("media_type") or metadata.get("media_type")),
        "tmdb_id": _int_or_none(content.get("tmdb_id") or metadata.get("tmdb_id")),
        "imdb_id": _text_or_none(content.get("imdb_id") or metadata.get("imdb_id")),
        "original_title": _text_or_none(content.get("original_title")),
        "aliases_json": _json_or_none(content.get("aliases")),
        "overview": _text_or_none(content.get("overview")),
        "tagline": _text_or_none(content.get("tagline")),
        "genres_json": _json_or_none(content.get("genres")),
        "keywords_json": _json_or_none(content.get("keywords")),
        "certification": _text_or_none(content.get("certification")),
        "certification_country": _text_or_none(content.get("certification_country")),
        "adult": _bool_or_none(content.get("adult")),
        "original_language": _text_or_none(content.get("original_language")),
        "origin_countries_json": _json_or_none(content.get("origin_countries")),
        "spoken_languages_json": _json_or_none(content.get("spoken_languages")),
        "release_date": _text_or_none(content.get("release_date") or content.get("first_air_date")),
        "runtime_minutes": _int_or_none(content.get("runtime_minutes")),
        "status": _text_or_none(content.get("status")),
        "first_air_date": _text_or_none(content.get("first_air_date")),
        "last_air_date": _text_or_none(content.get("last_air_date")),
        "number_of_seasons": _int_or_none(content.get("number_of_seasons")),
        "number_of_episodes": _int_or_none(content.get("number_of_episodes")),
        "in_production": _bool_or_none(content.get("in_production")),
        "networks_json": _json_or_none(content.get("networks")),
        "poster_path": poster_path,
        "backdrop_path": _text_or_none(content.get("backdrop_path")),
        "homepage": _text_or_none(content.get("homepage")),
        "tmdb_metadata_json": _json_or_none(tmdb_payload),
        "extra_metadata_json": "{}",
        "metadata_source": "tmdb" if tmdb_payload else None,
        "metadata_schema_version": _int_or_none(tmdb_payload.get("schema_version")),
        "tmdb_language": _text_or_none(tmdb_payload.get("language")),
        "tmdb_region": _text_or_none(tmdb_payload.get("region")),
        "tmdb_fetched_at": _text_or_none(tmdb_payload.get("fetched_at")),
        "created_at": now,
        "updated_at": now,
    }
    columns = tuple(values)
    update_columns = [
        column
        for column in columns
        if column not in {"movie_name", "extra_metadata_json", "created_at"}
    ]
    updates = ", ".join(
        f"{column} = COALESCE(excluded.{column}, movies_meta.{column})"
        for column in update_columns
    )
    placeholders = ", ".join(f":{column}" for column in columns)
    sql = (
        f"INSERT INTO movies_meta ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(movie_name) DO UPDATE SET {updates}"
    )

    conn = get_db_connection(db_path)
    try:
        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


def _text_or_none(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _json_or_none(value) -> str | None:
    if value in (None, "", [], {}):
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
