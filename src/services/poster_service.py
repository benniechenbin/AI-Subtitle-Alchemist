from dataclasses import dataclass
from typing import Callable

from src import db
from src.config.settings import settings
from src.observability.logger import logger
from src.services.tmdb_service import fetch_tmdb_poster


PosterFetcher = Callable[..., str | None]
ProgressCallback = Callable[[str], None]


@dataclass
class PosterBackfillResult:
    pending_count: int = 0
    success_count: int = 0
    not_found_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    skipped_reason: str = ""


def backfill_missing_posters(
    db_path: str | None = None,
    api_key: str | None = None,
    fetcher: PosterFetcher = fetch_tmdb_poster,
    progress_callback: ProgressCallback | None = None,
) -> PosterBackfillResult:
    db.sync_movies_to_meta(db_path)
    movies = db.get_movies_missing_posters(db_path)
    result = PosterBackfillResult(pending_count=len(movies))

    if not movies:
        _emit(progress_callback, "🖼️ 海报已齐全，无需抓取。")
        return result

    effective_key = api_key or settings.env.tmdb_api_key
    if not effective_key:
        result.skipped_count = len(movies)
        result.skipped_reason = "missing_tmdb_api_key"
        logger.warning("跳过自动海报抓取：缺少 TMDB API Key")
        _emit(progress_callback, "⚠️ 缺少 TMDB API Key，已跳过自动海报抓取。")
        return result

    for movie in movies:
        movie_name = movie["movie_name"]
        release_year = movie.get("release_year")
        _emit(progress_callback, f"🖼️ 正在抓取海报: {movie_name}")

        try:
            poster_url = fetcher(
                movie_name,
                api_key=effective_key,
                release_year=release_year,
            )
            if poster_url:
                db.update_movie_poster(db_path, movie_name, poster_url)
                result.success_count += 1
                _emit(progress_callback, f"✅ 海报已补齐: {movie_name}")
            else:
                result.not_found_count += 1
                _emit(progress_callback, f"⚠️ 未找到海报: {movie_name}")
        except Exception as e:
            result.error_count += 1
            logger.error(f"自动抓取海报失败: {movie_name}: {e}")
            _emit(progress_callback, f"❌ 海报抓取失败: {movie_name} ({e})")

    return result


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)
