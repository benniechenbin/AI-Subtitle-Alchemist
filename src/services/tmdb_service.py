import requests

from src.config.constants import TMDB_POSTER_BASE_URL
from src.config.settings import settings
from src.observability.logger import logger


def search_tmdb_metadata(
    movie_name: str,
    api_key: str = None,
    release_year: int | str | None = None,
) -> dict | None:
    """
    通过 TMDB API 搜索影视剧并获取最佳匹配的元数据。
    
    :return: 包含 title, year, poster_path 的字典，如果没找到则返回 None
    """
    effective_key = api_key or settings.env.tmdb_api_key

    if not effective_key:
        logger.error("❌ 缺少 TMDB API Key，请检查 .env 配置")
        return None

    # 使用 search/multi 接口，这样电影和动漫（剧集）都能同时搜到
    url = "https://api.themoviedb.org/3/search/multi"
    params = {
        "api_key": effective_key,
        "query": movie_name,
        "language": "zh-CN",  # 优先返回中文海报和译名
        "page": 1,
        "include_adult": "false" # 过滤成人内容
    }

    try:
        # 发起请求，设置 10 秒超时防止卡死
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # 如果状态码不是 200，主动抛出异常
        data = response.json()
        results = data.get("results") or []

        if results:
            selected = _select_best_result(results, release_year)
            if selected:
                title = selected.get("title") or selected.get("name")
                if not title:
                    title = selected.get("original_title") or selected.get("original_name")
                return {
                    "title": title,
                    "year": _extract_result_year(selected),
                    "poster_path": selected.get("poster_path"),
                }

            logger.warning(f"⚠️ TMDB 收录了《{movie_name}》，但未能找到有效匹配。")
            return None
        else:
            logger.warning(f"⚠️ 在 TMDB 中未找到关于《{movie_name}》的记录。")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 网络请求 TMDB 失败: {e}")
        return None


def fetch_tmdb_poster(
    movie_name: str,
    api_key: str = None,
    release_year: int | str | None = None,
) -> str | None:
    """
    通过 TMDB API 搜索影视剧并获取海报 URL。
    """
    metadata = search_tmdb_metadata(movie_name, api_key, release_year)
    if metadata and metadata.get("poster_path"):
        return f"{TMDB_POSTER_BASE_URL}{metadata['poster_path']}"
    return None


def _select_best_result(
    results: list[dict],
    release_year: int | str | None = None,
) -> dict | None:
    if not results:
        return None

    target_year = _normalize_year(release_year)
    if target_year is not None:
        for result in results:
            if _extract_result_year(result) == target_year:
                return result

    poster_candidates = [result for result in results if result.get("poster_path")]
    if poster_candidates:
        return poster_candidates[0]

    return results[0]


def _extract_result_year(result: dict) -> int | None:
    date_value = result.get("release_date") or result.get("first_air_date") or ""
    if not isinstance(date_value, str) or len(date_value) < 4:
        return None
    return _normalize_year(date_value[:4])


def _normalize_year(value: int | str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if year <= 0:
        return None
    return year
