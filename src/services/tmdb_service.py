import requests

from src.config.settings import settings
from src.observability.logger import logger

def fetch_tmdb_poster(movie_name: str, api_key: str = None) -> str | None:
    """
    通过 TMDB API 搜索影视剧并获取海报 URL。
    
    :param movie_name: 影视剧名称 (如 "葬送的芙莉莲")
    :param api_key: TMDB 的 v3 API Key
    :return: 完整的海报图片 URL，如果没找到则返回 None
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

        if data.get("results"):
            # 取第一个最匹配的结果
            first_result = data["results"][0]
            poster_path = first_result.get("poster_path")

            if poster_path:
                # 拼接完整图片地址 (w500 是中等分辨率，适合在 UI 列表中展示)
                full_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                return full_url
            else:
                logger.warning(f"⚠️ TMDB 收录了《{movie_name}》，但目前没有上传海报。")
                return None
        else:
            logger.warning(f"⚠️ 在 TMDB 中未找到关于《{movie_name}》的记录。")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 网络请求 TMDB 失败: {e}")
        return None