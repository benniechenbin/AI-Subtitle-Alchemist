import re

import guessit


def analyze_filenames(file_names: list[str]) -> list[dict]:
    """根据文件名列表预识别元数据。"""
    results = []
    for name in file_names:
        info = guessit.guessit(name)
        title = info.get("title")
        if not title or title == "未知电影" or "[" in str(title):
            anime_match = re.search(r"^\[.*?\]\[(.*?)\]", name)
            if anime_match:
                title = anime_match.group(1).replace("_", " ").strip()
            else:
                movie_match = re.search(r"^\[(.*?)\]", name)
                if movie_match:
                    title = movie_match.group(1).strip()
        if not title:
            title = name
        s_num = info.get("season", 0)
        e_num = info.get("episode", 0)
        results.append(
            {
                "原始文件名": name,
                "识别片名": title,
                "年份": info.get("year", 0),
                "season_num": s_num,
                "episode_num": e_num,
                "剧集": f"S{str(s_num).zfill(2)}E{str(e_num).zfill(2)}" if e_num else "",
                "状态": "待确认",
            }
        )
    return results
