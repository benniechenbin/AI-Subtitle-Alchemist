from src import db

def test_db_public_api_exports():
    """验证 db 模块的公开 API 导出是否完整，防止重构导致外部调用中断。"""
    required = [
        "init_db",
        "get_db_connection",
        "insert_subtitles_batch",
        "fetch_vectors_for_search",
        "get_all_movies",
        "get_library_stats",
        "get_golden_quotes",
        "insert_or_update_bgm",
        "get_all_bgm",
        "add_tag",
        "get_tags_by_type",
        "link_tag_to_movie",
        "link_tag_to_quote",
        "link_tag_to_subtitle",
        "get_movie_tags",
        "get_subtitle_tags",
        "search_subtitles_by_tags",
    ]

    for name in required:
        assert hasattr(db, name), f"db 模块缺失公开 API: {name}"
