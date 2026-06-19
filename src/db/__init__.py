from .base import get_db_connection
from .migrations import init_db
from .repos.subtitle_repo import (
    insert_subtitles_batch,
    fetch_subtitles_by_ids,
    iter_subtitles_for_vector_rebuild,
    update_subtitle_embedding_metadata_batch,
    get_context_by_id,
    search_keyword,
    check_file_exists,
    get_all_file_paths,
    get_subtitle_ids_by_paths,
    delete_records_by_path,
)
from .repos.media_repo import (
    get_all_movies,
    get_library_stats,
    sync_movies_to_meta,
    get_movies_with_meta,
    get_movies_missing_posters,
    update_movie_poster,
    upsert_movie_metadata,
)
from .repos.quote_repo import get_golden_quotes
from .repos.bgm_repo import insert_or_update_bgm, get_all_bgm
from .repos.tag_repo import (
    get_tags_by_type,
    add_tag,
    link_tag_to_movie,
    link_tag_to_quote,
    get_movie_tags,
    link_tag_to_subtitle,
    get_subtitle_tags,
    search_subtitles_by_tags,
)

__all__ = [
    "get_db_connection",
    "init_db",
    "insert_subtitles_batch",
    "fetch_subtitles_by_ids",
    "iter_subtitles_for_vector_rebuild",
    "update_subtitle_embedding_metadata_batch",
    "get_context_by_id",
    "search_keyword",
    "check_file_exists",
    "get_all_file_paths",
    "get_subtitle_ids_by_paths",
    "delete_records_by_path",
    "get_all_movies",
    "get_library_stats",
    "sync_movies_to_meta",
    "get_movies_with_meta",
    "get_movies_missing_posters",
    "update_movie_poster",
    "upsert_movie_metadata",
    "get_golden_quotes",
    "insert_or_update_bgm",
    "get_all_bgm",
    "add_tag",
    "get_tags_by_type",
    "link_tag_to_movie",
    "link_tag_to_quote",
    "get_movie_tags",
    "link_tag_to_subtitle",
    "get_subtitle_tags",
    "search_subtitles_by_tags",
]
