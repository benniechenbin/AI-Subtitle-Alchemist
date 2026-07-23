import time
from src.observability.logger import logger
from src.db.base import get_db_connection
from . import schema


def init_db(db_path=None):
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()

        # 0. 建立迁移版本控制表
        c.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        conn.commit()

        # 检查当前版本
        c.execute("SELECT MAX(version) FROM schema_migrations")
        row = c.fetchone()
        current_version = row[0] if row and row[0] else 0

        if current_version == 0:
            logger.info("🎬 初始化数据库 V1...")
            c.execute(schema.SQL_CREATE_SUBTITLES)
            c.execute("CREATE INDEX IF NOT EXISTS idx_content ON subtitles (content)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_hash ON subtitles (file_hash)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_movie ON subtitles (movie_name)")
            c.execute(schema.SQL_CREATE_MOVIES_META)
            c.execute(schema.SQL_CREATE_GOLDEN_QUOTES)
            c.execute(schema.SQL_CREATE_BGM_LIBRARY)

            c.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (1, ?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            conn.commit()
            current_version = 1

        # ================= V2 资产模型升级 =================
        if current_version < 2:
            logger.info("🚀 正在升级数据库至 V2 (资产模型升级)...")
            _migration_v2(conn)
            c.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (2, ?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            conn.commit()
            current_version = 2
            logger.info("✅ 数据库已成功升级至 V2")

        # ================= V3 向量索引注册表 =================
        if current_version < 3:
            logger.info("🧭 正在升级数据库至 V3 (向量索引注册表)...")
            _migration_v3(conn)
            c.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (3, ?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            conn.commit()
            current_version = 3
            logger.info("✅ 数据库已成功升级至 V3")

        # ================= V4 sqlite-vec 单份向量存储 =================
        if current_version < 4:
            logger.info("🧭 正在升级数据库至 V4 (sqlite-vec 单份向量存储)...")
            _migration_v4(conn)
            c.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (4, ?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            conn.commit()
            current_version = 4
            logger.info("✅ 数据库已成功升级至 V4")

        # ================= V5 TMDB 富元数据 =================
        if current_version < 5:
            logger.info("🎞️ 正在升级数据库至 V5 (TMDB 富元数据)...")
            _migration_v5(conn)
            c.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (5, ?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            conn.commit()
            logger.info("✅ 数据库已成功升级至 V5")
    finally:
        conn.close()


def _migration_v2(conn):
    """V2 迁移：新增统一标签系统，升级 BGM 模型"""
    c = conn.cursor()

    # 1. 新增标签系统相关表
    c.execute(schema.SQL_CREATE_TAGS)
    c.execute(schema.SQL_CREATE_SUBTITLE_TAGS)
    c.execute(schema.SQL_CREATE_MOVIE_TAGS)
    c.execute(schema.SQL_CREATE_QUOTE_TAGS)

    # 2. 升级 BGM 表结构
    columns_to_add = [
        ("artist", "TEXT"),
        ("normalized_title", "TEXT"),
        ("source", "TEXT DEFAULT 'llm_prior'"),
        ("confidence", "REAL DEFAULT 0.5"),
        ("user_verified", "INTEGER DEFAULT 0"),
        ("raw_metadata", "TEXT"),
        ("created_at", "TEXT"),
        ("updated_at", "TEXT"),
    ]

    c.execute("PRAGMA table_info(bgm_library)")
    existing_cols = [row[1] for row in c.fetchall()]

    for col_name, col_type in columns_to_add:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE bgm_library ADD COLUMN {col_name} {col_type}")

    # 3. 预置基础标签字典
    c.executemany(
        "INSERT OR IGNORE INTO tags (name, type) VALUES (?, ?)", schema.DEFAULT_TAGS
    )


def _migration_v3(conn):
    """V3 迁移：新增向量索引注册表。"""
    c = conn.cursor()
    c.execute(schema.SQL_CREATE_VECTOR_INDEX_REGISTRY)


def _migration_v4(conn):
    """V4 迁移：新写入不再把 embedding BLOB 存入 subtitles 表。"""
    # 旧库里的 embedding 列保留不动，避免 destructive migration。
    # 新库 schema 已移除该列，插入逻辑会用显式列名兼容两种表结构。
    return None


def _migration_v5(conn):
    """V5 迁移：为影片主表增加可追溯的 TMDB 内容元数据列。"""
    c = conn.cursor()
    c.execute(schema.SQL_CREATE_MOVIES_META)
    columns_to_add = [
        ("media_key", "TEXT"),
        ("media_type", "TEXT"),
        ("tmdb_id", "INTEGER"),
        ("imdb_id", "TEXT"),
        ("original_title", "TEXT"),
        ("aliases_json", "TEXT"),
        ("overview", "TEXT"),
        ("tagline", "TEXT"),
        ("genres_json", "TEXT"),
        ("keywords_json", "TEXT"),
        ("certification", "TEXT"),
        ("certification_country", "TEXT"),
        ("adult", "INTEGER"),
        ("original_language", "TEXT"),
        ("origin_countries_json", "TEXT"),
        ("spoken_languages_json", "TEXT"),
        ("release_date", "TEXT"),
        ("runtime_minutes", "INTEGER"),
        ("status", "TEXT"),
        ("first_air_date", "TEXT"),
        ("last_air_date", "TEXT"),
        ("number_of_seasons", "INTEGER"),
        ("number_of_episodes", "INTEGER"),
        ("in_production", "INTEGER"),
        ("networks_json", "TEXT"),
        ("poster_path", "TEXT"),
        ("backdrop_path", "TEXT"),
        ("homepage", "TEXT"),
        ("tmdb_metadata_json", "TEXT"),
        ("extra_metadata_json", "TEXT DEFAULT '{}'"),
        ("metadata_source", "TEXT"),
        ("metadata_schema_version", "INTEGER"),
        ("tmdb_language", "TEXT"),
        ("tmdb_region", "TEXT"),
        ("tmdb_fetched_at", "TEXT"),
        ("created_at", "TEXT"),
        ("updated_at", "TEXT"),
    ]
    c.execute("PRAGMA table_info(movies_meta)")
    existing_cols = {row[1] for row in c.fetchall()}
    for col_name, col_type in columns_to_add:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE movies_meta ADD COLUMN {col_name} {col_type}")
    c.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_movies_meta_media_key "
        "ON movies_meta(media_key) WHERE media_key IS NOT NULL"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_movies_meta_tmdb "
        "ON movies_meta(media_type, tmdb_id)"
    )
