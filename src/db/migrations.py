import time
from src.observability.logger import logger
from src.db.base import get_db_connection
from . import schema

def init_db(db_path=None):
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        
        # 0. 建立迁移版本控制表
        c.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)")
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
            
            c.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (1, ?)", (time.strftime("%Y-%m-%d %H:%M:%S"),))
            conn.commit()
            current_version = 1

        # ================= V2 资产模型升级 =================
        if current_version < 2:
            logger.info("🚀 正在升级数据库至 V2 (资产模型升级)...")
            _migration_v2(conn)
            c.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (2, ?)", (time.strftime("%Y-%m-%d %H:%M:%S"),))
            conn.commit()
            logger.info("✅ 数据库已成功升级至 V2")
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
        ("updated_at", "TEXT")
    ]
    
    c.execute("PRAGMA table_info(bgm_library)")
    existing_cols = [row[1] for row in c.fetchall()]
    
    for col_name, col_type in columns_to_add:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE bgm_library ADD COLUMN {col_name} {col_type}")

    # 3. 预置基础标签字典
    c.executemany("INSERT OR IGNORE INTO tags (name, type) VALUES (?, ?)", schema.DEFAULT_TAGS)
