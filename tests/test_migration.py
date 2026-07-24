import os
import sqlite3

from src.db import get_db_connection, init_db


def test_migration_empty_db():
    print("--- Testing: Migration from Empty DB ---")
    test_db = "data/test_empty.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    # 模拟从 0 开始初始化
    init_db(test_db)

    conn = get_db_connection(test_db)
    c = conn.cursor()

    # 验证版本
    c.execute("SELECT MAX(version) FROM schema_migrations")
    version = c.fetchone()[0]
    print(f"Final Version: {version}")
    assert version == 5

    c.execute("PRAGMA table_info(movies_meta)")
    movie_columns = {row[1] for row in c.fetchall()}
    assert {
        "media_key",
        "tmdb_id",
        "overview",
        "genres_json",
        "certification",
        "tmdb_metadata_json",
        "extra_metadata_json",
    } <= movie_columns

    # 验证新表是否存在
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tags'")
    assert c.fetchone() is not None
    print("Table 'tags' exists.")

    # 验证预置标签
    c.execute("SELECT COUNT(*) FROM tags")
    tag_count = c.fetchone()[0]
    print(f"Pre-populated tags: {tag_count}")
    assert tag_count >= 28

    conn.close()
    os.remove(test_db)
    print("SUCCESS: Empty DB migration passed.\n")


def test_migration_v1_upgrade():
    print("--- Testing: Migration from V1 to V2 ---")
    test_db = "data/test_v1_upgrade.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    conn = sqlite3.connect(test_db)
    c = conn.cursor()

    # 手动创建 V1 结构的最小子集
    c.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
    )
    c.execute(
        "CREATE TABLE bgm_library (id INTEGER PRIMARY KEY AUTOINCREMENT, track_name TEXT UNIQUE, valence REAL, energy REAL, tempo INTEGER, tags TEXT)"
    )
    c.execute("INSERT INTO bgm_library (track_name, valence) VALUES ('old_song', 0.8)")
    c.execute(
        "INSERT INTO schema_migrations (version, applied_at) VALUES (1, '2026-01-01')"
    )
    conn.commit()
    conn.close()

    # 执行 init_db 触发升级
    init_db(test_db)

    conn = get_db_connection(test_db)
    c = conn.cursor()

    # 验证版本
    c.execute("SELECT MAX(version) FROM schema_migrations")
    version = c.fetchone()[0]
    print(f"Upgraded to Version: {version}")
    assert version == 5

    # 验证旧数据保留
    c.execute(
        "SELECT track_name, valence, artist FROM bgm_library WHERE track_name='old_song'"
    )
    row = c.fetchone()
    print(f"Data integrity check: {row}")
    assert row[0] == "old_song"
    assert row[1] == 0.8
    assert row[2] is None  # 新列默认值为 None

    conn.close()
    os.remove(test_db)
    print("SUCCESS: V1 to V2 upgrade passed.\n")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    try:
        test_migration_empty_db()
        test_migration_v1_upgrade()
        print("🎉 ALL MIGRATION TESTS PASSED")
    except Exception as e:  # noqa: BLE001
        print(f"❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
