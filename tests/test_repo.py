import os
import time

from src import db


def test_repository_logic():
    print("--- Testing: Repository Logic ---")
    test_db = "data/test_repo.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    db.init_db(test_db)

    # 1. 验证标签新增与查询
    tag_id = db.add_tag(test_db, "测试标签", "theme")
    assert tag_id > 0
    tags = db.get_tags_by_type(test_db, "theme")
    assert any(t["name"] == "测试标签" for t in tags)
    print("Tag logic passed.")

    # 2. 验证 BGM 插入与更新 (created_at 稳定性)
    bgm_data = {
        "track_name": "test_bgm",
        "valence": 0.5,
        "energy": 0.5,
        "tempo": 120,
        "tags": ["tag1"],
    }
    db.insert_or_update_bgm(test_db, bgm_data)

    all_bgm = db.get_all_bgm(test_db)
    original_bgm = next(b for b in all_bgm if b["track_name"] == "test_bgm")
    assert original_bgm["track_name"] == "test_bgm"
    # 注意：get_all_bgm 可能不返回 created_at，我们通过 select 验证
    conn = db.get_db_connection(test_db)
    c = conn.cursor()
    c.execute(
        "SELECT created_at, updated_at FROM bgm_library WHERE track_name='test_bgm'"
    )
    row1 = c.fetchone()
    print(f"BGM Initial: created={row1[0]}, updated={row1[1]}")

    time.sleep(1.1)  # 确保时间戳变更

    # 更新 BGM
    bgm_data["valence"] = 0.9
    db.insert_or_update_bgm(test_db, bgm_data)

    c.execute(
        "SELECT created_at, updated_at FROM bgm_library WHERE track_name='test_bgm'"
    )
    row2 = c.fetchone()
    print(f"BGM Updated: created={row2[0]}, updated={row2[1]}")

    assert row2[0] == row1[0]  # created_at 不变
    assert row2[1] != row1[1]  # updated_at 变了
    print("BGM timestamp logic passed.")

    # 3. 验证电影标签关联
    c.execute(
        "INSERT INTO movies_meta (movie_name, release_year) VALUES ('Test Movie', 2025)"
    )
    conn.commit()
    db.link_tag_to_movie(test_db, "Test Movie", tag_id)
    movie_tags = db.get_movie_tags(test_db, "Test Movie")
    assert any(t["name"] == "测试标签" for t in movie_tags)
    print("Movie tag link passed.")

    conn.close()
    os.remove(test_db)
    print("SUCCESS: Repository logic tests passed.\n")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    try:
        test_repository_logic()
    except Exception as e:  # noqa: BLE001
        print(f"❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
