import time
from src.db.base import get_db_connection


def insert_or_update_bgm(db_path: str, bgm_data: dict) -> None:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        c.execute(
            """
            REPLACE INTO bgm_library (
                track_name, artist, normalized_title,
                valence, energy, tempo, tags,
                source, confidence, user_verified,
                raw_metadata, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM bgm_library WHERE track_name=?), ?), ?)
        """,
            (
                bgm_data["track_name"],
                bgm_data.get("artist", ""),
                bgm_data.get("normalized_title", bgm_data["track_name"]),
                bgm_data["valence"],
                bgm_data["energy"],
                bgm_data["tempo"],
                ",".join(bgm_data["tags"])
                if isinstance(bgm_data["tags"], list)
                else bgm_data["tags"],
                bgm_data.get("source", "llm_prior"),
                bgm_data.get("confidence", 0.5),
                bgm_data.get("user_verified", 0),
                bgm_data.get("raw_metadata", ""),
                bgm_data["track_name"],
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_bgm(db_path: str) -> list[dict]:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT
                track_name, artist, valence, energy, tempo, tags,
                source, confidence, user_verified, updated_at
            FROM bgm_library
            ORDER BY id DESC
        """)
        columns = [
            "track_name",
            "artist",
            "valence",
            "energy",
            "tempo",
            "tags",
            "source",
            "confidence",
            "user_verified",
            "updated_at",
        ]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    finally:
        conn.close()
