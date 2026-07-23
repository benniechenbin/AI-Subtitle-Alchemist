from src.db.base import get_db_connection


def get_tags_by_type(db_path: str, tag_type: str) -> list[dict]:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute(
            "SELECT id, name, type FROM tags WHERE type = ? ORDER BY name ASC",
            (tag_type,),
        )
        columns = ["id", "name", "type"]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    finally:
        conn.close()


def add_tag(db_path: str, name: str, tag_type: str) -> int:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT OR IGNORE INTO tags (name, type) VALUES (?, ?)", (name, tag_type)
        )
        conn.commit()
        c.execute("SELECT id FROM tags WHERE name = ? AND type = ?", (name, tag_type))
        row = c.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to insert or retrieve tag: {name}/{tag_type}")
        return row[0]
    finally:
        conn.close()


def link_tag_to_movie(
    db_path: str,
    movie_name: str,
    tag_id: int,
    confidence: float = 1.0,
    source: str = "manual",
):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT OR REPLACE INTO movie_tags (movie_name, tag_id, confidence, source)
            VALUES (?, ?, ?, ?)
        """,
            (movie_name, tag_id, confidence, source),
        )
        conn.commit()
    finally:
        conn.close()


def link_tag_to_quote(
    db_path: str,
    quote_id: int,
    tag_id: int,
    confidence: float = 1.0,
    source: str = "llm",
):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT OR REPLACE INTO quote_tags (quote_id, tag_id, confidence, source)
            VALUES (?, ?, ?, ?)
        """,
            (quote_id, tag_id, confidence, source),
        )
        conn.commit()
    finally:
        conn.close()


def get_movie_tags(db_path: str, movie_name: str) -> list[dict]:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute(
            """
            SELECT t.name, t.type, mt.confidence, mt.source
            FROM tags t
            JOIN movie_tags mt ON t.id = mt.tag_id
            WHERE mt.movie_name = ?
        """,
            (movie_name,),
        )
        columns = ["name", "type", "confidence", "source"]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    finally:
        conn.close()


def link_tag_to_subtitle(
    db_path: str,
    subtitle_id: int,
    tag_id: int,
    confidence: float = 1.0,
    source: str = "manual",
):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT OR REPLACE INTO subtitle_tags (subtitle_id, tag_id, confidence, source)
            VALUES (?, ?, ?, ?)
        """,
            (subtitle_id, tag_id, confidence, source),
        )
        conn.commit()
    finally:
        conn.close()


def get_subtitle_tags(db_path: str, subtitle_id: int) -> list[dict]:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute(
            """
            SELECT t.name, t.type, st.confidence, st.source
            FROM tags t
            JOIN subtitle_tags st ON t.id = st.tag_id
            WHERE st.subtitle_id = ?
        """,
            (subtitle_id,),
        )
        columns = ["name", "type", "confidence", "source"]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    finally:
        conn.close()


def search_subtitles_by_tags(db_path: str, tag_ids: list[int]) -> list[dict]:
    if not tag_ids:
        return []

    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        placeholders = ",".join(["?"] * len(tag_ids))
        c.execute(
            f"""
            SELECT s.id, s.movie_name, s.season, s.episode, s.start_time, s.content
            FROM subtitles s
            JOIN subtitle_tags st ON s.id = st.subtitle_id
            WHERE st.tag_id IN ({placeholders})
            GROUP BY s.id
            HAVING COUNT(DISTINCT st.tag_id) = ?
        """,
            (*tag_ids, len(tag_ids)),
        )

        columns = ["id", "movie_name", "season", "episode", "start_time", "content"]
        return [dict(zip(columns, row)) for row in c.fetchall()]
    finally:
        conn.close()
