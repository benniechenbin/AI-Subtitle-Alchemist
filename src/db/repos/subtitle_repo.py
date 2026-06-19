import os
from src.db.base import get_db_connection

def insert_subtitles_batch(db_path, data_rows):
    if not data_rows:
        return []
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        c.executemany(
            """
            INSERT INTO subtitles (
                file_hash, file_path, movie_name, year, season, episode,
                line_index, start_time, end_time, content, embedding_model, embedding_dim
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            data_rows,
        )
        last_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        count = len(data_rows)
        return list(range(last_id - count + 1, last_id + 1))
    finally:
        conn.close()



def fetch_subtitles_by_ids(db_path, subtitle_ids):
    if not subtitle_ids:
        return []
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        placeholders = ",".join("?" for _ in subtitle_ids)
        cursor.execute(
            f"""
            SELECT id, movie_name, season, episode, start_time, content
            FROM subtitles
            WHERE id IN ({placeholders})
            """,
            tuple(subtitle_ids),
        )
        rows_by_id = {row[0]: row for row in cursor.fetchall()}
        return [rows_by_id[item_id] for item_id in subtitle_ids if item_id in rows_by_id]
    finally:
        conn.close()



def iter_subtitles_for_vector_rebuild(db_path, batch_size=10000):
    last_id = 0
    limit = max(1, int(batch_size))
    while True:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT id, content
                FROM subtitles
                WHERE content IS NOT NULL
                  AND content != ''
                  AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (last_id, limit),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            break
        yield rows
        last_id = int(rows[-1][0])

def update_subtitle_embedding_metadata_batch(db_path, updates):
    if not updates:
        return
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        c.executemany(
            """
            UPDATE subtitles
            SET embedding_model = ?, embedding_dim = ?
            WHERE id = ?
            """,
            updates,
        )
        conn.commit()
    finally:
        conn.close()

def get_context_by_id(db_path, center_id, movie_name, window=1):
    from src.observability.logger import logger
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        query = """
            SELECT content FROM subtitles
            WHERE id BETWEEN ? AND ? AND movie_name = ?
            ORDER BY id ASC
        """
        cursor.execute(query, (center_id - window, center_id + window, movie_name))
        return " ".join([row[0] for row in cursor.fetchall()])
    except Exception as e:
        logger.error(f"⚠️ 获取字幕上下文失败 (ID:{center_id}): {e}")
        return ""
    finally:
        conn.close()

def search_keyword(db_path, keyword):
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        c.execute(
            "SELECT movie_name, season, episode, start_time, content FROM subtitles WHERE content LIKE ? LIMIT 50",
            (f"%{keyword}%",),
        )
        rows = c.fetchall()
        return [
            {
                "movie": r[0],
                "season": r[1],
                "episode": r[2],
                "time": r[3],
                "content": r[4],
            }
            for r in rows
        ]
    finally:
        conn.close()

def check_file_exists(db_path, file_hash):
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        c.execute("SELECT movie_name FROM subtitles WHERE file_hash = ?", (file_hash,))
        return c.fetchone()
    finally:
        conn.close()

def get_all_file_paths(db_path):
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        c.execute("SELECT file_path FROM subtitles")
        return {os.path.normpath(row[0]) for row in c.fetchall()}
    finally:
        conn.close()

def get_subtitle_ids_by_paths(db_path, paths):
    if not paths:
        return []
    normalized_paths = [os.path.normpath(path) for path in paths]
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        placeholders = ",".join("?" for _ in normalized_paths)
        c.execute(
            f"SELECT id FROM subtitles WHERE file_path IN ({placeholders})",
            tuple(normalized_paths),
        )
        return [row[0] for row in c.fetchall()]
    finally:
        conn.close()

def delete_records_by_path(db_path, paths):
    if not paths:
        return []
    normalized_paths = [os.path.normpath(path) for path in paths]
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        placeholders = ",".join("?" for _ in normalized_paths)
        c.execute(
            f"SELECT id FROM subtitles WHERE file_path IN ({placeholders})",
            tuple(normalized_paths),
        )
        deleted_ids = [row[0] for row in c.fetchall()]
        for p in normalized_paths:
            c.execute("DELETE FROM subtitles WHERE file_path = ?", (p,))
        conn.commit()
        return deleted_ids
    finally:
        conn.close()

