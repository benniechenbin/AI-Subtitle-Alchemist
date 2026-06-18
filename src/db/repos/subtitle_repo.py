import os
from src.db.base import get_db_connection

def insert_subtitles_batch(db_path, data_rows):
    if not data_rows:
        return
    conn = get_db_connection(db_path)
    try:
        c = conn.cursor()
        c.executemany(
            "INSERT INTO subtitles VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            data_rows,
        )
        conn.commit()
    finally:
        conn.close()

def fetch_vectors_for_search(db_path, target_movie=None, embedding_model=None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        sql = "SELECT id, movie_name, season, episode, start_time, content, embedding FROM subtitles WHERE 1=1"
        args = []
        if target_movie:
            sql += " AND movie_name = ?"
            args.append(target_movie)
        if embedding_model:
            sql += " AND embedding_model = ?"
            args.append(embedding_model)
        cursor.execute(sql, tuple(args))
        return cursor.fetchall()
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
    c = conn.cursor()
    c.execute(
        "SELECT movie_name, season, episode, start_time, content FROM subtitles WHERE content LIKE ? LIMIT 50",
        (f"%{keyword}%",),
    )
    rows = c.fetchall()
    conn.close()
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

def check_file_exists(db_path, file_hash):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    c.execute("SELECT movie_name FROM subtitles WHERE file_hash = ?", (file_hash,))
    exists = c.fetchone()
    conn.close()
    return exists

def get_all_file_paths(db_path):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    c.execute("SELECT file_path FROM subtitles")
    files = {os.path.normpath(row[0]) for row in c.fetchall()}
    conn.close()
    return files

def delete_records_by_path(db_path, paths):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    for p in paths:
        c.execute("DELETE FROM subtitles WHERE file_path = ?", (p,))
    conn.commit()
    conn.close()
    return True
