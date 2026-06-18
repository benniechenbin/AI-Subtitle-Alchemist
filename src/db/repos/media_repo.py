import time
from src.db.base import get_db_connection
from src.observability.logger import logger

def get_all_movies(db_path=None):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    c.execute("SELECT DISTINCT movie_name FROM subtitles ORDER BY movie_name")
    movies = [row[0] for row in c.fetchall()]
    conn.close()
    return movies

def get_library_stats(db_path=None):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(DISTINCT movie_name) FROM subtitles")
        m_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM subtitles")
        l_count = c.fetchone()[0]
        return {
            "movie_count": m_count,
            "line_count": l_count,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logger.error(f"⚠️ 统计数据获取失败: {e}")
        return {"movie_count": 0, "line_count": 0, "last_update": "从未"}
    finally:
        conn.close()

def sync_movies_to_meta(db_path=None):
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT OR IGNORE INTO movies_meta (movie_name, release_year)
            SELECT movie_name, MAX(year) 
            FROM subtitles 
            WHERE movie_name IS NOT NULL
            GROUP BY movie_name
        """)
        conn.commit()
    finally:
        conn.close()

def get_movies_with_meta(db_path=None) -> list[dict]:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT 
                m.movie_name, 
                m.poster_url, 
                m.release_year, 
                m.highlight_status,
                COUNT(s.id) as line_count
            FROM movies_meta m
            LEFT JOIN subtitles s ON m.movie_name = s.movie_name
            GROUP BY m.movie_name
            ORDER BY line_count DESC
        """)
        columns = ['movie_name', 'poster_url', 'release_year', 'highlight_status', 'line_count']
        return [dict(zip(columns, row)) for row in c.fetchall()]
    finally:
        conn.close()

def update_movie_poster(db_path: str, movie_name: str, poster_url: str) -> None:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute(
            "UPDATE movies_meta SET poster_url = ? WHERE movie_name = ?",
            (poster_url, movie_name)
        )
        conn.commit()
    finally:
        conn.close()
