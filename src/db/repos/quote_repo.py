from src.db.base import get_db_connection

def get_golden_quotes(db_path: str, movie_name: str) -> list[dict]:
    conn = get_db_connection(db_path)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT quote_content, timestamp, reason 
            FROM golden_quotes 
            WHERE movie_name = ?
            ORDER BY timestamp ASC
        """, (movie_name,))
        columns = ['content', 'timestamp', 'reason']
        return [dict(zip(columns, row)) for row in c.fetchall()]
    finally:
        conn.close()
