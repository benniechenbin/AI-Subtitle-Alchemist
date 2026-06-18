import os
import sqlite3
from src.config import settings
from src.config.settings import DEFAULT_DB_PATH, ensure_data_dir

def _resolve_db_path(db_path=None) -> str:
    if db_path:
        return str(db_path)
    return settings.prefs.db_path or str(DEFAULT_DB_PATH)

def get_db_connection(db_path=None):
    target_path = _resolve_db_path(db_path)
    ensure_data_dir()
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    conn = sqlite3.connect(target_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn
