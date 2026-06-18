import streamlit as st

from src import db
from src.config import settings
from src.config.settings import LOGS_DIR, migrate_legacy_db
from src.observability.logger import setup_logger


def init_app() -> None:
    st.set_page_config(page_title="AI 字幕炼金术师", page_icon="🎬", layout="wide")

    migrate_legacy_db()
    settings.migrate_legacy_config()

    if "db_inited" not in st.session_state:
        db.init_db(settings.prefs.db_path)
        st.session_state["db_inited"] = True

    if "logger_inited" not in st.session_state:
        setup_logger(log_dir=LOGS_DIR, log_level="INFO")
        st.session_state["logger_inited"] = True


def save_preferences(updates: dict) -> None:
    settings.save_preferences(updates)


def get_effective_api_key() -> str:
    """获取最终生效的 API Key：优先从 .env，其次从 st.session_state (侧边栏输入)"""
    return settings.get_llm_api_key() or st.session_state.get("llm_key", "")
