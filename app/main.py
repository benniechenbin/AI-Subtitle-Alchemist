import streamlit as st

from app.bootstrap import init_app
from app.sidebar import render_sidebar
from app.tabs import (
    render_clean_import_tab,
    render_database_tab,
    render_script_writer_tab,
    render_bgm_library_tab, 
)


def run() -> None:
    init_app()

    with st.sidebar:
        ctx = render_sidebar()

    st.title("🎬 AI 字幕炼金术师 V2.0")
    tab1, tab2, tab3, tab4 = st.tabs(["🧹 清洗入库", "🗄️ 数据库", "🤖 AI 编剧", "🎵 BGM 音乐库"])

    with tab1:
        render_clean_import_tab(ctx)
    with tab2:
        render_database_tab(ctx)
    with tab3:
        render_script_writer_tab(ctx)
    with tab4:
        render_bgm_library_tab(ctx)


if __name__ == "__main__":
    run()
