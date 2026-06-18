import time
import pandas as pd
import streamlit as st

from app.bootstrap import get_effective_api_key
from app.cached import cached_embedding_model, cached_library_stats
from src import db
from src.config import settings
from src.models import ScanDoneResult
from src.services.scan_service import scan_library
from src.services.search import search_semantic
from src.services.tmdb_service import fetch_tmdb_poster
from src.services.highlight_service import HighlightService
from src.observability.logger import logger


def render_database_tab(ctx: dict) -> None:
    library_path = ctx["library_path"]
    embedding_model = ctx["embedding_model"]
    db_path = settings.prefs.db_path

    # 1. 顶部统计面板
    stats = cached_library_stats(db_path)
    c1, c2, c3, c4 = st.columns([1, 1, 1.5, 1.2])
    with c1:
        st.metric("🎬 电影总数", f"{stats['movie_count']}")
    with c2:
        st.metric("💬 核心台词", f"{stats['line_count']}")
    with c3:
        st.metric("📅 最后同步", stats["last_update"])
    with c4:
        st.markdown('<div style="margin-top: 18px;"></div>', unsafe_allow_html=True)
        rebuild_btn = st.button("🔄 全盘重建索引", width="stretch", type="primary")

    if rebuild_btn:
        _run_rebuild_index(library_path, embedding_model, db_path)

    st.divider()

    # 2. 🎬 核心亮点：智能海报墙看板
    st.subheader("🖼️ 素材库海报墙")
    
    # 每次渲染时先做一次弱同步，保证新增的电影能进 meta 表
    db.sync_movies_to_meta(db_path)
    movies_list = db.get_movies_with_meta(db_path)

    if not movies_list:
        st.info("💡 库里空空如也，请先去「清洗入库」或点击上方「重建索引」吧！")
    else:
        # 每行雷打不动展示 4 部影视剧
        cols_per_row = 4
        for i in range(0, len(movies_list), cols_per_row):
            row_movies = movies_list[i : i + cols_per_row]
            cols = st.columns(cols_per_row)
            
            for col, movie in zip(cols, row_movies):
                with col:
                    with st.container(border=True):
                        # 如果没有海报 URL，使用现代主义灰色方块占位
                        if movie["poster_url"]:
                            st.image(movie["poster_url"], width='stretch')
                        else:
                            st.markdown(
                                """
                                <div style="background-color: #262730; height: 260px; border-radius: 5px; 
                                display: flex; align-items: center; justify-content: center; color: #808495; 
                                font-size: 14px; margin-bottom: 10px; border: 1px dashed #434651;">
                                🎬 暂无海报数据
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # 电影文本元数据展示
                        st.markdown(f"**{movie['movie_name']}**")
                        year_str = f" ({movie['release_year']})" if movie['release_year'] else ""
                        st.caption(f"📅 年份: {year_str} | 💬 台词: {movie['line_count']} 条")
                        st.divider()
                        # 如果缺失海报，提供动态补取按钮
                        if not movie["poster_url"]:
                            if st.button("🖼️ 抓取海报", key=f"poster_{movie['movie_name']}", width="stretch"):
                                with st.spinner("逆向抓取中..."):
                                    poster_url = fetch_tmdb_poster(movie["movie_name"])
                                    if poster_url:
                                        db.update_movie_poster(db_path, movie["movie_name"], poster_url)
                                        st.toast(f"🎉 海报抓取成功！")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("未找到海报。")

                        # 渲染金句挖掘按钮状态机
                        status = movie.get("highlight_status", 0)
                        if status == 0 or status == 3:
                            btn_label = "🪄 提炼金句" if status == 0 else "❌ 挖掘失败，点击重试"
                            if st.button(btn_label, key=f"hl_{movie['movie_name']}", width="stretch"):
                                with st.spinner("AI 正在逐帧通读全片台词，可能需要 1-3 分钟，请稍候..."):
                                    try:
                                        HighlightService.extract_movie_highlights(
                                            movie["movie_name"], 
                                            api_key=get_effective_api_key()
                                        )
                                        st.toast("🎉 金句提炼完成！")
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"挖掘失败: {e}")
                        elif status == 1:
                            st.button("⏳ 疯狂挖掘中...", key=f"hl_ing_{movie['movie_name']}", disabled=True, width="stretch")
                        elif status == 2:
                            if st.button("✨ 查看金句", key=f"hl_done_{movie['movie_name']}", width="stretch", type="primary"):
                                _show_quotes_dialog(movie["movie_name"], db_path)

        st.divider()

    # 3. 库内检索区域（保持原有逻辑不变）
    st.subheader("🔍 库内检索")
    search_col, btn_col1, btn_col2 = st.columns([3, 0.6, 0.6])
    with search_col:
        query = st.text_input(
            "输入关键词...",
            placeholder="例如：遗憾 或者 我爱你",
            label_visibility="collapsed",
        )
    with btn_col1:
        precise_search = st.button("🔎 精确查找", width="stretch")
    with btn_col2:
        semantic_search = st.button("🧠 AI 语义", width="stretch")

    search_results = None
    if precise_search and query:
        search_results = db.search_keyword(db_path, query)

    if semantic_search and query:
        with st.spinner("AI 正在理解意图..."):
            model = cached_embedding_model(embedding_model)
            search_results = search_semantic(
                query,
                model,
                db_path,
                embedding_model_name=embedding_model,
                final_k=20,
                allow_duplicates=True,
            )

    if search_results is not None:
        _render_search_results(search_results)

    # 4. 🧪 资产模型调试面板 (Phase 2.5)
    st.divider()
    with st.expander("🧪 标签系统调试面板 (资产模型验证)", expanded=False):
        _render_debug_tag_panel(db_path)

def _render_debug_tag_panel(db_path: str):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. 标签字典概览")
        tag_type = st.selectbox("标签维度", ["theme", "emotion", "scene", "usage", "style"])
        tags = db.get_tags_by_type(db_path, tag_type)
        if tags:
            st.write(pd.DataFrame(tags))
        else:
            st.info("该维度暂无标签")
    
    with c2:
        st.markdown("#### 2. 手动关联测试")
        movies = db.get_all_movies(db_path)
        if movies:
            sel_movie = st.selectbox("选择电影", movies, key="debug_sel_movie")
            all_tags = []
            for ttype in ["theme", "emotion", "scene", "usage", "style"]:
                all_tags.extend(db.get_tags_by_type(db_path, ttype))
            
            tag_options = {f"{t['name']} ({t['type']})": t['id'] for t in all_tags}
            sel_tag_label = st.selectbox("选择标签", list(tag_options.keys()), key="debug_sel_tag")
            tag_id = tag_options[sel_tag_label]
            
            if st.button("🔗 关联标签", type="primary"):
                db.link_tag_to_movie(db_path, sel_movie, tag_id)
                st.toast(f"✅ 已将 {sel_tag_label} 关联至 {sel_movie}")
            
            st.markdown("---")
            st.markdown(f"**《{sel_movie}》已有标签：**")
            movie_tags = db.get_movie_tags(db_path, sel_movie)
            if movie_tags:
                tag_str = " ".join([f"`{mt['name']}`" for mt in movie_tags])
                st.markdown(tag_str)
            else:
                st.caption("暂无标签")
        else:
            st.info("库中暂无电影，无法测试关联。")

@st.dialog("✨ 黄金台词大赏")
def _show_quotes_dialog(movie_name: str, db_path: str):
    st.markdown(f"### 《{movie_name}》")
    quotes = db.get_golden_quotes(db_path, movie_name)
    
    if not quotes:
        st.info("大模型未能从该片中提取出符合标准的金句，或提取过程出错。")
        return
        
    st.success(f"共为您提炼出 {len(quotes)} 条直击灵魂的台词。")
    for q in quotes:
        with st.container(border=True):
            st.markdown(f"#### 「 {q['content']} 」")
            st.caption(f"⏱️ 出现时间: `{q['timestamp']}`")
            st.markdown(f"💡 **推荐理由**: {q['reason']}")


def _run_rebuild_index(library_path: str, embedding_model: str, db_path: str) -> None:
    with st.status("🚀 正在启动扫描引擎...", expanded=True) as status:
        st.write("🧠 正在加载 AI 模型...")
        try:
            model = cached_embedding_model(embedding_model)
            st.write(f"📂 开始扫描目录: {library_path}")

            for log, data in scan_library(
                library_path, model, embedding_model, db_path
            ):
                if log == "DONE" and isinstance(data, ScanDoneResult):
                    status.update(
                        label=f"🎉 处理完成！新增 {data.new_added} 个文件",
                        state="complete",
                        expanded=False,
                    )
                    if data.missing_files:
                        st.error(
                            f"发现 {len(data.missing_files)} 个文件已从硬盘移除。"
                        )
                        if st.button("🧹 立即清理无效记录"):
                            db.delete_records_by_path(db_path, data.missing_files)
                            st.rerun()
                else:
                    st.write(log)

            cached_library_stats.clear()
            time.sleep(1)
            st.rerun()
        except Exception as e:
            status.update(label="❌ 发生错误", state="error")
            st.error(f"扫描中断: {str(e)}")


def _render_search_results(search_results: list[dict]) -> None:
    if len(search_results) > 0:
        st.success(f"找到 {len(search_results)} 条结果：")
        df_res = pd.DataFrame(search_results)
        display_cols = {
            "movie": "电影",
            "season": "季",
            "episode": "集",
            "time": "时间",
            "content": "台词内容",
            "score": "匹配度",
        }
        actual_cols = [c for c in display_cols if c in df_res.columns]
        st.dataframe(
            df_res[actual_cols].rename(columns=display_cols),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("未找到匹配的内容。")
