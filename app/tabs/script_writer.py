import time

import streamlit as st

from app.bootstrap import get_effective_api_key
from app.cached import cached_embedding_model
from src import db
from src.config import settings
from src.models import ScriptGenerationRequest
from src.prompts import DEFAULT_DIRECTOR_PROMPT
from src.services.llm import LLMCallError
from src.services.script_service import ScriptService


def render_script_writer_tab(ctx: dict) -> None:
    st.header("🤖 AI 编剧助手")
    embedding_model = ctx["embedding_model"]
    prefs = settings.prefs

    has_key = bool(get_effective_api_key()) or "ollama" in prefs.llm_base_url.lower()
    if has_key:
        st.success(f"🟢 AI 引擎就绪: {prefs.llm_provider} ({prefs.llm_model_name})")
    else:
        st.warning(
            "⚠️ AI 引擎未配置：请点击左侧侧边栏的「🔑 LLM 大模型配置」填入 API Key。"
        )

    st.divider()

    c1, c2 = st.columns([3, 1])
    movies = db.get_all_movies(prefs.db_path)
    selected_movie = c1.selectbox("核心素材", ["(全库综合)"] + movies)
    script_style = c2.selectbox(
        "风格", ["情感混剪 (遗憾/治愈)", "燃向踩点 (动作/励志)", "预告片 (悬疑/惊悚)"]
    )

    prompt = st.text_area(
        "📝 导演指令 (Prompt)", value=DEFAULT_DIRECTOR_PROMPT, height=200
    )

    col1, _col2 = st.columns([1, 1])
    with col1:
        allow_dup = st.checkbox(
            "允许重复台词 (台词混剪模式)",
            value=False,
            help="混剪模式下，允许同一句台词出现多次。",
        )

    generate_btn = st.button(
        "🚀 生成混剪脚本", type="primary", width="stretch", disabled=not has_key
    )

    if generate_btn:
        _run_script_generation(
            embedding_model=embedding_model,
            selected_movie=selected_movie,
            script_style=script_style,
            prompt=prompt,
            allow_dup=allow_dup,
        )


def _run_script_generation(
    embedding_model: str,
    selected_movie: str,
    script_style: str,
    prompt: str,
    allow_dup: bool,
) -> None:
    prefs = settings.prefs
    with st.status("🎬 AI 正在创作剧本...", expanded=True) as status:
        try:
            st.write("1. 🔍 正在从数据库检索相关台词与时间码...")
            model = cached_embedding_model(embedding_model)
            target = None if selected_movie == "(全库综合)" else selected_movie

            request = ScriptGenerationRequest(
                db_path=prefs.db_path,
                prompt=prompt,
                script_style=script_style,
                target_movie=target,
                allow_duplicates=allow_dup,
                llm_key=get_effective_api_key(),
                llm_model_name=prefs.llm_model_name,
                llm_base_url=prefs.llm_base_url,
                embedding_model=embedding_model,
            )
            st.write(f"2. ✍️ 正在请求 {prefs.llm_model_name} 生成分镜...")
            result = ScriptService.generate(request, model)
            st.write(f"3. 🧠 已组装 {result.material_count} 条素材")
            status.update(label="✅ 创作完成！", state="complete", expanded=False)
            st.divider()
            st.subheader("📄 包含时间码的混剪脚本")
            st.caption(f"共引用 {result.material_count} 条素材")
            st.markdown(result.script)

            st.download_button(
                label="📥 导出脚本 (.md)",
                data=result.script,
                file_name=f"script_{int(time.time())}.md",
                mime="text/markdown",
            )
        except (ValueError, LLMCallError) as e:
            status.update(label="❌ 生成失败", state="error")
            st.error(str(e))
        except Exception as e:
            status.update(label="❌ 生成失败", state="error")
            st.error(f"出错: {str(e)}")
