import time

import streamlit as st

from app.bootstrap import save_preferences
from src.config import settings
from src.services.harvester_api import HarvesterApiClient, HarvesterApiError


def render_sidebar() -> dict:
    """渲染侧边栏，返回当前运行上下文。"""
    prefs = settings.prefs
    st.title("🎛️ 控制台")

    def save_path():
        save_preferences({"library_path": st.session_state["path_input"]})
        st.toast("✅ 路径已保存")

    library_path = st.text_input(
        "📂 本地字幕库",
        value=prefs.library_path,
        key="path_input",
        on_change=save_path,
    )

    st.divider()
    harvester_api_base_url, harvester_health = _render_harvester_config()

    st.divider()
    model_options = {
        "paraphrase-multilingual-MiniLM-L12-v2": "🚀 极速版 (MiniLM - 推荐)",
        "moka-ai/m3e-base": "🇨🇳 中文标准版 (M3E - 默认)",
        "BAAI/bge-m3": "🔥 旗舰版 (BGE-M3 - 较慢)",
        "custom": "🛠️ 自定义 (手动输入模型ID)",
    }

    cur_mod = prefs.embedding_model
    default_idx = (
        list(model_options.keys()).index(cur_mod) if cur_mod in model_options else 3
    )

    selected_option = st.selectbox(
        "🧠 语义模型",
        list(model_options.keys()),
        index=default_idx,
        format_func=lambda x: model_options.get(x, x),
        help="决定了AI如何理解中文。建议使用 M3E。",
    )

    if selected_option == "custom":
        sel_mod = st.text_input(
            "输入模型 ID 或本地路径",
            value=cur_mod if cur_mod not in model_options else "",
        )
    else:
        sel_mod = selected_option

    if sel_mod != cur_mod and sel_mod != "":
        st.divider()
        st.warning(
            "⚠️ 模型已变更！请去「数据库管理」点击「重建索引」，否则搜索将失效！",
            icon="🚨",
        )
        save_preferences({"embedding_model": sel_mod})

    st.divider()
    st.toggle(
        "🎉 启用入库彩蛋",
        value=st.session_state.get("easter_egg", True),
        key="easter_egg",
    )

    st.divider()
    with st.expander("🔑 LLM 大模型配置", expanded=False):
        _render_llm_config()

    return {
        "library_path": library_path,
        "embedding_model": sel_mod,
        "harvester_api_base_url": harvester_api_base_url,
        "harvester_health": harvester_health,
    }


def _render_harvester_config() -> tuple[str, dict | None]:
    prefs = settings.prefs

    def save_harvester_url():
        save_preferences(
            {"harvester_api_base_url": st.session_state["harvester_api_input"]}
        )
        st.toast("✅ Harvester API 已保存")

    with st.expander("🌾 字幕采集服务", expanded=False):
        base_url = st.text_input(
            "Harvester API",
            value=prefs.harvester_api_base_url,
            key="harvester_api_input",
            on_change=save_harvester_url,
        )

        health = None
        try:
            health = HarvesterApiClient(base_url).health()
            st.success("✅ Harvester 已连接")
            output_dir = health.get("output_dir")
            if output_dir:
                st.caption(f"output_dir: `{output_dir}`")
        except HarvesterApiError as exc:
            st.warning(f"未连接 Harvester：{exc}")

    return base_url, health


def _render_llm_config() -> None:
    prefs = settings.prefs
    providers = ["DeepSeek", "OpenAI", "Google", "Custom", "Local (Ollama)"]
    default_provider = prefs.llm_provider
    provider = st.selectbox(
        "厂商",
        providers,
        index=providers.index(default_provider) if default_provider in providers else 0,
    )

    env_key = settings.get_llm_api_key()
    if env_key:
        st.success("✅ 已从 .env 加载 API Key")
        key_placeholder = "••••••••"
    else:
        st.info("💡 未在当前系统中检测到 API Key，请在下方填写以进行临时会话：")
        key_placeholder = "在此输入临时 API Key..."

    api_key = st.text_input(
        "API Key (优先使用 .env)",
        value=st.session_state.get("llm_key", ""),
        type="password",
        placeholder=key_placeholder,
    )
    if api_key:
        st.session_state["llm_key"] = api_key

    model_presets = {
        "DeepSeek": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "Google": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "Local (Ollama)": ["llama3", "qwen2.5", "mistral"],
        "Custom": [],
    }

    current_model_val = prefs.llm_model_name

    if provider in model_presets and model_presets[provider]:
        options = model_presets[provider] + ["✍️ 手动输入..."]
        try:
            pre_index = options.index(current_model_val)
        except ValueError:
            pre_index = 0
        selected_preset = st.selectbox("模型版本", options, index=pre_index)
        model_name = (
            st.text_input("请输入模型 ID", value=current_model_val)
            if selected_preset == "✍️ 手动输入..."
            else selected_preset
        )
    else:
        model_name = st.text_input("模型名称", value=current_model_val)

    default_base = prefs.llm_base_url
    auto_fill_base = default_base
    placeholder_base = "https://api.deepseek.com"

    if provider == "DeepSeek":
        placeholder_base = "https://api.deepseek.com"
        if "openai" in default_base or "googleapis" in default_base:
            auto_fill_base = placeholder_base
    elif provider == "OpenAI":
        placeholder_base = "https://api.openai.com/v1"
        if "deepseek" in default_base:
            auto_fill_base = placeholder_base
    elif provider == "Google":
        placeholder_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
        if "deepseek" in default_base or "openai" in default_base:
            auto_fill_base = placeholder_base
    elif provider == "Local (Ollama)":
        placeholder_base = "http://localhost:11434/v1"
        if "api" in default_base:
            auto_fill_base = placeholder_base

    base_url = st.text_input(
        "Base URL", value=auto_fill_base, placeholder=placeholder_base
    )

    if st.button("💾 保存配置", width="stretch"):
        save_preferences(
            {
                "llm_provider": provider,
                "llm_base_url": base_url,
                "llm_model_name": model_name,
            }
        )
        st.toast(f"✅ 已切换至 {provider}")
        time.sleep(1)
        st.rerun()
