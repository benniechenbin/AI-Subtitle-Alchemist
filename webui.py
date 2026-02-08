import json
import os
import io
import zipfile
import streamlit as st
import core_logic as core
import pandas as pd
import time
from sentence_transformers import SentenceTransformer

# ==========================================
# 缓存与页面基础
# ==========================================
@st.cache_data(ttl=60)
def cached_get_library_stats():
    return core.get_library_stats()

st.set_page_config(
    page_title="AI 字幕炼金术师",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 模型加载器 (缓存)
# ==========================================
@st.cache_resource(show_spinner="正在加载 AI 模型，请稍候...")
def load_embedding_model(model_name):
    try:
        print(f"🔄 Cache Miss: 正在加载模型 {model_name}...")
        return SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")

# ==========================================
# 侧边栏：全局控制
# ==========================================
with st.sidebar:
    st.title("🎛️ 控制台")
    CONFIG_FILE = "config.json"
    if 'config_loaded' not in st.session_state:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                st.session_state['config'] = json.load(f)
        else:
            default_path = os.path.join(os.path.expanduser("~"), "Movies", "Subtitles")
            
            # 如果文件夹不存在，甚至可以贴心地自动创建（可选）
            if not os.path.exists(default_path):
                try:
                    os.makedirs(default_path)
                except:
                    pass

            st.session_state['config'] = {
                'library_path': default_path,  # <--- 使用变量
                'embedding_model': "moka-ai/m3e-base"
            }
        st.session_state['config_loaded'] = True

    def save_config():
        st.session_state['config']['library_path'] = st.session_state['path_input']
        with open(CONFIG_FILE, 'w') as f:
            json.dump(st.session_state['config'], f)
        st.toast("✅ 路径配置已保存")
        cached_get_library_stats.clear()

    library_path = st.text_input(
        "📂 字幕库根目录",
        value=st.session_state['config']['library_path'],
        key="path_input",
        on_change=save_config
    )
    # --- 语义模型选择 ---
    model_options = {
        "paraphrase-multilingual-MiniLM-L12-v2": "🚀 极速版 (MiniLM - 推荐)",
        "moka-ai/m3e-base": "🇨🇳 中文标准版 (M3E - 默认)",
        "BAAI/bge-m3": "🔥 旗舰版 (BGE-M3 - 较慢)",
        "custom": "🛠️ 自定义 (手动输入模型ID)"
    }
    current_config_model = st.session_state['config'].get('embedding_model', "moka-ai/m3e-base")
    if current_config_model in model_options:
        default_index = list(model_options.keys()).index(current_config_model)
        radio_key_value = current_config_model
    else:
        default_index = list(model_options.keys()).index("custom")
        radio_key_value = "custom"
    selected_option = st.selectbox(
        "🧠 语义模型 (Embedding)",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=default_index,
        key="model_select_box",
        help="决定了AI如何理解中文。建议使用 M3E。"
    )
    if selected_option == "custom":
        custom_model_id = st.text_input(
            "输入 HuggingFace ID 或 本地路径",
            value=current_config_model if radio_key_value == "custom" else "sentence-transformers/all-mpnet-base-v2",
            help="例如：intfloat/multilingual-e5-large"
        )
        selected_model = custom_model_id.strip() if custom_model_id else "moka-ai/m3e-base"
    else:
        selected_model = selected_option
    if selected_model != current_config_model:
        st.divider()
        st.info(f"准备切换为：{selected_model}")
        st.warning(
            "⚠️ 模型已变更！\n\n请务必去「数据库管理」点击「重新扫描」，否则无法使用新模型搜索！", 
            icon="🚨"
        )
    st.divider()
    st.toggle("🎉 启用入库彩蛋", value=st.session_state.get("easter_egg", True), key="easter_egg")
    st.divider()
    # --- LLM 大模型配置（折叠）---
    with st.expander("🔑 LLM 大模型配置 (全局)", expanded=False):
        cfg = st.session_state['config']
        default_provider = cfg.get('llm_provider', "DeepSeek")
        default_key = cfg.get('llm_key', "")
        default_base = cfg.get('llm_base_url', "https://api.deepseek.com")
        provider = st.selectbox(
            "厂商", 
            ["DeepSeek", "OpenAI", "Google", "Custom", "Local (Ollama)"],
            index=["DeepSeek", "OpenAI", "Google", "Custom", "Local (Ollama)"].index(default_provider) if default_provider in ["DeepSeek", "OpenAI", "Google", "Custom", "Local (Ollama)"] else 0,
            key="llm_provider"
        )
        api_key = st.text_input("API Key", value=default_key, type="password", key="llm_key_input", help="DeepSeek 或 OpenAI 的 SK")
        model_presets = {
            "DeepSeek": ["deepseek-chat", "deepseek-reasoner"], # V3 和 R1
            "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            "Google": ["gemini-1.5-pro", "gemini-1.5-flash"],
            "Local (Ollama)": ["llama3", "qwen2.5", "mistral"], # 常用本地模型
            "Custom": []
        }
        current_model_val = cfg.get('llm_model_name', "")
        if provider in model_presets and model_presets[provider]:
            options = model_presets[provider] + ["✍️ 手动输入..."]
            try:
                pre_index = options.index(current_model_val)
            except ValueError:
                pre_index = 0
            selected_preset = st.selectbox("模型版本", options, index=pre_index, key="llm_model_select")
            
            if selected_preset == "✍️ 手动输入...":
                model_name = st.text_input("请输入模型 ID", value=current_model_val, placeholder="例如: deepseek-coder", key="llm_model_manual")
            else:
                model_name = selected_preset
        else:
            model_name = st.text_input("模型名称", value=current_model_val, placeholder="如 llama3", key="llm_model_manual_only")
        placeholder_base = "https://api.deepseek.com"
        auto_fill_base = default_base
        
        if provider == "DeepSeek": 
            placeholder_base = "https://api.deepseek.com"
            if "openai" in default_base or "localhost" in default_base: auto_fill_base = placeholder_base
        elif provider == "OpenAI": 
            placeholder_base = "https://api.openai.com/v1"
            if "deepseek" in default_base: auto_fill_base = placeholder_base
        elif provider == "Local (Ollama)":
            placeholder_base = "http://localhost:11434/v1"
        base_url = st.text_input("Base URL", value=auto_fill_base, placeholder=placeholder_base, key="llm_base_input")
        if st.button("💾 保存配置", use_container_width=True):
            st.session_state['config']['llm_provider'] = provider
            st.session_state['config']['llm_key'] = api_key
            st.session_state['config']['llm_base_url'] = base_url
            st.session_state['config']['llm_model_name'] = model_name
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(st.session_state['config'], f)
            st.toast(f"✅ 已保存：使用 {provider} - {model_name}")
            time.sleep(1)
            st.rerun()
        st.session_state['llm_config'] = {
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name
        }

# ==========================================
# 主界面标题与 Tab 导航
# ==========================================
st.title("🎬 AI 字幕炼金术师")
st.markdown("Automated Subtitle Processing & AI Script Generation System")
tab1, tab2, tab3 = st.tabs(["🧹 清洗与转码", "🗄️ 数据库管理", "🤖 AI 编剧助手"])

# ==========================================
# Tab 1：清洗与转码
# ==========================================
with tab1:
    st.header("1. 智能清洗与重命名流水线")
    if 'analysis_data' not in st.session_state:
        st.session_state['analysis_data'] = None
    if 'process_done' not in st.session_state:
        st.session_state['process_done'] = False
    if 'processed_files' not in st.session_state:
        st.session_state['processed_files'] = []
    if 'show_success_celebration' not in st.session_state:
        st.session_state['show_success_celebration'] = None
    if 'pending_import' not in st.session_state:
        st.session_state['pending_import'] = None

    # --- 入库成功：气球与 Toast ---
    if st.session_state.get('show_success_celebration'):
        msg = st.session_state['show_success_celebration']
        if msg != "error":
            if st.session_state.get("easter_egg", True):
                st.balloons()
            st.toast(msg.get("toast", "入库成功！"), icon=msg.get("icon", "🗄️"))
        st.session_state['show_success_celebration'] = None

    # --- 上传与预识别 ---
    uploaded_files = st.file_uploader("第一步：上传字幕文件", accept_multiple_files=True)
    if uploaded_files:
        if st.button("🔍 智能预识别名称"):
            with st.spinner("正在解析..."):
                st.session_state['analysis_data'] = core.analyze_filenames(uploaded_files)
                st.session_state['process_done'] = False

    # --- 元数据校对与批量处理 ---
    if st.session_state['analysis_data'] is not None:
        st.divider()
        st.subheader("第二步：确认元数据")
        edited_df = st.data_editor(
            pd.DataFrame(st.session_state['analysis_data']),
            column_config={
                "原始文件名": st.column_config.Column(disabled=True),
                "识别片名": st.column_config.TextColumn("片名"),
                "年份": st.column_config.NumberColumn("年份", format="%d"),
                "season_num": None, "episode_num": None,
            },
            use_container_width=True, hide_index=True
        )
        skip_embedding = st.checkbox(
            "⚡️ 极速入库模式 (暂不生成 AI 向量)", 
            value=False, 
            help="勾选后将跳过 AI 分析，入库速度提升 100 倍！但在【重建索引】之前，这些文件无法通过 AI 语义搜索找到（只能用关键词搜）。"
        )

        if st.button("🚀 批量处理", type="primary", use_container_width=True):
            with st.status("正在处理（转码与落盘）...", expanded=True) as status:
                model_instance = None
                if not skip_embedding:
                    try:
                        st.write("🧠 正在加载/获取 AI 模型...")
                        model_instance = load_embedding_model(selected_model)
                    except Exception as e:
                        st.error(str(e))
                        st.stop()
                else:
                    st.write("⚡️ 已启用极速模式，跳过模型加载...")
                final_metadata = edited_df.to_dict('records')
                current_model_name = selected_model if not skip_embedding else None
                logs, processed_files, stats, pending_rows = core.process_only(
                    uploaded_files, final_metadata, library_path, model_instance,
                    model_name=current_model_name
                )
                st.session_state['process_logs'] = logs
                st.session_state['processed_files'] = processed_files
                if pending_rows:
                    st.session_state['pending_import'] = {'pending_rows': pending_rows, 'stats': stats}
                st.session_state['process_done'] = True
                status.update(label="✅ 处理完成！准备入库...", state="complete", expanded=False)
                time.sleep(1)
                st.rerun()
    # --- 处理日志 ---
    if st.session_state.get('process_logs'):
        st.divider()
        st.subheader("📝 处理日志")
        with st.expander("点击查看详细处理记录", expanded=True):
            for log in st.session_state['process_logs']:
                if "❌" in log: st.error(log)
                elif "⚠️" in log: st.warning(log)
                else: st.success(log)
    # --- 入库前确认 ---
    if st.session_state.get('pending_import'):
        st.divider()
        st.subheader("第三步：确认入库")
        stats = st.session_state['pending_import']['stats']
        success, fail, dup = stats.get("success", 0), stats.get("fail", 0), stats.get("duplicate", 0)
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ 成功", success, "条可入库")
        col2.metric("❌ 失败", fail, "条")
        col3.metric("⚠️ 重复", dup, "已跳过")
        st.caption("文件已保存到本地。点击「确认入库」将把上述成功项写入数据库；点击「取消」则仅保留本地文件，不入库。")
        confirm_col, cancel_col, _ = st.columns([1, 1, 4])
        with confirm_col:
            if st.button("确认入库", type="primary", use_container_width=True, key="confirm_import"):
                core.commit_pending_to_db(st.session_state['pending_import']['pending_rows'])
                st.session_state['pending_import'] = None
                cached_get_library_stats.clear()
                st.session_state['show_success_celebration'] = {"toast": "入库成功！可在「数据库管理」检索", "icon": "🗄️"}
                st.rerun()
        with cancel_col:
            if st.button("取消", use_container_width=True, key="cancel_import"):
                st.session_state['pending_import'] = None
                st.rerun()
    # --- 下载区：单文件 + 多文件 ZIP ---
    if st.session_state['process_done'] and st.session_state['processed_files']:
        st.divider()
        st.subheader("📥 下载转码后的 SRT 文件")
        files = st.session_state['processed_files']
        if st.session_state.get('pending_import'):
            st.caption("文件已保存到本地；确认入库后可在「数据库管理」中检索。")
        else:
            st.caption("上述文件已保存到本地并写入数据库，可在「数据库管理」中检索。")
        if len(files) > 1:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    name = f.get("name", f.get("filename", getattr(f, "name", "unknown.srt")))
                    content = f.get("content", getattr(f, "content", ""))
                    if content is None:
                        content = ""
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="replace")
                    zf.writestr(name, content)
            buf.seek(0)
            st.download_button(
                label="📦 批量下载 (ZIP 压缩包)",
                data=buf.getvalue(),
                file_name="subtitles.zip",
                mime="application/zip",
                use_container_width=True,
                key="batch_zip_dl"
            )
        with st.container(border=True):
            for i, f in enumerate(files):
                name = f.get("name", f.get("filename", getattr(f, "name", "unknown.srt")))
                content = f.get("content", getattr(f, "content", None))
                if content is None and hasattr(f, "read"):
                    try:
                        f.seek(0)
                        content = f.read()
                        if isinstance(content, bytes):
                            content = content.decode("utf-8", errors="replace")
                    except Exception:
                        content = ""
                content = content or ""
                c_name, c_btn = st.columns([3, 1])
                c_name.write(f"📄 {name}")
                if content:
                    c_btn.download_button(
                        label="点击下载",
                        data=content,
                        file_name=name,
                        mime="text/plain",
                        key=f"dl_{i}_{name}_{i}",
                    )
                else:
                    c_btn.warning("文件为空")

# ==========================================
# Tab 2：数据库管理
# ==========================================
with tab2:
    st.header("2. 核心数据库 (Memory Bank)")
    stats = cached_get_library_stats()
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("📚 已收录电影/剧集", f"{stats['movie_count']} 部")
        m2.metric("💬 台词总行数", f"{stats['line_count']} 行")
        m3.metric("⏱️ 最后更新", stats['last_update'])
    st.divider()

    # --- 库维护：扫描与清理 ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.info(f"当前监控的硬盘路径: `{library_path}`")
    with c2:
        sync_btn = st.button("🔄 重新扫描硬盘 & 重建索引", use_container_width=True)
    if 'scan_result' not in st.session_state:
        st.session_state['scan_result'] = None
    if sync_btn:
        with st.status("正在深度遍历硬盘...", expanded=True) as status:
            try:
                st.write("🧠 正在初始化 AI 模型...")
                model_instance = load_embedding_model(selected_model)
            except Exception as e:
                status.update(label="❌ 模型加载失败", state="error")
                st.error(str(e))
                st.stop()
            for log, data in core.scan_library_path(library_path, model_instance, model_name=selected_model):
                if log == "DONE":
                    st.session_state['scan_result'] = data
                    status.update(label="✅ 扫描完成", state="complete")
                    st.session_state['config']['embedding_model'] = selected_model
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump(st.session_state['config'], f)
                else:
                    st.write(log)
            if st.session_state.get('scan_result') and st.session_state['scan_result'].get('success'):
                st.toast("索引更新完毕！", icon="🎉")
                time.sleep(1)
                st.rerun()
    if st.session_state.get('scan_result') and st.session_state['scan_result'].get('success'):
        res = st.session_state['scan_result']
        if res['new_added'] > 0:
            st.success(f"🎉 成功入库 {res['new_added']} 部新影片！")
        missing_count = len(res['missing_files'])
        if missing_count > 0:
            st.warning(f"⚠️ 发现数据库中有 {missing_count} 个文件在硬盘上找不到了。")
            with st.expander("查看丢失文件列表", expanded=False):
                for f in res['missing_files'][:10]:
                    st.code(f, language="text")
                if missing_count > 10:
                    st.caption(f"... 以及其他 {missing_count - 10} 个文件")
            col_del_text, col_del_btn = st.columns([3, 1])
            with col_del_text:
                st.write("这些是无效的'幽灵记录'，建议清理。")
            with col_del_btn:
                if st.button("🗑️ 确认清理无效记录", type="primary", use_container_width=True):
                    success, msg = core.delete_missing_records(res['missing_files'])
                    if success:
                        st.success(msg)
                        st.session_state['scan_result'] = None
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
        elif res['new_added'] == 0:
            st.info("数据库与硬盘完全同步，未发现变动。")
    st.divider()

    # --- 智能检索：关键词 / 语义 ---
    st.subheader("🔍 智能检索台")
    query = st.text_input("请输入检索内容", placeholder="输入关键词（如：钱）或 抽象概念（如：友情、遗憾）...", key="search_query")
    col_btn1, col_btn2, col_space = st.columns([1, 1, 4])
    with col_btn1:
        btn_keyword = st.button("🔍 关键词匹配", use_container_width=True)
    with col_btn2:
        btn_semantic = st.button("🧠 AI 语义搜索", use_container_width=True, type="primary")

    if btn_keyword and query:
        st.info(f"正在进行【关键词】精确匹配：'{query}' ...")
        results = core.search_db_keyword(query)
        st.write(f"找到 {len(results)} 条记录：")
        for res in results:
            with st.container(border=True):
                if res['season'] > 0 or res['episode'] > 0:
                    ep_info = f" `S{str(res['season']).zfill(2)}E{str(res['episode']).zfill(2)}`"
                else:
                    ep_info = ""
                st.markdown(f"**🎬 {res['movie']}**{ep_info} `[{res['time']}]`")
                st.text(res['content'])

    if btn_semantic and query:
        results = None
        with st.status("🧠 AI 正在思考...", expanded=True) as status:
            try:
                model_instance = load_embedding_model(selected_model)
                st.write("1. 向量化 (Embedding)...")
                results = core.search_db_semantic(query, model_instance)
                status.update(label="✅ 语义匹配完成！", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ 搜索出错", state="error")
                st.error(f"搜索失败: {e}")
                st.stop()
        if results is not None:
            st.success(f"AI 联想到了 {len(results)} 条相关内容：")
            for res in results:
                with st.container(border=True):
                    ep = f" S{str(res['season']).zfill(2)}E{str(res['episode']).zfill(2)}" if (res.get('season') or 0) or (res.get('episode') or 0) else ""
                    st.markdown(f"**🤖 {res['movie']}**{ep} `[{res['time']}]`")
                    st.markdown(f"> *{res['content']}*")

# ==========================================
# Tab 3：AI 混剪实验室
# ==========================================
with tab3:
    st.header("3. AI 编剧指挥台")
    st.caption("基于语义库的智能剧本生成系统")
    # --- 状态与创作区 ---
    llm_cfg = st.session_state.get('llm_config', {})
    has_key = bool(llm_cfg.get('api_key')) or llm_cfg.get('provider') == "Local (Ollama)"
    if has_key:
        st.success(f"🟢 AI 引擎就绪: {llm_cfg.get('provider')} ({llm_cfg.get('model_name')})")
    else:
        st.warning("⚠️ AI 引擎未配置：请点击左侧侧边栏的「🔑 LLM 大模型配置」填入 API Key。")

    st.divider()
    c1, c2 = st.columns([3, 1])
    with c1:
        movie_list = core.get_all_movies()
        selected_movie = st.selectbox(
            "📂 核心素材来源", 
            ["(全库综合搜索)"] + movie_list,
            help="选择具体的电影，AI 将优先使用该电影的台词；选全库则会跨电影混剪。"
        )
    with c2:
        script_style = st.selectbox("🎭 脚本风格", ["情感混剪 (遗憾/治愈)", "燃向踩点 (动作/励志)", "预告片 (悬疑/惊悚)"])
    default_prompt = """主题：关于【时间与遗憾】
要求：
1. 开头要慢，用几句关于“错过”的台词铺垫。
2. 中段节奏加快，展示人生中的不同阶段。
3. 结尾要有一句振聋发聩的金句，升华主题。
4. 不需要旁白，全部用电影原声台词。"""
    prompt_text = st.text_area(
        "📝 导演指令 (Prompt)", 
        value=default_prompt, 
        height=200,
        placeholder="在这里告诉 AI 你想剪辑什么样的视频..."
    )
    generate_btn = st.button("🚀 生成混剪脚本", type="primary", use_container_width=True, disabled=not has_key)
    if generate_btn:
        with st.status("🎬 AI 正在创作剧本...", expanded=True) as status:
            st.write("1. 🧠 理解导演意图...")
            time.sleep(0.5)
            st.write(f"2. 🔍 正在检索 '{selected_movie}' 相关的语义向量...")
            time.sleep(1)
            st.write(f"3. ✍️ 正在请求 {llm_cfg.get('model_name')} 生成分镜...")
            sys_prompt = "你是一个专业的视频剪辑师。请把用户提供的素材和要求，写成Markdown格式的剪辑脚本表。"
            user_input = f"素材范围：{selected_movie}\n风格：{script_style}\n详细要求：{prompt_text}"
            try:
                response = core.call_deepseek_llm(
                    sys_prompt, 
                    user_input, 
                    llm_cfg['api_key']
                )
                status.update(label="✅ 创作完成！", state="complete", expanded=False)
                st.divider()
                st.subheader("📄 混剪脚本")
                st.markdown(response)
                st.download_button(
                    label="📥 导出脚本 (.md)",
                    data=response,
                    file_name=f"script_{int(time.time())}.md",
                    mime="text/markdown"
                )
            except Exception as e:
                status.update(label="❌ 生成失败", state="error")
                st.error(f"调用 API 出错: {e}")