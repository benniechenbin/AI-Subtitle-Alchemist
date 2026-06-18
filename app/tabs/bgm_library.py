import time
import pandas as pd
import streamlit as st

from app.bootstrap import get_effective_api_key
from src import db
from src.config import settings
from src.services.bgm_service import BgmService

@st.dialog("⚙️ 确认并微调 BGM 元数据")
def _edit_and_confirm_dialog(analysis_result: dict, db_path: str):
    st.markdown(f"### 🎵 {analysis_result['normalized_title']}")
    st.caption(f"艺术家: {analysis_result['artist']} | AI 置信度: {analysis_result['confidence']:.2f}")
    
    # 允许用户直接在弹窗里滑动修改大模型给出的数值
    c1, c2 = st.columns(2)
    with c1:
        new_title = st.text_input("曲目名称", value=analysis_result['normalized_title'])
        new_artist = st.text_input("艺术家", value=analysis_result['artist'])
        new_valence = st.slider("情绪 (0=致郁, 1=治愈)", min_value=0.0, max_value=1.0, value=analysis_result['valence'], step=0.05)
    with c2:
        new_tempo = st.number_input("BPM 节拍", value=analysis_result['tempo'], step=1)
        new_energy = st.slider("能量 (0=安静, 1=高燃)", min_value=0.0, max_value=1.0, value=analysis_result['energy'], step=0.05)
        # 将列表转为逗号分隔的字符串供用户编辑
        tags_str = ", ".join(analysis_result['tags'])
        new_tags_str = st.text_input("剪辑标签 (逗号分隔)", value=tags_str)

    if st.button("💾 确认无误，立即入库", type="primary", width="stretch"):
        final_data = {
            "track_name": analysis_result['track_name'],
            "normalized_title": new_title,
            "artist": new_artist,
            "valence": new_valence,
            "energy": new_energy,
            "tempo": new_tempo,
            "tags": [t.strip() for t in new_tags_str.split(",") if t.strip()],
            "source": analysis_result.get('source', 'llm_prior'),
            "confidence": analysis_result.get('confidence', 1.0),
            "user_verified": 1,
            "raw_metadata": analysis_result.get('raw_metadata', "")
        }
        db.insert_or_update_bgm(db_path, final_data)
        st.session_state.pop("pending_bgm_result", None)
        st.toast(f"✅ 《{final_data['normalized_title']}》已收录至 BGM 弹药库！")
        time.sleep(1)
        st.rerun()

def render_bgm_library_tab(ctx: dict) -> None:
    st.header("🎧 音乐资产库")
    db_path = settings.prefs.db_path

    # 1. 顶部录入区
    st.subheader("1. 录入新 BGM")
    c_input, c_btn = st.columns([4, 1])
    with c_input:
        bgm_query = st.text_input("输入歌曲名称 (建议带上艺术家或出处)", placeholder="例如：Hans Zimmer - Time (盗梦空间)", label_visibility="collapsed")
    with c_btn:
        analyze_btn = st.button("✨ AI 听感分析", width="stretch", type="primary")

    if analyze_btn and bgm_query:
        with st.spinner(f"AI 正在提取《{bgm_query}》的声学特征..."):
            result = BgmService.analyze_bgm(bgm_query, api_key=get_effective_api_key())
            if result:
                # 不要直接调弹窗，而是把结果存入状态
                st.session_state["pending_bgm_result"] = result
            else:
                st.error("大模型分析失败，请检查网络或控制台日志。")

    # 在按钮逻辑之外，只要状态里有数据，就保持弹窗开启
    if st.session_state.get("pending_bgm_result"):
        _edit_and_confirm_dialog(st.session_state["pending_bgm_result"], db_path)

    st.divider()

    # 2. 底部展示区：音乐资产数据网格
    st.subheader("2. 本地 BGM 弹药库")
    bgm_list = db.get_all_bgm(db_path)
    
    if not bgm_list:
        st.info("💡 你的 BGM 弹药库目前空空如也，试着录入第一首歌吧！")
    else:
        df = pd.DataFrame(bgm_list)
        # 优化数据表格的展示列名
        display_df = df.rename(columns={
            "track_name": "🎶 原始名称",
            "artist": "👨‍🎨 艺术家",
            "valence": "情绪",
            "energy": "能量",
            "tempo": "节拍",
            "tags": "🏷️ 标签",
            "source": "来源",
            "confidence": "置信度",
            "user_verified": "验证",
            "updated_at": "更新时间"
        })
        # 格式化展示
        display_df["验证"] = display_df["验证"].apply(lambda x: "✅" if x == 1 else "🤖")
        st.dataframe(display_df, width='stretch', hide_index=True)
