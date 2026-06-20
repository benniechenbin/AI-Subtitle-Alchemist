import io
import hashlib
import time
import zipfile

import pandas as pd
import streamlit as st

from app.cached import cached_embedding_model, cached_library_stats
from src import db
from src.models import UploadedFileInput
from src.services.import_service import process_uploaded_files
from src.services.upload_manifest_adapter import (
    prepare_upload_analysis,
    split_uploaded_files,
)
from src.services.vector_index import get_vector_index_service
from src.config import settings
from src.observability.logger import logger


def render_clean_import_tab(ctx: dict) -> None:
    st.header("1. 智能清洗流水线")
    library_path = ctx["library_path"]
    embedding_model = ctx["embedding_model"]
    db_path = settings.prefs.db_path

    if "analysis_data" not in st.session_state:
        st.session_state["analysis_data"] = None
    if "process_done" not in st.session_state:
        st.session_state["process_done"] = False
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = []
    if "pending_import" not in st.session_state:
        st.session_state["pending_import"] = None
    if "subtitle_upload_files" not in st.session_state:
        st.session_state["subtitle_upload_files"] = []
    if "tmdb_matches" not in st.session_state:
        st.session_state["tmdb_matches"] = []
    if "upload_signature" not in st.session_state:
        st.session_state["upload_signature"] = None

    uploaded_files = st.file_uploader(
        "第一步：上传字幕文件",
        accept_multiple_files=True,
        key="subtitle_file_uploader",
        on_change=_clear_upload_state,
    )
    upload_signature = _upload_signature(uploaded_files)
    if st.session_state["upload_signature"] != upload_signature:
        st.session_state["upload_signature"] = upload_signature
        _clear_upload_state()

    if uploaded_files:
        subtitle_files, manifest_files = split_uploaded_files(uploaded_files)
        if manifest_files:
            st.info(
                f"已检测到 {len(manifest_files)} 个 Harvester manifest、"
                f"{len(subtitle_files)} 个字幕文件，将优先使用 JSON 中的片名和年份。"
            )
        elif not subtitle_files:
            st.warning("未检测到可处理的字幕文件。")

        if st.button("🔍 智能预识别名称"):
            with st.spinner("正在解析文件名..."):
                upload_analysis = prepare_upload_analysis(uploaded_files)
                for warning in upload_analysis.warnings:
                    st.warning(warning)
                st.session_state["analysis_data"] = _filter_manifest_rows(
                    upload_analysis.analysis_data
                )
                st.session_state["subtitle_upload_files"] = (
                    upload_analysis.subtitle_files
                )
                st.session_state["tmdb_matches"] = upload_analysis.tmdb_matches
                st.session_state["process_done"] = False

    if (
        st.session_state["analysis_data"] is not None
        and not st.session_state.get("pending_import")
    ):
        subtitle_upload_files = st.session_state.get("subtitle_upload_files", [])
        if subtitle_upload_files:
            _render_metadata_editor(
                subtitle_upload_files, library_path, embedding_model, db_path
            )
        else:
            st.warning("未检测到可处理的字幕文件。")

    if st.session_state.get("process_logs"):
        _render_process_logs()

    if st.session_state.get("pending_import"):
        _render_import_confirmation(db_path)

    if st.session_state["process_done"] and st.session_state["processed_files"]:
        _render_download_section()


def _render_metadata_editor(uploaded_files, library_path, embedding_model, db_path):
    st.divider()
    st.session_state["analysis_data"] = _filter_manifest_rows(
        st.session_state["analysis_data"]
    )

    with st.expander("🛠️ 批量改名与年代工具", expanded=False):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            bulk_name = st.text_input("统一修改片名为...", placeholder="例如：葬送的芙莉莲")
        with c2:
            bulk_year = st.number_input(
                "统一修改年份", min_value=1900, max_value=2100, value=2025
            )
        with c3:
            st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
            if st.button("🪄 立即应用", width="stretch"):
                for item in st.session_state["analysis_data"]:
                    if bulk_name:
                        item["识别片名"] = bulk_name
                    item["年份"] = bulk_year
                st.success("已批量更新元数据！")
                st.rerun()

    st.subheader("第二步：确认元数据")
    editor_key = _metadata_editor_key()
    edited_df = st.data_editor(
        pd.DataFrame(st.session_state["analysis_data"]),
        column_config={
            "原始文件名": st.column_config.Column(disabled=True),
            "识别片名": st.column_config.TextColumn("片名"),
            "年份": st.column_config.NumberColumn("年份", format="%d"),
        },
        width="stretch",
        hide_index=True,
        key=editor_key,
    )
    st.caption("修改 TMDB 匹配后的片名或年份，会自动取消该 TMDB 元数据绑定。")

    st.divider()
    col_opt, col_btn = st.columns([2, 1])
    with col_opt:
        skip_embedding = st.checkbox(
            "⚡️ 极速入库模式 (暂不生成 AI 向量)",
            value=False,
            help="勾选后将跳过 AI 分析，入库速度提升 100 倍！但在【重建索引】之前，无法使用 AI 语义搜索。",
        )
    with col_btn:
        start_process = st.button("🚀 批量处理", type="primary", width="stretch")

    if start_process:
        _run_batch_process(
            uploaded_files,
            edited_df,
            library_path,
            embedding_model,
            db_path,
            skip_embedding,
            st.session_state.get("tmdb_matches", []),
        )


def _run_batch_process(
    uploaded_files,
    edited_df,
    library_path,
    embedding_model,
    db_path,
    skip_embedding,
    tmdb_matches,
):
    with st.status("正在处理（转码与落盘）...", expanded=True) as status:
        model_instance = None
        current_model_name = "None"

        if not skip_embedding:
            try:
                st.write("🧠 正在加载/获取 AI 模型...")
                model_instance = cached_embedding_model(embedding_model)
                current_model_name = embedding_model
            except Exception as e:
                st.error(str(e))
                st.stop()
        else:
            st.write("⚡️ 已启用极速模式，跳过模型加载...")

        file_inputs = []
        for f in uploaded_files:
            f.seek(0)
            file_inputs.append(UploadedFileInput(name=f.name, raw_bytes=f.read()))

        (
            logs,
            processed_files,
            stats,
            pending_rows,
            pending_vectors,
            pending_meta,
        ) = process_uploaded_files(
            file_inputs,
            edited_df.to_dict("records"),
            library_path,
            model_instance,
            model_name=current_model_name,
            db_path=db_path,
            tmdb_matches=tmdb_matches,
        )

        st.session_state["process_logs"] = logs
        st.session_state["processed_files"] = processed_files
        if pending_rows:
            st.session_state["pending_import"] = {
                "pending_rows": pending_rows,
                "pending_vectors": pending_vectors,
                "pending_meta": pending_meta,
                "stats": stats,
            }

        st.session_state["process_done"] = True
        status.update(label="✅ 处理完成！准备入库...", state="complete", expanded=False)
        time.sleep(1)
        st.rerun()


def _upload_signature(uploaded_files) -> tuple:
    return tuple(
        (
            getattr(uploaded_file, "name", ""),
            getattr(uploaded_file, "size", None),
        )
        for uploaded_file in (uploaded_files or [])
    )


def _metadata_editor_key() -> str:
    signature = repr(st.session_state.get("upload_signature", ())).encode("utf-8")
    digest = hashlib.md5(signature).hexdigest()[:10]
    return f"metadata_editor_{digest}"


def _clear_upload_state() -> None:
    st.session_state["analysis_data"] = None
    st.session_state["subtitle_upload_files"] = []
    st.session_state["tmdb_matches"] = []
    st.session_state["process_done"] = False
    st.session_state["processed_files"] = []
    st.session_state["pending_import"] = None
    st.session_state["process_logs"] = []


def _filter_manifest_rows(rows):
    if not rows:
        return rows
    return [
        row
        for row in rows
        if str(row.get("原始文件名", "")).replace("\\", "/").rsplit("/", 1)[-1].lower()
        != "harvester_import_manifest.json"
    ]


def _render_process_logs():
    st.divider()
    st.subheader("📝 处理日志")
    with st.expander("点击查看详细处理记录", expanded=True):
        for log in st.session_state["process_logs"]:
            if "❌" in log:
                st.error(log)
            elif "⚠️" in log:
                st.warning(log)
            else:
                st.success(log)


def _render_import_confirmation(db_path):
    st.divider()
    st.subheader("第三步：确认入库")
    stats = st.session_state["pending_import"]["stats"]

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ 成功", stats.get("success", 0), "条")
    c2.metric("❌ 失败", stats.get("fail", 0), "条")
    c3.metric("⚠️ 重复", stats.get("duplicate", 0), "跳过")

    st.caption("文件已保存到本地。点击「确认入库」将写入数据库。")
    confirm_col, cancel_col, _ = st.columns([1, 1, 4])

    with confirm_col:
        if st.button("确认入库", type="primary", width="stretch", key="confirm_import"):
            pending_rows = st.session_state["pending_import"]["pending_rows"]
            pending_vectors = st.session_state["pending_import"].get("pending_vectors", [])
            pending_meta = st.session_state["pending_import"].get("pending_meta", [])

            pending_payload = {
                "pending_rows": pending_rows,
                "pending_vectors": pending_vectors,
                "pending_meta": pending_meta,
            }
            try:
                _inserted_ids, vector_error = _commit_pending_import(
                    db_path, pending_payload
                )
            except Exception as exc:
                logger.exception("Clean import transaction failed: %s", exc)
                st.error("入库失败，本次数据已全部回滚，可安全重试。")
                return

            st.session_state["pending_import"] = None
            if vector_error:
                st.warning("字幕已入库，但向量索引更新失败，请稍后重建索引。")

            cached_library_stats.clear()
            if st.session_state.get("easter_egg", True):
                st.balloons()
            if not vector_error:
                st.toast("入库成功！可在「数据库管理」检索", icon="🗄️")
            time.sleep(2)
            st.rerun()

    with cancel_col:
        if st.button("取消", width="stretch", key="cancel_import"):
            st.session_state["pending_import"] = None
            st.rerun()


def _commit_pending_import(db_path, pending_payload, vector_service=None):
    inserted_ids = db.insert_subtitles_with_metadata_batch(
        db_path,
        pending_payload.get("pending_rows", []),
        pending_payload.get("pending_meta", []),
    )
    service = vector_service or get_vector_index_service()
    try:
        service.upsert_vector_rows(
            db_path,
            inserted_ids,
            pending_payload.get("pending_vectors", []),
        )
    except Exception as exc:
        logger.exception("Vector index update failed after import: %s", exc)
        return inserted_ids, exc
    return inserted_ids, None


def _render_download_section():
    st.divider()
    st.subheader("📥 下载转码后的 SRT")
    files = st.session_state["processed_files"]

    if len(files) > 1:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                name = f.get("name", f.get("filename", "unknown.srt"))
                content = f.get("content", "")
                if content is None:
                    content = ""
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                zf.writestr(name, content)
        buf.seek(0)
        st.download_button(
            "📦 批量下载 (ZIP)",
            data=buf.getvalue(),
            file_name="subtitles.zip",
            mime="application/zip",
            width="stretch",
        )

    with st.container(border=True):
        for i, f in enumerate(files):
            name = f.get("name", "unknown.srt")
            content = f.get("content", "")
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")

            c_name, c_btn = st.columns([3, 1])
            c_name.write(f"📄 {name}")
            if content:
                c_btn.download_button(
                    "下载",
                    data=content,
                    file_name=name,
                    mime="text/plain",
                    key=f"dl_{i}",
                )
