from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.services.harvester_api import HarvesterApiClient, HarvesterApiError

KEY_CANDIDATE_ROWS = "harvester_candidate_rows"
KEY_CANDIDATE_SOURCE_PATH = "harvester_candidate_source_path"
KEY_LAST_DISCOVERY = "harvester_last_discovery"
KEY_LAST_COLLECTION = "harvester_last_collection"
KEY_MANIFEST_ROWS = "harvester_manifest_rows"
KEY_LAST_IMPORT = "harvester_last_import"

COUNTRIES = {
    "全部": None,
    "中国": "CN",
    "日本": "JP",
    "韩国": "KR",
    "美国": "US",
    "中国香港": "HK",
    "中国台湾": "TW",
    "泰国": "TH",
    "印度": "IN",
}
LANGUAGES = {
    "全部": None,
    "中文": "zh",
    "日语": "ja",
    "韩语": "ko",
    "英语": "en",
}
SORTS = {
    "热度": "popularity.desc",
    "上映日期": "primary_release_date.desc",
    "评分": "vote_average.desc",
    "投票数": "vote_count.desc",
}
MEDIA_TYPES = {
    "全部": "all",
    "电影": "movie",
    "剧集": "tv",
}


def render_harvester_discovery_tab(ctx: dict) -> None:
    st.header("候选发现")
    client = _client(ctx)

    with st.form("harvester_discovery_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            year = st.number_input(
                "年份",
                min_value=1874,
                max_value=2100,
                value=datetime.now().year,
            )
        with col2:
            month_label = st.selectbox(
                "月份",
                ["全年", *[str(item) for item in range(1, 13)]],
            )
        with col3:
            media_label = st.selectbox("媒体类型", list(MEDIA_TYPES.keys()))
        with col4:
            max_pages = st.selectbox("最大页数", [1, 2, 3, 5], index=2)

        col5, col6, col7 = st.columns(3)
        with col5:
            country_label = st.selectbox("国家/地区", list(COUNTRIES.keys()))
        with col6:
            language_label = st.selectbox("原始语言", list(LANGUAGES.keys()))
        with col7:
            sort_label = st.selectbox("排序", list(SORTS.keys()))

        col8, col9, col10, col11 = st.columns(4)
        with col8:
            min_vote_count = st.number_input("最低投票数", min_value=0, value=0)
        with col9:
            min_runtime = st.number_input("最低片长", min_value=0, value=0)
        with col10:
            update_state = st.checkbox("更新采集记录", value=True)
        with col11:
            force_refresh = st.checkbox("强制刷新", value=False)

        submitted = st.form_submit_button("开始发现", type="primary")

    if submitted:
        payload = {
            "year": int(year),
            "month": None if month_label == "全年" else int(month_label),
            "media_type": MEDIA_TYPES[media_label],
            "max_pages": int(max_pages),
            "force_refresh": force_refresh,
            "update_state": update_state,
            "origin_country": COUNTRIES[country_label],
            "original_language": LANGUAGES[language_label],
            "sort_by": SORTS[sort_label],
            "min_vote_count": _none_if_zero(min_vote_count),
            "min_runtime": _none_if_zero(min_runtime),
        }
        _run_discovery(client, payload)

    _render_discovery_result()

    st.divider()
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("刷新候选", width="stretch"):
            _load_candidates(client)

    if KEY_CANDIDATE_ROWS not in st.session_state:
        _load_candidates(client, quiet=True)

    _render_candidate_editor(client)


def render_harvester_collection_tab(ctx: dict) -> None:
    st.header("字幕搜索与下载")
    client = _client(ctx)
    default_input = st.session_state.get(KEY_CANDIDATE_SOURCE_PATH, "")

    with st.form("harvester_collection_form"):
        input_text = st.text_input(
            "候选输入路径",
            value=str(default_input or ""),
            placeholder="留空时使用 Harvester 最新候选",
        )
        provider_labels = st.multiselect(
            "字幕源",
            ["Assrt", "SubDL"],
            default=["Assrt", "SubDL"],
        )

        st.markdown("#### 本次处理范围")
        col_scope1, col_scope2 = st.columns(2)
        with col_scope1:
            process_all = st.checkbox("处理全部候选", value=False)
        with col_scope2:
            limit = st.number_input(
                "最多处理候选数",
                min_value=1,
                value=20,
                disabled=process_all,
            )

        st.markdown("#### 请求频率")
        col_rate1, col_rate2 = st.columns(2)
        with col_rate1:
            assrt_rate = st.selectbox(
                "Assrt 请求频率",
                [5, 10, 20],
                index=0,
                format_func=lambda value: f"{value} 次/分钟",
            )
        with col_rate2:
            subdl_rate_option = st.selectbox(
                "SubDL 请求频率",
                ["不限制", "20"],
                index=0,
                format_func=lambda value: (
                    f"{value} 次/分钟" if value != "不限制" else value
                ),
            )

        overwrite = st.checkbox("覆盖已有缓存", value=False)
        submitted = st.form_submit_button("开始采集", type="primary")

    if submitted:
        if not provider_labels:
            st.error("请至少选择一个字幕源。")
            return

        payload = {
            "providers": provider_labels,
            "process_all": process_all,
            "limit": None if process_all else int(limit),
            "overwrite": overwrite,
            "assrt_requests_per_minute": int(assrt_rate),
            "subdl_requests_per_minute": (
                None if subdl_rate_option == "不限制" else int(subdl_rate_option)
            ),
            "input_path": input_text.strip() or None,
        }
        try:
            with st.spinner("正在搜索并下载字幕..."):
                result = client.collect_subtitles(payload)
            st.session_state[KEY_LAST_COLLECTION] = result
            st.success("字幕采集完成")
        except HarvesterApiError as exc:
            st.error(str(exc))

    _render_collection_result()


def render_harvester_import_tab(ctx: dict) -> None:
    st.header("导入字幕库")
    client = _client(ctx)
    library_path = _library_dir(ctx)
    st.caption(f"本地字幕库：`{library_path}`")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("刷新列表", width="stretch"):
            _load_manifest_rows(client)

    if KEY_MANIFEST_ROWS not in st.session_state:
        _load_manifest_rows(client, quiet=True)

    rows = st.session_state.get(KEY_MANIFEST_ROWS, [])
    if not rows:
        st.info("还没有可导入的字幕 manifest。")
        return

    st.dataframe(rows, width="stretch", hide_index=True)
    labels = [_manifest_label(row) for row in rows]
    selected_label = st.selectbox("查看 manifest", labels)
    selected_row = rows[labels.index(selected_label)]
    _render_manifest_detail(client, selected_row)

    st.divider()
    select_all = st.checkbox("默认选择全部 manifest", value=False)
    default_selection = labels if select_all else [selected_label]
    selected_labels = st.multiselect(
        "选择要导入本地字幕库的 manifest",
        labels,
        default=default_selection,
    )

    if st.button("导入到本地字幕库", type="primary"):
        manifest_paths = [
            str(rows[labels.index(label)]["manifest_path"])
            for label in selected_labels
        ]
        if not manifest_paths:
            st.error("请至少选择一个 manifest。")
            return

        try:
            with st.spinner("正在导入本地字幕库..."):
                result = client.export_library(
                    manifest_paths=manifest_paths,
                    library_dir=library_path,
                )
            st.session_state[KEY_LAST_IMPORT] = result
            st.success(f"已导入 {result.get('imported_count', 0)} 个字幕文件")
            st.info("导入完成后，请到「数据库」页点击「全盘重建索引」。")
        except HarvesterApiError as exc:
            st.error(str(exc))

    if st.session_state.get(KEY_LAST_IMPORT):
        st.write(st.session_state[KEY_LAST_IMPORT])


def _client(ctx: dict) -> HarvesterApiClient:
    return HarvesterApiClient(ctx["harvester_api_base_url"])


def _library_dir(ctx: dict) -> str:
    return str(Path(ctx["library_path"]).expanduser().resolve(strict=False))


def _run_discovery(client: HarvesterApiClient, payload: dict[str, Any]) -> None:
    try:
        with st.spinner("正在发现候选..."):
            result = client.run_discovery(payload)
        st.session_state[KEY_LAST_DISCOVERY] = result
        st.success("候选发现完成")
        _load_candidates(client)
    except HarvesterApiError as exc:
        st.error(str(exc))


def _render_discovery_result() -> None:
    result = st.session_state.get(KEY_LAST_DISCOVERY)
    if not result:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("候选总数", result.get("total_candidates", 0))
    col2.metric("新增候选", result.get("new_candidates", 0))
    col3.metric("本轮候选", result.get("candidate_count", 0))
    col4.metric("已更新记录", "是" if result.get("state_updated") else "否")

    with st.expander("发现结果", expanded=False):
        st.write(
            {
                "snapshot_path": result.get("snapshot_path"),
                "batch_path": result.get("batch_path"),
                "state_path": result.get("state_path"),
            }
        )
        _render_logs(result)


def _load_candidates(client: HarvesterApiClient, quiet: bool = False) -> None:
    try:
        result = client.list_candidates()
        st.session_state[KEY_CANDIDATE_ROWS] = result.get("rows", [])
        st.session_state[KEY_CANDIDATE_SOURCE_PATH] = result.get("source_path")
        if result.get("warnings") and not quiet:
            for warning in result["warnings"]:
                st.warning(warning)
    except HarvesterApiError as exc:
        if not quiet:
            st.error(str(exc))


def _render_candidate_editor(client: HarvesterApiClient) -> None:
    rows = st.session_state.get(KEY_CANDIDATE_ROWS, [])
    source_path = st.session_state.get(KEY_CANDIDATE_SOURCE_PATH)
    if not rows:
        st.info("还没有可编辑的候选文件。")
        return

    st.caption(f"候选来源：`{source_path}`")
    edited_df = st.data_editor(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        key="harvester_candidate_editor",
        column_config={
            "keep": st.column_config.CheckboxColumn("保留"),
            "priority": st.column_config.NumberColumn("优先级", step=1),
            "note": st.column_config.TextColumn("备注"),
            "tmdb_metadata": st.column_config.TextColumn("TMDB metadata", disabled=True),
        },
    )

    if st.button("固化候选列表", type="primary"):
        try:
            result = client.curate_candidates(_records(edited_df))
            st.success(f"已固化 {result.get('count', 0)} 条候选")
            _load_candidates(client)
        except HarvesterApiError as exc:
            st.error(str(exc))


def _render_collection_result() -> None:
    result = st.session_state.get(KEY_LAST_COLLECTION)
    if not result:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("成功", result.get("succeeded", 0))
    col2.metric("失败", result.get("failed", 0))
    col3.metric("跳过", result.get("skipped", 0))
    col4.metric("缓存", result.get("cached", 0))

    st.write(
        {
            "input_path": result.get("input_path"),
            "output_dir": result.get("output_dir"),
            "total_candidates": result.get("total_candidates"),
            "processed_candidates": result.get("processed_candidates"),
        }
    )
    _render_logs(result)

    rows = []
    for item in result.get("results", []):
        rows.append(
            {
                "title": item.get("title"),
                "year": item.get("year"),
                "provider": item.get("provider"),
                "status": item.get("status"),
                "search_results_count": item.get("search_results_count"),
                "downloadable_count": item.get("downloadable_count"),
                "error_message": item.get("error_message"),
                "subtitle_files": "\n".join(item.get("subtitle_files") or []),
                "media_dir": item.get("media_dir"),
            }
        )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)


def _load_manifest_rows(client: HarvesterApiClient, quiet: bool = False) -> None:
    try:
        result = client.list_manifests()
        st.session_state[KEY_MANIFEST_ROWS] = result.get("manifests", [])
        if result.get("warnings") and not quiet:
            for warning in result["warnings"]:
                st.warning(warning)
    except HarvesterApiError as exc:
        if not quiet:
            st.error(str(exc))


def _render_manifest_detail(client: HarvesterApiClient, row: dict[str, Any]) -> None:
    media_key = str(row.get("media_key") or "")
    if not media_key:
        return

    try:
        detail = client.get_manifest(media_key)
    except HarvesterApiError as exc:
        st.error(str(exc))
        return

    st.markdown("#### Manifest")
    st.json(detail.get("payload", {}))
    subtitle_files = detail.get("subtitle_files") or []
    if subtitle_files:
        st.markdown("#### 字幕文件")
        for subtitle_file in subtitle_files:
            st.write(str(subtitle_file))


def _manifest_label(row: dict[str, Any]) -> str:
    return " | ".join(
        str(value or "")
        for value in (
            row.get("media_key"),
            row.get("title"),
            row.get("year"),
            row.get("provider"),
        )
    )


def _render_logs(result: dict[str, Any]) -> None:
    logs = result.get("logs") or []
    warnings = result.get("warnings") or []
    if not logs and not warnings:
        return

    with st.expander("日志", expanded=False):
        for warning in warnings:
            st.warning(warning)
        for log in logs:
            st.write(log)


def _records(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        raw_records = value.to_dict("records")
    elif isinstance(value, list):
        raw_records = value
    else:
        return []
    return [_clean_record(record) for record in raw_records if isinstance(record, dict)]


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float) and pd.isna(value):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def _none_if_zero(value: int | float) -> int | None:
    parsed = int(value)
    return parsed if parsed > 0 else None
