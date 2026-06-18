import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.services.harvester_manifest import MANIFEST_NAME, normalize_manifest_path
from src.utils import analyze_filenames


VALID_SUBTITLE_EXTS = {".srt", ".ass", ".ssa", ".vtt", ".txt"}


@dataclass
class UploadAnalysisResult:
    subtitle_files: list[Any]
    analysis_data: list[dict]
    manifest_detected: bool
    warnings: list[str]


def prepare_upload_analysis(uploaded_files: list[Any]) -> UploadAnalysisResult:
    subtitle_files, manifest_files = split_uploaded_files(uploaded_files)
    warnings: list[str] = []
    manifest_items: list[dict] = []

    if manifest_files:
        manifest_items, warnings = _load_first_manifest_items(manifest_files)

    fallback_rows = analyze_filenames([f.name for f in subtitle_files])
    rel_path_index, basename_index = _build_manifest_indexes(manifest_items)
    analysis_data = []

    for upload_file, fallback_row in zip(subtitle_files, fallback_rows):
        row = dict(fallback_row)
        item, conflict = _match_manifest_item(
            upload_file.name, rel_path_index, basename_index
        )
        if item:
            title = _clean_text(item.get("title"))
            if title:
                row["识别片名"] = title

            year = _coerce_year(item.get("year"), row.get("年份", 0))
            row["年份"] = year
            row["状态"] = "来自Harvester JSON"
        elif conflict:
            row["状态"] = "JSON匹配冲突，请手动确认"

        analysis_data.append(row)

    if uploaded_files and not subtitle_files:
        warnings.append("未检测到可处理的字幕文件。")

    return UploadAnalysisResult(
        subtitle_files=subtitle_files,
        analysis_data=analysis_data,
        manifest_detected=bool(manifest_files),
        warnings=warnings,
    )


def split_uploaded_files(uploaded_files: list[Any]) -> tuple[list[Any], list[Any]]:
    subtitle_files = []
    manifest_files = []

    for upload_file in uploaded_files or []:
        basename = _basename(getattr(upload_file, "name", ""))
        lower_basename = basename.lower()
        if lower_basename == MANIFEST_NAME:
            manifest_files.append(upload_file)
            continue

        ext = _extension(lower_basename)
        if ext in VALID_SUBTITLE_EXTS:
            subtitle_files.append(upload_file)

    return subtitle_files, manifest_files


def has_harvester_manifest(uploaded_files: list[Any]) -> bool:
    _subtitle_files, manifest_files = split_uploaded_files(uploaded_files)
    return bool(manifest_files)


def _load_first_manifest_items(manifest_files: list[Any]) -> tuple[list[dict], list[str]]:
    warnings = []
    for manifest_file in manifest_files:
        try:
            payload = _read_uploaded_bytes(manifest_file).decode("utf-8")
            manifest = json.loads(payload)
            items = _extract_manifest_items(manifest)
            return items, warnings
        except Exception as e:
            warnings.append(
                f"Harvester manifest 读取失败，已回退到文件名识别: "
                f"{getattr(manifest_file, 'name', MANIFEST_NAME)} ({e})"
            )
    return [], warnings


def _extract_manifest_items(manifest: Any) -> list[dict]:
    if isinstance(manifest, dict):
        raw_items = manifest.get("items", [])
    elif isinstance(manifest, list):
        raw_items = manifest
    else:
        return []

    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, dict)]


def _build_manifest_indexes(
    manifest_items: list[dict],
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    rel_path_index: dict[str, dict] = {}
    basename_index: dict[str, list[dict]] = defaultdict(list)

    for item in manifest_items:
        subtitle_file = item.get("subtitle_file")
        if not subtitle_file:
            continue

        rel_path = normalize_manifest_path(subtitle_file)
        if not rel_path:
            continue

        rel_path_index[rel_path] = item
        basename_index[_basename(rel_path)].append(item)

    return rel_path_index, dict(basename_index)


def _match_manifest_item(
    upload_name: str,
    rel_path_index: dict[str, dict],
    basename_index: dict[str, list[dict]],
) -> tuple[dict | None, bool]:
    upload_path = normalize_manifest_path(upload_name)
    item = rel_path_index.get(upload_path)
    if item:
        return item, False

    basename_matches = basename_index.get(_basename(upload_path), [])
    if len(basename_matches) == 1:
        return basename_matches[0], False
    if len(basename_matches) > 1:
        return None, True
    return None, False


def _read_uploaded_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()

    current_pos = None
    if hasattr(uploaded_file, "tell"):
        try:
            current_pos = uploaded_file.tell()
        except Exception:
            current_pos = None

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    raw = uploaded_file.read()

    if current_pos is not None and hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(current_pos)
        except Exception:
            pass

    return raw


def _basename(path: str) -> str:
    normalized = normalize_manifest_path(path)
    if not normalized:
        return ""
    return normalized.rsplit("/", 1)[-1]


def _extension(basename: str) -> str:
    if "." not in basename:
        return ""
    return "." + basename.rsplit(".", 1)[-1]


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_year(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
