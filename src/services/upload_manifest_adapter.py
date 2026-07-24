import json
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from src.services.harvester_manifest import MANIFEST_NAME, normalize_manifest_path
from src.services.tmdb_service import search_tmdb_metadata
from src.utils import analyze_filenames

VALID_SUBTITLE_EXTS = {".srt", ".ass", ".ssa", ".vtt", ".txt"}


@dataclass
class UploadAnalysisResult:
    subtitle_files: list[Any]
    analysis_data: list[dict]
    tmdb_matches: list[dict | None]
    manifest_detected: bool
    warnings: list[str]


def prepare_upload_analysis(
    uploaded_files: list[Any],
    tmdb_search: Callable[..., dict | None] | None = None,
) -> UploadAnalysisResult:
    subtitle_files, manifest_files = split_uploaded_files(uploaded_files)
    warnings: list[str] = []
    manifest_items: list[dict] = []

    if manifest_files:
        manifest_items, warnings = _load_first_manifest_items(manifest_files)

    fallback_rows = analyze_filenames([f.name for f in subtitle_files])
    rel_path_index, basename_index = _build_manifest_indexes(manifest_items)
    analysis_data: list[dict] = []
    query_keys: list[tuple[str, Any] | None] = []

    for upload_file, fallback_row in zip(subtitle_files, fallback_rows):
        row = dict(fallback_row)
        query_key = None
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
        else:
            guessed_title = row.get("识别片名")
            guessed_year = row.get("年份")
            if guessed_title:
                query_key = (str(guessed_title), guessed_year)

        analysis_data.append(row)
        query_keys.append(query_key)

    searcher = tmdb_search or search_tmdb_metadata
    tmdb_cache, search_warnings = _search_tmdb_queries(query_keys, searcher)
    warnings.extend(search_warnings)

    tmdb_matches: list[dict | None] = []
    for row, query_key in zip(analysis_data, query_keys):
        metadata = tmdb_cache.get(query_key) if query_key else None
        if metadata and metadata.get("title"):
            row["识别片名"] = metadata["title"]
            if metadata.get("year"):
                row["年份"] = metadata["year"]
            row["状态"] = "✅ TMDB精准匹配"
            tmdb_matches.append(
                {
                    "expected_title": row.get("识别片名"),
                    "expected_year": row.get("年份"),
                    "metadata": metadata,
                }
            )
        else:
            if query_key and ("状态" not in row or not row["状态"]):
                row["状态"] = "已从文件名提取"
            tmdb_matches.append(None)

    if uploaded_files and not subtitle_files:
        warnings.append("未检测到可处理的字幕文件。")

    return UploadAnalysisResult(
        subtitle_files=subtitle_files,
        analysis_data=analysis_data,
        tmdb_matches=tmdb_matches,
        manifest_detected=bool(manifest_files),
        warnings=warnings,
    )


def _search_tmdb_queries(
    query_keys: list[tuple[str, Any] | None],
    searcher: Callable[..., dict | None],
) -> tuple[dict[tuple[str, Any], dict | None], list[str]]:
    unique_keys = list(dict.fromkeys(key for key in query_keys if key is not None))
    if not unique_keys:
        return {}, []

    results: dict[tuple[str, Any], dict | None] = {}
    warnings: list[str] = []
    max_workers = min(4, len(unique_keys))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(searcher, title, release_year=year): (title, year)
            for title, year in unique_keys
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[key] = None
                warnings.append(
                    f"TMDB 匹配失败，已保留文件名识别结果: {key[0]} ({exc})"
                )

    return results, warnings


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


def _load_first_manifest_items(
    manifest_files: list[Any],
) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    for manifest_file in manifest_files:
        try:
            payload = _read_uploaded_bytes(manifest_file).decode("utf-8")
            manifest = json.loads(payload)
            items = _extract_manifest_items(manifest)
            return items, warnings
        except Exception as e:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            current_pos = None

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:  # noqa: BLE001, S110
            pass

    raw = uploaded_file.read()

    if current_pos is not None and hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(current_pos)
        except Exception:  # noqa: BLE001, S110
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
