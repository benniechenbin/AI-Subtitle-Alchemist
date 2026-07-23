import os
import re

from src import db
from src.models import ScanDoneResult
from src.services.harvester_manifest import (
    load_harvester_manifest,
    resolve_subtitle_metadata,
)
from src.services.vector_index import get_vector_index_service
from src.utils import calculate_file_hash, decode_subtitle_bytes, parse_subtitle_content


def scan_library(
    library_path: str,
    model,
    model_name: str = "",
    db_path: str | None = None,
    sync_vector_index: bool = True,
):
    if not os.path.exists(library_path):
        yield "❌ 路径不存在", None
        return
    if not model:
        yield "❌ 模型未加载", None
        return

    new_count = 0
    valid_exts = (".srt", ".txt", ".ass", ".ssa", ".vtt")
    manifest_index = load_harvester_manifest(library_path)
    scan_files = _collect_scan_files(library_path, valid_exts)
    disk_files = set(scan_files)
    seen_hashes: dict[str, str] = {}

    for message in _cleanup_indexed_duplicates(db_path, sync_vector_index):
        yield message, new_count
    db_files = db.get_all_file_paths(db_path)

    for full_path in scan_files:
        file = os.path.basename(full_path)
        if full_path in db_files:
            continue
        try:
            with open(full_path, "rb") as f:
                raw = f.read()
            raw_hash = calculate_file_hash(raw)

            content, encoding, _had_replace = decode_subtitle_bytes(raw)
            f_hash = calculate_file_hash(content.encode("utf-8"))

            # 规范化媒体库：如果不是 UTF-8 且没有容错替换，则安全地原子性覆写为 UTF-8
            if encoding not in ("utf-8", "utf-8-sig") and not _had_replace:
                temp_path = full_path + ".tmp"
                try:
                    with open(temp_path, "w", encoding="utf-8") as f_out:
                        f_out.write(content)
                    os.replace(temp_path, full_path)
                except Exception as e:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                    yield f"⚠️ 无法规范化文件编码 {file}: {e}", new_count

            existing_paths = db.get_file_paths_by_hash(db_path, f_hash)
            if raw_hash != f_hash:
                existing_paths.update(db.get_file_paths_by_hash(db_path, raw_hash))
            live_existing_paths = {
                path for path in existing_paths if os.path.exists(path)
            }
            if live_existing_paths:
                existing_path = sorted(live_existing_paths)[0]
                yield (
                    (
                        f"⚠️ 跳过同内容旧副本: {file} "
                        f"(已由 {os.path.basename(existing_path)} 入库)"
                    ),
                    new_count,
                )
                continue

            if f_hash in seen_hashes:
                yield (
                    (
                        f"⚠️ 跳过同内容旧副本: {file} "
                        f"(保留 {os.path.basename(seen_hashes[f_hash])})"
                    ),
                    new_count,
                )
                continue

            ext = file.lower().split(".")[-1]
            subs = parse_subtitle_content(content, ext)
            if not subs:
                continue

            seen_hashes[f_hash] = full_path

            texts = [s["text"] for s in subs]
            embeddings = []
            if texts:
                embeddings = model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=16,
                )

            metadata = resolve_subtitle_metadata(
                full_path, library_path, manifest_index
            )
            rows = []
            vector_rows = []
            dim = (
                embeddings.shape[1]
                if (len(embeddings) > 0 and hasattr(embeddings, "shape"))
                else 0
            )
            row_model_name = model_name if dim else None

            for i, s in enumerate(subs):
                emb = embeddings[i].tobytes() if i < len(embeddings) else None
                rows.append(
                    (
                        f_hash,
                        full_path,
                        metadata["movie_name"],
                        metadata["year"],
                        metadata["season"],
                        metadata["episode"],
                        i,
                        s["start"],
                        s["end"],
                        s["text"],
                        row_model_name,
                        dim,
                    )
                )
                vector_rows.append((emb, row_model_name, dim))

            inserted_ids = db.insert_subtitles_batch(db_path, rows)
            db.upsert_movie_metadata(db_path, metadata)
            if sync_vector_index:
                get_vector_index_service().upsert_vector_rows(
                    db_path, inserted_ids, vector_rows
                )
            new_count += 1
            yield f"✅ 已入库: {file}", new_count
        except Exception as e:
            yield f"❌ 失败 {file}: {e}", new_count

    missing = list(db_files - disk_files)
    yield "DONE", ScanDoneResult(new_added=new_count, missing_files=missing)


def _collect_scan_files(library_path: str, valid_exts: tuple[str, ...]) -> list[str]:
    paths = []
    for root, _dirs, files in os.walk(library_path):
        for file in files:
            if file.lower().endswith(valid_exts):
                paths.append(os.path.normpath(os.path.join(root, file)))
    return sorted(paths, key=_scan_priority)


def _scan_priority(path: str) -> tuple[int, str]:
    filename = os.path.basename(path)
    is_clean_name = bool(
        re.search(r"\.[0-9a-fA-F]{8}\.(?:srt|txt|ass|ssa|vtt)$", filename)
    )
    return (0 if is_clean_name else 1, os.path.normcase(path))


def _cleanup_indexed_duplicates(
    db_path: str | None, sync_vector_index: bool
) -> list[str]:
    messages = []
    for _file_hash, indexed_paths in db.get_file_hash_path_index(db_path).items():
        live_paths = [path for path in indexed_paths if os.path.exists(path)]
        if len(live_paths) < 2:
            continue

        keep_path = sorted(live_paths, key=_scan_priority)[0]
        duplicate_paths = [path for path in live_paths if path != keep_path]
        try:
            deleted_ids = db.delete_records_by_path(db_path, duplicate_paths)
            if sync_vector_index and deleted_ids:
                get_vector_index_service().delete(db_path, deleted_ids)
            messages.append(
                "⚠️ 已清理重复数据库记录，磁盘文件保留: "
                f"{', '.join(os.path.basename(path) for path in duplicate_paths)} "
                f"(保留 {os.path.basename(keep_path)})"
            )
        except Exception as exc:
            messages.append(
                "⚠️ 重复记录清理失败，已保留原数据: "
                f"{', '.join(os.path.basename(path) for path in duplicate_paths)} "
                f"({exc})"
            )
    return messages
