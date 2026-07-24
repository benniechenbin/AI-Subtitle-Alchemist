import os
import tempfile

from src import db
from src.models import UploadedFileInput
from src.observability.logger import logger
from src.utils import (
    calculate_file_hash,
    decode_subtitle_bytes,
    parse_subtitle_content,
)


def process_uploaded_files(
    files: list[UploadedFileInput],
    metadata_list: list[dict],
    target_folder: str,
    model,
    model_name: str = "",
    db_path: str | None = None,
    tmdb_matches: list[dict | None] | None = None,
) -> tuple[list[str], list[dict], dict, list, list, list]:

    os.makedirs(target_folder, exist_ok=True)

    logs: list[str] = []
    processed_files: list[dict] = []
    pending_rows: list = []
    pending_vectors: list = []
    pending_meta: list = []
    stats = {"success": 0, "fail": 0, "duplicate": 0}

    match_list = tmdb_matches or []
    for file_index, (file_input, meta) in enumerate(zip(files, metadata_list)):
        raw_bytes = file_input.raw_bytes
        name = file_input.name

        source_hash = calculate_file_hash(raw_bytes)

        content, encoding, had_replace = decode_subtitle_bytes(raw_bytes)
        if had_replace:
            logs.append(f"⚠️ 编码识别不确定，已强制解码: {name} ({encoding})")

        stored_bytes = content.encode("utf-8")
        f_hash = calculate_file_hash(stored_bytes)
        if db.check_file_exists(db_path, source_hash) or (
            f_hash != source_hash and db.check_file_exists(db_path, f_hash)
        ):
            logs.append(f"⚠️ 跳过重复: {name}")
            stats["duplicate"] += 1
            continue

        ext = (name or "").split(".")[-1].lower()
        if ext not in ("srt", "ass", "ssa", "vtt", "txt"):
            ext = "srt"  # fallback

        subs = parse_subtitle_content(content, ext)

        if not subs:
            logs.append(f"⚠️ 无有效字幕: {name}")
            stats["fail"] += 1
            continue

        embeddings = []
        if model:
            try:
                texts = [s["text"] for s in subs]
                if texts:
                    embeddings = model.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=16,
                    )
            except Exception as e:  # noqa: BLE001
                logs.append(f"⚠️ 向量化失败: {e}")
                logger.error(f"文件 {name} 向量化失败: {e}")
        clean_title = str(meta.get("识别片名", "")).replace("/", "_").replace(":", " ")
        new_name = f"{clean_title}"
        if meta.get("年份"):
            new_name += f" ({meta['年份']})"
        if meta.get("season_num") or meta.get("episode_num"):
            new_name += (
                f".S{str(meta.get('season_num', 0)).zfill(2)}"
                f"E{str(meta.get('episode_num', 0)).zfill(2)}"
            )

        # 加入 8 位 hash 防止同名覆盖，并保持原后缀
        new_name += f".{f_hash[:8]}.{ext}"

        save_path = os.path.join(target_folder, new_name)
        try:
            file_action = _persist_clean_file(
                target_folder=target_folder,
                original_name=name,
                source_bytes=raw_bytes,
                source_hash=source_hash,
                stored_bytes=stored_bytes,
                stored_hash=f_hash,
                save_path=save_path,
            )
        except Exception as exc:  # noqa: BLE001
            logs.append(f"❌ 落盘失败: {name} ({exc})")
            stats["fail"] += 1
            continue

        processed_files.append({"name": new_name, "content": content})
        current_dim = (
            embeddings.shape[1]
            if (model and len(embeddings) > 0 and hasattr(embeddings, "shape"))
            else 0
        )

        for i, s in enumerate(subs):
            emb_blob = (
                embeddings[i].tobytes() if (model and i < len(embeddings)) else None
            )
            row = (
                f_hash,
                save_path,
                meta.get("识别片名"),
                meta.get("年份"),
                meta.get("season_num"),
                meta.get("episode_num"),
                i,
                s["start"],
                s["end"],
                s["text"],
                model_name if model else None,
                current_dim,
            )
            pending_rows.append(row)
            pending_vectors.append(
                (emb_blob, model_name if model else None, current_dim)
            )

        logs.append(f"✅ {file_action}: {new_name}")
        stats["success"] += 1

        tmdb_match = match_list[file_index] if file_index < len(match_list) else None
        if tmdb_match and _tmdb_match_is_valid(meta, tmdb_match):
            tmdb_metadata = tmdb_match.get("metadata") or {}
            raw_tmdb = tmdb_metadata.get("raw")
            if raw_tmdb:
                pending_meta.append(
                    {
                        "movie_name": meta.get("识别片名"),
                        "year": meta.get("年份"),
                        "media_type": tmdb_metadata.get("media_type"),
                        "tmdb_id": tmdb_metadata.get("tmdb_id"),
                        "tmdb_metadata": {
                            "content": raw_tmdb,
                            "source": "tmdb",
                            "schema_version": 1,
                        },
                    }
                )
        elif tmdb_match:
            logs.append(f"ℹ️ 片名或年份已手工修改，已取消 TMDB 绑定: {name}")

    return logs, processed_files, stats, pending_rows, pending_vectors, pending_meta


def _persist_clean_file(
    *,
    target_folder: str,
    original_name: str,
    source_bytes: bytes,
    source_hash: str,
    stored_bytes: bytes,
    stored_hash: str,
    save_path: str,
) -> str:
    source_path = _library_source_candidate(target_folder, original_name)
    source_is_confirmed = False
    if source_path and os.path.isfile(source_path):
        with open(source_path, "rb") as source_file:
            source_is_confirmed = calculate_file_hash(source_file.read()) == source_hash

    if os.path.exists(save_path):
        with open(save_path, "rb") as existing_file:
            if calculate_file_hash(existing_file.read()) != stored_hash:
                raise FileExistsError(f"目标文件已存在且内容不同: {save_path}")

    if (
        source_is_confirmed
        and source_path is not None
        and source_path != os.path.abspath(save_path)
    ):
        if source_bytes == stored_bytes:
            os.replace(source_path, save_path)
        else:
            _write_bytes_atomically(save_path, stored_bytes)
            os.remove(source_path)
        return "已安全改名"

    _write_bytes_atomically(save_path, stored_bytes)
    return "处理完成"


def _library_source_candidate(target_folder: str, original_name: str) -> str | None:
    basename = str(original_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not basename:
        return None
    target_root = os.path.abspath(target_folder)
    candidate = os.path.abspath(os.path.join(target_root, basename))
    if os.path.commonpath((target_root, candidate)) != target_root:
        return None
    return candidate


def _write_bytes_atomically(path: str, content: bytes) -> None:
    target_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(target_dir, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(content)
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _tmdb_match_is_valid(meta: dict, tmdb_match: dict) -> bool:
    return _normalized_title(meta.get("识别片名")) == _normalized_title(
        tmdb_match.get("expected_title")
    ) and _normalized_year(meta.get("年份")) == _normalized_year(
        tmdb_match.get("expected_year")
    )


def _normalized_title(value) -> str:
    return str(value or "").strip()


def _normalized_year(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
