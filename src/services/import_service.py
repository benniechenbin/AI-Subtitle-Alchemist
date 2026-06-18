import os

from src import db
from src.models import UploadedFileInput
from src.utils import (
    calculate_file_hash,
    decode_subtitle_bytes,
    parse_subtitle_content,
)
from src.observability.logger import logger


def process_uploaded_files(
    files: list[UploadedFileInput],
    metadata_list: list[dict],
    target_folder: str,
    model,
    model_name: str = "",
    db_path: str | None = None,
) -> tuple[list[str], list[dict], dict, list]:

    os.makedirs(target_folder, exist_ok=True)

    logs: list[str] = []
    processed_files: list[dict] = []
    pending_rows: list = []
    stats = {"success": 0, "fail": 0, "duplicate": 0}

    for file_input, meta in zip(files, metadata_list):
        raw_bytes = file_input.raw_bytes
        name = file_input.name

        f_hash = calculate_file_hash(raw_bytes)
        if db.check_file_exists(db_path, f_hash):
            logs.append(f"⚠️ 跳过重复: {name}")
            stats["duplicate"] += 1
            continue

        content, encoding, had_replace = decode_subtitle_bytes(raw_bytes)
        if had_replace:
            logs.append(f"⚠️ 编码识别不确定，已强制解码: {name} ({encoding})")

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
                        batch_size=16  
                    )
            except Exception as e:
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
        with open(save_path, "w", encoding="utf-8") as f_out:
            f_out.write(content)

        processed_files.append({"name": new_name, "content": content})
        current_dim = embeddings.shape[1] if (model and len(embeddings) > 0) else 0

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
                emb_blob,
                model_name if model else None,
                current_dim,
            )
            pending_rows.append(row)

        logs.append(f"✅ 处理完成: {new_name}")
        stats["success"] += 1

    return logs, processed_files, stats, pending_rows
