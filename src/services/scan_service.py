import os

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

    db_files = db.get_all_file_paths(db_path)
    new_count = 0
    valid_exts = (".srt", ".txt", ".ass", ".ssa", ".vtt")
    disk_files = set()
    manifest_index = load_harvester_manifest(library_path)

    for root, _dirs, files in os.walk(library_path):
        for file in files:
            if not file.lower().endswith(valid_exts):
                continue
            full_path = os.path.normpath(os.path.join(root, file))
            disk_files.add(full_path)
            if full_path in db_files:
                continue
            try:
                with open(full_path, "rb") as f:
                    raw = f.read()
                f_hash = calculate_file_hash(raw)
                
                content, encoding, _had_replace = decode_subtitle_bytes(raw)
                
                # 规范化媒体库：如果不是 UTF-8 且没有容错替换，则安全地原子性覆写为 UTF-8
                if encoding not in ("utf-8", "utf-8-sig") and not _had_replace:
                    temp_path = full_path + ".tmp"
                    try:
                        with open(temp_path, "w", encoding="utf-8") as f_out:
                            f_out.write(content)
                        os.replace(temp_path, full_path)
                        # 重新计算 hash 以保证一致性（虽然文本内容没变，但字节变了）
                        with open(full_path, "rb") as f_re:
                            raw = f_re.read()
                            f_hash = calculate_file_hash(raw)
                    except Exception as e:
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except Exception:
                                pass
                        yield f"⚠️ 无法规范化文件编码 {file}: {e}", new_count

                ext = file.lower().split(".")[-1]
                subs = parse_subtitle_content(content, ext)
                if not subs:
                    continue

                texts = [s["text"] for s in subs]
                embeddings = []
                if texts:
                    embeddings = model.encode(
                        texts, 
                        convert_to_numpy=True, 
                        show_progress_bar=False,
                        batch_size=16
                    )

                metadata = resolve_subtitle_metadata(
                    full_path, library_path, manifest_index
                )
                rows = []
                vector_rows = []
                dim = embeddings.shape[1] if len(embeddings) > 0 else 0
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
