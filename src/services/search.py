import difflib
import numpy as np
from src import db

class VectorSearchService:
    """
    向量检索服务类。
    目前底层采用 SQLite 存储 + NumPy 计算余弦相似度。
    未来可无缝升级为 FAISS, sqlite-vec 或 Qdrant。
    """

    @staticmethod
    def search_semantic(
        query: str,
        model,
        db_path: str,
        embedding_model_name: str | None = None,
        final_k: int = 20,
        fetch_ratio: int = 4,
        allow_duplicates: bool = False,
        target_movie: str | None = None,
    ) -> list[dict]:
        """
        核心语义检索逻辑。
        """
        # 1. 将 query 向量化
        query_vector = model.encode(query)
        q_vec = np.array(query_vector)

        # 2. 从数据库拉取匹配该模型的向量数据
        rows = db.fetch_vectors_for_search(
            db_path, 
            target_movie=target_movie, 
            embedding_model=embedding_model_name
        )
        if not rows:
            return []

        # 3. 计算余弦相似度
        candidates = []
        for row in rows:
            embedding_blob = row[6]
            if not embedding_blob:
                continue
            try:
                db_vec = np.frombuffer(embedding_blob, dtype=np.float32)
            except Exception:
                try:
                    db_vec = np.array(eval(embedding_blob))
                except Exception:
                    continue
            
            if db_vec.shape != q_vec.shape:
                continue
            
            # 余弦相似度
            score = np.dot(q_vec, db_vec) / (
                np.linalg.norm(q_vec) * np.linalg.norm(db_vec)
            )
            candidates.append(
                {
                    "id": row[0],
                    "movie": row[1],
                    "season": row[2],
                    "episode": row[3],
                    "time": row[4],
                    "content": row[5],
                    "score": score,
                }
            )

        # 4. 排序并初步截断
        candidates.sort(key=lambda x: x["score"], reverse=True)
        raw_results = candidates[: final_k * fetch_ratio]

        # 5. 去重与上下文展开
        unique_results = []
        seen_contents = []

        for res in raw_results:
            if not allow_duplicates:
                expanded = db.get_context_by_id(
                    db_path, res["id"], res["movie"], window=1
                )
                if expanded:
                    res["content"] = expanded

            new_text = res["content"]

            if allow_duplicates:
                # 混剪模式：同一电影同一时间的台词不重复即可
                is_same_source = any(
                    exist["movie"] == res["movie"] and exist["time"] == res["time"]
                    for exist in unique_results
                )
                if not is_same_source:
                    unique_results.append(res)
                if len(unique_results) >= final_k:
                    break
                continue

            if len(new_text) < 2 or new_text in seen_contents:
                continue

            # 语义去重 (difflib)
            is_duplicate = any(
                difflib.SequenceMatcher(None, new_text, seen).ratio() > 0.85
                for seen in seen_contents
            )
            if is_duplicate:
                continue

            unique_results.append(res)
            seen_contents.append(new_text)
            if len(unique_results) >= final_k:
                break

        return unique_results

# 为了保持向下兼容，暂时保留模块级别的函数名
def search_semantic(*args, **kwargs):
    return VectorSearchService.search_semantic(*args, **kwargs)
