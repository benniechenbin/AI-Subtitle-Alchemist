from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src import db
from src.config import settings
from src.db import schema

logger = logging.getLogger(__name__)
_TABLE_NAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class VectorIndexUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorIndexItem:
    subtitle_id: int
    embedding: np.ndarray
    embedding_model: str
    embedding_dim: int


@dataclass(frozen=True)
class VectorSearchHit:
    subtitle_id: int
    distance: float
    score: float


class VectorIndexBackend(Protocol):
    name: str

    def ensure_index(
        self, db_path: str | None, embedding_model: str, embedding_dim: int
    ) -> None: ...

    def upsert(self, db_path: str | None, items: list[VectorIndexItem]) -> None: ...

    def delete(self, db_path: str | None, subtitle_ids: list[int]) -> None: ...

    def search(
        self,
        db_path: str | None,
        query_vector: np.ndarray,
        embedding_model: str | None,
        embedding_dim: int,
        top_k: int,
        target_movie: str | None = None,
    ) -> list[VectorSearchHit]: ...

    def rebuild(self, db_path: str | None, items: list[VectorIndexItem]) -> None: ...

    def clear(
        self, db_path: str | None, embedding_model: str | None = None
    ) -> None: ...

    def stats(self, db_path: str | None) -> dict: ...


class SqliteVecBackend:
    name = "sqlite_vec"

    def ensure_index(
        self, db_path: str | None, embedding_model: str, embedding_dim: int
    ) -> None:
        conn = db.get_db_connection(db_path)
        try:
            self._ensure_index_on_connection(conn, embedding_model, embedding_dim)
            conn.commit()
        finally:
            conn.close()

    def upsert(self, db_path: str | None, items: list[VectorIndexItem]) -> None:
        for group_key, group_items in _group_items(items).items():
            embedding_model, embedding_dim = group_key
            conn = db.get_db_connection(db_path)
            try:
                table_name = self._ensure_index_on_connection(
                    conn, embedding_model, embedding_dim
                )
                for item in group_items:
                    conn.execute(
                        f"DELETE FROM {table_name} WHERE subtitle_id = ?",
                        (item.subtitle_id,),
                    )
                    conn.execute(
                        f"INSERT INTO {table_name}(subtitle_id, embedding) VALUES (?, ?)",
                        (item.subtitle_id, _vector_blob(item.embedding)),
                    )
                conn.commit()
            finally:
                conn.close()

    def delete(self, db_path: str | None, subtitle_ids: list[int]) -> None:
        if not subtitle_ids:
            return
        conn = db.get_db_connection(db_path)
        try:
            self._load_extension(conn)
            table_names = _registry_table_names(conn, self.name)
            for table_name in table_names:
                for subtitle_id in subtitle_ids:
                    conn.execute(
                        f"DELETE FROM {table_name} WHERE subtitle_id = ?",
                        (subtitle_id,),
                    )
            conn.commit()
        finally:
            conn.close()

    def search(
        self,
        db_path: str | None,
        query_vector: np.ndarray,
        embedding_model: str | None,
        embedding_dim: int,
        top_k: int,
        target_movie: str | None = None,
    ) -> list[VectorSearchHit]:
        if not embedding_model:
            raise VectorIndexUnavailable("sqlite-vec search requires embedding_model.")

        table_name = _vec_table_name(embedding_model, embedding_dim)
        conn = db.get_db_connection(db_path)
        try:
            self._load_extension(conn)
            if not _table_exists(conn, table_name):
                return []
            rows = conn.execute(
                f"""
                SELECT subtitle_id, distance
                FROM {table_name}
                WHERE embedding MATCH ?
                  AND k = ?
                """,
                (_vector_blob(query_vector), int(top_k)),
            ).fetchall()
        finally:
            conn.close()

        return [
            VectorSearchHit(
                subtitle_id=int(row[0]),
                distance=float(row[1]),
                score=_score_from_distance(float(row[1])),
            )
            for row in rows
        ]

    def rebuild(self, db_path: str | None, items: list[VectorIndexItem]) -> None:
        for group_key, group_items in _group_items(items).items():
            embedding_model, embedding_dim = group_key
            conn = db.get_db_connection(db_path)
            try:
                table_name = self._ensure_index_on_connection(
                    conn, embedding_model, embedding_dim
                )
                conn.execute(f"DELETE FROM {table_name}")
                for item in group_items:
                    conn.execute(
                        f"INSERT INTO {table_name}(subtitle_id, embedding) VALUES (?, ?)",
                        (item.subtitle_id, _vector_blob(item.embedding)),
                    )
                conn.commit()
            finally:
                conn.close()

    def clear(self, db_path: str | None, embedding_model: str | None = None) -> None:
        conn = db.get_db_connection(db_path)
        try:
            self._load_extension(conn)
            for table_name in _registry_table_names(
                conn, self.name, embedding_model=embedding_model
            ):
                conn.execute(f"DELETE FROM {table_name}")
            conn.commit()
        finally:
            conn.close()

    def stats(self, db_path: str | None) -> dict:
        conn = db.get_db_connection(db_path)
        try:
            self._load_extension(conn)
            _ensure_registry_table(conn)
            rows = conn.execute(
                """
                SELECT embedding_model, embedding_dim, vec_table_name
                FROM vector_index_registry
                WHERE backend = ?
                ORDER BY embedding_model, embedding_dim
                """,
                (self.name,),
            ).fetchall()
            return {"backend": self.name, "indexes": [tuple(row) for row in rows]}
        finally:
            conn.close()

    def _ensure_index_on_connection(
        self, conn, embedding_model: str, embedding_dim: int
    ) -> str:
        table_name = _vec_table_name(embedding_model, embedding_dim)
        self._load_extension(conn)
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}
            USING vec0(
                subtitle_id INTEGER PRIMARY KEY,
                embedding FLOAT[{int(embedding_dim)}] distance_metric=cosine
            )
            """
        )
        _upsert_registry(
            conn,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            backend=self.name,
            table_name=table_name,
        )
        return table_name

    def _load_extension(self, conn) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:
            raise VectorIndexUnavailable("sqlite-vec is not installed.") from exc

        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:
            try:
                conn.enable_load_extension(False)
            except Exception:
                pass
            raise VectorIndexUnavailable(f"sqlite-vec failed to load: {exc}") from exc


class VectorIndexService:
    def __init__(
        self,
        *,
        primary_backend: VectorIndexBackend | None = None,
    ) -> None:
        self.primary_backend = primary_backend or SqliteVecBackend()

    def upsert_vector_rows(
        self,
        db_path: str | None,
        subtitle_ids: list[int],
        vector_rows: list[tuple],
    ) -> None:
        items = _items_from_vector_rows(subtitle_ids, vector_rows)
        self.upsert(db_path, items)

    def upsert(self, db_path: str | None, items: list[VectorIndexItem]) -> None:
        if not items:
            return
        try:
            self.primary_backend.upsert(db_path, items)
        except VectorIndexUnavailable as exc:
            logger.warning(
                "Vector index unavailable; vector rows were not persisted. %s", exc
            )

    def delete(self, db_path: str | None, subtitle_ids: list[int]) -> None:
        if not subtitle_ids:
            return
        try:
            self.primary_backend.delete(db_path, subtitle_ids)
        except VectorIndexUnavailable as exc:
            logger.warning("Vector index delete skipped. %s", exc)

    def rebuild(
        self,
        db_path: str | None,
        embedding_model: str | None = None,
        model=None,
        batch_size: int = 10000,
    ) -> bool:
        """Rebuild the vector index. Returns True if rebuilt, False if skipped."""
        if model is None:
            logger.warning("Vector index rebuild skipped: embedding model is required.")
            return False
        model_name = embedding_model or getattr(settings.prefs, "embedding_model", "")
        if not model_name:
            logger.warning(
                "Vector index rebuild skipped: embedding model name is required."
            )
            return False
        try:
            self.primary_backend.clear(db_path, embedding_model=embedding_model)
            for rows in db.iter_subtitles_for_vector_rebuild(
                db_path,
                batch_size=batch_size,
            ):
                items = _items_from_text_rows(
                    rows,
                    model=model,
                    embedding_model=model_name,
                )
                db.update_subtitle_embedding_metadata_batch(
                    db_path,
                    [
                        (item.embedding_model, item.embedding_dim, item.subtitle_id)
                        for item in items
                    ],
                )
                self.primary_backend.upsert(db_path, items)
            return True
        except VectorIndexUnavailable as exc:
            logger.warning("Vector index rebuild skipped. %s", exc)
            return False

    def search(
        self,
        *,
        query: str,
        model,
        db_path: str | None,
        embedding_model_name: str | None = None,
        final_k: int = 20,
        fetch_ratio: int = 4,
        allow_duplicates: bool = False,
        target_movie: str | None = None,
    ) -> list[dict]:
        query_vector = _as_float32(model.encode(query))
        top_k = max(final_k * fetch_ratio, final_k)
        if target_movie:
            top_k *= 5

        hits = self._search_hits(
            db_path=db_path,
            query_vector=query_vector,
            embedding_model_name=embedding_model_name,
            top_k=top_k,
            target_movie=target_movie,
        )
        rows = _rows_for_hits(db_path, hits)
        raw_results = _candidate_rows(rows, hits, target_movie=target_movie)

        return _dedupe_results(
            raw_results,
            db_path=db_path,
            final_k=final_k,
            allow_duplicates=allow_duplicates,
        )

    def _search_hits(
        self,
        *,
        db_path: str | None,
        query_vector: np.ndarray,
        embedding_model_name: str | None,
        top_k: int,
        target_movie: str | None,
    ) -> list[VectorSearchHit]:
        try:
            return self.primary_backend.search(
                db_path,
                query_vector,
                embedding_model_name,
                len(query_vector),
                top_k,
                target_movie=target_movie,
            )
        except VectorIndexUnavailable as exc:
            logger.warning("Vector index unavailable. %s", exc)
            return []


_default_service: VectorIndexService | None = None


def get_vector_index_service() -> VectorIndexService:
    global _default_service
    if _default_service is None:
        _default_service = VectorIndexService()
    return _default_service


def _items_from_vector_rows(
    subtitle_ids: list[int],
    vector_rows: list[tuple],
) -> list[VectorIndexItem]:
    items: list[VectorIndexItem] = []
    for subtitle_id, row in zip(subtitle_ids, vector_rows):
        embedding, embedding_model, embedding_dim = row
        if embedding is None or not embedding_model or not embedding_dim:
            continue
        vector = _embedding_from_blob(embedding)
        if vector is None:
            continue
        items.append(
            VectorIndexItem(
                subtitle_id=int(subtitle_id),
                embedding=vector,
                embedding_model=str(embedding_model),
                embedding_dim=int(embedding_dim),
            )
        )
    return items


def _items_from_text_rows(
    rows: list, *, model, embedding_model: str
) -> list[VectorIndexItem]:
    texts = [row[1] for row in rows]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=16,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.size == 0:
        return []
    embedding_dim = int(vectors.shape[1])
    return [
        VectorIndexItem(
            subtitle_id=int(row[0]),
            embedding=_as_float32(vector),
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
        for row, vector in zip(rows, vectors)
    ]


def _rows_for_hits(db_path: str | None, hits: list[VectorSearchHit]) -> list:
    return db.fetch_subtitles_by_ids(db_path, [hit.subtitle_id for hit in hits])


def _candidate_rows(
    rows: list, hits: list[VectorSearchHit], *, target_movie: str | None
) -> list[dict]:
    hit_by_id = {hit.subtitle_id: hit for hit in hits}
    candidates = []
    for row in rows:
        hit = hit_by_id.get(int(row[0]))
        if hit is None:
            continue
        if target_movie and row[1] != target_movie:
            continue
        candidates.append(
            {
                "id": row[0],
                "movie": row[1],
                "season": row[2],
                "episode": row[3],
                "time": row[4],
                "content": row[5],
                "score": hit.score,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _dedupe_results(
    raw_results: list[dict],
    *,
    db_path: str | None,
    final_k: int,
    allow_duplicates: bool,
) -> list[dict]:
    unique_results: list[dict] = []
    seen_contents: list[str] = []

    for res in raw_results:
        if not allow_duplicates:
            expanded = db.get_context_by_id(db_path, res["id"], res["movie"], window=1)
            if expanded:
                res["content"] = expanded

        new_text = res["content"]

        if allow_duplicates:
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


def _group_items(
    items: list[VectorIndexItem],
) -> dict[tuple[str, int], list[VectorIndexItem]]:
    groups: dict[tuple[str, int], list[VectorIndexItem]] = {}
    for item in items:
        if item.embedding.size == 0:
            continue
        key = (item.embedding_model, item.embedding_dim)
        groups.setdefault(key, []).append(item)
    return groups


def _vec_table_name(embedding_model: str, embedding_dim: int) -> str:
    digest = hashlib.sha1(embedding_model.encode("utf-8")).hexdigest()[:12]
    name = f"vec_subtitles_{int(embedding_dim)}_{digest}"
    return _validate_table_name(name)


def _validate_table_name(table_name: str) -> str:
    if not _TABLE_NAME_RE.fullmatch(table_name):
        raise ValueError(f"Invalid vector index table name: {table_name}")
    return table_name


def _upsert_registry(
    conn,
    *,
    embedding_model: str,
    embedding_dim: int,
    backend: str,
    table_name: str,
) -> None:
    _ensure_registry_table(conn)
    conn.execute(
        """
        INSERT INTO vector_index_registry (
            embedding_model, embedding_dim, backend, vec_table_name, updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(embedding_model, embedding_dim, backend)
        DO UPDATE SET vec_table_name = excluded.vec_table_name,
                      updated_at = excluded.updated_at
        """,
        (
            embedding_model,
            int(embedding_dim),
            backend,
            table_name,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )


def _ensure_registry_table(conn) -> None:
    conn.execute(schema.SQL_CREATE_VECTOR_INDEX_REGISTRY)


def _registry_table_names(
    conn, backend: str, embedding_model: str | None = None
) -> list[str]:
    _ensure_registry_table(conn)
    sql = "SELECT vec_table_name FROM vector_index_registry WHERE backend = ?"
    args = [backend]
    if embedding_model:
        sql += " AND embedding_model = ?"
        args.append(embedding_model)
    rows = conn.execute(
        sql,
        tuple(args),
    ).fetchall()
    return [_validate_table_name(str(row[0])) for row in rows]


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _vector_blob(vector: np.ndarray) -> bytes:
    return _as_float32(vector).tobytes()


def _as_float32(vector) -> np.ndarray:
    return np.asarray(vector, dtype=np.float32).reshape(-1)


def _embedding_from_blob(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.frombuffer(value, dtype=np.float32)
    except Exception:
        try:
            parsed = json.loads(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
            )
            return _as_float32(parsed)
        except Exception:
            return None


def _score_from_distance(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance))
