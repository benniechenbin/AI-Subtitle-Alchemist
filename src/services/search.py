from src.services.vector_index import get_vector_index_service


class VectorSearchService:
    """
    语义检索入口。

    底层优先使用 sqlite-vec 向量索引；如果 sqlite-vec 不可用或索引为空，
    自动回退到原有 SQLite BLOB + NumPy 全量扫描。
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
        return get_vector_index_service().search(
            query=query,
            model=model,
            db_path=db_path,
            embedding_model_name=embedding_model_name,
            final_k=final_k,
            fetch_ratio=fetch_ratio,
            allow_duplicates=allow_duplicates,
            target_movie=target_movie,
        )


def search_semantic(*args, **kwargs):
    return VectorSearchService.search_semantic(*args, **kwargs)
