from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e!s}") from e
