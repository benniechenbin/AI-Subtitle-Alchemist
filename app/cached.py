import streamlit as st

from src import db
from src.services.embedding import load_embedding_model


@st.cache_data(ttl=60)
def cached_library_stats(db_path: str) -> dict:
    return db.get_library_stats(db_path)


@st.cache_resource(show_spinner="正在加载 AI 模型...")
def cached_embedding_model(model_name: str):
    return load_embedding_model(model_name)
