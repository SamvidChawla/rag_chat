from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    database_url: str
    db_pool_min: int = 1
    db_pool_max: int = 2

    # Gemini API
    gemini_api_key: str
    gemini_llm_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_llm_temperature: float = 0.2 
    gemini_llm_max_tokens: int = 1024
    embedding_dim: int = 3072

    # Retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 10
    rerank_top_k: int = 5
    enable_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # App
    app_env: str = "dev"
    max_upload_size_mb: int = 20

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()