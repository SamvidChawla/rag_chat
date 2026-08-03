from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    database_url: str

    # Gemini API
    gemini_api_key: str
    gemini_llm_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 3072

    # Retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 10
    rerank_top_k: int = 5

    # App
    app_env: str = "dev"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()