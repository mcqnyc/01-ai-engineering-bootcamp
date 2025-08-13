from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    """
    Configuration settings for the chatbot UI.
    """
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str
    QDRANT_URL: str
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_PROVIDER: str
    GENERATION_MODEL: str
    GENERATION_MODEL_PROVIDER: str
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

    model_config = SettingsConfigDict(env_file='.env')

class Settings(BaseSettings):
    API_URL: str = "http://api:8000"

    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=True,
        extra='ignore'
    )

config = Config()
settings = Settings()
