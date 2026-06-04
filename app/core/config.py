"""Application settings loaded from environment / .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Healthcare AI"
    debug: bool = False
    database_url: str = "postgresql+asyncpg://user:pass@localhost/healthcareai"
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"


settings = Settings()
