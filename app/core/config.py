"""애플리케이션 설정. 모든 설정은 이 모듈의 get_settings()로만 접근한다.

비밀값은 SecretStr로 감싸 repr/로그에 값이 찍히지 않게 한다.
"""

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: SecretStr
    openai_stt_model: str = "whisper-1"
    openai_llm_model: str = ""

    # 인증
    jwt_secret: SecretStr
    jwt_expire_hours: int = 24

    # DB / 큐
    database_url: str = "postgresql+psycopg://presento:presento@localhost:5432/presento"
    redis_url: str = "redis://localhost:6379/0"

    # 파일 저장
    s3_endpoint_url: str | None = None
    s3_bucket: str = "presento"
    s3_access_key: SecretStr = SecretStr("")
    s3_secret_key: SecretStr = SecretStr("")
    s3_region: str = "ap-northeast-2"

    # CORS (쉼표로 구분)
    cors_origins: str = "http://localhost:5173"

    # 비용 보호
    max_video_seconds: int = 600
    max_video_bytes: int = 524_288_000
    daily_analysis_limit: int = 20

    @field_validator("openai_api_key")
    @classmethod
    def _openai_key_not_empty(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY must not be empty")
        return v

    @field_validator("jwt_secret")
    @classmethod
    def _jwt_secret_strong(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
