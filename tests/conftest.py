import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """셸에 export된 설정 변수가 테스트에 섞이지 않게 지운다."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    # _env_file=None: 테스트는 로컬 .env를 절대 읽지 않는다.
    return Settings(
        _env_file=None,
        openai_api_key=SecretStr("test-openai-key"),
        openai_llm_model="test-model",
        jwt_secret=SecretStr("test-jwt-secret-for-pytest-only-0000"),
        database_url="sqlite+pysqlite:///:memory:",
        redis_url="redis://localhost:6379/15",
        cors_origins="http://localhost:5173",
    )


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as c:
        yield c
