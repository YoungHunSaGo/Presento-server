import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.main import create_app
from app.models import Base
from tests.fakes import FakeStorage


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
        bcrypt_rounds=4,
        database_url="sqlite+pysqlite:///:memory:",
        redis_url="redis://localhost:6379/15",
        cors_origins="http://localhost:5173",
    )


@pytest.fixture
def storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def app(settings: Settings, storage: FakeStorage):
    app = create_app(settings)
    app.state.storage = storage  # 테스트는 실제 S3/MinIO를 호출하지 않는다
    Base.metadata.create_all(app.state.engine)
    yield app
    app.state.engine.dispose()


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db(app):
    session = app.state.session_factory()
    yield session
    session.close()


@pytest.fixture
def signup(client):
    """회원가입 후 응답 JSON을 돌려주는 헬퍼."""

    def _signup(
        email: str = "student@ycu.ac.kr",
        password: str = "presento1234",
        nickname: str = "영훈",
    ) -> dict:
        res = client.post(
            "/api/v1/auth/signup",
            json={"email": email, "password": password, "nickname": nickname},
        )
        assert res.status_code == 201, res.text
        return res.json()

    return _signup


@pytest.fixture
def auth_headers(signup) -> dict[str, str]:
    return {"Authorization": f"Bearer {signup()['accessToken']}"}


@pytest.fixture
def other_headers(signup) -> dict[str, str]:
    """FORBIDDEN 테스트용 두 번째 사용자."""
    token = signup(email="other@ycu.ac.kr", nickname="다른사람")["accessToken"]
    return {"Authorization": f"Bearer {token}"}
