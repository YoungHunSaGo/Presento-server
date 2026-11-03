import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.core.errors import AppError
from app.main import create_app


class _Body(BaseModel):
    email: str
    password: str = Field(min_length=8)


@pytest.fixture
def err_client(settings):
    app = create_app(settings)
    router = APIRouter()

    @router.get("/_test/app-error")
    def app_error():
        raise AppError("EMAIL_ALREADY_EXISTS", "이미 가입된 이메일이에요.", 409)

    @router.post("/_test/validate")
    def validate(body: _Body):
        return {"ok": True}

    @router.get("/_test/boom")
    def boom():
        raise RuntimeError("secret internal detail")

    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _assert_error_shape(body: dict, code: str) -> None:
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def test_app_error_format(err_client):
    res = err_client.get("/_test/app-error")
    assert res.status_code == 409
    _assert_error_shape(res.json(), "EMAIL_ALREADY_EXISTS")
    assert res.json()["error"]["message"] == "이미 가입된 이메일이에요."
    assert res.json()["error"]["details"] is None


def test_validation_error_is_400_with_field_details(err_client):
    res = err_client.post("/_test/validate", json={"password": "short"})
    assert res.status_code == 400
    body = res.json()
    _assert_error_shape(body, "VALIDATION_ERROR")
    fields = {d["field"] for d in body["error"]["details"]}
    assert fields == {"email", "password"}
    # 입력값(비밀번호)이 응답에 새지 않는다.
    assert "short" not in res.text


def test_malformed_json_is_validation_error(err_client):
    res = err_client.post(
        "/_test/validate", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert res.status_code == 400
    _assert_error_shape(res.json(), "VALIDATION_ERROR")


def test_unknown_path_is_not_found(err_client):
    res = err_client.get("/api/v1/does-not-exist")
    assert res.status_code == 404
    _assert_error_shape(res.json(), "NOT_FOUND")


def test_unhandled_error_hides_internal_detail(err_client):
    res = err_client.get("/_test/boom")
    assert res.status_code == 500
    _assert_error_shape(res.json(), "INTERNAL_ERROR")
    assert "secret internal detail" not in res.text


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("get", "/_test/boom", 500),
        ("get", "/api/v1/does-not-exist", 404),
        ("get", "/_test/app-error", 409),
    ],
)
def test_error_responses_include_cors_header(err_client, method, path, status):
    res = getattr(err_client, method)(path, headers={"Origin": "http://localhost:5173"})
    assert res.status_code == status
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_validation_error_cors_header(err_client):
    res = err_client.post("/_test/validate", json={}, headers={"Origin": "http://localhost:5173"})
    assert res.status_code == 400
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_validation_reasons_are_korean(err_client):
    res = err_client.post("/_test/validate", json={"password": "short"})
    reasons = {d["field"]: d["reason"] for d in res.json()["error"]["details"]}
    assert reasons["email"] == "필수 값이에요."
    assert reasons["password"] == "8자 이상 입력해 주세요."


def test_malformed_json_has_empty_field(err_client):
    res = err_client.post(
        "/_test/validate", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert res.json()["error"]["details"][0]["field"] == ""


def test_method_not_allowed_uses_error_format(err_client):
    res = err_client.delete("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert res.status_code == 405
    _assert_error_shape(res.json(), "METHOD_NOT_ALLOWED")
    assert res.json()["error"]["message"] == "지원하지 않는 요청 방식이에요."
    assert res.json()["error"]["details"] is None
    assert "GET" in res.headers["allow"]
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_openapi_documents_400_not_422(err_client):
    spec = err_client.get("/openapi.json").json()
    op = spec["paths"]["/_test/validate"]["post"]
    assert "422" not in op["responses"]
    assert op["responses"]["400"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }
    assert "HTTPValidationError" not in spec["components"]["schemas"]


def test_settings_repr_hides_secrets(settings):
    text = repr(settings) + str(settings.model_dump())
    assert "test-openai-key" not in text
    assert "test-jwt-secret-for-pytest-only-0000" not in text
