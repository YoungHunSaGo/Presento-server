import re
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.security import create_access_token, decode_access_token, verify_password
from app.models import User

SIGNUP = "/api/v1/auth/signup"
LOGIN = "/api/v1/auth/login"
ME = "/api/v1/users/me"
KST_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00$")


def _err(res) -> str:
    return res.json()["error"]["code"]


# ---------- A1 회원가입 ----------


def test_signup_returns_spec_shape(signup):
    body = signup()
    assert set(body) == {"accessToken", "user"}
    user = body["user"]
    assert set(user) == {"id", "email", "nickname", "createdAt"}
    assert user["id"].startswith("u-")
    assert user["email"] == "student@ycu.ac.kr"
    assert user["nickname"] == "영훈"
    assert KST_ISO.match(user["createdAt"]), user["createdAt"]


def test_signup_stores_hashed_password(signup, db):
    signup()
    user = db.query(User).one()
    assert user.password_hash != "presento1234"
    assert user.password_hash.startswith("$2")
    assert verify_password("presento1234", user.password_hash)


def test_signup_response_does_not_leak_password(client):
    res = client.post(
        SIGNUP, json={"email": "a@b.com", "password": "presento1234", "nickname": "n"}
    )
    assert "presento1234" not in res.text
    assert "passwordHash" not in res.text


def test_signup_duplicate_email_conflict(signup, client):
    signup()
    res = client.post(
        SIGNUP, json={"email": "STUDENT@ycu.ac.kr", "password": "presento1234", "nickname": "x"}
    )
    assert res.status_code == 409
    assert _err(res) == "EMAIL_ALREADY_EXISTS"


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"email": "a@b.com", "password": "1234567", "nickname": "n"}, "password"),
        ({"email": "not-an-email", "password": "presento1234", "nickname": "n"}, "email"),
        ({"email": "a@b.com", "password": "presento1234"}, "nickname"),
        ({"email": "a@b.com", "password": "presento1234", "nickname": "   "}, "nickname"),
        ({"password": "presento1234", "nickname": "n"}, "email"),
    ],
)
def test_signup_validation_errors(client, payload, field):
    res = client.post(SIGNUP, json=payload)
    assert res.status_code == 400
    assert _err(res) == "VALIDATION_ERROR"
    assert field in {d["field"] for d in res.json()["error"]["details"]}


def test_signup_accepts_long_password_over_bcrypt_limit(client):
    long_pw = "가" * 100  # 300바이트: bcrypt 72바이트 제한 초과
    res = client.post(SIGNUP, json={"email": "long@b.com", "password": long_pw, "nickname": "n"})
    assert res.status_code == 201
    res = client.post(LOGIN, json={"email": "long@b.com", "password": long_pw})
    assert res.status_code == 200
    res = client.post(LOGIN, json={"email": "long@b.com", "password": long_pw[:-1] + "나"})
    assert res.status_code == 401


# ---------- A2 로그인 ----------


def test_login_success_same_shape_as_signup(signup, client):
    created = signup()
    res = client.post(LOGIN, json={"email": "student@ycu.ac.kr", "password": "presento1234"})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"accessToken", "user"}
    assert body["user"] == created["user"]


def test_login_email_case_insensitive(signup, client):
    signup()
    res = client.post(LOGIN, json={"email": " Student@YCU.ac.kr ", "password": "presento1234"})
    assert res.status_code == 200


def test_login_wrong_password(signup, client):
    signup()
    res = client.post(LOGIN, json={"email": "student@ycu.ac.kr", "password": "wrong-password"})
    assert res.status_code == 401
    assert _err(res) == "INVALID_CREDENTIALS"


def test_login_unknown_email_same_error(client):
    res = client.post(LOGIN, json={"email": "nobody@ycu.ac.kr", "password": "presento1234"})
    assert res.status_code == 401
    assert _err(res) == "INVALID_CREDENTIALS"


def test_login_missing_fields_is_validation_error(client):
    res = client.post(LOGIN, json={"email": "a@b.com"})
    assert res.status_code == 400
    assert _err(res) == "VALIDATION_ERROR"


# ---------- A3 내 정보 ----------


def test_me_returns_user(signup, client):
    created = signup()
    res = client.get(ME, headers={"Authorization": f"Bearer {created['accessToken']}"})
    assert res.status_code == 200
    assert res.json() == created["user"]


def _bad_tokens(settings):
    secret = settings.jwt_secret.get_secret_value()
    now = datetime.now(UTC)
    return {
        "expired": jwt.encode(
            {"sub": "u-x", "iat": now - timedelta(hours=25), "exp": now - timedelta(hours=1)},
            secret,
            algorithm="HS256",
        ),
        "wrong_secret": jwt.encode(
            {"sub": "u-x", "iat": now, "exp": now + timedelta(hours=1)},
            "another-secret-another-secret-0000",
            algorithm="HS256",
        ),
        "alg_none": jwt.encode(
            {"sub": "u-x", "iat": now, "exp": now + timedelta(hours=1)}, None, algorithm="none"
        ),
        "no_exp": jwt.encode({"sub": "u-x", "iat": now}, secret, algorithm="HS256"),
        "garbage": "not-a-jwt",
    }


@pytest.mark.parametrize("kind", ["expired", "wrong_secret", "alg_none", "no_exp", "garbage"])
def test_me_rejects_bad_tokens(client, settings, kind):
    token = _bad_tokens(settings)[kind]
    res = client.get(ME, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 401
    assert _err(res) == "UNAUTHORIZED"


def test_me_without_token(client):
    res = client.get(ME)
    assert res.status_code == 401
    assert _err(res) == "UNAUTHORIZED"


def test_me_with_non_bearer_scheme(signup, client):
    token = signup()["accessToken"]
    res = client.get(ME, headers={"Authorization": f"Basic {token}"})
    assert res.status_code == 401
    assert _err(res) == "UNAUTHORIZED"


def test_me_token_for_deleted_user(signup, client, db):
    token = signup()["accessToken"]
    db.query(User).delete()
    db.commit()
    res = client.get(ME, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 401
    assert _err(res) == "UNAUTHORIZED"


# ---------- 토큰 ----------


def test_token_expires_in_24_hours(settings):
    token = create_access_token("u-abc", settings)
    payload = jwt.decode(token, options={"verify_signature": False})
    assert payload["sub"] == "u-abc"
    assert payload["exp"] - payload["iat"] == 24 * 3600
    assert jwt.get_unverified_header(token)["alg"] == "HS256"
    assert decode_access_token(token, settings) == "u-abc"


def test_login_preflight_cors(client):
    res = client.options(
        LOGIN,
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_openapi_lists_auth_paths(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/auth/signup", "/api/v1/auth/login", "/api/v1/users/me"} <= set(paths)
    assert "422" not in paths["/api/v1/auth/signup"]["post"]["responses"]
