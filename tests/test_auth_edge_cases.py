from datetime import UTC, datetime, timedelta

import jwt
import pytest
from sqlalchemy.orm import Session

from app.api.v1 import auth as auth_module
from app.schemas.common import to_kst_iso

SIGNUP = "/api/v1/auth/signup"
LOGIN = "/api/v1/auth/login"
ME = "/api/v1/users/me"


def _err(res) -> str:
    return res.json()["error"]["code"]


def test_signup_race_on_unique_email_returns_409(signup, client, monkeypatch):
    """중복 확인을 통과한 두 요청이 동시에 INSERT하는 상황을 흉내 낸다."""
    signup()
    monkeypatch.setattr(Session, "scalar", lambda self, *a, **kw: None)
    res = client.post(
        SIGNUP, json={"email": "student@ycu.ac.kr", "password": "presento1234", "nickname": "x"}
    )
    assert res.status_code == 409
    assert _err(res) == "EMAIL_ALREADY_EXISTS"
    monkeypatch.undo()
    # 롤백 뒤에도 다른 가입은 정상 처리된다.
    assert signup(email="other@ycu.ac.kr")["user"]["email"] == "other@ycu.ac.kr"


def test_nickname_too_long(client):
    res = client.post(
        SIGNUP, json={"email": "a@b.com", "password": "presento1234", "nickname": "가" * 51}
    )
    assert res.status_code == 400
    assert res.json()["error"]["details"][0]["field"] == "nickname"


def test_signup_password_max_length_128(client):
    ok = client.post(SIGNUP, json={"email": "max@b.com", "password": "a" * 128, "nickname": "n"})
    assert ok.status_code == 201

    res = client.post(SIGNUP, json={"email": "over@b.com", "password": "a" * 129, "nickname": "n"})
    assert res.status_code == 400
    assert _err(res) == "VALIDATION_ERROR"
    detail = res.json()["error"]["details"][0]
    assert detail == {"field": "password", "reason": "128자 이하로 입력해 주세요."}
    assert "a" * 129 not in res.text


def test_login_password_max_length_128(signup, client):
    signup(email="max@b.com", password="a" * 128)
    ok = client.post(LOGIN, json={"email": "max@b.com", "password": "a" * 128})
    assert ok.status_code == 200

    res = client.post(LOGIN, json={"email": "max@b.com", "password": "a" * 129})
    assert res.status_code == 400
    assert _err(res) == "VALIDATION_ERROR"
    assert res.json()["error"]["details"][0]["field"] == "password"


def test_nickname_is_trimmed(signup):
    assert signup(nickname="  영훈  ")["user"]["nickname"] == "영훈"


def test_login_with_punycode_domain_matches_signup(signup, client):
    signup(email="a@xn--r8jz45g.jp")
    res = client.post(LOGIN, json={"email": "a@xn--r8jz45g.jp", "password": "presento1234"})
    assert res.status_code == 200


def test_unknown_email_runs_dummy_password_check(client, monkeypatch):
    calls = []
    monkeypatch.setattr(auth_module, "burn_password_check", lambda pw, r: calls.append(r))
    res = client.post(LOGIN, json={"email": "nobody@ycu.ac.kr", "password": "presento1234"})
    assert res.status_code == 401
    assert calls == [4]


def test_token_with_future_iat_rejected(client, settings, signup):
    user_id = signup()["user"]["id"]
    now = datetime.now(UTC)
    token = jwt.encode(
        {"sub": user_id, "iat": now + timedelta(hours=1), "exp": now + timedelta(hours=2)},
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    res = client.get(ME, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 401


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (datetime(2026, 11, 3, 8, 4, 0, 123456, tzinfo=UTC), "2026-11-03T17:04:00+09:00"),
        (datetime(2026, 11, 3, 8, 4, 0), "2026-11-03T17:04:00+09:00"),  # naive = UTC(SQLite)
    ],
)
def test_to_kst_iso(dt, expected):
    assert to_kst_iso(dt) == expected


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (
            {"json": {"email": "a@b.com", "password": 12345678, "nickname": "n"}},
            "문자열로 입력해 주세요.",
        ),
        ({"json": ["a", "b"]}, "JSON 객체 형식으로 보내 주세요."),
        ({"data": {"email": "a@b.com"}}, "JSON 객체 형식으로 보내 주세요."),
    ],
)
def test_validation_reasons_never_english(client, kwargs, reason):
    res = client.post(SIGNUP, **kwargs)
    assert res.status_code == 400
    reasons = [d["reason"] for d in res.json()["error"]["details"]]
    assert reason in reasons
    assert all(not r.isascii() for r in reasons), reasons
