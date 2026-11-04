import sys

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings

_VALID = {
    "openai_api_key": SecretStr("test-openai-key"),
    "jwt_secret": SecretStr("x" * 32),
}


def test_empty_openai_key_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{**_VALID, "openai_api_key": SecretStr("")})


def test_short_jwt_secret_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{**_VALID, "jwt_secret": SecretStr("short")})


def test_missing_secrets_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_cors_origin_list_parses_comma_separated():
    s = Settings(_env_file=None, **_VALID, cors_origins="http://a.com, http://b.com,")
    assert s.cors_origin_list == ["http://a.com", "http://b.com"]


def test_importing_celery_app_does_not_load_settings():
    sys.modules.pop("app.workers.celeryconfig", None)
    import app.workers.celery_app  # noqa: F401

    assert "app.workers.celeryconfig" not in sys.modules


@pytest.mark.parametrize(("key", "secret"), [("only-key", ""), ("", "only-secret")])
def test_s3_keys_must_be_pair(key, secret):
    with pytest.raises(ValidationError, match="S3_ACCESS_KEY"):
        Settings(
            _env_file=None,
            openai_api_key=SecretStr("k"),
            jwt_secret=SecretStr("x" * 32),
            s3_access_key=SecretStr(key),
            s3_secret_key=SecretStr(secret),
        )


def test_s3_keys_both_empty_is_iam_role():
    s = Settings(
        _env_file=None,
        openai_api_key=SecretStr("k"),
        jwt_secret=SecretStr("x" * 32),
        s3_access_key=SecretStr(""),
        s3_secret_key=SecretStr(""),
    )
    assert s.s3_access_key.get_secret_value() == ""


def test_validation_errors_do_not_show_secret_values():
    # 시작 실패 로그에 비밀값이 찍히지 않아야 한다
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            openai_api_key=SecretStr("sk-very-secret-openai"),
            jwt_secret=SecretStr("short-jwt-secret"),
            s3_access_key=SecretStr("AKIAONLYKEY"),
        )
    text = str(exc_info.value)
    for secret in ("sk-very-secret", "short-jwt-secret", "AKIAONLYKEY"):
        assert secret not in text
