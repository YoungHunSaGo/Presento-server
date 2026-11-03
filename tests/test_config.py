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
