"""S3Storage 단위 테스트. presigned URL 서명은 네트워크 호출 없이 로컬에서 계산된다."""

from urllib.parse import parse_qs, urlparse

from pydantic import SecretStr

from app.core.config import Settings
from app.services.storage import S3Storage


def _settings(**kw) -> Settings:
    base = {
        "openai_api_key": SecretStr("test-openai-key"),
        "jwt_secret": SecretStr("x" * 32),
        "s3_endpoint_url": "http://minio:9000",
        "s3_public_endpoint_url": "http://localhost:9000",
        "s3_bucket": "presento",
        "s3_access_key": SecretStr("test-access"),
        "s3_secret_key": SecretStr("test-secret"),
    }
    return Settings(_env_file=None, **{**base, **kw})


def test_presign_put_uses_public_endpoint_and_signs_content_type():
    url = S3Storage(_settings()).presign_put("slides/p-1/a.pdf", "application/pdf", 600)
    parsed = urlparse(url)
    assert parsed.netloc == "localhost:9000"
    assert parsed.path == "/presento/slides/p-1/a.pdf"
    q = parse_qs(parsed.query)
    assert q["X-Amz-Expires"] == ["600"]
    assert "content-type" in q["X-Amz-SignedHeaders"][0].split(";")
    assert "X-Amz-Signature" in q
    assert "test-secret" not in url


def test_presign_get_expiry():
    url = S3Storage(_settings()).presign_get("videos/a-1", 3600)
    q = parse_qs(urlparse(url).query)
    assert q["X-Amz-Expires"] == ["3600"]


def test_public_endpoint_falls_back_to_internal():
    url = S3Storage(_settings(s3_public_endpoint_url="")).presign_get("k", 60)
    assert urlparse(url).netloc == "minio:9000"


def test_blank_endpoint_settings_become_none():
    s = _settings(s3_endpoint_url="  ", s3_public_endpoint_url="")
    assert s.s3_endpoint_url is None
    assert s.s3_public_endpoint_url is None
