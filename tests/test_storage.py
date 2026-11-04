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


# ---------- 운영: AWS S3 + EC2 IAM 역할(키 없음) ----------


def _aws_role_env(monkeypatch, tmp_path):
    """EC2 인스턴스 역할 대신 boto3 기본 자격 증명 체인의 환경 변수로 임시 자격 증명을 준다.
    메타데이터 서버(IMDS)와 ~/.aws 파일은 보지 않게 막는다(네트워크 호출 없음)."""
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ASIAROLEKEY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "role-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "role-session-token")


def _aws_settings(**kw):
    base = {
        "s3_endpoint_url": "",
        "s3_public_endpoint_url": "",
        "s3_bucket": "presento-prod",
        "s3_access_key": SecretStr(""),
        "s3_secret_key": SecretStr(""),
    }
    return _settings(**{**base, **kw})


def test_blank_keys_use_default_credentials_on_aws(monkeypatch, tmp_path):
    _aws_role_env(monkeypatch, tmp_path)
    url = S3Storage(_aws_settings()).presign_put("videos/a-1", "video/webm", 900)
    parsed = urlparse(url)
    # 가상 호스트 방식 + 서울 리전
    assert parsed.netloc == "presento-prod.s3.ap-northeast-2.amazonaws.com"
    assert parsed.path == "/videos/a-1"
    q = parse_qs(parsed.query)
    assert q["X-Amz-Credential"][0].startswith("ASIAROLEKEY/")
    assert q["X-Amz-Credential"][0].split("/")[2] == "ap-northeast-2"
    assert q["X-Amz-Security-Token"] == ["role-session-token"]  # 역할의 임시 자격 증명
    assert "role-secret" not in url


def test_blank_keys_do_not_override_default_chain_with_empty_strings(monkeypatch, tmp_path):
    _aws_role_env(monkeypatch, tmp_path)
    storage = S3Storage(_aws_settings(s3_access_key=SecretStr("  "), s3_secret_key=SecretStr(" ")))
    creds = storage._client._request_signer._credentials
    assert creds.access_key == "ASIAROLEKEY"


def test_minio_keeps_path_style():
    url = S3Storage(_settings()).presign_get("k", 60)
    assert urlparse(url).path == "/presento/k"
