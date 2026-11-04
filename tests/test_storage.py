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


# ---------- 집 서버: 서버 안은 http://minio:9000, 브라우저용 서명은 공개 주소 ----------

HOME_PUBLIC = "http://100.101.102.103:9000"  # Tailscale 주소 예시


def _home_settings(**kw):
    base = {"s3_endpoint_url": "http://minio:9000", "s3_public_endpoint_url": HOME_PUBLIC}
    return _settings(**{**base, **kw})


class _Raw:
    def stream(self, **_kw):
        yield b""


def _capture_requests(client, status=404):
    """보내려던 요청 URL을 기록하고 네트워크 대신 가짜 응답을 돌려준다(외부 호출 없음)."""
    from botocore.awsrequest import AWSResponse

    sent: list[str] = []

    def before_send(request, **_kw):
        sent.append(request.url)
        return AWSResponse(request.url, status, {}, _Raw())

    client.meta.events.register("before-send.s3", before_send)
    return sent


def test_home_presigned_urls_use_public_address():
    storage = S3Storage(_home_settings())
    put = urlparse(storage.presign_put("videos/a-1", "video/webm", 900))
    get = urlparse(storage.presign_get("videos/a-1", 3600))
    for url in (put, get):
        assert f"{url.scheme}://{url.netloc}" == HOME_PUBLIC  # 브라우저가 접속할 호스트로 서명
        assert url.path == "/presento/videos/a-1"  # MinIO는 경로 방식


def test_home_server_side_calls_use_internal_address():
    storage = S3Storage(_home_settings())
    sent = _capture_requests(storage._client)
    assert storage.head("videos/a-1") is None  # 404 → 아직 업로드 안 됨
    assert sent and all(urlparse(u).netloc == "minio:9000" for u in sent)
    internal = urlparse(storage.presign_get_internal("videos/a-1", 60))
    assert internal.netloc == "minio:9000"  # 워커의 ffprobe도 서버 안 주소로


def test_home_public_address_unset_keeps_previous_behavior():
    # 공개 주소를 비우면 예전처럼 내부 주소로 서명한다
    storage = S3Storage(_home_settings(s3_public_endpoint_url=""))
    assert urlparse(storage.presign_put("k", "video/webm", 60)).netloc == "minio:9000"
    assert urlparse(storage.presign_get("k", 60)).netloc == "minio:9000"
