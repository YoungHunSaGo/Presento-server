"""S3 호환 스토리지: presigned PUT/GET URL 발급, 파일 존재 확인, 삭제."""

import logging
from typing import Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """스토리지 호출 실패(네트워크, 권한 등). 파일이 없는 경우는 에러가 아니다."""


class Storage(Protocol):
    def presign_put(self, key: str, content_type: str, expires_in: int) -> str: ...
    def presign_get(self, key: str, expires_in: int) -> str: ...
    def presign_get_internal(self, key: str, expires_in: int) -> str: ...
    def head(self, key: str) -> int | None: ...
    def delete(self, key: str) -> None: ...


class S3Storage:
    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket
        common = {
            "region_name": settings.s3_region,
            "aws_access_key_id": settings.s3_access_key.get_secret_value() or None,
            "aws_secret_access_key": settings.s3_secret_key.get_secret_value() or None,
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=3,
                read_timeout=10,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        }
        # 서버 → 스토리지 호출용(컨테이너 안 주소)
        self._client = boto3.client("s3", endpoint_url=settings.s3_endpoint_url, **common)
        # 브라우저용 URL 서명(서명은 네트워크 호출 없음). SigV4는 host가 서명에 들어가므로
        # 브라우저가 실제로 접속할 주소로 서명해야 한다.
        public = settings.s3_public_endpoint_url or settings.s3_endpoint_url
        self._signer = boto3.client("s3", endpoint_url=public, **common)

    def presign_put(self, key: str, content_type: str, expires_in: int) -> str:
        return self._signer.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires_in,
            HttpMethod="PUT",
        )

    def presign_get(self, key: str, expires_in: int) -> str:
        return self._signer.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_in
        )

    def presign_get_internal(self, key: str, expires_in: int) -> str:
        """서버(API·워커)가 ffprobe 등으로 직접 읽을 때 쓰는 URL. 컨테이너 안 주소로 서명한다."""
        return self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_in
        )

    def head(self, key: str) -> int | None:
        """파일 크기(바이트). 없으면 None."""
        try:
            res = self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise StorageError(str(e)) from e
        except BotoCoreError as e:
            raise StorageError(str(e)) from e
        return int(res["ContentLength"])

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except (ClientError, BotoCoreError) as e:
            raise StorageError(str(e)) from e


def delete_quietly(storage: Storage, keys: list[str | None]) -> None:
    """DB 커밋 뒤 정리용. 실패해도 요청은 성공시키고 로그만 남긴다."""
    for key in keys:
        if not key:
            continue
        try:
            storage.delete(key)
        except StorageError:
            logger.warning("Failed to delete storage object %s", key, exc_info=True)
