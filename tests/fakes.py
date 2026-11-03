from app.services.storage import StorageError


class FakeStorage:
    """S3 대신 쓰는 메모리 스토리지. objects: key → 파일 크기."""

    def __init__(self) -> None:
        self.objects: dict[str, int] = {}
        self.deleted: list[str] = []
        self.presigned: list[tuple[str, str, str, int]] = []  # (method, key, content_type, exp)
        self.fail_head = False
        self.fail_delete = False

    def presign_put(self, key: str, content_type: str, expires_in: int) -> str:
        self.presigned.append(("PUT", key, content_type, expires_in))
        return f"https://fake-s3.local/presento/{key}?X-Amz-Signature=fake"

    def presign_get(self, key: str, expires_in: int) -> str:
        self.presigned.append(("GET", key, "", expires_in))
        return f"https://fake-s3.local/presento/{key}?X-Amz-Signature=fake"

    def head(self, key: str) -> int | None:
        if self.fail_head:
            raise StorageError("fake head failure")
        return self.objects.get(key)

    def delete(self, key: str) -> None:
        if self.fail_delete:
            raise StorageError("fake delete failure")
        self.objects.pop(key, None)
        self.deleted.append(key)
