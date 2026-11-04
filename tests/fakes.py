from app.services.storage import StorageError


class FakeStorage:
    """S3 대신 쓰는 메모리 스토리지. objects: key → 파일 크기."""

    def __init__(self) -> None:
        self.objects: dict[str, int] = {}
        self.deleted: list[str] = []
        self.presigned: list[tuple[str, str, str, int]] = []  # (method, key, content_type, exp)
        self.head_calls: list[str] = []
        self.fail_head = False
        self.fail_delete = False
        self.files: dict[str, bytes] = {}  # download 내용
        self.downloads: list[str] = []
        self.fail_download = False

    def presign_put(self, key: str, content_type: str, expires_in: int) -> str:
        self.presigned.append(("PUT", key, content_type, expires_in))
        return f"https://fake-s3.local/presento/{key}?X-Amz-Signature=fake"

    def presign_get(self, key: str, expires_in: int) -> str:
        self.presigned.append(("GET", key, "", expires_in))
        return f"https://fake-s3.local/presento/{key}?X-Amz-Signature=fake"

    def presign_get_internal(self, key: str, expires_in: int) -> str:
        return f"http://fake-minio:9000/presento/{key}?X-Amz-Signature=fake"

    def head(self, key: str) -> int | None:
        self.head_calls.append(key)
        if self.fail_head:
            raise StorageError("fake head failure")
        return self.objects.get(key)

    def delete(self, key: str) -> None:
        if self.fail_delete:
            raise StorageError("fake delete failure")
        self.objects.pop(key, None)
        self.deleted.append(key)

    def download(self, key: str, dest) -> None:
        """files에 넣은 내용(없으면 크기만큼 0바이트)을 dest에 쓴다."""
        self.downloads.append(key)
        if self.fail_download or key not in self.objects and key not in self.files:
            raise StorageError("fake download failure")
        dest.write_bytes(self.files.get(key, b"\0" * self.objects.get(key, 0)))


class FakeQueue:
    """Celery 대신 쓰는 큐. 넣은 작업만 기록한다."""

    def __init__(self) -> None:
        self.analyses: list[str] = []
        self.deletes: list[list[str]] = []
        self.fail = False

    def enqueue_analysis(self, analysis_id: str) -> None:
        if self.fail:
            raise ConnectionError("broker down")
        self.analyses.append(analysis_id)

    def enqueue_delete(self, keys: list[str]) -> None:
        if self.fail:
            raise ConnectionError("broker down")
        self.deletes.append(list(keys))

    @property
    def deleted_keys(self) -> set[str]:
        return {k for batch in self.deletes for k in batch}


class FakeProbe:
    """ffprobe 대신. 기본 30초짜리 영상."""

    def __init__(self, duration: float | None = 30.0) -> None:
        self.duration = duration
        self.error: Exception | None = None
        self.calls: list[tuple[str, str, float, bool]] = []

    def __call__(self, source: str, content_type: str, timeout: float, full_scan: bool):
        self.calls.append((source, content_type, timeout, full_scan))
        if self.error:
            raise self.error
        return self.duration
