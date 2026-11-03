"""발표 자료 PDF 업로드 확인.

명세에는 슬라이드 업로드 완료 알림 API가 없다. 그래서 P6에서 발급한 키를 대기 상태로 두고,
발표를 조회할 때 스토리지에 실제 파일이 있는지 확인되면 hasSlides로 반영한다.
"""

import logging
import secrets
from datetime import UTC, datetime, timedelta

from app.models import Presentation
from app.services.storage import Storage, StorageError

logger = logging.getLogger(__name__)

# 업로드 URL이 만료되고도 이만큼 지나도록 파일이 없으면 대기를 포기한다.
_PENDING_GRACE = timedelta(minutes=30)


def new_slides_key(presentation_id: str) -> str:
    return f"slides/{presentation_id}/{secrets.token_hex(8)}.pdf"


def _as_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def confirm_pending_slides(
    presentation: Presentation,
    storage: Storage,
    max_bytes: int,
    url_expires_sec: int,
    now: datetime | None = None,
) -> list[str]:
    """대기 중인 슬라이드가 업로드됐는지 확인해 반영한다.

    presentation을 수정만 하고 커밋은 호출자가 한다. 커밋 뒤 지울 스토리지 키 목록을 돌려준다.
    """
    pending = presentation.slides_pending_key
    if not pending:
        return []
    now = now or datetime.now(UTC)

    try:
        size = storage.head(pending)
    except StorageError:
        logger.warning("Slides HEAD failed for %s", presentation.id, exc_info=True)
        return []

    if size is None:
        issued = presentation.slides_pending_at
        deadline = timedelta(seconds=url_expires_sec) + _PENDING_GRACE
        if issued is None or now - _as_aware(issued) > deadline:
            presentation.slides_pending_key = None
            presentation.slides_pending_at = None
        return []

    presentation.slides_pending_key = None
    presentation.slides_pending_at = None
    if size > max_bytes:
        # presigned PUT은 크기를 강제할 수 없어 여기서 거른다.
        logger.info("Slides for %s exceed limit (%d bytes); discarded", presentation.id, size)
        return [pending]

    old = presentation.slides_key
    presentation.slides_key = pending
    presentation.updated_at = now  # hasSlides가 바뀌는 사용자에게 보이는 변경
    return [old] if old else []
