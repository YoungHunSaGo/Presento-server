"""발표 자료 PDF 업로드 확인.

명세에는 슬라이드 업로드 완료 알림 API가 없다. 그래서 P6에서 발급한 키를 대기 상태로 두고,
발표를 조회할 때 스토리지에 실제 파일이 있는지 확인되면 hasSlides로 반영한다.

- hasSlides가 한 번 true가 되면(slides_key 저장) 그 뒤로는 HEAD를 하지 않는다.
  다시 올릴 때는 P6가 같은 키로 덮어쓰는 URL을 주므로 확인할 필요가 없다.
- 모든 쓰기는 조건부 UPDATE로 해서, 동시에 P6가 키를 바꾼 경우 덮어쓰지 않는다.
"""

import logging
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.models import Presentation
from app.services.storage import Storage, StorageError, delete_quietly

logger = logging.getLogger(__name__)

# 업로드 URL이 만료되고도 이만큼 지나도록 파일이 없으면 대기를 포기한다.
PENDING_GRACE = timedelta(minutes=30)
# 스토리지 장애 시 목록 조회가 오래 멈추지 않게 요청당 HEAD 횟수를 제한한다.
MAX_CHECKS_PER_REQUEST = 5


def new_slides_key(presentation_id: str) -> str:
    return f"slides/{presentation_id}/{secrets.token_hex(8)}.pdf"


def _as_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def needs_check(p: Presentation) -> bool:
    return p.slides_key is None and p.slides_pending_key is not None


def refresh_pending_slides(
    db: Session,
    storage: Storage,
    presentations: list[Presentation],
    *,
    max_bytes: int,
    url_expires_sec: int,
    now: datetime | None = None,
) -> None:
    """대기 중인 업로드를 확인해 반영한다. 바뀐 게 있으면 커밋하고 객체를 새로 읽는다."""
    now = now or datetime.now(UTC)
    targets = [p for p in presentations if needs_check(p)][:MAX_CHECKS_PER_REQUEST]
    to_delete: list[str] = []
    touched: list[Presentation] = []

    for p in targets:
        pending = p.slides_pending_key
        if pending is None:
            continue
        try:
            size = storage.head(pending)
        except StorageError:
            logger.warning("Slides HEAD failed for %s", p.id, exc_info=True)
            continue

        same_pending = (Presentation.id == p.id) & (Presentation.slides_pending_key == pending)
        clear = {"slides_pending_key": None, "slides_pending_at": None}

        if size is None:
            issued = p.slides_pending_at
            deadline = timedelta(seconds=url_expires_sec) + PENDING_GRACE
            if issued is not None and now - _as_aware(issued) <= deadline:
                continue  # 아직 업로드 중일 수 있다
            stmt = update(Presentation).where(same_pending).values(**clear)
        elif size > max_bytes:
            # presigned PUT은 크기를 강제할 수 없어 여기서 거른다.
            logger.info("Slides for %s exceed limit (%d bytes); discarded", p.id, size)
            stmt = update(Presentation).where(same_pending).values(**clear)
        else:
            stmt = (
                update(Presentation)
                .where(same_pending, Presentation.slides_key.is_(None))
                .values(slides_key=pending, updated_at=now, **clear)
            )

        if db.execute(stmt).rowcount == 1:
            touched.append(p)
            if size is not None and size > max_bytes:
                to_delete.append(pending)

    if touched:
        db.commit()
        for p in touched:
            db.refresh(p)
        delete_quietly(storage, list(to_delete))
