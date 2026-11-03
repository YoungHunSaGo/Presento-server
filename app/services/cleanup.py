"""DB에서 지운 발표·회차의 스토리지 파일 정리."""

import logging

from app.services.queue import TaskQueue
from app.services.storage import Storage, delete_quietly

logger = logging.getLogger(__name__)


def delete_objects_later(queue: TaskQueue, storage: Storage, keys: list[str | None]) -> None:
    """워커에게 삭제를 맡긴다. 큐가 죽어 있으면 요청 안에서 바로 지운다(실패는 로그만)."""
    real = [k for k in keys if k]
    if not real:
        return
    try:
        queue.enqueue_delete(real)
    except Exception:
        logger.warning("Failed to enqueue delete of %d objects; deleting inline", len(real))
        delete_quietly(storage, list(real))
