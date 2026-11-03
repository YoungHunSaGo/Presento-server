"""Celery 작업. 실제 로직은 app.services에 두고 여기서는 자원만 연결한다."""

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.database import make_engine, make_session_factory
from app.services.media import probe_duration
from app.services.pipeline.fake import run_fake_pipeline
from app.services.queue import TASK_DELETE_OBJECTS, TASK_RUN_ANALYSIS
from app.services.storage import S3Storage, StorageError
from app.workers.celery_app import celery_app


@lru_cache
def _resources():
    # 워커 프로세스(포크 후)마다 한 번 만든다.
    settings: Settings = get_settings()
    session_factory = make_session_factory(make_engine(settings.database_url))
    return settings, session_factory, S3Storage(settings)


@celery_app.task(name=TASK_RUN_ANALYSIS)
def run_analysis(analysis_id: str) -> str:
    settings, session_factory, storage = _resources()
    return run_fake_pipeline(
        analysis_id,
        session_factory=session_factory,
        storage=storage,
        settings=settings,
        probe=probe_duration,
    )


@celery_app.task(
    name=TASK_DELETE_OBJECTS,
    autoretry_for=(StorageError,),
    retry_backoff=True,
    max_retries=5,
)
def delete_objects(keys: list[str]) -> int:
    _, _, storage = _resources()
    for key in keys:
        storage.delete(key)  # S3 삭제는 멱등이라 재시도해도 안전
    return len(keys)
