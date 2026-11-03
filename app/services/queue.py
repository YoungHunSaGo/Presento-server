"""API → 워커 작업 전달. API는 워커 코드를 import하지 않고 작업 이름으로만 보낸다."""

from typing import Protocol

TASK_RUN_ANALYSIS = "presento.run_analysis"
TASK_DELETE_OBJECTS = "presento.delete_objects"


class TaskQueue(Protocol):
    def enqueue_analysis(self, analysis_id: str) -> None: ...
    def enqueue_delete(self, keys: list[str]) -> None: ...


class CeleryTaskQueue:
    def enqueue_analysis(self, analysis_id: str) -> None:
        from app.workers.celery_app import celery_app

        celery_app.send_task(TASK_RUN_ANALYSIS, args=[analysis_id])

    def enqueue_delete(self, keys: list[str]) -> None:
        from app.workers.celery_app import celery_app

        celery_app.send_task(TASK_DELETE_OBJECTS, args=[keys])
