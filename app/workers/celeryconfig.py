"""Celery 설정 모듈. Celery가 설정에 처음 접근할 때 import된다."""

from app.core.config import get_settings

_settings = get_settings()

broker_url = _settings.redis_url
result_backend = _settings.redis_url
result_expires = 60 * 60 * 24

task_serializer = "json"
accept_content = ["json"]
result_serializer = "json"
timezone = "Asia/Seoul"

task_acks_late = True
# 워커 프로세스가 죽으면(OOM, 강제 종료) 메시지를 버리지 않고 다시 넣는다.
# 다시 받은 작업은 _claim의 조건 때문에 중복 실행되지 않고, 끝내 멈춘 회차는 N3에서 FAILED 처리된다.
task_reject_on_worker_lost = True
worker_prefetch_multiplier = 1
# FFmpeg·외부 API가 멈춰도 워커 슬롯을 영원히 잡지 않게 한다.
task_soft_time_limit = 15 * 60
task_time_limit = 16 * 60

# 워커가 시작할 때 작업 모듈을 등록한다(API는 이름으로만 send_task).
imports = ("app.workers.tasks",)

# Redis가 죽어 있을 때 API 요청이 오래 멈추지 않게 한다.
broker_connection_timeout = 3
broker_connection_retry_on_startup = True
task_publish_retry = True
task_publish_retry_policy = {
    "max_retries": 2,
    "interval_start": 0,
    "interval_step": 0.5,
    "interval_max": 1,
}
