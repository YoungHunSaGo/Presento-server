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
worker_prefetch_multiplier = 1
# FFmpeg·외부 API가 멈춰도 워커 슬롯을 영원히 잡지 않게 한다.
task_soft_time_limit = 15 * 60
task_time_limit = 16 * 60
