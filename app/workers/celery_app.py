"""Celery 앱. 실행: celery -A app.workers.celery_app worker -l info

설정은 app.workers.celeryconfig에서 지연 로딩한다(import만으로 .env를 읽지 않게).
"""

from celery import Celery

celery_app = Celery("presento")
celery_app.config_from_object("app.workers.celeryconfig")
