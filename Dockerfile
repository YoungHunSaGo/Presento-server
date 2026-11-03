# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg: 음성 추출·프레임 샘플링
# libegl1 libgles2 libgl1 libglib2.0-0: MediaPipe(Face/Pose Landmarker)가 불러오는 공유 라이브러리
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libegl1 libgles2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# MediaPipe 모델 파일은 저장소에 커밋하지 않고 빌드 때 공식 URL에서 받는다.
# --checksum으로 SHA-256을 검증하므로 파일이 바뀌면 빌드가 실패한다.
# URL에서 받은 파일은 기본 권한이 600이라 비루트 사용자가 읽도록 644로 둔다.
# ADD가 자동으로 만드는 상위 디렉터리도 비루트가 못 들어가므로 미리 755로 만든다.
RUN install -d -m 755 /opt/models
ADD --checksum=sha256:64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff \
    --chmod=644 \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    /opt/models/face_landmarker.task
ADD --checksum=sha256:5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1 \
    --chmod=644 \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task \
    /opt/models/pose_landmarker_full.task
ENV MEDIAPIPE_FACE_MODEL_PATH=/opt/models/face_landmarker.task \
    MEDIAPIPE_POSE_MODEL_PATH=/opt/models/pose_landmarker_full.task

WORKDIR /srv

# 의존성 레이어를 먼저 설치해 코드만 바뀌면 캐시를 재사용한다.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install . && pip uninstall -y presento-server && rm -rf app build *.egg-info
COPY app ./app
RUN pip install --no-deps .
COPY alembic.ini ./
COPY alembic ./alembic

RUN useradd --create-home --uid 1000 presento
USER presento

EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
