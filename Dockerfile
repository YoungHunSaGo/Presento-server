FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# 의존성 레이어를 먼저 설치해 코드만 바뀌면 캐시를 재사용한다.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install . && pip uninstall -y presento-server && rm -rf app build *.egg-info
COPY app ./app
RUN pip install --no-deps .

RUN useradd --create-home --uid 1000 presento
USER presento

EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
