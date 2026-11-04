# Presento Backend — Claude Code 작업 규칙

Presento는 발표 영상을 분석해 타임라인 기반 피드백 리포트를 주는 서비스다. 이 저장소는 백엔드 API와 AI 분석 파이프라인이다. 대회용 프로젝트이며, 모델을 직접 학습하지 않고 외부 API(OpenAI)와 MediaPipe를 조합한다.

## 0. 가장 중요한 규칙

1. **API 명세는 `docs/api-spec.md`가 기준이다.** 경로, 필드명(camelCase), 상태값, 에러 코드를 명세와 한 글자도 다르게 만들지 않는다. 명세를 바꿔야 하면 코드보다 먼저 사람에게 묻는다. 프론트 팀원이 이 명세로 이미 개발 중이다.
2. **비밀값을 절대 노출하지 않는다.** `.env`의 값(특히 `OPENAI_API_KEY`)을 출력·로그·커밋·테스트 픽스처에 넣지 않는다. `.env`는 읽을 필요가 없고, 설정은 `app/core/config.py`로만 접근한다. `.env`가 `.gitignore`에 있는지 커밋 전에 확인한다.
3. **수치는 코드가, 해석은 LLM이.** 말 속도·추임새 수·침묵·시선 비율·점수 같은 숫자는 전부 파이썬 코드로 계산한다. LLM은 계산된 수치와 대본을 입력받아 피드백 문장만 작성한다. LLM이 만든 숫자를 그대로 리포트에 넣지 않는다.
4. **완료 선언 전에 반드시 검증한다.** 기능 하나를 끝낼 때마다 (a) `pytest` 전체 통과, (b) `ruff check`, (c) 작성자와 분리된 리뷰 서브에이전트가 보안·예외 처리·성능·명세 일치 관점에서 비판적으로 검토, (d) 지적 사항 수정 후 다시 테스트. 테스트 없이 "완료"라고 하지 않는다.
5. **API 크레딧을 아낀다(잔액 약 19달러).** 테스트는 OpenAI 호출을 전부 목(mock)으로 대체한다. 실제 호출은 사람이 요청했을 때만, 30초 이하 샘플 영상으로 한다.

## 1. 기술 스택

| 영역 | 선택 |
| --- | --- |
| 언어 | Python 3.11+ |
| 웹 프레임워크 | FastAPI, Pydantic v2, pydantic-settings |
| DB | PostgreSQL + SQLAlchemy 2.0 + Alembic (테스트는 SQLite) |
| 작업 큐 | Redis + Celery |
| 파일 저장 | S3 호환 스토리지 (로컬은 MinIO, 배포는 AWS S3), boto3 presigned URL |
| 인증 | JWT (HS256, 24시간), 비밀번호는 bcrypt |
| 영상 처리 | FFmpeg (음성 추출: 모노 16kHz, 프레임 샘플링: 초당 1~2장) |
| 받아쓰기 | OpenAI `whisper-1` (`response_format=verbose_json`, `timestamp_granularities=["word","segment"]`) — 단어 타임스탬프가 필수라 다른 STT 모델로 바꾸지 않는다 |
| 피드백 생성 | OpenAI LLM, Structured Outputs(JSON 스키마)로 응답 강제. 모델명은 `.env`의 `OPENAI_LLM_MODEL` |
| 비언어 분석 | MediaPipe (Face Landmarker, Pose Landmarker) — 로컬 실행, API 비용 없음 |
| 테스트 | pytest, httpx(TestClient), 외부 호출은 전부 mock |
| 린트 | ruff |
| 로컬 실행 | docker-compose (api, worker, postgres, redis, minio) |

## 2. 폴더 구조

```
app/
  main.py                  # FastAPI 앱, CORS, 에러 핸들러, /api/v1 라우터 등록
  core/
    config.py              # pydantic-settings로 .env 로드
    security.py            # JWT, 비밀번호 해시
    errors.py              # AppError(code, message, status) + 명세 형식 에러 응답
  api/v1/
    auth.py                # A1~A3
    presentations.py       # P1~P7
    analyses.py            # N1~N6
  schemas/                 # Pydantic 요청/응답 모델 (camelCase alias)
  models/                  # SQLAlchemy 모델: User, Presentation, Analysis, Report
  services/
    storage.py             # presigned PUT/GET URL, 파일 존재 확인
    pipeline/
      audio.py             # FFmpeg 음성 추출, 프레임 샘플링
      stt.py               # whisper-1 호출, 결과를 segments/words로 정규화
      speech_metrics.py    # 말 속도(분당 음절), 추임새, 침묵, 시간 초과 계산
      nonverbal.py         # MediaPipe로 시선 정면 비율, 자세 안정성, 제스처 빈도
      content.py           # LLM으로 구조·명확성 평가
      feedback.py          # LLM 피드백 생성 + 스키마 검증 + 근거 대조 + 재시도
      scoring.py           # 항목별 점수, 목적별 가중치로 종합 점수
      report.py            # 최종 리포트 JSON 조립
      artifacts.py         # 회차별 중간 결과 저장·재계산(LLM 없이 scoring·report 다시)
  tools/
    recompute.py           # python -m app.tools.recompute: 중간 결과로 점수·리포트 재계산
  workers/
    celery_app.py
    tasks.py               # run_analysis(analysis_id): stage/progress 갱신하며 파이프라인 실행
  prompts/                 # 프롬프트 파일 (버전 이름 포함: feedback_v1.md 등)
tests/
  fixtures/                # 짧은 샘플 오디오, whisper 응답 JSON 샘플, 기대 리포트
eval/                      # 평가셋(사람 점수)과 AI 점수 비교 스크립트
docs/
  api-spec.md              # API 명세 (기준 문서)
```

## 3. 분석 파이프라인 규칙

- 단계와 진행률은 명세의 `stage` 값(`EXTRACTING_AUDIO` → `TRANSCRIBING` → `ANALYZING_NONVERBAL` → `ANALYZING_CONTENT` → `GENERATING_FEEDBACK`)과 progress 범위를 그대로 따른다.
- 한국어 말 속도는 **분당 음절 수**로 계산한다(공백·문장부호 제외 한글 글자 수 기준). 30초 구간별 `paceSeries`도 만든다.
- 추임새 사전은 `app/services/pipeline/fillers_ko.py`에 상수로 둔다(음, 어, 그, 저, 약간, 이제, 뭐, 막 등). 단어 경계로만 매칭해 "그래서"의 "그"를 세지 않는다.
- 침묵: 연속 단어 사이 공백이 2초 이상인 구간. 기준값은 설정 상수로 둔다.
- 얼굴 인식 신뢰도가 낮은 구간은 평가에서 빼고 `lowConfidenceRanges`로 돌려준다.
- 피드백 LLM 응답은 JSON 스키마로 강제하고, 다음 중 하나라도 실패하면 최대 2번 재생성한다. 그래도 실패한 항목은 버린다.
  - 스키마 검증 실패
  - `start`/`end`가 영상 길이 밖
  - `evidence` 문장이 실제 대본에 없음(공백 정규화 후 부분 일치)
- 외부 AI 호출 실패는 `AI_SERVICE_ERROR`로 분석을 `FAILED` 처리한다. 목소리가 없으면 `NO_SPEECH_DETECTED`, 얼굴이 거의 없으면 `NO_FACE_DETECTED`.
- 영상 10분 초과는 `VIDEO_TOO_LONG`. whisper 업로드 25MB 제한을 넘지 않게 음성은 모노 16kHz로 변환해 보낸다.
- 같은 영상 해시의 결과는 재사용해 API를 다시 호출하지 않는다.
- **평가와 시연은 Docker 안에서만 실행한다.** 같은 영상이라도 ffmpeg 버전(YUV→RGB 색 변환)과 MediaPipe 플랫폼 빌드(macOS/Linux)에 따라 랜드마크가 달라져 제스처 같은 지표가 바뀐다(실측: 같은 영상에서 분당 제스처 Docker 8.57 / Mac 13.33. 안정화 뒤 손이 가려진 이 영상은 양쪽 모두 측정 불가로 일치, 시선 0.25/0.23). 로컬(Mac) 실행 결과는 점수 비교에 쓰지 않는다.

## 4. 작업 순서 (마일스톤)

각 마일스톤이 끝날 때마다 0장 4번 검증을 하고, 커밋 후 사람에게 결과를 요약한다.

1. **뼈대:** 폴더 구조, config, 에러 형식, CORS(`http://localhost:5173`), `GET /api/v1/health`, docker-compose, pytest 설정.
2. **인증:** A1~A3.
3. **발표:** P1~P7 (P6은 presigned URL 발급까지).
4. **분석 API + 가짜 파이프라인:** N1~N6을 먼저 완성하고, 워커는 단계를 몇 초씩 진행시킨 뒤 `docs/api-spec.md` 5장의 예시 리포트를 저장한다. 프론트가 실제 서버에 일찍 붙을 수 있게 하기 위함이다.
5. **실제 파이프라인:** audio → stt → speech_metrics → nonverbal → content → feedback → scoring → report 순서로 하나씩 교체한다. 각 모듈은 단위 테스트를 먼저 쓴다.
6. **평가셋:** `eval/`에 샘플 영상별 사람 점수를 넣고, AI 점수와 차이를 출력하는 스크립트.

## 5. 코드 규칙

- 응답 스키마는 Pydantic 모델로 정의하고 `alias_generator=to_camel`, `populate_by_name=True`를 쓴다. 엔드포인트에 `response_model`을 반드시 단다.
- 에러는 `AppError`로만 던지고, 전역 핸들러가 명세 형식(`{"error": {"code", "message", "details"}}`)으로 바꾼다. FastAPI 기본 422 응답도 `VALIDATION_ERROR` 형식으로 바꾼다.
- 다른 사용자의 발표·회차 접근은 모든 엔드포인트에서 `FORBIDDEN`으로 막는다(테스트 필수).
- 사용자에게 보이는 `message`는 한국어 존댓말 해요체.
- 커밋은 마일스톤 또는 기능 단위로 작게. 커밋 메시지는 `feat:`, `fix:`, `test:`, `docs:`, `chore:` 접두사 + 한국어 설명.
- 새 패키지는 `pyproject.toml`(또는 `requirements.txt`)에 버전과 함께 추가한다.
