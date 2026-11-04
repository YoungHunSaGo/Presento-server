"""애플리케이션 설정. 모든 설정은 이 모듈의 get_settings()로만 접근한다.

비밀값은 SecretStr로 감싸 repr/로그에 값이 찍히지 않게 한다.
"""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # 설정 검증 실패 메시지에 입력값(비밀값)을 넣지 않는다(시작 실패 로그로 새지 않게).
        hide_input_in_errors=True,
    )

    # OpenAI
    openai_api_key: SecretStr
    openai_llm_model: str = ""
    # 받아쓰기 모델은 whisper-1로 고정(stt.WHISPER_MODEL). 단어 타임스탬프가 필수라 설정에 없다.
    # OpenAI 호출 제한 시간(초)과 SDK 자동 재시도 횟수. 10분 음성 받아쓰기도 들어오게 넉넉히.
    # 시간 초과 후 재시도는 이중 과금될 수 있어 재시도는 1번만.
    openai_timeout_sec: float = Field(default=180.0, gt=0, le=600)
    openai_max_retries: int = Field(default=1, ge=0, le=3)
    # 내용 분석·피드백 LLM 예상 비용 계산용 단가(USD / 100만 토큰). 기본값은 gpt-5-mini.
    # 모델을 바꾸면 .env로 맞춘다.
    openai_llm_input_usd_per_1m: float = Field(default=0.25, ge=0)
    openai_llm_output_usd_per_1m: float = Field(default=2.0, ge=0)

    # 인증
    jwt_secret: SecretStr
    jwt_expire_hours: int = Field(default=24, ge=1)
    bcrypt_rounds: int = Field(default=12, ge=4, le=16)

    # DB / 큐
    database_url: str = "postgresql+psycopg://presento:presento@localhost:5432/presento"
    redis_url: str = "redis://localhost:6379/0"

    # 파일 저장
    s3_endpoint_url: str | None = None
    # presigned URL 서명용 주소. 브라우저가 접속할 수 있어야 한다(compose: http://localhost:9000).
    # 비우면 s3_endpoint_url을 쓴다(AWS S3는 둘 다 비움).
    s3_public_endpoint_url: str | None = None
    s3_bucket: str = "presento"
    # 둘 다 비우면 boto3 기본 자격 증명(EC2 인스턴스 IAM 역할, AWS_* 환경 변수 등)을 쓴다.
    # 하나만 채우는 것은 설정 실수라 시작할 때 막는다.
    s3_access_key: SecretStr = SecretStr("")
    s3_secret_key: SecretStr = SecretStr("")
    s3_region: str = "ap-northeast-2"

    # CORS (쉼표로 구분)
    cors_origins: str = "http://localhost:5173"

    # 비용 보호
    max_video_seconds: int = 600
    max_video_bytes: int = 524_288_000
    max_slides_bytes: int = 20 * 1024 * 1024
    daily_analysis_limit: int = 20

    # presigned URL 만료(초): 명세 P6 600, N1 900, N5 3600
    slides_upload_url_expires_sec: int = 600
    video_upload_url_expires_sec: int = 900
    video_play_url_expires_sec: int = 3600

    # MediaPipe 모델 파일(Dockerfile이 빌드 때 받아 둔다). 로컬 실행은 .env로 경로 지정
    mediapipe_face_model_path: str = "/opt/models/face_landmarker.task"
    mediapipe_pose_model_path: str = "/opt/models/pose_landmarker_full.task"

    # true면 실제 분석 대신 명세 예시 리포트를 쓰는 가짜 파이프라인(프론트 개발용, API 비용 없음)
    use_fake_pipeline: bool = False
    # 워커 임시 파일 위치(비우면 시스템 임시 폴더). 분석이 끝나면 성공·실패 모두 지운다.
    work_dir: str | None = None

    # 가짜 파이프라인(마일스톤 4): 단계당 걸리는 시간(초). 명세 요청: 1~2초
    fake_stage_seconds: float = Field(default=1.5, ge=0, le=10)
    # N2에서 영상 길이를 빠르게 확인할 때 ffprobe 제한 시간(초)
    start_probe_timeout_sec: float = 10.0

    @field_validator("s3_endpoint_url", "s3_public_endpoint_url", mode="before")
    @classmethod
    def _blank_url_is_none(cls, v: object) -> object:
        # .env의 "S3_PUBLIC_ENDPOINT_URL=" 같은 빈 값은 "설정 안 함"으로 본다.
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("openai_api_key")
    @classmethod
    def _openai_key_not_empty(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY must not be empty")
        return v

    @field_validator("jwt_secret")
    @classmethod
    def _jwt_secret_strong(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")
        return v

    @model_validator(mode="after")
    def _s3_keys_pair(self) -> "Settings":
        has_key = bool(self.s3_access_key.get_secret_value().strip())
        has_secret = bool(self.s3_secret_key.get_secret_value().strip())
        if has_key != has_secret:
            raise ValueError(
                "S3_ACCESS_KEY and S3_SECRET_KEY must both be set or both be empty (IAM role)"
            )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
