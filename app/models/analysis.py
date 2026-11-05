from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, new_id, utcnow
from app.models.enums import AnalysisStatus

if TYPE_CHECKING:
    from app.models.artifact import AnalysisArtifact
    from app.models.presentation import Presentation
    from app.models.report import Report


class Analysis(Base):
    """발표 한 회차."""

    __tablename__ = "analyses"
    __table_args__ = (
        UniqueConstraint("presentation_id", "attempt_no"),
        Index(
            "ix_analyses_presentation_id_status_completed_at",
            "presentation_id",
            "status",
            "completed_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("a"))
    presentation_id: Mapped[str] = mapped_column(
        ForeignKey("presentations.id", ondelete="CASCADE"), index=True
    )
    attempt_no: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default=AnalysisStatus.WAITING_UPLOAD)
    stage: Mapped[str | None] = mapped_column(String(30))
    progress: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    error_code: Mapped[str | None] = mapped_column(String(40))
    error_message: Mapped[str | None] = mapped_column(String(255))

    # 업로드 정보(N1)
    source: Mapped[str | None] = mapped_column(String(20))
    content_type: Mapped[str | None] = mapped_column(String(50))
    file_size: Mapped[int | None] = mapped_column(BigInteger)
    video_key: Mapped[str | None] = mapped_column(String(255))
    duration_sec: Mapped[float | None] = mapped_column(Float)

    # 완료 시 채우는 점수 요약(P2·P3·P7 및 M1~M3 마이페이지용). 전체 리포트는 Report에 둔다.
    overall_score: Mapped[int | None] = mapped_column(Integer)
    grade: Mapped[str | None] = mapped_column(String(4))
    scores: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    metric_details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    filler_counts: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    feedback_counts: Mapped[dict[str, int] | None] = mapped_column(JSON)
    filler_count: Mapped[int | None] = mapped_column(Integer)
    syllables_per_min: Mapped[int | None] = mapped_column(Integer)

    # 결과 재사용(같은 영상 해시). cache_key = 영상 해시 + 목적·제한 시간·슬라이드 + 파이프라인 버전
    video_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    cache_key: Mapped[str | None] = mapped_column(String(64), index=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(20))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    presentation: Mapped[Presentation] = relationship(back_populates="analyses")
    report: Mapped[Report | None] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    # 파이프라인 중간 결과(사용자 대본 포함). 회차와 함께 지운다.
    artifact: Mapped[AnalysisArtifact | None] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
