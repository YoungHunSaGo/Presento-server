"""analysis pipeline: analysis status fields, reports, usage, attempt_seq

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ANALYSIS_COLUMNS = [
    sa.Column("stage", sa.String(length=30), nullable=True),
    sa.Column("progress", sa.Integer(), server_default="0", nullable=False),
    sa.Column("error_code", sa.String(length=40), nullable=True),
    sa.Column("error_message", sa.String(length=255), nullable=True),
    sa.Column("source", sa.String(length=20), nullable=True),
    sa.Column("content_type", sa.String(length=50), nullable=True),
    sa.Column("file_size", sa.BigInteger(), nullable=True),
    sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
]


def upgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        for column in _ANALYSIS_COLUMNS:
            batch.add_column(column)

    with op.batch_alter_table("presentations") as batch:
        batch.add_column(sa.Column("attempt_seq", sa.Integer(), server_default="0", nullable=False))
    # 기존 회차가 있으면 번호를 이어서 쓰도록 맞춘다.
    op.execute(
        "UPDATE presentations SET attempt_seq = COALESCE("
        "(SELECT MAX(attempt_no) FROM analyses WHERE analyses.presentation_id = presentations.id)"
        ", 0)"
    )

    op.create_table(
        "reports",
        sa.Column("analysis_id", sa.String(length=32), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["analyses.id"],
            name=op.f("fk_reports_analysis_id_analyses"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("analysis_id", name=op.f("pk_reports")),
    )

    op.create_table(
        "analysis_usage",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("analysis_id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_analysis_usage_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analysis_usage")),
    )
    op.create_index(
        "ix_analysis_usage_user_id_created_at",
        "analysis_usage",
        ["user_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_usage_user_id_created_at", table_name="analysis_usage")
    op.drop_table("analysis_usage")
    op.drop_table("reports")
    with op.batch_alter_table("presentations") as batch:
        batch.drop_column("attempt_seq")
    with op.batch_alter_table("analyses") as batch:
        for column in reversed(_ANALYSIS_COLUMNS):
            batch.drop_column(column.name)
