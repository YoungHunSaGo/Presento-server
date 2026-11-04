"""analysis result reuse: video hash, cache key, pipeline version

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.add_column(sa.Column("video_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("cache_key", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("pipeline_version", sa.String(length=20), nullable=True))
        batch.create_index("ix_analyses_video_sha256", ["video_sha256"])
        batch.create_index("ix_analyses_cache_key", ["cache_key"])


def downgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.drop_index("ix_analyses_cache_key")
        batch.drop_index("ix_analyses_video_sha256")
        batch.drop_column("pipeline_version")
        batch.drop_column("cache_key")
        batch.drop_column("video_sha256")
