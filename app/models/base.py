import secrets
from datetime import UTC, datetime

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# 제약조건 이름을 고정해 Alembic 마이그레이션과 모델이 항상 같은 이름을 쓰게 한다.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def new_id(prefix: str) -> str:
    """명세의 문자열 ID 형식(예: u-1a2b...)."""
    return f"{prefix}-{secrets.token_hex(8)}"


def utcnow() -> datetime:
    return datetime.now(UTC)
