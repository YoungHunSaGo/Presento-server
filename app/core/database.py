"""DB 엔진과 세션 팩토리."""

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def _enable_sqlite_foreign_keys(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        # 테스트용 인메모리 SQLite: 모든 스레드가 같은 연결을 공유해야 테이블이 보인다.
        engine = create_engine(url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
        # PostgreSQL처럼 ON DELETE CASCADE가 동작하게 한다.
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
        return engine
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
