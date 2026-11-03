"""FastAPI 앱 팩토리.

실행: uvicorn app.main:create_app --factory
모듈 import만으로 .env를 읽지 않도록 전역 app 객체를 두지 않는다.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.api.v1 import api_router
from app.core.config import Settings, get_settings
from app.core.database import make_engine, make_session_factory
from app.core.errors import CatchAllMiddleware, register_error_handlers
from app.core.security import warm_dummy_hash
from app.schemas.common import ErrorResponse

_ERROR_REF = {"$ref": "#/components/schemas/ErrorResponse"}


def _install_openapi(app: FastAPI) -> None:
    """Swagger에 FastAPI 기본 422 대신 실제 응답인 400 VALIDATION_ERROR를 적는다."""

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                responses = operation.get("responses", {})
                if responses.pop("422", None) is not None:
                    responses["400"] = {
                        "description": "VALIDATION_ERROR",
                        "content": {"application/json": {"schema": _ERROR_REF}},
                    }
        components = schema.get("components", {}).get("schemas", {})
        components.pop("HTTPValidationError", None)
        components.pop("ValidationError", None)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 첫 '없는 이메일' 로그인이 더 느려 보이지 않게 더미 해시를 미리 만든다.
    warm_dummy_hash(app.state.settings.bcrypt_rounds)
    yield
    app.state.engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Presento API",
        version="0.1.0",
        lifespan=_lifespan,
        redirect_slashes=False,
        responses={500: {"model": ErrorResponse}},
    )
    app.state.settings = settings
    app.state.engine = make_engine(settings.database_url)
    app.state.session_factory = make_session_factory(app.state.engine)
    app.dependency_overrides[get_settings] = lambda: settings

    # 나중에 추가한 미들웨어가 바깥쪽이다: CORS(바깥) → CatchAll(안쪽) → 라우터.
    app.add_middleware(CatchAllMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,  # 쿠키 없이 Bearer 토큰만 사용
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    register_error_handlers(app)
    app.include_router(api_router)
    _install_openapi(app)
    return app
