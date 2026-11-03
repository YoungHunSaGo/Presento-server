"""명세 형식 에러 응답: {"error": {"code", "message", "details"}}.

에러는 AppError로만 던지고, 여기 등록된 전역 핸들러가 명세 형식으로 바꾼다.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        details: Any = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


def _response(
    status_code: int, code: str, message: str, details: Any = None, headers: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_body(code, message, details),
        headers=headers,
    )


# Starlette가 직접 내는 HTTP 오류(없는 경로 등)를 명세 코드로 바꾼다.
_HTTP_STATUS_MAP: dict[int, tuple[str, str]] = {
    401: ("UNAUTHORIZED", "로그인이 필요해요."),
    403: ("FORBIDDEN", "접근 권한이 없어요."),
    404: ("NOT_FOUND", "요청한 정보를 찾을 수 없어요."),
}


async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return _response(exc.status_code, exc.code, exc.message, exc.details)


def _reason_ko(err: dict[str, Any]) -> str:
    """Pydantic 오류 타입을 한국어 이유로 바꾼다. 모르는 타입은 일반 문구로 대신한다."""
    ctx = err.get("ctx") or {}
    match err.get("type"):
        case "missing":
            return "필수 값이에요."
        case "json_invalid":
            return "JSON 형식이 올바르지 않아요."
        case "string_too_short":
            return f"{ctx.get('min_length')}자 이상 입력해 주세요."
        case "string_too_long":
            return f"{ctx.get('max_length')}자 이하로 입력해 주세요."
        case "int_parsing" | "int_type" | "float_parsing" | "float_type":
            return "숫자를 입력해 주세요."
        case "enum" | "literal_error":
            return "허용되지 않는 값이에요."
        case "greater_than" | "greater_than_equal" | "less_than" | "less_than_equal":
            return "허용 범위를 벗어난 값이에요."
        case "value_error" if "email" in str(err.get("msg", "")).lower():
            return "올바른 이메일 형식이 아니에요."
        case "string_type":
            return "문자열로 입력해 주세요."
        case "bool_type" | "bool_parsing":
            return "true 또는 false로 입력해 주세요."
        case "list_type":
            return "목록 형식으로 입력해 주세요."
        case "dict_type" | "model_type" | "model_attributes_type":
            return "JSON 객체 형식으로 보내 주세요."
        case _:
            return "형식이 올바르지 않아요."


async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ())]
        if loc and loc[0] in ("body", "query", "path", "header", "cookie"):
            loc = loc[1:]
        if err.get("type") == "json_invalid":
            loc = []  # loc가 문자 위치라 필드 이름이 아니다
        # 입력값(err["input"])은 비밀번호 등이 섞일 수 있어 응답에 넣지 않는다.
        details.append({"field": ".".join(loc), "reason": _reason_ko(err)})
    return _response(400, "VALIDATION_ERROR", "입력값을 다시 확인해 주세요.", details)


async def _http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    if exc.status_code in _HTTP_STATUS_MAP:
        code, message = _HTTP_STATUS_MAP[exc.status_code]
    elif exc.status_code >= 500:
        code, message = "INTERNAL_ERROR", "서버에 문제가 생겼어요. 잠시 후 다시 시도해 주세요."
    else:
        code, message = "VALIDATION_ERROR", "잘못된 요청이에요."
    return _response(exc.status_code, code, message, headers=getattr(exc, "headers", None))


class CatchAllMiddleware:
    """처리되지 않은 예외를 500 INTERNAL_ERROR로 바꾸는 ASGI 미들웨어.

    Starlette의 Exception 핸들러는 CORSMiddleware 바깥에서 돌아 500 응답에 CORS 헤더가
    빠진다. 이 미들웨어를 CORS보다 안쪽에 두어 프론트가 에러 본문을 읽을 수 있게 한다.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # 내부 예외 내용은 응답에 넣지 않고 서버 로그에만 남긴다.
            logger.exception("Unhandled error on %s %s", scope.get("method"), scope.get("path"))
            if started:
                raise
            response = _response(
                500, "INTERNAL_ERROR", "서버에 문제가 생겼어요. 잠시 후 다시 시도해 주세요."
            )
            await response(scope, receive, send)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)  # type: ignore[arg-type]
