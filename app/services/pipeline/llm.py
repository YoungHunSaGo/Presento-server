"""OpenAI LLM 호출 공용 도우미(content, feedback).

- Responses API + Structured Outputs(strict JSON 스키마). gpt-5 계열은 reasoning effort low.
- API 실패는 SDK가 이미 재시도하므로 여기서는 다시 보내지 않고 LlmCallError로 바꾼다.
  원본 예외(요청 헤더에 API 키)는 cause·context로도 달고 다니지 않는다.
- 호출 로그: 모델·토큰 수·예상 비용만. 키와 입력 전문은 남기지 않는다.
"""

import logging
from typing import Any, Protocol

import openai

from app.core.config import Settings

logger = logging.getLogger(__name__)

REASONING_EFFORT = "low"


class LlmCallError(Exception):
    """OpenAI 호출 자체가 실패했다(네트워크, 4xx/5xx)."""


class _ResponsesAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class LlmClient(Protocol):
    """openai.OpenAI 중 쓰는 부분. 테스트에서 가짜로 바꾼다."""

    responses: _ResponsesAPI


def make_client(settings: Settings) -> LlmClient:
    return openai.OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_sec,
        max_retries=settings.openai_max_retries,
    )


def structured_kwargs(
    *,
    model: str,
    name: str,
    schema: dict[str, Any],
    instructions: str,
    user_input: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "store": False,
        "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
    }
    if model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": REASONING_EFFORT}
    return kwargs


def call(client: LlmClient, kwargs: dict[str, Any], label: str) -> Any:
    """responses.create 한 번. 실패하면 LlmCallError(상태 코드·종류만)."""
    failure: str | None = None
    response: Any = None
    try:
        response = client.responses.create(**kwargs)
    except openai.APIStatusError as e:
        failure = f"status={e.status_code} type={type(e).__name__}"
    except openai.OpenAIError as e:
        failure = f"type={type(e).__name__}"
    if failure is not None:
        logger.warning("%s llm request failed: %s", label, failure)
        raise LlmCallError(f"llm request failed: {failure}")
    return response


def _int(v: Any) -> int:
    return int(v) if isinstance(v, int | float) and not isinstance(v, bool) else 0


def usage_tokens(response: Any) -> tuple[int, int, int, int]:
    """(입력, 출력, 그중 추론, 그중 캐시된 입력) 토큰."""
    usage = getattr(response, "usage", None)
    out_details = getattr(usage, "output_tokens_details", None)
    in_details = getattr(usage, "input_tokens_details", None)
    return (
        _int(getattr(usage, "input_tokens", 0)),
        _int(getattr(usage, "output_tokens", 0)),
        _int(getattr(out_details, "reasoning_tokens", 0)),
        _int(getattr(in_details, "cached_tokens", 0)),
    )


def estimate_cost(settings: Settings, input_tokens: int, output_tokens: int) -> float:
    """예상 비용(USD). 캐시 할인은 빼고 계산한다(실제보다 조금 높게 잡힘)."""
    return (
        input_tokens * settings.openai_llm_input_usd_per_1m
        + output_tokens * settings.openai_llm_output_usd_per_1m
    ) / 1_000_000


def log_usage(
    label: str, model: str, attempt: int, response: Any, settings: Settings
) -> tuple[int, int]:
    inp, out, reasoning, cached = usage_tokens(response)
    logger.info(
        "%s llm call: model=%s attempt=%d input_tokens=%d cached_tokens=%d "
        "output_tokens=%d reasoning_tokens=%d est_cost_usd=%.5f",
        label,
        model,
        attempt,
        inp,
        cached,
        out,
        reasoning,
        estimate_cost(settings, inp, out),
    )
    return inp, out


def is_refusal(response: Any) -> bool:
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return True
    return False


def incomplete_reason(response: Any) -> str | None:
    """잘린 응답이면 이유("max_output_tokens" 등), 아니면 None."""
    if getattr(response, "status", None) != "incomplete":
        return None
    return getattr(getattr(response, "incomplete_details", None), "reason", None) or "unknown"
