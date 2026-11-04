"""내용 분석: LLM으로 발표 구조·명확성 평가, 슬라이드 일치도, 예상 질문.

OpenAI Responses API + Structured Outputs(JSON 스키마, strict)로 응답 형식을 강제한다.
LLM은 문장 ID만 쓰고, 시간(초)은 코드가 대본에서 찾아 붙인다(수치는 코드가, 해석은 LLM이).

숫자 검사(NumberGuard): LLM 글에 시간·점수·지표 숫자가 있으면 그 항목을 버린다.
- "12초", "3분", "1:30", "80점", "85%", "341음절", "12번", "3회", "1.5배" 같은 숫자+단위
- 수치 요약(metrics_summary)에 있는 값과 같은 숫자("341", "0.42", 비율이면 "42")
단, 같은 숫자+단위가 대본·슬라이드에 그대로 있으면 발표 내용을 옮긴 것이라 허용한다
("1분 자기소개", "매출 30% 성장"). "1분기", "프로젝트"처럼 단위가 아닌 낱말은 처음부터 안 잡는다.

입력 자르기(토큰 상한):
- 문장 하나는 MAX_SEGMENT_CHARS까지.
- 대본 전체가 MAX_TRANSCRIPT_CHARS를 넘으면 앞(HEAD_RATIO)과 뒤를 남기고 가운데를 줄인다
  (도입·결론이 구조 평가에 가장 중요하다). 줄인 범위는 "(중략: s-xxx ~ s-yyy)"로 알려준다.
  LLM이 보지 못한(줄인) 문장 ID를 쓰면 지어낸 것으로 보고 버린다.
- 슬라이드는 MAX_SLIDE_PAGES 페이지, 페이지당 MAX_SLIDE_PAGE_CHARS, 합계 MAX_SLIDES_CHARS까지.

재시도(최대 MAX_RETRIES번 더). 그래도 실패하면 ContentServiceError.
- 응답이 JSON 스키마에 맞지 않음
- 쓸 수 있는 예상 질문이 MIN_QUESTIONS개 미만
- 출력 토큰 상한으로 잘림: 같은 상한으로 다시 보내면 또 잘리므로 상한을 MAX_OUTPUT_TOKENS_RETRY로
  한 번만 올려 재시도하고, 그래도 잘리면 바로 실패
바로 실패(재시도 안 함, 같은 결과라 크레딧만 쓴다): 거절(refusal), 그 밖의 잘림, API 호출 실패
(SDK가 이미 재시도함).
항목 단위로 버리는 경우: 없는(또는 보지 못한) 문장 ID, 너무 흩어진 문장 묶음, 숫자 검사에 걸린 글,
없는 슬라이드 페이지. keyMessage가 숫자 검사에 걸리면 None.

말투: 모든 글은 해요체. 합쇼체("~습니다/합니다")로 끝나는 문장이 있으면 경고 로그만 남긴다
(재시도하지 않음).

로그: 호출마다 모델·토큰 수·예상 비용(USD)만 남긴다. API 키와 대본·슬라이드 전문은 남기지 않는다.

에러 구분: ContentServiceError → AI_SERVICE_ERROR
"""

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.config import Settings
from app.models.enums import Purpose
from app.schemas.report import ExpectedQuestion, SlideCoverage, Transcript
from app.services.pipeline import llm
from app.services.pipeline.korean_style import count_formal
from app.services.pipeline.llm import LlmClient

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v2"  # 지시문(content_system_*.md)
RUBRIC_VERSION = "v1"  # 목적별 루브릭(content_rubric_*_*.md)
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# ---- 입력 상한 ----
MAX_SEGMENT_CHARS = 300
MAX_TRANSCRIPT_CHARS = 9000  # 10분 발표(약 3500음절)는 보통 다 들어간다
HEAD_RATIO = 0.6
MAX_SLIDE_PAGES = 40
MAX_SLIDE_PAGE_CHARS = 500
MAX_SLIDES_CHARS = 6000

# ---- 호출 ----
MAX_OUTPUT_TOKENS = 6000  # gpt-5 계열은 추론 토큰도 여기에 포함된다
MAX_OUTPUT_TOKENS_RETRY = 10000  # 상한으로 잘렸을 때 한 번만 올려 본다
MAX_RETRIES = 2
REASONING_EFFORT = "low"

# ---- 출력 ----
MAX_ISSUES = 6
MAX_ISSUE_SPAN_SEGMENTS = 8  # 문제 하나가 가리키는 첫~끝 문장 간격(문장 수) 상한
QUESTION_COUNT = 5
MIN_QUESTIONS = 3

# 숫자 + 시간·점수·지표 단위. 단위 뒤에 다른 글자가 붙어 다른 낱말이 되는 경우는 뺀다.
_UNIT_NUMBER = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:"
    r"초(?!등|반|안|과|기|대)"
    r"|분(?!기|면|위|야|석|량|할|류|배|모|자|당|산|포|리|노|위기)"
    r"|시간(?!표|대)"
    r"|점(?!검|포|유|프|심|선|자)"
    r"|%|퍼센트|프로(?!젝트|그램|세스|필|모|토|덕|그)"
    r"|음절"
    r"|번(?!째|지|호|길|역)"
    r"|회(?!차|사|의|장|계|원|로|관|식|색)"
    r"|배(?!송|치|포|경|터|우|달|당|열)"
    r")|\b\d{1,2}:\d{2}\b"
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SEGMENT_ID = re.compile(r"^s-\d{3,}$")


class ContentServiceError(Exception):
    """LLM 호출 실패, 또는 재시도해도 쓸 수 있는 응답을 못 받았다."""


# ---------------------------------------------------------------- 결과


class SectionKind(StrEnum):
    INTRO = "INTRO"
    BODY = "BODY"
    CONCLUSION = "CONCLUSION"


class IssueType(StrEnum):
    STRUCTURE = "STRUCTURE"
    CLARITY = "CLARITY"


@dataclass(frozen=True)
class Section:
    kind: SectionKind
    start_id: str
    end_id: str
    start: float
    end: float


@dataclass(frozen=True)
class ContentIssue:
    type: IssueType
    segment_ids: list[str]
    start: float
    end: float
    reason: str


@dataclass(frozen=True)
class LlmUsage:
    model: str
    attempts: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class ContentAnalysis:
    sections: list[Section]
    key_message: str | None
    issues: list[ContentIssue]
    slide_coverage: list[SlideCoverage] | None  # 슬라이드가 없으면 None(명세 5장)
    expected_questions: list[ExpectedQuestion]
    usage: LlmUsage | None  # LLM을 부르지 않았으면 None
    transcript_truncated: bool = False
    dropped_items: int = 0
    # LLM이 판단을 빠뜨린 페이지, 페이지 상한(MAX_SLIDE_PAGES)으로 보내지 않은 페이지가 있는지
    slides_missing_pages: list[int] = field(default_factory=list)
    slides_truncated: bool = False


@dataclass(frozen=True)
class ContentInput:
    transcript: Transcript
    purpose: Purpose
    metrics_summary: dict[str, Any] = field(default_factory=dict)  # 이름 → 값(숫자·None)
    slides: list[str] | None = None  # 페이지별 텍스트(1페이지부터)


# ---------------------------------------------------------------- LLM 응답 스키마


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Range(_Strict):
    startId: str  # noqa: N815 - LLM 스키마 필드명
    endId: str  # noqa: N815


class _Sections(_Strict):
    intro: _Range | None
    body: _Range | None
    conclusion: _Range | None


class _Issue(_Strict):
    type: Literal["STRUCTURE", "CLARITY"]
    segmentIds: list[str]  # noqa: N815
    reason: str


class _Slide(_Strict):
    page: int
    covered: bool
    note: str | None


class _Question(_Strict):
    question: str
    intent: str
    answerTip: str  # noqa: N815


class _LlmContent(_Strict):
    sections: _Sections
    keyMessage: str  # noqa: N815
    issues: list[_Issue]
    slides: list[_Slide]
    expectedQuestions: list[_Question]  # noqa: N815


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    # Structured Outputs 문서·SDK(to_strict_json_schema)가 쓰는 형태
    return {"anyOf": [schema, {"type": "null"}]}


_STR = {"type": "string"}
_RANGE = {"startId": _STR, "endId": _STR}

RESPONSE_SCHEMA: dict[str, Any] = _obj(
    {
        "sections": _obj(
            {
                "intro": _nullable(_obj(_RANGE)),
                "body": _nullable(_obj(_RANGE)),
                "conclusion": _nullable(_obj(_RANGE)),
            }
        ),
        "keyMessage": _STR,
        "issues": {
            "type": "array",
            "items": _obj(
                {
                    "type": {"type": "string", "enum": ["STRUCTURE", "CLARITY"]},
                    "segmentIds": {"type": "array", "items": _STR},
                    "reason": _STR,
                }
            ),
        },
        "slides": {
            "type": "array",
            "items": _obj(
                {
                    "page": {"type": "integer"},
                    "covered": {"type": "boolean"},
                    "note": _nullable(_STR),
                }
            ),
        },
        "expectedQuestions": {
            "type": "array",
            "items": _obj({"question": _STR, "intent": _STR, "answerTip": _STR}),
        },
    }
)


# ---------------------------------------------------------------- 숫자 검사


def _compact(text: str) -> str:
    return "".join(text.split())


def _metric_tokens(values: Iterable[Any]) -> set[str]:
    """수치 요약 값이 글에 나타날 수 있는 모양들. 한 자리 정수는 흔해서 뺀다."""
    out: set[str] = set()
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int | float):
            continue
        forms = {f"{v:g}", str(round(v)), f"{v:.1f}", f"{v:.2f}"}
        if 0 < v <= 1:
            forms.add(str(round(v * 100)))
        out |= {f for f in forms if len(f.replace(".", "").lstrip("0")) >= 2}
    return out


class NumberGuard:
    """LLM 글에 시간·점수·지표 숫자가 섞였는지 본다. 대본·슬라이드에 있는 숫자는 허용."""

    def __init__(self, source_text: str = "", metrics: Mapping[str, Any] | None = None) -> None:
        self._source = _compact(source_text)
        self._source_numbers = set(_NUMBER.findall(source_text))
        self._metric = _metric_tokens((metrics or {}).values()) - self._source_numbers

    def bad(self, text: str | None) -> bool:
        if not text:
            return False
        for m in _UNIT_NUMBER.finditer(text):
            if _compact(m.group(0)) not in self._source:
                return True
        return any(n in self._metric for n in _NUMBER.findall(text))


def has_time_or_score(text: str | None) -> bool:
    """대본·지표 없이 본 숫자 검사(시간·점수·지표 단위가 붙은 숫자)."""
    return NumberGuard().bad(text)


# ---------------------------------------------------------------- 프롬프트


@lru_cache
def load_system_prompt(version: str = PROMPT_VERSION) -> str:
    return (PROMPTS_DIR / f"content_system_{version}.md").read_text(encoding="utf-8").strip()


@lru_cache
def load_rubric(purpose: Purpose, version: str = RUBRIC_VERSION) -> str:
    path = PROMPTS_DIR / f"content_rubric_{purpose.value.lower()}_{version}.md"
    return path.read_text(encoding="utf-8").strip()


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_transcript_block(transcript: Transcript) -> tuple[str, bool, set[str]]:
    """'s-001: 문장' 줄들. 상한을 넘으면 가운데를 줄인다. (본문, 잘렸는지, 보낸 문장 ID)"""
    segs = transcript.segments
    lines = [f"{s.id}: {_clip(s.text, MAX_SEGMENT_CHARS)}" for s in segs]
    total = sum(len(line) + 1 for line in lines)
    if total <= MAX_TRANSCRIPT_CHARS:
        return "\n".join(lines), False, {s.id for s in segs}

    head_budget = int(MAX_TRANSCRIPT_CHARS * HEAD_RATIO)
    tail_budget = MAX_TRANSCRIPT_CHARS - head_budget
    n_head = used = 0
    for line in lines:
        if used + len(line) + 1 > head_budget:
            break
        n_head += 1
        used += len(line) + 1
    n_tail = used = 0
    for line in reversed(lines[n_head:]):
        if used + len(line) + 1 > tail_budget:
            break
        n_tail += 1
        used += len(line) + 1
    head, tail = lines[:n_head], lines[len(lines) - n_tail :]
    marker = f"(중략: {segs[n_head].id} ~ {segs[len(segs) - n_tail - 1].id})"
    sent = {s.id for s in segs[:n_head]} | {s.id for s in segs[len(segs) - n_tail :]}
    return "\n".join([*head, marker, *tail]), True, sent


def build_slides_block(slides: Sequence[str]) -> str:
    pages = list(slides)[:MAX_SLIDE_PAGES]
    per_page = min(MAX_SLIDE_PAGE_CHARS, max(50, MAX_SLIDES_CHARS // max(1, len(pages))))
    return "\n".join(
        f"[p{i}] {_clip(t, per_page) or '(텍스트 없음)'}" for i, t in enumerate(pages, 1)
    )


def _format_metric(value: Any) -> str:
    if value is None:
        return "측정 불가"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_user_input(data: ContentInput) -> tuple[str, bool, set[str]]:
    transcript, truncated, sent_ids = build_transcript_block(data.transcript)
    parts = [
        f"[발표 목적] {data.purpose.value}",
        "[평가 기준]\n" + load_rubric(data.purpose),
    ]
    if data.metrics_summary:
        metrics = "\n".join(f"- {k}: {_format_metric(v)}" for k, v in data.metrics_summary.items())
        parts.append("[수치 요약(참고용, 글에 옮기지 말 것)]\n" + metrics)
    parts.append("[대본]\n" + transcript)
    if data.slides:
        parts.append("[슬라이드]\n" + build_slides_block(data.slides))
    else:
        parts.append("[슬라이드] 없음. slides는 빈 배열로.")
    return "\n\n".join(parts), truncated, sent_ids


# ---------------------------------------------------------------- 호출


def make_client(settings: Settings) -> LlmClient:
    return llm.make_client(settings)


def _request_kwargs(model: str, instructions: str, user_input: str) -> dict[str, Any]:
    return llm.structured_kwargs(
        model=model,
        name=f"content_analysis_{PROMPT_VERSION}",
        schema=RESPONSE_SCHEMA,
        instructions=instructions,
        user_input=user_input,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _cost(settings: Settings, input_tokens: int, output_tokens: int) -> float:
    return llm.estimate_cost(settings, input_tokens, output_tokens)


class _Invalid(Exception):
    """재시도할 만한 응답 문제."""


class _TokenLimit(Exception):
    """출력 토큰 상한에 걸려 잘렸다."""


def _parse(response: Any) -> _LlmContent:
    if llm.is_refusal(response):
        raise ContentServiceError("llm refused")
    reason = llm.incomplete_reason(response)
    if reason == "max_output_tokens":
        raise _TokenLimit
    if reason is not None:
        raise ContentServiceError(f"llm response incomplete: {reason}")
    status = getattr(response, "status", None)
    if status not in (None, "completed"):
        raise ContentServiceError(f"llm response status {status}")
    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        raise _Invalid("empty output")
    try:
        return _LlmContent.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise _Invalid(f"schema violation: {type(e).__name__}") from None


# ---------------------------------------------------------------- 후처리


@dataclass
class _Post:
    transcript: Transcript
    sent_ids: set[str]
    guard: NumberGuard
    seg_index: dict[str, int] = field(init=False)
    dropped: int = 0

    def __post_init__(self) -> None:
        self.seg_index = {s.id: i for i, s in enumerate(self.transcript.segments)}

    def times(self, ids: Sequence[str]) -> tuple[float, float]:
        segs = [self.transcript.segments[self.seg_index[i]] for i in ids]
        return min(s.start for s in segs), max(s.end for s in segs)

    def valid(self, ids: Sequence[str]) -> bool:
        return bool(ids) and all(
            _SEGMENT_ID.match(i) and i in self.seg_index and i in self.sent_ids for i in ids
        )


def _sections(raw: _Sections, post: _Post) -> list[Section]:
    out: list[Section] = []
    for kind, rng in (
        (SectionKind.INTRO, raw.intro),
        (SectionKind.BODY, raw.body),
        (SectionKind.CONCLUSION, raw.conclusion),
    ):
        if rng is None:
            continue
        if not post.valid([rng.startId, rng.endId]):
            post.dropped += 1
            continue
        a, b = sorted((rng.startId, rng.endId), key=post.seg_index.__getitem__)
        start, end = post.times([a, b])
        out.append(Section(kind=kind, start_id=a, end_id=b, start=start, end=end))
    return out


def _issues(raw: list[_Issue], post: _Post) -> list[ContentIssue]:
    out: list[ContentIssue] = []
    for item in raw:
        ids = list(dict.fromkeys(item.segmentIds))  # 중복 제거, 순서 유지
        reason = item.reason.strip()
        if not post.valid(ids) or not reason or post.guard.bad(reason):
            post.dropped += 1
            continue
        ids.sort(key=post.seg_index.__getitem__)
        span = post.seg_index[ids[-1]] - post.seg_index[ids[0]] + 1
        if span > MAX_ISSUE_SPAN_SEGMENTS:
            # 멀리 떨어진 문장들을 묶으면 타임라인에 엉뚱하게 긴 구간이 생긴다.
            post.dropped += 1
            continue
        start, end = post.times(ids)
        out.append(
            ContentIssue(
                type=IssueType(item.type), segment_ids=ids, start=start, end=end, reason=reason
            )
        )
    if len(out) > MAX_ISSUES:
        post.dropped += len(out) - MAX_ISSUES
        out = out[:MAX_ISSUES]
    return out


def _slides(
    raw: list[_Slide], page_count: int | None, post: _Post
) -> tuple[list[SlideCoverage] | None, list[int]]:
    if page_count is None:
        post.dropped += len(raw)
        return None, []
    by_page: dict[int, SlideCoverage] = {}
    for s in raw:
        if not 1 <= s.page <= page_count or s.page in by_page:
            post.dropped += 1
            continue
        note = s.note.strip() if s.note and not s.covered else None
        if note and post.guard.bad(note):
            note = None  # 판단(covered)은 살리고 글만 버린다
            post.dropped += 1
        by_page[s.page] = SlideCoverage(page=s.page, covered=s.covered, note=note or None)
    missing = [p for p in range(1, page_count + 1) if p not in by_page]
    if missing:
        logger.info("slide coverage missing for %d page(s)", len(missing))
    return [by_page[p] for p in sorted(by_page)], missing


def _questions(raw: list[_Question], post: _Post) -> list[ExpectedQuestion]:
    out: list[ExpectedQuestion] = []
    for q in raw:
        fields = (q.question.strip(), q.intent.strip(), q.answerTip.strip())
        if not all(fields) or any(post.guard.bad(f) for f in fields):
            post.dropped += 1
            continue
        out.append(ExpectedQuestion(question=fields[0], intent=fields[1], answer_tip=fields[2]))
    return out[:QUESTION_COUNT]


def _postprocess(raw: _LlmContent, data: ContentInput, sent_ids: set[str]) -> ContentAnalysis:
    source = "\n".join(s.text for s in data.transcript.segments)
    if data.slides:
        source += "\n" + "\n".join(data.slides)
    post = _Post(data.transcript, sent_ids, NumberGuard(source, data.metrics_summary))

    questions = _questions(raw.expectedQuestions, post)
    if len(questions) < MIN_QUESTIONS:
        raise _Invalid(f"only {len(questions)} usable questions")
    key: str | None = raw.keyMessage.strip() or None
    if key and post.guard.bad(key):
        key = None
        post.dropped += 1
    page_count = min(len(data.slides), MAX_SLIDE_PAGES) if data.slides else None
    coverage, missing = _slides(raw.slides, page_count, post)
    sections = _sections(raw.sections, post)
    issues = _issues(raw.issues, post)
    return ContentAnalysis(
        sections=sections,
        key_message=key,
        issues=issues,
        slide_coverage=coverage,
        expected_questions=questions,
        usage=None,
        dropped_items=post.dropped,
        slides_missing_pages=missing,
        slides_truncated=bool(data.slides) and len(data.slides) > MAX_SLIDE_PAGES,
    )


# ---------------------------------------------------------------- 진입점


def _warn_formal(result: ContentAnalysis) -> None:
    texts: list[str | None] = [result.key_message]
    texts += [i.reason for i in result.issues]
    texts += [s.note for s in result.slide_coverage or []]
    for q in result.expected_questions:
        texts += [q.question, q.intent, q.answer_tip]
    n = count_formal(texts)
    if n:
        # 글 내용은 남기지 않는다(대본 인용이 섞일 수 있음).
        logger.warning("content: %d text field(s) not in 해요체", n)


def _empty_result(data: ContentInput) -> ContentAnalysis:
    page_count = min(len(data.slides), MAX_SLIDE_PAGES) if data.slides else 0
    return ContentAnalysis(
        sections=[],
        key_message=None,
        issues=[],
        slide_coverage=(
            [SlideCoverage(page=p, covered=False, note=None) for p in range(1, page_count + 1)]
            if data.slides
            else None
        ),
        expected_questions=[],
        usage=None,
        slides_truncated=bool(data.slides) and len(data.slides) > MAX_SLIDE_PAGES,
    )


def analyze_content(data: ContentInput, client: LlmClient, settings: Settings) -> ContentAnalysis:
    """대본·목적·수치·슬라이드로 내용 분석을 한다."""
    if not any(s.text.strip() for s in data.transcript.segments):
        # 빈 대본은 LLM을 부르지 않는다(비용). 보통은 stt 단계에서 이미 NO_SPEECH로 끝난다.
        return _empty_result(data)

    model = settings.openai_llm_model.strip()
    if not model:
        raise ContentServiceError("OPENAI_LLM_MODEL is not set")
    user_input, truncated, sent_ids = build_user_input(data)
    kwargs = _request_kwargs(model, load_system_prompt(), user_input)

    total_in = total_out = 0
    last_problem = ""
    for attempt in range(1, MAX_RETRIES + 2):
        # API 실패는 SDK가 이미 재시도했다. 같은 요청을 더 보내 크레딧을 쓰지 않는다.
        error: str | None = None
        try:
            response = llm.call(client, kwargs, "content")
        except llm.LlmCallError as e:
            error = str(e)
        if error is not None:
            raise ContentServiceError(error)
        inp, out = llm.log_usage("content", model, attempt, response, settings)
        total_in += inp
        total_out += out
        try:
            result = _postprocess(_parse(response), data, sent_ids)
        except _TokenLimit:
            if kwargs["max_output_tokens"] >= MAX_OUTPUT_TOKENS_RETRY:
                raise ContentServiceError("llm output hit token limit twice") from None
            logger.warning("content llm hit max_output_tokens (attempt %d); raising cap", attempt)
            kwargs = {**kwargs, "max_output_tokens": MAX_OUTPUT_TOKENS_RETRY}
            last_problem = "max_output_tokens"
            continue
        except _Invalid as e:
            last_problem = str(e)
            logger.warning("content llm response rejected (attempt %d): %s", attempt, e)
            continue
        if result.dropped_items:
            logger.info("content: dropped %d invalid item(s)", result.dropped_items)
        _warn_formal(result)
        usage = LlmUsage(
            model=model,
            attempts=attempt,
            input_tokens=total_in,
            output_tokens=total_out,
            estimated_cost_usd=round(_cost(settings, total_in, total_out), 6),
        )
        return ContentAnalysis(
            sections=result.sections,
            key_message=result.key_message,
            issues=result.issues,
            slide_coverage=result.slide_coverage,
            expected_questions=result.expected_questions,
            usage=usage,
            transcript_truncated=truncated,
            dropped_items=result.dropped_items,
            slides_missing_pages=result.slides_missing_pages,
            slides_truncated=result.slides_truncated,
        )
    raise ContentServiceError(
        f"no usable response after {MAX_RETRIES + 1} attempts: {last_problem}"
    )


# ---------------------------------------------------------------- 슬라이드 PDF


def extract_slide_texts(pdf_path: Path, max_pages: int = MAX_SLIDE_PAGES) -> list[str] | None:
    """pypdf로 페이지별 텍스트를 뽑는다. 읽을 수 없는 PDF면 None(슬라이드 없이 분석).

    PDF는 사용자가 올린 파일이라 깨졌거나 악의적일 수 있다. pypdf는 깨진 파일에서 여러 종류의
    예외를 내므로 모두 잡고, 실패해도 분석 전체를 멈추지 않는다. 처리 시간은 워커 작업 시간 제한
    (Celery soft limit)이 막는다. 업로드 크기는 P6에서 20MB로 제한된다.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted and not reader.decrypt(""):
            logger.info("slides pdf is encrypted; skipping")
            return None
        pages: list[str] = []
        for page in reader.pages[:max_pages]:
            try:
                pages.append(" ".join((page.extract_text() or "").split()))
            except Exception:  # noqa: BLE001 - 페이지 하나가 깨져도 나머지는 쓴다
                pages.append("")
        return pages or None
    except Exception as e:  # noqa: BLE001 - 신뢰할 수 없는 PDF 경계
        logger.info("slides pdf could not be read: %s", type(e).__name__)
        return None
