"""content: LLM 내용 분석. OpenAI는 전부 가짜 클라이언트로 대신한다(실제 호출 없음)."""

import copy
import json
import logging
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.models.enums import Purpose
from app.schemas.report import Segment, Transcript
from app.services.pipeline import content
from app.services.pipeline.content import (
    ContentInput,
    ContentServiceError,
    IssueType,
    SectionKind,
    analyze_content,
    build_transcript_block,
    extract_slide_texts,
    has_time_or_score,
)


def transcript(n: int = 6, text: str = "문장 {i}입니다") -> Transcript:
    return Transcript(
        segments=[
            Segment(
                id=f"s-{i:03d}",
                start=i * 5.0 - 5,
                end=i * 5.0 - 0.5,
                text=text.format(i=i),
                words=[],
            )
            for i in range(1, n + 1)
        ]
    )


def good_payload(**over) -> dict:
    data = {
        "sections": {
            "intro": {"startId": "s-001", "endId": "s-002"},
            "body": {"startId": "s-003", "endId": "s-005"},
            "conclusion": {"startId": "s-006", "endId": "s-006"},
        },
        "keyMessage": "연습할 때마다 고칠 점을 바로 확인할 수 있어요.",
        "issues": [
            {
                "type": "STRUCTURE",
                "segmentIds": ["s-004", "s-003"],
                "reason": "순서가 뒤바뀌었어요.",
            },
            {"type": "CLARITY", "segmentIds": ["s-005"], "reason": "결론이 모호해요."},
        ],
        "slides": [],
        "expectedQuestions": [
            {"question": f"질문 {c}?", "intent": "의도", "answerTip": "요령"} for c in "가나다라마"
        ],
    }
    data.update(over)
    return data


def response(payload, status: str = "completed", tokens: tuple[int, int] = (1000, 500)):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(
        status=status,
        output_text=text,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


class FakeLlm:
    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def llm_settings(settings):
    settings.openai_llm_model = "gpt-5-mini"
    return settings


def run(client, *, slides=None, t=None, purpose=Purpose.CLASS, settings=None, metrics=None):
    data = ContentInput(
        transcript=t or transcript(),
        purpose=purpose,
        metrics_summary=metrics
        if metrics is not None
        else {"syllablesPerMin": 341, "gazeFrontRatio": None},
        slides=slides,
    )
    return analyze_content(data, client, settings)


# ---------- 요청 ----------


def test_request_uses_strict_json_schema_and_low_reasoning(llm_settings):
    client = FakeLlm(response(good_payload()))
    run(client, settings=llm_settings)
    kw = client.calls[0]
    assert kw["model"] == "gpt-5-mini"
    assert kw["reasoning"] == {"effort": "low"}
    assert kw["max_output_tokens"] == content.MAX_OUTPUT_TOKENS
    assert kw["store"] is False
    fmt = kw["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["schema"] == content.RESPONSE_SCHEMA
    assert kw["instructions"] == content.load_system_prompt()


def test_non_gpt5_model_has_no_reasoning(settings):
    settings.openai_llm_model = "gpt-4.1-mini"
    client = FakeLlm(response(good_payload()))
    run(client, settings=settings)
    assert "reasoning" not in client.calls[0]


def test_strict_schema_requires_every_property():
    def walk(node):
        if isinstance(node, dict):
            if "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)

    walk(content.RESPONSE_SCHEMA)


def test_input_contains_ids_purpose_rubric_and_metrics(llm_settings):
    client = FakeLlm(response(good_payload()))
    run(client, settings=llm_settings, purpose=Purpose.PITCH)
    text = client.calls[0]["input"]
    assert "[발표 목적] PITCH" in text
    assert content.load_rubric(Purpose.PITCH) in text
    assert "s-001: 문장 1입니다" in text and "s-006: 문장 6입니다" in text
    assert "- syllablesPerMin: 341" in text
    assert "- gazeFrontRatio: 측정 불가" in text
    assert "[슬라이드] 없음" in text


@pytest.mark.parametrize("purpose", list(Purpose))
def test_rubric_file_exists_for_every_purpose(purpose):
    assert purpose.value in content.load_rubric(purpose)


def test_missing_model_setting_is_error(settings):
    settings.openai_llm_model = " "
    client = FakeLlm(response(good_payload()))
    with pytest.raises(ContentServiceError):
        run(client, settings=settings)
    assert client.calls == []


# ---------- 정상 응답 → 시간 붙이기 ----------


def test_ids_are_mapped_to_times_by_code(llm_settings):
    r = run(FakeLlm(response(good_payload())), settings=llm_settings)
    intro, body, concl = r.sections
    assert (intro.kind, intro.start_id, intro.end_id, intro.start, intro.end) == (
        SectionKind.INTRO, "s-001", "s-002", 0.0, 9.5,
    )  # fmt: skip
    assert (body.start, body.end) == (10.0, 24.5)
    assert concl.kind is SectionKind.CONCLUSION
    structure = r.issues[0]
    assert structure.type is IssueType.STRUCTURE
    assert structure.segment_ids == ["s-003", "s-004"]  # 대본 순서로 정렬
    assert (structure.start, structure.end) == (10.0, 19.5)
    assert r.key_message.startswith("연습할 때마다")
    assert [q.question for q in r.expected_questions] == [f"질문 {c}?" for c in "가나다라마"]
    assert r.slide_coverage is None
    assert r.usage.attempts == 1 and r.dropped_items == 0


def test_reversed_section_range_is_sorted(llm_settings):
    p = good_payload()
    p["sections"]["body"] = {"startId": "s-005", "endId": "s-003"}
    r = run(FakeLlm(response(p)), settings=llm_settings)
    body = next(s for s in r.sections if s.kind is SectionKind.BODY)
    assert (body.start_id, body.end_id) == ("s-003", "s-005")


def test_null_section_is_omitted(llm_settings):
    p = good_payload()
    p["sections"]["conclusion"] = None
    r = run(FakeLlm(response(p)), settings=llm_settings)
    assert [s.kind for s in r.sections] == [SectionKind.INTRO, SectionKind.BODY]


# ---------- 없는 문장 ID ----------


@pytest.mark.parametrize("bad_id", ["s-099", "s-1", "S-001", "", "s-001 "])
def test_issue_with_unknown_id_is_dropped(llm_settings, bad_id):
    p = good_payload()
    p["issues"][0]["segmentIds"] = ["s-003", bad_id]
    r = run(FakeLlm(response(p)), settings=llm_settings)
    assert [i.type for i in r.issues] == [IssueType.CLARITY]
    assert r.dropped_items == 1


def test_issue_without_ids_is_dropped(llm_settings):
    p = good_payload()
    p["issues"][0]["segmentIds"] = []
    assert len(run(FakeLlm(response(p)), settings=llm_settings).issues) == 1


def test_section_with_unknown_id_is_dropped(llm_settings):
    p = good_payload()
    p["sections"]["intro"] = {"startId": "s-000", "endId": "s-002"}
    r = run(FakeLlm(response(p)), settings=llm_settings)
    assert SectionKind.INTRO not in [s.kind for s in r.sections]
    assert r.dropped_items == 1


def test_duplicate_ids_in_issue_are_collapsed(llm_settings):
    p = good_payload()
    p["issues"][1]["segmentIds"] = ["s-005", "s-005"]
    assert run(FakeLlm(response(p)), settings=llm_settings).issues[1].segment_ids == ["s-005"]


def test_issues_capped(llm_settings):
    p = good_payload(
        issues=[{"type": "CLARITY", "segmentIds": ["s-001"], "reason": "모호해요."}] * 10
    )
    assert len(run(FakeLlm(response(p)), settings=llm_settings).issues) == content.MAX_ISSUES


# ---------- 시간·점수 숫자 금지 ----------


@pytest.mark.parametrize(
    ("text", "bad"),
    [("12초 동안 멈췄어요", True), ("3분이 넘어요", True), ("1:30 지점", True),
     ("80점이에요", True), ("85% 정도", True), ("70 퍼센트", True), ("2.5초", True),
     ("분당 341음절이에요", True), ("추임새를 12번 썼어요", True), ("침묵이 3회", True),
     ("1.5배 빨라요", True),
     ("두 가지 이유", False), ("1단계부터 설명해요", False), ("3가지 기능", False),
     ("2024년 자료", False), ("1분기 매출", False), ("10프로젝트", False), ("4분면", False),
     ("5분위", False), ("24시간표", False), ("2번째 슬라이드", False), ("3회차 발표", False),
     ("3점검", False), ("", False), (None, False)],
)  # fmt: skip
def test_has_time_or_score(text, bad):
    assert has_time_or_score(text) is bad


@pytest.mark.parametrize(
    ("source", "text"),
    [("저는 1분 자기소개를 준비했습니다", "1분 자기소개의 핵심을 먼저 말해요."),
     ("매출이 30% 성장했습니다", "매출 30% 성장을 근거로 들어요."),
     ("3점 슛 성공률을 분석했어요", "3점 슛 분석이 핵심이에요."),
     ("24 시간 운영합니다", "24시간 운영을 강조해요.")],
)  # fmt: skip
def test_numbers_quoted_from_transcript_are_allowed(source, text):
    assert has_time_or_score(text)  # 대본 없이 보면 걸리지만
    assert not content.NumberGuard(source).bad(text)  # 대본에 있으면 허용


def test_metric_values_are_rejected_unless_in_transcript():
    guard = content.NumberGuard(
        "발표를 시작합니다", {"syllablesPerMin": 341, "gazeFrontRatio": 0.42, "fillerCount": 3}
    )
    assert guard.bad("말 속도가 341로 빨라요.")
    assert guard.bad("정면 비율이 0.42예요.")
    assert guard.bad("정면을 42 정도만 봤어요.")
    assert not guard.bad("세 가지를 말해요.")
    assert not guard.bad("3가지 기능을 소개해요.")  # 한 자리 수는 흔해서 지표로 보지 않는다
    # 대본에 같은 숫자가 있으면 지표가 아니라 발표 내용
    assert not content.NumberGuard("341명이 참여했어요", {"syllablesPerMin": 341}).bad("341명 참여")


def test_items_with_time_or_score_are_dropped(llm_settings):
    p = good_payload()
    p["issues"][0]["reason"] = "15초 동안 같은 말을 해요."
    p["expectedQuestions"][0]["answerTip"] = "30초 안에 답하세요."
    r = run(FakeLlm(response(p)), settings=llm_settings)
    assert len(r.issues) == 1
    assert len(r.expected_questions) == 4
    assert r.dropped_items == 2


def test_key_message_with_number_is_dropped_without_retry(llm_settings):
    client = FakeLlm(response(good_payload(keyMessage="90점짜리 발표예요.")))
    r = run(client, settings=llm_settings)
    assert len(client.calls) == 1
    assert r.key_message is None
    assert r.dropped_items == 1


def test_key_message_quoting_transcript_number_is_kept(llm_settings):
    t = transcript(3, text="저희 서비스로 매출이 30% 성장했습니다 {i}")
    p = good_payload(keyMessage="매출 30% 성장으로 효과를 증명해요.")
    p["sections"] = {"intro": None, "body": None, "conclusion": None}
    p["issues"] = []
    r = run(FakeLlm(response(p)), settings=llm_settings, t=t)
    assert r.key_message == "매출 30% 성장으로 효과를 증명해요."


def test_metric_number_in_reason_is_dropped(llm_settings):
    p = good_payload()
    p["issues"][0]["reason"] = "말 속도가 341로 너무 빨라요."
    r = run(FakeLlm(response(p)), settings=llm_settings, metrics={"syllablesPerMin": 341})
    assert [i.type for i in r.issues] == [IssueType.CLARITY]


def test_retry_tokens_are_summed(llm_settings):
    client = FakeLlm(response("not json"), response(good_payload()))
    r = run(client, settings=llm_settings)
    assert r.usage.attempts == 2
    assert r.usage.input_tokens == 2000 and r.usage.output_tokens == 1000


# ---------- 스키마 위반·재시도 ----------


def _without(key):
    p = good_payload()
    del p[key]
    return p


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "",
        _without("issues"),
        good_payload(extra="x"),
        good_payload(keyMessage=3),
        good_payload(issues=[{"type": "PACE", "segmentIds": ["s-001"], "reason": "x"}]),
        good_payload(slides=[{"page": "1", "covered": True, "note": None}]),
    ],
)
def test_schema_violation_retries_then_succeeds(llm_settings, bad):
    client = FakeLlm(response(bad), response(good_payload()))
    r = run(client, settings=llm_settings)
    assert len(client.calls) == 2 and r.usage.attempts == 2


def test_gives_up_after_two_retries(llm_settings):
    client = FakeLlm(response("not json"))
    with pytest.raises(ContentServiceError):
        run(client, settings=llm_settings)
    assert len(client.calls) == 1 + content.MAX_RETRIES == 3


def _incomplete(reason):
    r = response(good_payload(), status="incomplete")
    r.incomplete_details = SimpleNamespace(reason=reason)
    return r


def test_token_limit_retries_once_with_higher_cap(llm_settings):
    client = FakeLlm(_incomplete("max_output_tokens"), response(good_payload()))
    r = run(client, settings=llm_settings)
    assert r.usage.attempts == 2
    assert client.calls[0]["max_output_tokens"] == content.MAX_OUTPUT_TOKENS
    assert client.calls[1]["max_output_tokens"] == content.MAX_OUTPUT_TOKENS_RETRY


def test_token_limit_twice_fails_fast(llm_settings):
    client = FakeLlm(_incomplete("max_output_tokens"))
    with pytest.raises(ContentServiceError):
        run(client, settings=llm_settings)
    assert len(client.calls) == 2  # 같은 상한으로 세 번째는 보내지 않는다


def test_other_incomplete_fails_without_retry(llm_settings):
    client = FakeLlm(_incomplete("content_filter"))
    with pytest.raises(ContentServiceError):
        run(client, settings=llm_settings)
    assert len(client.calls) == 1


def _real_response(output, status="completed", incomplete=None):
    """SDK의 실제 Response 타입으로 만든 응답."""
    from openai.types.responses import Response

    return Response.model_construct(
        id="resp_x",
        object="response",
        created_at=0,
        model="gpt-5-mini",
        status=status,
        output=output,
        incomplete_details=incomplete,
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _message(part):
    from openai.types.responses import ResponseOutputMessage

    return ResponseOutputMessage.model_construct(
        id="msg_x", type="message", role="assistant", status="completed", content=[part]
    )


def test_real_response_type_is_parsed(llm_settings):
    from openai.types.responses import ResponseOutputText

    text = ResponseOutputText.model_construct(
        type="output_text", text=json.dumps(good_payload(), ensure_ascii=False), annotations=[]
    )
    r = run(FakeLlm(_real_response([_message(text)])), settings=llm_settings)
    assert len(r.expected_questions) == 5


def test_refusal_fails_without_retry(llm_settings):
    from openai.types.responses import ResponseOutputRefusal

    refusal = ResponseOutputRefusal.model_construct(type="refusal", refusal="거절")
    client = FakeLlm(_real_response([_message(refusal)]))
    with pytest.raises(ContentServiceError):
        run(client, settings=llm_settings)
    assert len(client.calls) == 1


def test_none_status_with_valid_output_is_accepted(llm_settings):
    assert run(FakeLlm(response(good_payload(), status=None)), settings=llm_settings).usage


def test_too_few_questions_retries(llm_settings):
    few = good_payload()
    few["expectedQuestions"] = few["expectedQuestions"][:2]
    client = FakeLlm(response(few), response(good_payload()))
    assert run(client, settings=llm_settings).usage.attempts == 2


def test_extra_questions_trimmed_to_five(llm_settings):
    p = good_payload()
    p["expectedQuestions"] = p["expectedQuestions"] * 2
    assert len(run(FakeLlm(response(p)), settings=llm_settings).expected_questions) == 5


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")),
        openai.RateLimitError(
            "limited",
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com")),
            body=None,
        ),
    ],
)
def test_api_error_is_service_error_without_extra_calls(llm_settings, error):
    client = FakeLlm(error)
    with pytest.raises(ContentServiceError) as exc_info:
        run(client, settings=llm_settings)
    assert len(client.calls) == 1  # SDK가 이미 재시도함. 크레딧 보호
    assert exc_info.value.__context__ is None and exc_info.value.__cause__ is None


# ---------- 빈 대본 ----------


@pytest.mark.parametrize("t", [Transcript(segments=[]), transcript(2, text="  ")])
def test_empty_transcript_does_not_call_llm(llm_settings, t):
    client = FakeLlm(response(good_payload()))
    r = run(client, settings=llm_settings, t=t)
    assert client.calls == []
    assert r.sections == [] and r.issues == [] and r.expected_questions == []
    assert r.key_message is None and r.usage is None and r.slide_coverage is None


def test_empty_transcript_with_slides_marks_all_uncovered(llm_settings):
    r = run(FakeLlm(response(good_payload())), settings=llm_settings, t=Transcript(segments=[]),
            slides=["표지", "목차"])  # fmt: skip
    assert [(s.page, s.covered) for s in r.slide_coverage] == [(1, False), (2, False)]


# ---------- 슬라이드 ----------


def test_no_slides_gives_null_and_ignores_llm_slides(llm_settings):
    p = good_payload(slides=[{"page": 1, "covered": True, "note": None}])
    r = run(FakeLlm(response(p)), settings=llm_settings, slides=None)
    assert r.slide_coverage is None
    assert r.dropped_items == 1


def test_slide_coverage_validated(llm_settings):
    p = good_payload(
        slides=[
            {"page": 2, "covered": False, "note": "경쟁 서비스 비교를 설명하지 않았어요."},
            {"page": 1, "covered": True, "note": "잘 설명했어요."},  # covered면 note 없앰
            {"page": 1, "covered": False, "note": None},  # 중복
            {"page": 9, "covered": True, "note": None},  # 없는 페이지
            {"page": 0, "covered": True, "note": None},
            {"page": 3, "covered": False, "note": "20초 만에 넘겼어요."},  # 숫자 → note만 버림
        ]
    )
    client = FakeLlm(response(p))
    r = run(client, settings=llm_settings, slides=["표지", "비교", "결론", "부록"])
    assert [(s.page, s.covered, s.note) for s in r.slide_coverage] == [
        (1, True, None),
        (2, False, "경쟁 서비스 비교를 설명하지 않았어요."),
        (3, False, None),
    ]
    assert r.dropped_items == 4
    assert r.slides_missing_pages == [4]  # LLM이 판단을 빠뜨린 페이지
    assert not r.slides_truncated
    assert "[p2] 비교" in client.calls[0]["input"]


def test_slides_over_page_limit_are_flagged(llm_settings):
    pages = [f"페이지 {i}" for i in range(1, 46)]
    p = good_payload(slides=[{"page": 41, "covered": True, "note": None}])
    r = run(FakeLlm(response(p)), settings=llm_settings, slides=pages)
    assert r.slides_truncated
    assert r.slide_coverage == []  # 41페이지는 보내지 않았으므로 버림
    assert len(r.slides_missing_pages) == content.MAX_SLIDE_PAGES


# ---------- 입력 자르기 ----------


def test_short_transcript_is_not_truncated():
    text, cut, sent = build_transcript_block(transcript(5))
    assert not cut and text.count("\n") == 4
    assert sent == {f"s-{i:03d}" for i in range(1, 6)}


def test_long_transcript_keeps_head_and_tail():
    t = transcript(400, text="가" * 60 + " {i}")
    text, cut, sent = build_transcript_block(t)
    assert cut
    assert "s-001" in sent and "s-400" in sent and "s-200" not in sent
    assert len(text) <= content.MAX_TRANSCRIPT_CHARS + 50
    assert text.startswith("s-001: ")
    assert text.rstrip().splitlines()[-1].startswith("s-400: ")
    marker = next(line for line in text.splitlines() if line.startswith("(중략"))
    assert "~" in marker


def test_long_segment_is_clipped():
    t = transcript(1, text="가" * 1000)
    text, _, _ = build_transcript_block(t)
    assert len(text) <= len("s-001: ") + content.MAX_SEGMENT_CHARS


def test_slides_block_limits():
    block = content.build_slides_block(["나" * 2000] * 60)
    lines = block.splitlines()
    assert len(lines) == content.MAX_SLIDE_PAGES
    assert len(block) <= content.MAX_SLIDES_CHARS + content.MAX_SLIDE_PAGES * 10
    assert "(텍스트 없음)" in content.build_slides_block([""])


def test_ids_from_cut_middle_are_dropped(llm_settings):
    # LLM이 보지 못한(중략된) 문장 ID는 지어낸 것으로 보고 버린다
    t = transcript(400, text="가" * 60 + " {i}")
    p = good_payload()
    p["issues"] = [
        {"type": "CLARITY", "segmentIds": ["s-200"], "reason": "모호해요."},
        {"type": "CLARITY", "segmentIds": ["s-399"], "reason": "모호해요."},
    ]
    r = run(FakeLlm(response(p)), settings=llm_settings, t=t)
    assert r.transcript_truncated
    assert [i.segment_ids for i in r.issues] == [["s-399"]]


def test_scattered_issue_ids_are_dropped(llm_settings):
    t = transcript(20)
    p = good_payload()
    p["issues"] = [
        {"type": "STRUCTURE", "segmentIds": ["s-001", "s-015"], "reason": "흩어졌어요."},
        {"type": "STRUCTURE", "segmentIds": ["s-001", "s-008"], "reason": "가까워요."},
    ]
    r = run(FakeLlm(response(p)), settings=llm_settings, t=t)
    assert [i.segment_ids for i in r.issues] == [["s-001", "s-008"]]  # 8문장 간격까지 허용


# ---------- 로그·비용 ----------


def test_logs_model_tokens_cost_but_not_key_or_transcript(llm_settings, caplog):
    secret_line = "아무도 몰라야 하는 대본 내용"
    t = transcript(3, text=secret_line + " {i}")
    with caplog.at_level(logging.DEBUG):
        run(FakeLlm(response(good_payload(), tokens=(2000, 1000))), settings=llm_settings, t=t)
    assert "model=gpt-5-mini" in caplog.text
    assert "input_tokens=2000" in caplog.text and "output_tokens=1000" in caplog.text
    # gpt-5-mini 기본 단가: 2000 × 0.25 / 1M + 1000 × 2.0 / 1M = 0.0025
    assert "est_cost_usd=0.00250" in caplog.text
    assert "reasoning_tokens=" in caplog.text
    assert secret_line not in caplog.text
    assert llm_settings.openai_api_key.get_secret_value() not in caplog.text


def test_default_prices_are_gpt5_mini(settings):
    assert settings.openai_llm_input_usd_per_1m == 0.25
    assert settings.openai_llm_output_usd_per_1m == 2.0


def test_usage_cost(llm_settings):
    r = run(FakeLlm(response(good_payload(), tokens=(1_000_000, 100_000))), settings=llm_settings)
    assert r.usage.estimated_cost_usd == pytest.approx(0.25 + 0.2)


def test_payload_not_mutated(llm_settings):
    p = good_payload()
    before = copy.deepcopy(p)
    run(FakeLlm(response(p)), settings=llm_settings)
    assert p == before


# ---------- 슬라이드 PDF ----------


def _pdf(pages: list[str]) -> bytes:
    """글자가 든 최소 PDF(Helvetica, ASCII)."""
    objs: list[bytes] = []
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_id = 3 + len(pages) * 2
    for i, text in enumerate(pages):
        content_id = 4 + i * 2
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_extract_slide_texts(tmp_path):
    p = tmp_path / "s.pdf"
    p.write_bytes(_pdf(["Intro", "Market size", "Thanks"]))
    assert extract_slide_texts(p) == ["Intro", "Market size", "Thanks"]


def test_extract_slide_texts_respects_page_limit(tmp_path):
    p = tmp_path / "s.pdf"
    p.write_bytes(_pdf([f"Page {i}" for i in range(5)]))
    assert extract_slide_texts(p, max_pages=2) == ["Page 0", "Page 1"]


@pytest.mark.parametrize("data", [b"not a pdf", b"%PDF-1.4\n garbage", b""])
def test_unreadable_pdf_returns_none(tmp_path, data):
    p = tmp_path / "bad.pdf"
    p.write_bytes(data)
    assert extract_slide_texts(p) is None


def test_missing_pdf_returns_none(tmp_path):
    assert extract_slide_texts(tmp_path / "nope.pdf") is None
