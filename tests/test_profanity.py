"""욕설·비속어 사전(profanity_ko) 및 부적절 발표 대본 시나리오 테스트."""

from datetime import UTC, datetime

import pytest

from app.models.enums import Purpose, ScoreCategory
from app.schemas.report import (
    ExpectedQuestion,
    Feedback,
    FeedbackType,
    NonverbalMetrics,
    Segment,
    Severity,
    Transcript,
    Word,
)
from app.services.pipeline import artifacts as art
from app.services.pipeline import content as content_mod
from app.services.pipeline import report as rp
from app.services.pipeline import runner
from app.services.pipeline.content import (
    ContentAnalysis,
    ContentInput,
    ContentIssue,
    IssueType,
    LanguageSeverity,
    Section,
    SectionKind,
    analyze_content,
)
from app.services.pipeline.feedback import (
    FeedbackInput,
    FeedbackResult,
    OverallText,
    build_candidates,
    generate_feedback,
    verify_evidence,
)
from app.services.pipeline.nonverbal import NonverbalAnalysis
from app.services.pipeline.profanity_ko import (
    contains_profanity,
    detect_profanity,
    match_profanity_token,
    profanity_from_json,
    profanity_to_json,
)
from app.services.pipeline.scoring import score_presentation
from app.services.pipeline.speech_metrics import analyze_speech

# ---------- 1. 사전 변형·마스킹(단일·다중) 매칭 ----------


@pytest.mark.parametrize(
    "token",
    [
        "씨발",
        "시발",
        "씨바",
        "시바",
        "ㅅㅂ",
        "ㅆㅂ",
        "씨발놈",
        "시발아",
        "시바새끼",
        "시바꺼",
        "좆",
        "좆같네요",
        "좆같다",
        "좆같은",
        "ㅈ같네요",
        "ㅈ같다",
        "존나",
        "존나게",
        "졸라",
        "ㅈㄴ",
        "개새끼",
        "개새끼야",
        "씹새끼",
        "미친새끼",
        "새끼",
        "새끼야",
        "새끼들",
        "새끼들아",
        "병신",
        "병신아",
        "ㅂㅅ",
        "지랄",
        "지랄하네",
        "ㅈㄹ",
        "엿같네",
        "엿같다",
        "엿같은",
        "꺼져",
        "꺼져라",
        "닥쳐",
        "닥쳐라",
        "미친놈",
        "미친년",
        "썅",
        # Whisper 별표(*) 및 영문 X 마스킹(단일·다중)
        "X같네요",
        "x같네요",
        "XX같네요",
        "xxx같다",
        "*같네요",
        "**같네요",
        "*같다",
        "*같은",
        "X발",
        "x발",
        "XX발",
        "*발",
        "**발",
        "***발",
        "x나게",
        "*나",
        "*나게",
        "개*끼",
        "개**끼",
        "개*끼야",
    ],
)
def test_profanity_variants_and_whisper_masking_matched(token):
    assert match_profanity_token(token) is not None, token
    assert contains_profanity(f"아 진짜 {token}!")


@pytest.mark.parametrize(
    "quoted_sentence",
    [
        "발표에서 'X같네요'라는 표현은 피하세요.",
        "발표에서 '*같네요'라는 말은 쓰지 마세요.",
        "발표에서 '엿같네'라는 표현은 부적절해요.",
        "발표에서 '꺼져'라고 말하면 안 돼요.",
        "발표에서 '존나'라는 부사를 썼어요.",
        "발표에서 '졸라'라는 말은 빼세요.",
        "발표에서 'ㅅㅂ'이라는 표현이 나왔어요.",
        "씨발/병신 같은 비속어는 삼가세요.",
        '도입부에서 "**발"이라고 말했어요.',
        "마지막에 'XX같네요'라고 끝냈어요.",
    ],
)
def test_contains_profanity_catches_quoted_and_punctuated_particles(quoted_sentence):
    assert contains_profanity(quoted_sentence), quoted_sentence


# ---------- 2. 오탐 예외(allowlist) 및 일상·학술 문맥 안전성 ----------


@pytest.mark.parametrize(
    "safe_text",
    [
        "시발점",
        "시발점입니다",
        "시발역",
        "시발역에서",
        "시발택시",
        "시발차",
        "시발지",
        "병신년",
        "병신년에",
        "지랄탄",
        "지랄탄을",
        "새끼손가락",
        "새끼손가락이",
        "새끼발가락",
        "새끼줄",
        "그래서",
        "시소",
        "발표",
        "발표하기",
        "같네요",
        "똑같네요",
        "신발",
        "출발",
        "개발",
        "하나",
        "그러나",
        "불이 꺼져서",
        "화면이 꺼졌다",
        "일정 시간이 지나면 화면이 자동으로 꺼져요",
        "전원이 꺼져 있습니다",
        "불이 꺼져 있을 때",
        "위기가 닥쳐서",
        "위기가 닥쳐 오고 있습니다",
        "허리띠를 졸라매고",
        "허리띠를 졸라 매고",
        "아이가 졸라서",
        "변수 X나 Y를 선택합니다",
        "모델 X나 모델 Y 모두 가능합니다",
        "어미가 새끼를 낳고 기릅니다",
        "새끼가 태어났어요",
        "시바가 귀엽습니다",
        "시바견은 일본 개 품종이에요",
        "X축",
        "x값",
    ],
)
def test_allowlist_and_normal_words_not_matched(safe_text):
    assert not contains_profanity(safe_text), safe_text
    t = Transcript(
        segments=[
            Segment(
                id="s-001",
                start=0.0,
                end=3.0,
                text=safe_text,
                words=[
                    Word(text=w, start=i * 0.5, end=(i + 1) * 0.5, tag=None)
                    for i, w in enumerate(safe_text.split())
                ],
            )
        ]
    )
    assert detect_profanity(t).count == 0, safe_text


# ---------- 3. 타임스탬프·문장 ID 매핑 및 아티팩트 직렬화 ----------


def test_detect_profanity_maps_word_timestamps_and_sentence_id():
    # 실제 집 서버 a-90f307ad0087ee32 Whisper 대본 그대로
    t = Transcript(
        segments=[
            Segment(
                id="s-001",
                start=0.0,
                end=3.56,
                text="어 발표하기 싫네요 안 할래요 아 X같네요",
                words=[
                    Word(text="어", start=0.0, end=0.6, tag=None),
                    Word(text="발표하기", start=0.6, end=1.08, tag=None),
                    Word(text="싫네요", start=1.08, end=1.7, tag=None),
                    Word(text="안", start=2.16, end=2.16, tag=None),
                    Word(text="할래요", start=2.16, end=2.88, tag=None),
                    Word(text="아", start=3.14, end=3.14, tag=None),
                    Word(text="X같네요", start=3.14, end=3.56, tag=None),
                ],
            )
        ]
    )
    res = detect_profanity(t)
    assert res.count == 1
    (h,) = res.hits
    assert h.word == "x같네요"
    assert h.pattern == "x같"
    assert (h.start, h.end) == (3.14, 3.56)
    assert h.segment_id == "s-001" and h.sentence_id == "s-001"

    # 직렬화 왕복 및 이전 버전(None) 폴백
    dumped = profanity_to_json(res)
    assert dumped["count"] == 1
    restored = profanity_from_json(dumped)
    assert restored == res
    fallback = profanity_from_json(None, transcript=t)
    assert fallback == res


# ---------- 4. 시나리오 비교 및 리포트 명세 검증 ----------


def _make_nv(gaze: float = 0.85) -> NonverbalAnalysis:
    return NonverbalAnalysis(
        metrics=NonverbalMetrics(
            gaze_front_ratio=gaze,
            posture_stability=0.85,
            gesture_per_min=6.0,
            low_confidence_ranges=[],
        ),
        frame_count=20,
        face_frame_ratio=1.0,
        posture_measured=True,
        gesture_measured=True,
        gesture_times=[],
        gaze_away_ranges=[],
        posture_unstable_ranges=[],
        gesture_none_ranges=[],
        gesture_excess_ranges=[],
    )


@pytest.mark.parametrize(
    "bad_text,bad_word",
    [
        ("아 발표하기 싫네요, 좆같네요", "좆같네요"),
        ("어 발표하기 싫네요 안 할래요 아 X같네요", "X같네요"),
    ],
)
def test_profanity_and_refusal_transcript_scores_clearly_lower_than_normal(bad_text, bad_word):
    normal_t = Transcript(
        segments=[
            Segment(
                id="s-001",
                start=0.0,
                end=4.0,
                text="안녕하세요 오늘은 프레젠터 서비스를 소개해 드릴게요",
                words=[
                    Word(text="안녕하세요", start=0.0, end=0.8, tag=None),
                    Word(text="오늘은", start=0.9, end=1.4, tag=None),
                    Word(text="프레젠터", start=1.5, end=2.2, tag=None),
                    Word(text="서비스를", start=2.3, end=3.0, tag=None),
                    Word(text="소개해", start=3.1, end=3.5, tag=None),
                    Word(text="드릴게요", start=3.5, end=4.0, tag=None),
                ],
            )
        ]
    )
    bad_t = Transcript(
        segments=[
            Segment(
                id="s-001",
                start=0.0,
                end=4.0,
                text=bad_text,
                words=[
                    Word(text="발표하기", start=0.5, end=1.2, tag=None),
                    Word(text="싫네요", start=1.2, end=1.8, tag=None),
                    Word(text=bad_word, start=3.0, end=3.8, tag=None),
                ],
            )
        ]
    )

    normal_sp = analyze_speech(normal_t, duration_sec=4.5, time_limit_sec=None)
    bad_sp = analyze_speech(bad_t, duration_sec=4.5, time_limit_sec=None)
    assert normal_sp.profanity.count == 0
    assert bad_sp.profanity.count == 1

    normal_ct = ContentAnalysis(
        sections=[Section(SectionKind.INTRO, "s-001", "s-001", 0.0, 4.0)],
        key_message="프레젠터 서비스를 소개해요.",
        issues=[ContentIssue(IssueType.STRUCTURE, ["s-001"], 0.0, 4.0, "본론이 짧아요.")],
        slide_coverage=None,
        expected_questions=[
            ExpectedQuestion(question="질문?", intent="의도예요.", answer_tip="팁이에요.")
        ],
        usage=None,
    )
    bad_ct = ContentAnalysis(
        sections=[],
        key_message=None,
        issues=[
            ContentIssue(IssueType.STRUCTURE, ["s-001"], 0.0, 4.0, "구성이 없어요."),
            # 같은 문장(s-001)에 일반 CLARITY와 LANGUAGE가 함께 있어도 언어 후보끼리 병합된다
            ContentIssue(IssueType.CLARITY, ["s-001"], 0.0, 4.0, "주제가 모호해요."),
            ContentIssue(
                IssueType.LANGUAGE,
                ["s-001"],
                0.0,
                4.0,
                "발표를 거부하는 표현이 있어요. 발표 자리에서는 이 표현 대신 주제를 소개해 보세요.",
                severity=LanguageSeverity.SERIOUS,
            ),
        ],
        slide_coverage=None,
        expected_questions=[
            ExpectedQuestion(question="질문?", intent="의도예요.", answer_tip="팁이에요.")
        ],
        usage=None,
    )

    nv = _make_nv(gaze=1.0)
    normal_score = score_presentation(Purpose.CLASS, normal_sp, nv, normal_ct)
    bad_score = score_presentation(Purpose.CLASS, bad_sp, nv, bad_ct)

    # 욕설(-40) + LANGUAGE SERIOUS(-30) = -70 감점이 적용되어 CONTENT와 종합 점수가 확실히 낮다
    assert bad_score.categories[ScoreCategory.CONTENT] <= 25
    assert normal_score.categories[ScoreCategory.CONTENT] >= 90
    assert bad_score.overall < normal_score.overall - 20

    # 피드백 후보 및 명세 리포트 검증(새 type/tag 없음)
    cands = build_candidates(
        FeedbackInput(
            purpose=Purpose.CLASS,
            duration_sec=4.5,
            transcript=bad_sp.transcript,
            speech=bad_sp,
            nonverbal=nv,
            content=bad_ct,
            category_scores=bad_score.categories,
        )
    )
    # 일반 CLARITY(priority=0)가 끼어 있어도 같은 문장(s-001)의 욕설+LANGUAGE는 하나로 합쳐진다
    critical_clarity = [
        c for c in cands if c.type is FeedbackType.CLARITY and c.severity is Severity.CRITICAL
    ]
    assert len(critical_clarity) == 1
    ev = bad_sp.transcript.segments[0].text
    assert verify_evidence(ev, bad_sp.transcript)

    fb_res = FeedbackResult(
        feedbacks=[
            Feedback(
                id="f-01",
                category=ScoreCategory.CONTENT,
                type=FeedbackType.CLARITY,
                severity=Severity.CRITICAL,
                start=0.0,
                end=4.0,
                title="발표 자리에 맞지 않는 표현이 있어요",
                description="발표를 거부하거나 비공식적인 표현이 있어 신뢰도가 떨어져요.",
                evidence=ev,
                suggestion=(
                    "발표 자리에서는 이 표현 대신 준비한 주제와 순서를 차분히 소개해 보세요."
                ),
            )
        ],
        overall=OverallText(
            summary="발표 태도와 구성을 먼저 다듬어야 해요.",
            strengths=["카메라 응시"],
            improvements=["발표 자리에 맞는 표현 쓰기"],
        ),
        improved_scripts=[],
        candidates=cands,
    )
    report = rp.build_report(
        rp.ReportParts(
            analysis_id="a-bad1",
            presentation_id="p-1",
            attempt_no=1,
            purpose=Purpose.CLASS,
            duration_sec=4.42,
            created_at=datetime(2026, 11, 5, tzinfo=UTC),
            scores=bad_score,
            speech=bad_sp,
            nonverbal=nv,
            content=bad_ct,
            feedback=fb_res,
        )
    )
    rp.validate_report(report)
    assert {f["type"] for f in report["feedbacks"]} <= {"CLARITY", "STRUCTURE"}
    assert {
        w["tag"] for s in report["transcript"]["segments"] for w in s["words"]
    } <= {None, "FILLER", "REPEAT"}

    # 아티팩트 저장 후 recompute해도 동일한 점수가 재현된다
    artifact_data = art.build_artifacts(
        pipeline_version=runner.PIPELINE_VERSION,
        purpose=Purpose.CLASS,
        duration_sec=4.42,
        time_limit_sec=None,
        transcript=bad_t,
        speech_metrics=bad_sp.metrics.model_dump(mode="json", by_alias=True),
        frames=[],
        content=bad_ct,
        feedback=fb_res,
        profanity=bad_sp.profanity,
    )
    assert runner.PIPELINE_VERSION == "5"
    assert artifact_data["speech"]["profanity"]["count"] == 1
    assert artifact_data["content"]["result"]["promptVersion"] == "content_system_v3"

    # 구버전 아티팩트(promptVersion 없음) 복원 시 기본값은 content_system_v2
    legacy_content_dict = dict(artifact_data["content"]["result"])
    legacy_content_dict.pop("promptVersion", None)
    assert art.content_from_json(legacy_content_dict).prompt_version == "content_system_v2"


def test_content_sanitizes_profanity_in_language_reason_and_feedback_flags_script_profanity(
    settings,
):
    from tests.test_content import FakeLlm as ContentFakeLlm
    from tests.test_content import good_payload, response, transcript
    from tests.test_feedback import FakeLlm as FeedbackFakeLlm
    from tests.test_feedback import item, payload, resp, rich_input

    settings.openai_llm_model = "gpt-5-mini"
    p = good_payload(
        issues=[
            {
                "type": "LANGUAGE",
                "severity": "SERIOUS",
                "segmentIds": ["s-001"],
                "reason": "'X같네요'라는 비속어를 썼어요.",
            },
            {
                "type": "CLARITY",
                "segmentIds": ["s-002"],
                "reason": "'존나'라는 표현이 모호해요.",
            },
        ]
    )
    res = analyze_content(
        ContentInput(transcript=transcript(), purpose=Purpose.CLASS),
        ContentFakeLlm(response(p)),
        settings,
    )
    # LANGUAGE 이슈는 살리되 욕설이 포함된 reason은 안전한 문장으로 대체하고, 일반 이슈는 버린다
    assert len(res.issues) == 1
    assert res.issues[0].type is IssueType.LANGUAGE
    assert not contains_profanity(res.issues[0].reason)

    # feedback.py에서 improvedScripts에 욕설이 들어가면 'profanity' 규칙 안내가 재시도에 포함된다
    bad_fb = payload()
    bad_fb["improvedScripts"] = [
        {"segmentId": "s-008", "improved": "문장 8은 X같네요.", "reason": "다듬었어요."}
    ]
    fb_client = FeedbackFakeLlm(resp(bad_fb), resp(payload()))
    generate_feedback(rich_input(), fb_client, settings)
    assert len(fb_client.calls) == 2
    assert "욕설·비속어 단어를" in fb_client.calls[1]["input"]
    _ = (content_mod, item)
