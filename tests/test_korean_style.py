"""해요체 검사."""

import pytest

from app.services.pipeline.korean_style import count_formal, formal_sentences


@pytest.mark.parametrize(
    "text",
    ["결론이 모호합니다.", "근거가 부족해 보입니다", "어떤 점이 다릅니까?",
     "좋아요. 다만 짧습니다.", "설명했습니다”", "필요합니다!", "사례를 드십시오."],
)  # fmt: skip
def test_formal_endings_detected(text):
    assert formal_sentences(text)


@pytest.mark.parametrize(
    "text",
    ["결론이 모호해요.", "근거를 먼저 말하면 좋아요.", "어떤 점이 다른가요?",
     "'감사합니다'로 끝내요.", "정리해 주세요", "", None],
)  # fmt: skip
def test_haeyo_is_not_flagged(text):
    assert formal_sentences(text) == []


def test_only_offending_sentences_returned():
    assert formal_sentences("좋아요. 다만 짧습니다. 보완해 보세요.") == ["다만 짧습니다."]


def test_count_formal_counts_fields():
    assert count_formal(["좋아요.", "부족합니다.", None, "필요합니다. 그래요."]) == 2
