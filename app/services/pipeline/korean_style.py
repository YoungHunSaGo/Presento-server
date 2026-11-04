"""LLM이 쓴 한국어 문장이 해요체인지 검사한다(경고용, 재시도 근거로 쓰지 않음)."""

import re
from collections.abc import Iterable

# "~습니다", "~합니다", "~입니다", "~습니까", "~십시오" 등 합쇼체 어미로 끝나는 문장
_FORMAL_END = re.compile(r"(?:니다|니까|십시오)[\s.!?…\"'”’)\]]*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def formal_sentences(text: str | None) -> list[str]:
    """합쇼체로 끝나는 문장들."""
    if not text:
        return []
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if _FORMAL_END.search(s.strip())]


def count_formal(texts: Iterable[str | None]) -> int:
    """합쇼체 문장이 하나라도 있는 글의 수."""
    return sum(1 for t in texts if formal_sentences(t))
