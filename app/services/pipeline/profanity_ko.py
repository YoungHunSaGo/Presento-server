"""한국어 욕설·비속어 사전 및 탐지(결정적, 코드 계산).

원칙:
- Whisper 받아쓰기는 욕설을 원문 한글('좆같네요', '씨발')로 적기도 하고,
  영문 X나 별표('*')로 마스킹('X같네요', '*같네요', 'X발', '개*끼')하기도 한다.
  따라서 정규화할 때 알파벳·숫자·한글뿐 아니라 '*'도 보존한다.
- '시발점', '시발역', '병신년'(연도), '지랄탄', '새끼손가락' 같은 정상 단어는
  예외 목록(ALLOWLIST_PREFIXES)으로 먼저 걸러 오탐을 막는다.
- '그래서', '같네요', '출발'처럼 부분 문자열이 우연히 겹치는 일반 단어는
  어간(접두) + 허용 어미/조사 규칙으로만 매칭해 오탐이 나지 않게 한다.
"""

import re
from dataclasses import dataclass
from typing import Any

from app.schemas.report import Transcript

# 정상 단어 오탐 예외(이 접두로 시작하는 토큰은 욕설로 보지 않는다)
ALLOWLIST_PREFIXES: tuple[str, ...] = (
    "시발점",
    "시발역",
    "시발택시",
    "시발차",
    "시발지",
    "시발항",
    "시발열차",
    "시발버스",
    "병신년",
    "지랄탄",
    "새끼손가락",
    "새끼발가락",
    "새끼줄",
    "졸라매",
    "졸라서",
    "졸랐",
    "졸라대",
    "꺼져서",
    "꺼졌",
    "꺼져가",
    "꺼져있",
    "닥쳐서",
    "닥쳐올",
    "닥쳐온",
    "닥쳤",
)

# 단독(또는 부사형)으로 쓰일 때 바로 매칭되는 표기
EXACT_PROFANITIES: frozenset[str] = frozenset(
    {
        "씨발",
        "시발",
        "씨바",
        "시바",
        "씨팔",
        "시팔",
        "ㅅㅂ",
        "ㅆㅂ",
        "x발",
        "*발",
        "시*",
        "씨*",
        "좆",
        "존나",
        "존나게",
        "졸라",
        "졸라게",
        "ㅈㄴ",
        "x나",
        "x나게",
        "*나",
        "*나게",
        "개새끼",
        "새끼",
        "개*끼",
        "*새끼",
        "ㅅㄲ",
        "개ㅅㄲ",
        "병신",
        "ㅂㅅ",
        "병*",
        "지랄",
        "ㅈㄹ",
        "썅",
        "미친놈",
        "미친년",
        "꺼져",
        "꺼져라",
        "꺼져요",
        "닥쳐",
        "닥쳐라",
        "닥쳐요",
    }
)

# 명사형 욕설 어간 + 뒤에 붙을 수 있는 조사·호격·파생 접미사
NOUN_STEMS: tuple[str, ...] = (
    "씨발",
    "시발",
    "씨바",
    "시바",
    "씨팔",
    "시팔",
    "x발",
    "*발",
    "개새끼",
    "개*끼",
    "*새끼",
    "새끼",
    "병신",
    "지랄",
    "미친놈",
    "미친년",
    "썅놈",
    "썅년",
    "썅",
    "좆",
)

_NOUN_SUFFIX_RE = re.compile(
    r"^(?:"
    r"아|야|이|가|은|는|을|를|도|만|의|에|에서|으로|로|한테|에게|처럼"
    r"|이라|이라고|이랑|이나|이네|이냐|이지|이다|입니다|이에요|예요|임"
    r"|들|들아|들이|들은|들을|들도"
    r"|년|놈|새끼|짓|병"
    r"|같[가-힣]*|하[가-힣]*|스런|스러운|스럽게"
    r")+$"
)

# 용언형(형용사·동사) 욕설 어간 + 뒤에 붙는 어미
PREDICATE_STEMS: tuple[str, ...] = (
    "좆같",
    "ㅈ같",
    "엿같",
    "x같",
    "*같",
    "지랄같",
    "병신같",
    "개같",
)

_PREDICATE_SUFFIX_RE = re.compile(
    r"^(?:"
    r"|다|네|네요|아|아요|어|어요|은|으니|으니까|아서|어서|지|지요|죠|냐|니|군|구나|구만"
    r"|습니다|ㅂ니다|음|기|고|데|던|더니|게|겠다|겠네|겠네요|더라|더라고|던데"
    r")$"
)

# 문장부호 제거 시 알파벳·숫자·한글·자모·별표(*)만 남긴다
_KEEP_CHARS = re.compile(r"[0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ*]+")
_REPEAT_CHARS = re.compile(r"(.)\1{2,}")


@dataclass(frozen=True)
class ProfanityHit:
    """욕설·비속어가 검출된 단어 위치."""

    word: str
    pattern: str
    start: float
    end: float
    segment_id: str

    @property
    def sentence_id(self) -> str:
        return self.segment_id


@dataclass(frozen=True)
class ProfanityResult:
    """대본 전체의 욕설·비속어 검출 결과."""

    hits: tuple[ProfanityHit, ...] = ()

    @property
    def count(self) -> int:
        return len(self.hits)


def normalize_profanity_token(text: str) -> str:
    """문장부호·공백을 떼고 소문자로 바꾼다(별표 '*'와 한글 자모는 유지)."""
    joined = "".join(_KEEP_CHARS.findall(text.lower()))
    return _REPEAT_CHARS.sub(r"\1\1", joined)


def match_profanity_token(text: str) -> str | None:
    """단어 하나가 욕설·비속어면 매칭된 기준 어간(pattern)을, 아니면 None을 돌려준다."""
    token = normalize_profanity_token(text)
    if not token:
        return None
    if any(token.startswith(prefix) for prefix in ALLOWLIST_PREFIXES):
        return None
    if token in EXACT_PROFANITIES:
        for stem in (*PREDICATE_STEMS, *NOUN_STEMS):
            if token.startswith(stem):
                return stem
        return token
    for stem in PREDICATE_STEMS:
        if token.startswith(stem):
            rest = token[len(stem) :]
            if _PREDICATE_SUFFIX_RE.match(rest):
                return stem
    for stem in NOUN_STEMS:
        if token.startswith(stem):
            rest = token[len(stem) :]
            if not rest or _NOUN_SUFFIX_RE.match(rest):
                return stem
    return None


def contains_profanity(text: str) -> bool:
    """문장 안에 욕설·비속어 토큰이 하나라도 있는지 확인한다(피드백 문구 검증용)."""
    return any(match_profanity_token(part) is not None for part in text.split())


def detect_profanity(transcript: Transcript) -> ProfanityResult:
    """대본의 단어(words) 또는 문장 텍스트에서 욕설·비속어 위치와 개수를 찾는다."""
    hits: list[ProfanityHit] = []
    for seg in transcript.segments:
        if seg.words:
            for w in seg.words:
                pat = match_profanity_token(w.text)
                if pat is not None:
                    hits.append(
                        ProfanityHit(
                            word=normalize_profanity_token(w.text),
                            pattern=pat,
                            start=w.start,
                            end=w.end,
                            segment_id=seg.id,
                        )
                    )
        else:
            for raw in seg.text.split():
                pat = match_profanity_token(raw)
                if pat is not None:
                    hits.append(
                        ProfanityHit(
                            word=normalize_profanity_token(raw),
                            pattern=pat,
                            start=seg.start,
                            end=seg.end,
                            segment_id=seg.id,
                        )
                    )
    hits.sort(key=lambda h: (h.start, h.end))
    return ProfanityResult(hits=tuple(hits))


def profanity_to_json(result: ProfanityResult) -> dict[str, Any]:
    return {
        "count": result.count,
        "hits": [
            {
                "word": h.word,
                "pattern": h.pattern,
                "start": h.start,
                "end": h.end,
                "segmentId": h.segment_id,
            }
            for h in result.hits
        ],
    }


def profanity_from_json(
    data: dict[str, Any] | None, transcript: Transcript | None = None
) -> ProfanityResult:
    """저장된 JSON에서 복원하거나, 이전 버전 아티팩트면 대본으로 즉시 계산한다."""
    if data is None:
        return detect_profanity(transcript) if transcript is not None else ProfanityResult()
    hits = tuple(
        ProfanityHit(
            word=h["word"],
            pattern=h["pattern"],
            start=float(h["start"]),
            end=float(h["end"]),
            segment_id=h["segmentId"],
        )
        for h in data.get("hits", [])
    )
    return ProfanityResult(hits=hits)
