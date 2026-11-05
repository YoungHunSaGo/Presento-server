"""한국어 욕설·비속어 사전 및 탐지(결정적, 코드 계산).

원칙:
- Whisper 받아쓰기는 욕설을 원문 한글('좆같네요', '씨발')로 적기도 하고,
  영문 X나 별표('*')로 마스킹('X같네요', '*같네요', '**발', 'XX같네요', '개*끼')하기도 한다.
  따라서 정규화할 때 알파벳·숫자·한글뿐 아니라 '*'도 보존하고 연속 마스킹('**', 'xx')을 축약한다.
- '시발점', '시발역', '병신년'(연도), '지랄탄', '새끼손가락' 같은 정상 단어는
  예외 목록(ALLOWLIST_PREFIXES)으로 먼저 걸러 오탐을 막는다.
- '변수 X나 Y', '화면이 꺼져요', '불이 꺼져 있다', '위기가 닥쳐 오고', '허리띠를 졸라 매고',
  '새끼를 낳다', '시바가'처럼 일상·학술 발표에서 쓰이는 정상 표현이 오탐되지 않도록
  단독/조사 결합 규칙과 앞뒤 단어 문맥(보조용언·주어)을 함께 본다.
"""

import re
from collections.abc import Sequence
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
    "시바견",
    "시바이누",
    "졸라매",
    "졸라서",
    "졸랐",
    "졸라대",
    "꺼져서",
    "꺼졌",
    "꺼져요",
    "꺼져가",
    "꺼져있",
    "닥쳐서",
    "닥쳐올",
    "닥쳐온",
    "닥쳐요",
    "닥쳤",
)

# 단독(또는 부사·명령형)으로 쓰일 때 매칭되는 표기
# ('x나'는 '변수 X나 Y'와 겹치고, '꺼져요'는 '화면이 꺼져요'와 겹치므로 제외)
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
        "x나게",
        "*나",
        "*나게",
        "개새끼",
        "씹새끼",
        "미친새끼",
        "이새끼",
        "저새끼",
        "그새끼",
        "시바새끼",
        "시바꺼",
        "새끼",
        "새끼야",
        "새끼들",
        "새끼들아",
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
        "닥쳐",
        "닥쳐라",
    }
)

# 명사형 욕설 어간 + 뒤에 붙을 수 있는 조사·호격·파생 접미사
# (단독 '새끼', '시바'는 '새끼를 낳다', '시바가' 오탐 방지를 위해 복합 욕설 어간만 둔다)
NOUN_STEMS: tuple[str, ...] = (
    "씨발",
    "시발",
    "씨바",
    "시바새끼",
    "시바꺼",
    "시바놈",
    "시바년",
    "씨팔",
    "시팔",
    "x발",
    "*발",
    "개새끼",
    "씹새끼",
    "미친새끼",
    "이새끼",
    "저새끼",
    "그새끼",
    "개*끼",
    "*새끼",
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
    r"|이라|이라고|이랑|이나|이네|이냐|이지|이다|입니다|이에요|예요|임|이라는|이란"
    r"|라고|라는|란"
    r"|들|들아|들이|들은|들을|들도"
    r"|년|놈|새끼|짓|병"
    r"|같[가-힣]*|하[가-힣]*|스런|스러운|스럽게"
    r")+$"
)

# 용언형(형용사·동사) 욕설 어간 + 뒤에 붙는 어미(피드백 문구의 인용 조사 결합 포함)
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
    r")(?:이라고|이라는|이라|이란|라고|라는|란|요|도|만|은|는)?$"
)

# 따옴표·인용문 뒤에 붙는 인용 조사('꺼져'라고, '존나'라는, 'ㅅㅂ'이라는 등)
_QUOTED_SUFFIX_RE = re.compile(r"^(?:이라고|이라는|이라|이란|라고|라는|란)+$")

# 문장부호 경계 분리 시 알파벳·숫자·한글·자모·별표(*)만 남긴다
_KEEP_CHARS = re.compile(r"[0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ*]+")
_MULTI_STAR = re.compile(r"\*{2,}")
_MULTI_X_BEFORE_KO = re.compile(r"x{2,}(?=[가-힣ㄱ-ㅎㅏ-ㅣ])")
_REPEAT_CHARS = re.compile(r"([^0-9*])\1{2,}")

# 보조용언·자연어 문맥 예외(앞뒤 토큰 기준)
_BENIGN_NEXT_PREFIX: dict[str, tuple[str, ...]] = {
    "꺼져": ("있", "가", "버", "간"),
    "닥쳐": ("오", "왔", "올", "와", "있"),
    "졸라": ("매", "맸", "매고", "매어", "매는"),
    "새끼": ("손가락", "발가락", "낳", "기르", "키우", "동물", "강아지", "고양이", "곰", "새"),
    "시바": ("견", "이누", "강아지", "신"),
}
_BENIGN_PREV_PREFIX: dict[str, tuple[str, ...]] = {
    "꺼져": ("불", "전원", "화면", "조명", "촛불", "엔진", "시동", "컴퓨터", "모니터", "티비"),
    "닥쳐": ("위기", "폭풍", "재난", "시련", "위험", "고난", "불행", "추위", "파도", "문제"),
    "졸라": ("허리띠", "신발끈", "끈", "줄", "목"),
    "새끼": ("어미", "동물", "강아지", "고양이", "사자", "호랑이", "새", "곰"),
}


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
    """문장부호·공백을 떼고 소문자로 바꾼다(별표 '*'와 한글 자모는 유지, 다중 마스킹 축약)."""
    joined = "".join(_KEEP_CHARS.findall(text.lower()))
    joined = _MULTI_STAR.sub("*", joined)
    joined = _MULTI_X_BEFORE_KO.sub("x", joined)
    return _REPEAT_CHARS.sub(r"\1\1", joined)


def _match_normalized(token: str) -> str | None:
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
    for exact in EXACT_PROFANITIES:
        if len(exact) >= 2 and token.startswith(exact):
            rest = token[len(exact) :]
            if _QUOTED_SUFFIX_RE.match(rest):
                return exact
    return None


def match_profanity_token(text: str) -> str | None:
    """단어 하나가 욕설·비속어면 매칭된 기준 어간(pattern)을, 아니면 None을 돌려준다."""
    return _match_normalized(normalize_profanity_token(text))


def _is_benign_in_context(prev_norm: str, cur_norm: str, next_norm: str) -> bool:
    """'불이 꺼져 있다', '위기가 닥쳐 오고', '허리띠를 졸라 매고' 같은 정상 연어인지 확인."""
    next_prefixes = _BENIGN_NEXT_PREFIX.get(cur_norm, ())
    if next_norm and any(next_norm.startswith(p) for p in next_prefixes):
        return True
    prev_prefixes = _BENIGN_PREV_PREFIX.get(cur_norm, ())
    return bool(prev_norm and any(prev_norm.startswith(p) for p in prev_prefixes))


def contains_profanity(text: str) -> bool:
    """문장 안에 욕설·비속어 토큰이 하나라도 있는지 확인한다(피드백·이유 문구 검증용).

    따옴표·슬래시·괄호 등으로 감싸고 조사를 붙인 형태("'X같네요'라는", "'꺼져'라고", "씨발/병신")도
    문장부호 경계 토큰과 공백 경계 토큰 양쪽으로 검사해 놓치지 않는다.
    """
    if not text:
        return False
    # 1) 문장부호·따옴표 경계로 분리한 토큰 시퀀스 검사
    punct_tokens = [normalize_profanity_token(t) for t in _KEEP_CHARS.findall(text.lower())]
    punct_tokens = [t for t in punct_tokens if t]
    for i, tok in enumerate(punct_tokens):
        if _match_normalized(tok) is not None:
            prev_t = punct_tokens[i - 1] if i > 0 else ""
            next_t = punct_tokens[i + 1] if i + 1 < len(punct_tokens) else ""
            if not _is_benign_in_context(prev_t, tok, next_t):
                return True
    # 2) 별표 마스킹이나 자모 결합이 공백 단위로 묶인 경우 검사
    ws_tokens = [normalize_profanity_token(part) for part in text.split()]
    ws_tokens = [t for t in ws_tokens if t]
    for i, tok in enumerate(ws_tokens):
        if _match_normalized(tok) is not None:
            prev_t = ws_tokens[i - 1] if i > 0 else ""
            next_t = ws_tokens[i + 1] if i + 1 < len(ws_tokens) else ""
            if not _is_benign_in_context(prev_t, tok, next_t):
                return True
    return False


def _detect_from_tokens(
    items: Sequence[tuple[str, float, float]], segment_id: str
) -> list[ProfanityHit]:
    norms = [normalize_profanity_token(raw) for raw, _, _ in items]
    out: list[ProfanityHit] = []
    for i, (_, start, end) in enumerate(items):
        tok = norms[i]
        pat = _match_normalized(tok)
        if pat is None:
            continue
        prev_t = norms[i - 1] if i > 0 else ""
        next_t = norms[i + 1] if i + 1 < len(norms) else ""
        if _is_benign_in_context(prev_t, tok, next_t):
            continue
        out.append(
            ProfanityHit(word=tok, pattern=pat, start=start, end=end, segment_id=segment_id)
        )
    return out


def detect_profanity(transcript: Transcript) -> ProfanityResult:
    """대본의 단어(words) 또는 문장 텍스트에서 욕설·비속어 위치와 개수를 찾는다."""
    hits: list[ProfanityHit] = []
    for seg in transcript.segments:
        if seg.words:
            items = [(w.text, w.start, w.end) for w in seg.words]
        else:
            items = [(raw, seg.start, seg.end) for raw in seg.text.split()]
        hits.extend(_detect_from_tokens(items, seg.id))
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
