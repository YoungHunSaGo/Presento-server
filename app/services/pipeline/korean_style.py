"""LLM이 쓴 한국어 문장의 말투(해요체) 검사와 끝부분 변환.

- formal_sentences / count_formal: 합쇼체("~습니다/합니다/입니다/습니까/십시오")로 끝나는 문장 찾기
- to_haeyo: 문장 끝의 합쇼체 어미를 해요체로 바꾼다. 어간 모음 조화와 자주 쓰는 불규칙 활용
  (하다·되다, 르·ㅡ·ㅂ·ㄷ·ㅅ·ㅎ 불규칙, ㄹ 탈락)을 처리하고, 확신할 수 없는 형태는 그대로 둔다
  (그 문장은 경고 대상으로 남는다). 재시도 근거로 쓰지 않는다.

주의: 발표자가 실제로 말할 문장(improvedScripts.improved)은 발표 말투(합쇼체)가 자연스러우므로
이 모듈로 바꾸지 않는다. 화면 안내 문구(피드백·요약·이유·질문 등)에만 쓴다.
"""

import re
from collections.abc import Iterable

# "~습니다", "~합니다", "~입니다", "~습니까", "~십시오" 등 합쇼체 어미로 끝나는 문장
_FORMAL_END = re.compile(r"(?:니다|니까|십시오)[\s.!?…\"'”’)\]]*$")
# 문장 경계: 마침표 등 뒤의 공백, 또는 공백 없이 바로 한글이 오는 곳("…입니다.좋습니다.")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])(\s+|(?=[가-힣]))")
_TRAILING = re.compile(r"[\s.!?…\"'”’)\]]*$")


def formal_sentences(text: str | None) -> list[str]:
    """합쇼체로 끝나는 문장들."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [s for s in parts[::2] if _FORMAL_END.search(s.strip())]


def count_formal(texts: Iterable[str | None]) -> int:
    """합쇼체 문장이 하나라도 있는 글의 수."""
    return sum(1 for t in texts if formal_sentences(t))


# ---------------------------------------------------------------- 한글 음절


_BASE, _CHO, _JUNG = 0xAC00, 588, 28
# 중성 번호
_A, _AE, _EO, _E, _YEO, _O, _WA, _WAE, _OE, _U, _WO, _EU, _I = (
    0,
    1,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    13,
    14,
    18,
    20,
)
# 종성 번호
_NONE, _NIEUN, _RIEUL, _BIEUP, _SSANG_SIOS, _HIEUH = 0, 4, 8, 17, 20, 27


def _is_syllable(ch: str) -> bool:
    return len(ch) == 1 and "가" <= ch <= "힣"


def _split(ch: str) -> tuple[int, int, int]:
    code = ord(ch) - _BASE
    return code // _CHO, (code % _CHO) // _JUNG, code % _JUNG


def _join(cho: int, jung: int, jong: int = _NONE) -> str:
    return chr(_BASE + cho * _CHO + jung * _JUNG + jong)


def _has_final(ch: str) -> bool:
    return _is_syllable(ch) and _split(ch)[2] != _NONE


def _bright(jung: int) -> bool:
    """양성 모음(ㅏ·ㅗ 계열) → 아요, 아니면 어요."""
    return jung in (_A, _O, 2, 12)  # ㅏ ㅗ ㅑ ㅛ


# ㄹ 탈락 동사·형용사: "-ㅂ니다" 앞 낱말 전체 → 해요체 (알다→압니다→알아요 등).
# 낱말 전체로 맞춘다("남깁니다"의 "깁"이 길다로 잡히지 않게).
_RIEUL_DROP = {
    "압": "알아요", "삽": "살아요", "만듭": "만들어요", "깁": "길어요", "멉": "멀어요",
    "놉": "놀아요", "듭": "들어요", "엽": "열어요", "팝": "팔아요", "붑": "불어요",
}  # fmt: skip
# ㅂ 불규칙(어렵다→어려워요). 좁다·잡다 같은 규칙 활용은 넣지 않는다.
_B_IRREGULAR = {"렵", "쉽", "깝", "덥", "춥", "겁", "볍", "갑", "맙", "답", "럽", "롭", "냅"}
_B_IRREGULAR_WA = {"돕", "곱"}  # 돕다 → 도와요
_D_IRREGULAR = {"듣", "걷", "묻", "싣"}  # 깨닫다는 따로(닫다는 규칙 활용)
_S_IRREGULAR = {"낫", "짓", "붓", "잇", "긋"}
_H_IRREGULAR = {"렇", "떻"}  # 그렇다·어떻다 → 그래요·어때요(넣다·놓다는 규칙)
# 르로 끝나지만 르 불규칙이 아닌 어간(ㅡ 탈락·러 불규칙)
_REU_EXCEPTIONS = {"따르": "따라요", "치르": "치러요", "들르": "들러요", "푸르": "푸르러요"}
# "-이다" 동사(보이다·줄이다 등): "보입니다"를 "보+입니다"(서술격 조사)로 읽지 않게 한다
_I_VERB_STEMS = {
    "보", "줄", "높", "늘", "붙", "먹", "쓰", "기울", "쌓", "놓", "섞", "움직", "숙", "모", "들",
    "끓", "녹", "속", "죽", "낮", "깎", "꺾", "덮", "묶", "볶", "삭", "적", "졸", "뒤섞", "드러",
}  # fmt: skip
# 숫자·영문 끝 글자를 읽을 때 받침이 없는 것(→ 예요): 2(이) 4(사) 5(오) 9(구), %(퍼센트)
_NO_FINAL_READING = set("2459%") | set("ABCDEFGHIJKOPQSTUVWXYZabcdefghijkopqstuvwxyz")


def _open_stem(stem: str) -> str | None:
    """받침 없는 어간 + 아/어요(-ㅂ니다를 떼고 남은 어간)."""
    if not stem or not _is_syllable(stem[-1]):
        return None
    if stem.endswith("아니"):
        return stem + "에요"
    last = stem[-1]
    cho, jung, _ = _split(last)
    head = stem[:-1]
    if last == "하":
        return head + "해요"
    if jung == _OE:  # 되 → 돼
        return head + _join(cho, _WAE) + "요"
    for ending, conj in _REU_EXCEPTIONS.items():
        if stem.endswith(ending):
            return stem[: -len(ending)] + conj
    if last == "르" and head and _is_syllable(head[-1]) and not _has_final(head[-1]):
        pc, pj, _ = _split(head[-1])
        return head[:-1] + _join(pc, pj, _RIEUL) + ("라요" if _bright(pj) else "러요")
    if jung == _EU:  # ㅡ 탈락: 바쁘 → 바빠, 크 → 커
        bright = bool(head) and _is_syllable(head[-1]) and _bright(_split(head[-1])[1])
        return head + _join(cho, _A if bright else _EO) + "요"
    if jung in (_A, _AE, _EO, _E, _YEO, 2, 3, 7):  # ㅏ ㅐ ㅓ ㅔ ㅕ ㅑ ㅒ ㅖ
        return stem + "요"
    if jung == _O:
        return head + _join(cho, _WA) + "요"
    if jung == _U:
        return head + _join(cho, _WO) + "요"
    if jung == _I and cho != 11:  # 이(ㅇ)는 "이어요/여요"가 갈려 두지 않는다
        return head + _join(cho, _YEO) + "요"
    return None


def _closed_stem(stem: str) -> str | None:
    """받침 있는 어간 + 아/어요(-습니다를 떼고 남은 어간)."""
    if not stem or not _is_syllable(stem[-1]):
        return None
    last = stem[-1]
    cho, jung, jong = _split(last)
    head = stem[:-1]
    if jong == _SSANG_SIOS:  # 과거: 했·됐·왔 → 어요
        return stem + "어요"
    if last in _H_IRREGULAR:  # ㅎ 불규칙: 그렇 → 그래, 어떻 → 어때
        return head + _join(cho, _AE) + "요"
    if last in _B_IRREGULAR_WA:
        return head + _join(cho, jung) + "와요"
    if last in _B_IRREGULAR:
        return head + _join(cho, jung) + "워요"
    if stem.endswith("깨닫"):
        return head + _join(cho, jung, _RIEUL) + "아요"
    if last in _D_IRREGULAR:
        return head + _join(cho, jung, _RIEUL) + ("아요" if _bright(jung) else "어요")
    if last in _S_IRREGULAR:
        return head + _join(cho, jung) + ("아요" if _bright(jung) else "어요")
    return stem + ("아요" if _bright(jung) else "어요")


def _last_word(text: str) -> str:
    return re.split(r"[\s\"'“‘(\[]", text)[-1]


def _copula(prefix: str) -> str:
    """명사 + 이에요/예요: 끝 글자를 읽었을 때 받침이 있으면 이에요."""
    if not prefix:
        return "이에요"
    last = prefix[-1]
    if _is_syllable(last):
        return prefix + ("이에요" if _has_final(last) else "예요")
    return prefix + ("예요" if last in _NO_FINAL_READING else "이에요")


def _convert_core(core: str) -> str | None:
    """구두점을 뗀 문장 끝을 해요체로. 못 바꾸면 None."""
    if core.endswith("십시오"):
        return core[:-3] + "세요"
    if core.endswith("십니다"):  # 하십니다 → 하세요
        return core[:-3] + "세요"
    if core.endswith("십니까"):  # 하십니까 → 하시나요
        return core[:-3] + "시나요"
    if core.endswith("입니다") or core.endswith("입니까"):
        prefix = core[:-3]
        if _last_word(prefix) in _I_VERB_STEMS:  # 보입니다 → 보여요, 보입니까 → 보이나요
            return prefix + ("이나요" if core.endswith("까") else "여요")
        if core.endswith("까"):
            return prefix + "인가요"
        return _copula(prefix)
    if core.endswith("습니다") or core.endswith("습니까"):
        stem = core[:-3]
        if core.endswith("까"):
            if not stem or not _is_syllable(stem[-1]):
                return None
            cho, jung, jong = _split(stem[-1])
            if jong == _HIEUH and jung == _EO:  # 어떻습니까 → 어떤가요
                return stem[:-1] + _join(cho, jung, _NIEUN) + "가요"
            return stem + "나요"
        return _closed_stem(stem)
    if core.endswith("니다") or core.endswith("니까"):
        before = core[:-2]
        if not before or not _is_syllable(before[-1]):
            return None
        cho, jung, jong = _split(before[-1])
        if jong != _BIEUP:
            return None
        word = _last_word(before)
        if word in ("겁", "뭡"):  # 것입니다·뭐입니다의 준말: 개선될 겁니다 → 거예요
            base = "거" if word == "겁" else "뭐"
            if core.endswith("다"):
                return before[:-1] + base + "예요"
            cho_, jung_, _ = _split(base)
            return before[:-1] + _join(cho_, jung_, _NIEUN) + "가요"  # 건가요·뭔가요
        if word in _RIEUL_DROP and core.endswith("다"):
            return before[: -len(word)] + _RIEUL_DROP[word]
        stem = before[:-1] + _join(cho, jung)  # ㅂ을 뗀 어간
        if core.endswith("까"):
            return stem + "나요"
        return _open_stem(stem)
    return None


def to_haeyo(text: str | None) -> tuple[str | None, int]:
    """문장마다 끝의 합쇼체 어미를 해요체로 바꾼다. (결과, 못 바꾼 문장 수)"""
    if not text:
        return text, 0
    parts = _SENTENCE_SPLIT.split(text)
    left = 0
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        if not _FORMAL_END.search(sentence.strip()):
            continue
        tail = _TRAILING.search(sentence)
        cut = tail.start() if tail else len(sentence)
        converted = _convert_core(sentence[:cut])
        if converted is None:
            left += 1
            continue
        parts[i] = converted + sentence[cut:]
    return "".join(parts), left
