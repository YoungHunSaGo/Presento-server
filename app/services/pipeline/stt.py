"""OpenAI whisper-1 받아쓰기와 결과 정규화.

whisper-1을 language="ko", response_format="verbose_json",
timestamp_granularities=["word", "segment"]로 호출하고,
응답을 명세 5장 transcript 형태(segments[].words[])로 바꾼다.

- whisper는 words를 segments와 따로(최상위 목록으로) 준다. 단어 가운데 시각이 속한 문장에 넣는다.
- 무음 구간에서 whisper가 지어내는 문장("시청해 주셔서 감사합니다" 등)은
  whisper 자체 판정 기준(no_speech_prob 높음 + avg_logprob 낮음)으로 버린다.
- 단어 tag는 여기서 정하지 않는다(None). 추임새·반복어 표시는 speech_metrics가 한다.
- 단어 타임스탬프가 필수라 모델은 whisper-1로 고정한다(설정으로 바꾸지 않는다).

에러 구분(파이프라인이 분석 실패 코드로 바꾼다):
- SttServiceError: OpenAI 호출 실패·응답 형식 이상 → AI_SERVICE_ERROR
- NoSpeechDetected: 받아쓴 말이 없다 → NO_SPEECH_DETECTED
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import openai

from app.core.config import Settings
from app.schemas.report import Segment, Transcript, Word
from app.services.pipeline.audio import WHISPER_MAX_BYTES

logger = logging.getLogger(__name__)

WHISPER_MODEL = "whisper-1"
LANGUAGE = "ko"

# whisper(openai/whisper transcribe.py)의 무음 판정 기준과 같다.
NO_SPEECH_PROB_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0

TIME_DECIMALS = 2


class SttServiceError(Exception):
    """OpenAI 받아쓰기 호출이 실패했거나 응답 형식이 예상과 다르다."""


class NoSpeechDetected(Exception):
    """받아쓴 말이 없다."""


class _TranscriptionsAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _AudioAPI(Protocol):
    transcriptions: _TranscriptionsAPI


class SttClient(Protocol):
    """openai.OpenAI 중 이 모듈이 쓰는 부분. 테스트에서 가짜로 바꾼다."""

    audio: _AudioAPI


@dataclass(frozen=True)
class SttResult:
    transcript: Transcript
    text: str  # 버린 문장을 뺀 전체 대본(문장 사이 공백 하나)
    duration: float  # whisper가 잰 음성 길이(초)


def make_client(settings: Settings) -> SttClient:
    return openai.OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_sec,
        max_retries=settings.openai_max_retries,
    )


def transcribe(audio_path: Path, client: SttClient) -> SttResult:
    """audio_path(audio.extract_audio 결과)를 받아쓰고 정규화해 돌려준다."""
    size = audio_path.stat().st_size
    if size == 0:
        raise NoSpeechDetected("audio file is empty")
    if size > WHISPER_MAX_BYTES:
        # 업로드해도 400이 나므로 크레딧·시간을 쓰기 전에 막는다.
        raise SttServiceError(f"audio is too large for whisper: {size} bytes")

    # 실패 시 except 밖에서 새 예외를 던진다. 원본 예외(요청 헤더에 API 키가 있음)가
    # __context__로도 따라가지 않게 하기 위함이다. 로그에는 상태 코드와 종류만 남긴다.
    failure: str | None = None
    try:
        with audio_path.open("rb") as f:
            response = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                language=LANGUAGE,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
    except openai.APIStatusError as e:
        failure = f"status={e.status_code} type={type(e).__name__}"
    except openai.OpenAIError as e:
        failure = f"type={type(e).__name__}"
    if failure is not None:
        logger.warning("whisper request failed: %s", failure)
        raise SttServiceError(f"whisper request failed: {failure}")

    return normalize(_as_dict(response))


def _as_dict(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        data = dump()
        if isinstance(data, Mapping):
            return data
    raise SttServiceError("unexpected whisper response type")


def _num(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SttServiceError(f"whisper response field {field} is not a number")
    v = float(value)
    if not math.isfinite(v):
        raise SttServiceError(f"whisper response field {field} is not finite")
    return v


def _list(data: Mapping[str, Any], field: str) -> Sequence[Any]:
    value = data.get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        raise SttServiceError(f"whisper response field {field} is not a list")
    return value


def _is_hallucination(seg: Mapping[str, Any]) -> bool:
    no_speech = seg.get("no_speech_prob")
    logprob = seg.get("avg_logprob")
    if not isinstance(no_speech, int | float) or not isinstance(logprob, int | float):
        return False
    return no_speech > NO_SPEECH_PROB_THRESHOLD and logprob < LOGPROB_THRESHOLD


def _clamp(start: float, end: float, duration: float) -> tuple[float, float]:
    start = min(max(start, 0.0), duration)
    end = min(max(end, start), duration)
    return round(start, TIME_DECIMALS), round(end, TIME_DECIMALS)


@dataclass
class _Seg:
    start: float
    end: float
    text: str
    drop: bool
    words: list[Word]


def normalize(data: Mapping[str, Any]) -> SttResult:
    """whisper verbose_json(dict)을 명세 transcript 형태로 바꾼다."""
    duration = _num(data.get("duration"), "duration")
    if duration <= 0:
        raise NoSpeechDetected("audio duration is zero")

    raw_segments = _list(data, "segments")
    raw_words = _list(data, "words")
    if not raw_segments:
        raise NoSpeechDetected("whisper returned no segments")
    if not raw_words:
        # granularity=word를 보냈는데 words가 없으면 응답 이상이다(말이 있는데 단어가 없음).
        raise SttServiceError("whisper response has segments but no words")

    segs: list[_Seg] = []
    for i, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise SttServiceError(f"segments[{i}] is not an object")
        start = _num(raw.get("start"), f"segments[{i}].start")
        end = _num(raw.get("end"), f"segments[{i}].end")
        text = raw.get("text")
        if not isinstance(text, str):
            raise SttServiceError(f"segments[{i}].text is not a string")
        if start >= duration:
            continue
        s, e = _clamp(start, end, duration)
        segs.append(_Seg(s, e, " ".join(text.split()), _is_hallucination(raw), []))
    segs.sort(key=lambda x: x.start)

    words: list[tuple[float, float, str]] = []
    for i, raw in enumerate(raw_words):
        if not isinstance(raw, Mapping):
            raise SttServiceError(f"words[{i}] is not an object")
        start = _num(raw.get("start"), f"words[{i}].start")
        end = _num(raw.get("end"), f"words[{i}].end")
        text = raw.get("word")
        if not isinstance(text, str):
            raise SttServiceError(f"words[{i}].word is not a string")
        text = " ".join(text.split())
        if not text or start >= duration:
            continue
        s, e = _clamp(start, end, duration)
        words.append((s, e, text))
    words.sort(key=lambda w: (w[0], w[1]))

    # 지어낸 문장 안(가운데 시각 기준)의 단어만 버리고, 나머지 단어는 남은 문장 중
    # 가운데 시각 이전에 시작한 마지막 문장에 넣는다(첫 문장 앞 단어는 첫 문장에).
    dropped = [s for s in segs if s.drop]
    if dropped:
        logger.info("dropped %d whisper segment(s) judged as no speech", len(dropped))
    kept = [s for s in segs if not s.drop]
    if kept:
        idx = 0
        for s, e, text in words:
            mid = (s + e) / 2
            if any(d.start <= mid <= d.end for d in dropped):
                continue
            while idx + 1 < len(kept) and mid >= kept[idx + 1].start:
                idx += 1
            kept[idx].words.append(Word(text=text, start=s, end=e, tag=None))

    # 단어가 하나도 안 붙은 문장도 대본 탭에는 보여야 하므로 text가 있으면 남긴다.
    kept = [s for s in kept if s.words or s.text]
    if not any(s.words for s in kept):
        raise NoSpeechDetected("no speech after filtering")

    # 단어가 있는 문장은 text·시각을 단어로 다시 만든다. 경계에 걸친 단어가 옆 문장으로 가도
    # text(대본 탭·근거 대조)와 words(지표·하이라이트)가 어긋나지 않고, 문장끼리 겹치지 않는다.
    segments: list[Segment] = []
    prev_end = 0.0
    for n, s in enumerate(kept, start=1):
        if s.words:
            text = " ".join(w.text for w in s.words)
            start, end = s.words[0].start, max(w.end for w in s.words)
        else:
            text, start, end = s.text, s.start, s.end
        start = max(start, prev_end)
        end = max(end, start)
        prev_end = end
        segments.append(Segment(id=f"s-{n:03d}", start=start, end=end, text=text, words=s.words))

    return SttResult(
        transcript=Transcript(segments=segments),
        text=" ".join(seg.text for seg in segments),
        duration=round(duration, TIME_DECIMALS),
    )
