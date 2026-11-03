"""stt 모듈: whisper-1 호출 인자, 응답 정규화, 에러 구분.

OpenAI는 절대 실제로 호출하지 않는다. 가짜 클라이언트가 tests/fixtures의 응답 JSON을 돌려준다.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from openai.types.audio import TranscriptionVerbose

from app.services.pipeline import stt
from app.services.pipeline.stt import NoSpeechDetected, SttServiceError, normalize, transcribe

FIXTURE = Path(__file__).parent / "fixtures" / "whisper_verbose_ko.json"


@pytest.fixture
def whisper_json() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class FakeClient:
    """openai.OpenAI 대신 쓰는 가짜. 받은 인자를 기록하고 정해진 응답(또는 예외)을 낸다."""

    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        f = kwargs["file"]
        self.calls.append({**kwargs, "file_bytes": f.read()})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def audio_file(tmp_path) -> Path:
    p = tmp_path / "audio.mp3"
    p.write_bytes(b"ID3fake-mp3")
    return p


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")


# ---------- 호출 인자 ----------


def test_transcribe_sends_whisper1_ko_verbose_json_with_word_and_segment(audio_file, whisper_json):
    client = FakeClient(TranscriptionVerbose.model_validate(whisper_json))
    transcribe(audio_file, client)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "whisper-1"
    assert call["language"] == "ko"
    assert call["response_format"] == "verbose_json"
    assert call["timestamp_granularities"] == ["word", "segment"]
    assert call["file_bytes"] == b"ID3fake-mp3"


def test_transcribe_accepts_sdk_model_and_plain_dict(audio_file, whisper_json):
    a = transcribe(audio_file, FakeClient(TranscriptionVerbose.model_validate(whisper_json)))
    b = transcribe(audio_file, FakeClient(whisper_json))
    assert a == b


def test_make_client_uses_settings(settings):
    client = stt.make_client(settings)
    assert isinstance(client, openai.OpenAI)
    assert client.max_retries == settings.openai_max_retries
    assert client.timeout == settings.openai_timeout_sec


# ---------- 정규화 ----------


def test_normalize_matches_spec_transcript_shape(whisper_json):
    result = normalize(whisper_json)
    dumped = result.transcript.model_dump(by_alias=True)

    segs = dumped["segments"]
    assert [s["id"] for s in segs] == ["s-001", "s-002"]
    first = segs[0]
    assert set(first) == {"id", "start", "end", "text", "words"}
    assert first["start"] == 0.0 and first["end"] == 6.3  # 단어 범위로 맞춘다
    assert first["text"] == "안녕하세요 음 오늘 저희가 발표할 주제는"
    assert first["words"][:2] == [
        {"text": "안녕하세요", "start": 0.0, "end": 0.9, "tag": None},
        {"text": "음", "start": 1.1, "end": 1.4, "tag": None},
    ]
    assert [w["text"] for w in segs[1]["words"]] == ["그래서", "발표", "연습", "서비스입니다."]
    assert result.duration == 14.2


def test_normalize_drops_hallucinated_silence_segment(whisper_json):
    result = normalize(whisper_json)
    assert "감사합니다" not in result.text
    all_words = [w.text for s in result.transcript.segments for w in s.words]
    assert "시청해" not in all_words
    assert result.text == "안녕하세요 음 오늘 저희가 발표할 주제는 그래서 발표 연습 서비스입니다."


def test_every_word_lands_in_exactly_one_segment(whisper_json):
    whisper_json["segments"][2]["no_speech_prob"] = 0.0  # 버리지 않게
    result = normalize(whisper_json)
    words = [w.text for s in result.transcript.segments for w in s.words]
    assert words == [w["word"] for w in whisper_json["words"]]
    for seg in result.transcript.segments:
        for w in seg.words:
            assert seg.start <= w.start <= w.end <= seg.end


def test_word_straddling_boundary_goes_to_segment_of_its_midpoint(whisper_json):
    # 6.2~6.9: 가운데 6.55 → 두 번째 문장(6.4 시작)
    whisper_json["words"][5] = {"word": "주제는", "start": 6.2, "end": 6.9}
    result = normalize(whisper_json)
    segs = result.transcript.segments
    assert segs[0].words[-1].text == "발표할"
    assert segs[1].words[0].text == "주제는"
    assert segs[1].start == 6.2  # 단어를 덮도록 넓힌다
    # text도 단어를 따라가고, 문장끼리 겹치지 않는다
    assert segs[0].text == "안녕하세요 음 오늘 저희가 발표할"
    assert segs[1].text.startswith("주제는 그래서")
    assert segs[0].end <= segs[1].start


def test_segments_never_overlap(whisper_json):
    # 겹치는 단어(5.0~6.6, 6.5~7.1)가 있어도 문장 경계는 뒤로 밀린다
    whisper_json["words"][5] = {"word": "주제는", "start": 4.2, "end": 6.6}
    segs = normalize(whisper_json).transcript.segments
    for a, b in zip(segs, segs[1:], strict=False):
        assert a.end <= b.start


def test_real_word_outside_dropped_segment_is_kept(whisper_json):
    # 지어낸 문장(3~5초)이 맨 앞 문장이고, 그 앞(0.1초)에 실제 단어가 있다
    whisper_json["segments"] = [
        {"id": 0, "start": 3.0, "end": 5.0, "text": " 감사합니다.", "avg_logprob": -1.5,
         "no_speech_prob": 0.9},
        {"id": 1, "start": 5.0, "end": 10.0, "text": " 발표 시작", "avg_logprob": -0.2,
         "no_speech_prob": 0.01},
    ]  # fmt: skip
    whisper_json["words"] = [
        {"word": "네", "start": 0.1, "end": 0.4},
        {"word": "감사합니다.", "start": 3.2, "end": 4.8},
        {"word": "발표", "start": 5.2, "end": 5.8},
        {"word": "시작", "start": 6.0, "end": 6.5},
    ]
    segs = normalize(whisper_json).transcript.segments
    assert [w.text for s in segs for w in s.words] == ["네", "발표", "시작"]


def test_words_out_of_order_are_sorted(whisper_json):
    whisper_json["words"].reverse()
    result = normalize(whisper_json)
    starts = [w.start for s in result.transcript.segments for w in s.words]
    assert starts == sorted(starts)


def test_times_clamped_to_duration_and_rounded(whisper_json):
    whisper_json["duration"] = 10.0
    whisper_json["segments"][1]["end"] = 10.8
    whisper_json["words"][9] = {"word": "서비스입니다.", "start": 8.7333333, "end": 10.7}
    result = normalize(whisper_json)
    last = result.transcript.segments[-1]
    assert last.end == 10.0
    assert last.words[-1].start == 8.73
    assert last.words[-1].end == 10.0
    # 길이 밖에서 시작하는 문장(11.0~)은 빠진다
    assert len(result.transcript.segments) == 2


def test_negative_or_inverted_word_times_are_fixed(whisper_json):
    whisper_json["words"][0] = {"word": "안녕하세요", "start": -0.3, "end": 0.9}
    whisper_json["words"][1] = {"word": "음", "start": 1.4, "end": 1.1}
    words = normalize(whisper_json).transcript.segments[0].words
    assert words[0].start == 0.0
    assert words[1].start == 1.4 and words[1].end == 1.4


def test_blank_words_are_skipped_and_whitespace_collapsed(whisper_json):
    whisper_json["words"][2] = {"word": "  ", "start": 2.0, "end": 2.5}
    whisper_json["words"][3] = {"word": " 저희가\n", "start": 2.6, "end": 3.2}
    words = [w.text for w in normalize(whisper_json).transcript.segments[0].words]
    assert "" not in words and "  " not in words
    assert "저희가" in words


def test_segment_without_words_keeps_its_text(whisper_json):
    whisper_json["segments"].insert(
        1,
        {"id": 9, "start": 6.4, "end": 6.45, "text": " 네", "avg_logprob": -0.2,
         "no_speech_prob": 0.01},
    )  # fmt: skip
    segs = normalize(whisper_json).transcript.segments
    assert [s.text for s in segs][:2] == ["안녕하세요 음 오늘 저희가 발표할 주제는", "네"]
    assert segs[1].words == []
    assert [s.id for s in segs] == ["s-001", "s-002", "s-003"]


def test_words_before_first_segment_go_to_first_segment(whisper_json):
    for s in whisper_json["segments"]:
        s["start"] += 0.5
    result = normalize(whisper_json)
    first = result.transcript.segments[0]
    assert first.words[0].text == "안녕하세요"
    assert first.start == 0.0


# ---------- 말 없음 ----------


def test_no_segments_is_no_speech(whisper_json):
    whisper_json["segments"] = []
    whisper_json["words"] = []
    whisper_json["text"] = ""
    with pytest.raises(NoSpeechDetected):
        normalize(whisper_json)


def test_only_hallucinated_segments_is_no_speech(whisper_json):
    for s in whisper_json["segments"]:
        s["no_speech_prob"] = 0.9
        s["avg_logprob"] = -1.5
    with pytest.raises(NoSpeechDetected):
        normalize(whisper_json)


def test_zero_duration_is_no_speech(whisper_json):
    whisper_json["duration"] = 0
    with pytest.raises(NoSpeechDetected):
        normalize(whisper_json)


def test_empty_audio_file_is_no_speech_without_calling_api(tmp_path, whisper_json):
    p = tmp_path / "empty.mp3"
    p.write_bytes(b"")
    client = FakeClient(whisper_json)
    with pytest.raises(NoSpeechDetected):
        transcribe(p, client)
    assert client.calls == []


# ---------- 응답 형식 이상 ----------


def test_segments_without_words_is_service_error(whisper_json):
    del whisper_json["words"]
    with pytest.raises(SttServiceError):
        normalize(whisper_json)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(duration="14.2"),
        lambda d: d.update(duration=float("nan")),
        lambda d: d.update(segments="oops"),
        lambda d: d["segments"][0].update(start=None),
        lambda d: d["segments"][0].update(text=3),
        lambda d: d["words"][0].update(end=True),
        lambda d: d["words"][0].pop("word"),
        lambda d: d["words"].__setitem__(0, "안녕"),
    ],
)
def test_malformed_response_is_service_error(whisper_json, mutate):
    data = copy.deepcopy(whisper_json)
    mutate(data)
    with pytest.raises(SttServiceError):
        normalize(data)


def test_unexpected_response_type_is_service_error(audio_file):
    with pytest.raises(SttServiceError):
        transcribe(audio_file, FakeClient("plain text"))


# ---------- 호출 실패 ----------


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=_request()),
        openai.APITimeoutError(request=_request()),
        openai.RateLimitError(
            "rate limited", response=httpx.Response(429, request=_request()), body=None
        ),
        openai.AuthenticationError(
            "bad key", response=httpx.Response(401, request=_request()), body=None
        ),
        openai.InternalServerError(
            "boom", response=httpx.Response(500, request=_request()), body=None
        ),
    ],
)
def test_openai_errors_become_service_error(audio_file, error):
    with pytest.raises(SttServiceError) as exc_info:
        transcribe(audio_file, FakeClient(error))
    # 원본 예외(요청 헤더에 API 키)를 cause로도 context로도 달고 다니지 않는다
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_too_large_audio_is_rejected_before_upload(audio_file, whisper_json, monkeypatch):
    monkeypatch.setattr(stt, "WHISPER_MAX_BYTES", 5)
    client = FakeClient(whisper_json)
    with pytest.raises(SttServiceError):
        transcribe(audio_file, client)
    assert client.calls == []


def test_logs_do_not_contain_api_key(audio_file, settings, caplog):
    err = openai.AuthenticationError(
        "Incorrect API key provided: test-openai-key",
        response=httpx.Response(401, request=_request()),
        body=None,
    )
    with caplog.at_level("DEBUG"), pytest.raises(SttServiceError) as exc_info:
        transcribe(audio_file, FakeClient(err))
    key = settings.openai_api_key.get_secret_value()
    assert key not in caplog.text
    assert key not in str(exc_info.value)
