"""audio 모듈: 음성 추출(모노 16kHz mp3)과 프레임 샘플링(초당 1~2장).

앞부분은 ffmpeg 호출을 가짜로 바꿔 인자와 에러 구분을 보고,
뒷부분은 ffmpeg로 몇 초짜리 영상을 직접 만들어 실제 결과를 확인한다(픽스처 파일을 커밋하지 않음).
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.media import MediaError
from app.services.pipeline import audio
from app.services.pipeline.audio import (
    FFmpegUnavailable,
    NoAudioStream,
    NoVideoStream,
    extract_audio,
    sample_frames,
)

# ---------- ffmpeg 호출을 가짜로 바꾼 단위 테스트 ----------


class _Runner:
    def __init__(self, result, on_call=None):
        self.result = result
        self.on_call = on_call
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []

    def __call__(self, args, timeout):
        self.calls.append(args)
        self.timeouts.append(timeout)
        if isinstance(self.result, Exception):
            raise self.result
        if self.on_call:
            self.on_call(args)
        code, err = self.result
        return subprocess.CompletedProcess(args, code, "", err)


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "in.mp4"
    p.write_bytes(b"fake")
    return p


def _touch_output(args):
    # 출력 경로는 마지막 인자("file:/.../out.mp3")
    path = args[-1].removeprefix("file:")
    with open(path, "wb") as f:
        f.write(b"ID3")


def _install(monkeypatch, result, on_call=None):
    r = _Runner(result, on_call)
    monkeypatch.setattr(audio, "_run", r)
    return r


def _value(args, flag):
    return args[args.index(flag) + 1]


def test_extract_audio_args(monkeypatch, video, tmp_path):
    r = _install(monkeypatch, (0, ""), _touch_output)
    out = extract_audio(video, tmp_path / "out.mp3", "video/mp4", timeout=42)
    assert out == tmp_path / "out.mp3"
    args = r.calls[0]
    assert args[0] == "ffmpeg"
    assert "-nostdin" in args
    assert _value(args, "-f") == "mov"
    assert _value(args, "-protocol_whitelist") == "file"
    assert _value(args, "-i") == f"file:{video.resolve()}"
    assert _value(args, "-map") == "0:a:0"
    assert "-vn" in args
    assert _value(args, "-ac") == "1"
    assert _value(args, "-ar") == "16000"
    assert _value(args, "-c:a") == "libmp3lame"
    # 음성이 늦게 시작해도 0초부터 무음으로 채워 영상 시각과 맞춘다
    assert _value(args, "-af") == "aresample=async=1:first_pts=0"
    assert "-t" not in args
    assert r.timeouts == [42]


def test_extract_audio_webm_uses_matroska(monkeypatch, video, tmp_path):
    r = _install(monkeypatch, (0, ""), _touch_output)
    extract_audio(video, tmp_path / "out.mp3", "video/webm")
    assert _value(r.calls[0], "-f") == "matroska"


def test_extract_audio_max_seconds(monkeypatch, video, tmp_path):
    r = _install(monkeypatch, (0, ""), _touch_output)
    extract_audio(video, tmp_path / "out.mp3", "video/mp4", max_seconds=600)
    assert _value(r.calls[0], "-t") == "600"


def test_unsupported_content_type(monkeypatch, video, tmp_path):
    r = _install(monkeypatch, (0, ""))
    with pytest.raises(MediaError):
        extract_audio(video, tmp_path / "out.mp3", "video/x-flv")
    with pytest.raises(MediaError):
        sample_frames(video, tmp_path / "f", "application/x-mpegurl")
    assert r.calls == []


def test_no_audio_stream(monkeypatch, video, tmp_path):
    _install(monkeypatch, (1, "Stream map '0:a:0' matches no streams."))
    with pytest.raises(NoAudioStream):
        extract_audio(video, tmp_path / "out.mp3", "video/mp4")


def test_corrupted_input(monkeypatch, video, tmp_path):
    _install(monkeypatch, (1, "in.mp4: Invalid data found when processing input"))
    with pytest.raises(MediaError) as e:
        extract_audio(video, tmp_path / "out.mp3", "video/mp4")
    # 에러 메시지에 서버 경로가 그대로 남지 않는다
    assert str(tmp_path) not in str(e.value)


def test_ffmpeg_missing(monkeypatch, video, tmp_path):
    _install(monkeypatch, FileNotFoundError("ffmpeg"))
    with pytest.raises(FFmpegUnavailable):
        extract_audio(video, tmp_path / "out.mp3", "video/mp4")


def test_ffmpeg_timeout(monkeypatch, video, tmp_path):
    _install(monkeypatch, subprocess.TimeoutExpired(["ffmpeg"], 1))
    with pytest.raises(FFmpegUnavailable):
        sample_frames(video, tmp_path / "f", "video/mp4")


def test_missing_input_file(monkeypatch, tmp_path):
    r = _install(monkeypatch, (0, ""))
    with pytest.raises(MediaError):
        extract_audio(tmp_path / "nope.mp4", tmp_path / "out.mp3", "video/mp4")
    assert r.calls == []


def test_audio_over_whisper_limit(monkeypatch, video, tmp_path):
    def big(args):
        path = args[-1].removeprefix("file:")
        with open(path, "wb") as f:
            f.truncate(audio.WHISPER_MAX_BYTES + 1)

    _install(monkeypatch, (0, ""), big)
    with pytest.raises(MediaError):
        extract_audio(video, tmp_path / "out.mp3", "video/mp4")


def test_empty_audio_output_is_no_audio(monkeypatch, video, tmp_path):
    def empty(args):
        open(args[-1].removeprefix("file:"), "wb").close()

    _install(monkeypatch, (0, ""), empty)
    with pytest.raises(NoAudioStream):
        extract_audio(video, tmp_path / "out.mp3", "video/mp4")


@pytest.mark.parametrize("fps", [0, 0.5, 2.5, 30])
def test_fps_out_of_range(monkeypatch, video, tmp_path, fps):
    r = _install(monkeypatch, (0, ""))
    with pytest.raises(ValueError):
        sample_frames(video, tmp_path / "f", "video/mp4", fps=fps)
    assert r.calls == []


def _write_frames(*numbers):
    # 출력 패턴("file:/.../%08d.jpg")의 디렉터리에 가짜 프레임을 만든다
    def on_call(args):
        out_dir = Path(args[-1].removeprefix("file:")).parent
        for n in numbers:
            (out_dir / n).write_bytes(b"\xff\xd8")

    return on_call


def test_sample_frames_args(monkeypatch, video, tmp_path):
    r = _install(monkeypatch, (0, ""), _write_frames("00000000.jpg"))
    sample_frames(video, tmp_path / "f", "video/quicktime", fps=2, max_seconds=600)
    args = r.calls[0]
    assert _value(args, "-f") == "mov"
    assert _value(args, "-protocol_whitelist") == "file"
    # 첨부 이미지(앨범 커버)가 아닌 실제 영상 트랙만
    assert _value(args, "-map") == "0:V:0"
    assert "-an" in args
    assert "fps=2" in _value(args, "-vf")
    assert _value(args, "-t") == "600"
    assert _value(args, "-frame_pts") == "1"
    assert _value(args, "-threads") == str(audio.FFMPEG_THREADS)
    # 길이를 주지 않으면 max_seconds 기준으로 timeout을 잡는다
    assert r.timeouts == [audio.TIMEOUT_BASE_SEC + audio.TIMEOUT_PER_VIDEO_SEC * 600]


def test_frame_timestamps_and_negative_pts_ignored(monkeypatch, video, tmp_path):
    _install(monkeypatch, (0, ""), _write_frames("00000002.jpg", "00000000.jpg", "-0000001.jpg"))
    frames = sample_frames(video, tmp_path / "f", "video/mp4", fps=2)
    assert [f.timestamp for f in frames] == [0.0, 1.0]


def test_zero_frames_is_media_error(monkeypatch, video, tmp_path):
    _install(monkeypatch, (0, ""))
    with pytest.raises(MediaError):
        sample_frames(video, tmp_path / "f", "video/mp4")


def test_frames_shorter_than_expected(monkeypatch, video, tmp_path):
    _install(monkeypatch, (0, ""), _write_frames("00000000.jpg", "00000001.jpg"))
    with pytest.raises(MediaError):
        sample_frames(video, tmp_path / "f", "video/mp4", expected_duration=60)


def test_frames_expected_duration_capped_by_max_seconds(monkeypatch, video, tmp_path):
    names = [f"{n:08d}.jpg" for n in range(10)]
    r = _install(monkeypatch, (0, ""), _write_frames(*names))
    frames = sample_frames(
        video, tmp_path / "f", "video/mp4", expected_duration=900, max_seconds=10
    )
    assert len(frames) == 10
    assert r.timeouts == [audio.TIMEOUT_BASE_SEC + audio.TIMEOUT_PER_VIDEO_SEC * 10]


def test_percent_in_out_dir_rejected(monkeypatch, video, tmp_path):
    r = _install(monkeypatch, (0, ""))
    with pytest.raises(ValueError):
        sample_frames(video, tmp_path / "pct%d", "video/mp4")
    assert r.calls == []


def test_missing_stream_detected_after_long_warnings(monkeypatch, video, tmp_path):
    noise = "[mov] some warning\n" * 50
    _install(monkeypatch, (1, noise + "Stream map '0:a:0' matches no streams."))
    with pytest.raises(NoAudioStream):
        extract_audio(video, tmp_path / "out.mp3", "video/mp4")


def test_no_video_stream(monkeypatch, video, tmp_path):
    _install(monkeypatch, (1, "Stream map '0:V:0' matches no streams."))
    with pytest.raises(NoVideoStream):
        sample_frames(video, tmp_path / "f", "video/mp4")


def test_frames_out_dir_must_be_empty(monkeypatch, video, tmp_path):
    out = tmp_path / "f"
    out.mkdir()
    (out / "old.jpg").write_bytes(b"x")
    r = _install(monkeypatch, (0, ""))
    with pytest.raises(ValueError):
        sample_frames(video, out, "video/mp4")
    assert r.calls == []


# ---------- 실제 ffmpeg로 확인 ----------

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg가 설치되어 있지 않음",
)


def _make_video(path, seconds=3, size="320x240", with_audio=True, codec="mpeg4"):
    args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10"]
    if with_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
        args += ["-ac", "2"]
    args += ["-t", str(seconds), "-c:v", codec]
    if with_audio:
        args += ["-c:a", "libopus" if path.suffix == ".webm" else "aac"]
    args.append(str(path))
    subprocess.run(args, check=True, capture_output=True)  # noqa: S603
    return path


def _probe(path):
    args = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    out = subprocess.run(  # noqa: S603
        args,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


@needs_ffmpeg
def test_real_extract_audio_mono_16k_mp3(tmp_path):
    src = _make_video(tmp_path / "talk.mp4")
    out = extract_audio(src, tmp_path / "out.mp3", "video/mp4")
    info = _probe(out)
    (stream,) = info["streams"]
    assert stream["codec_name"] == "mp3"
    assert stream["channels"] == 1
    assert stream["sample_rate"] == "16000"
    assert float(info["format"]["duration"]) == pytest.approx(3.0, abs=0.2)


@needs_ffmpeg
def test_real_extract_audio_from_webm(tmp_path):
    src = _make_video(tmp_path / "rec.webm", codec="libvpx")
    out = extract_audio(src, tmp_path / "out.mp3", "video/webm")
    (stream,) = _probe(out)["streams"]
    assert (stream["channels"], stream["sample_rate"]) == (1, "16000")


@needs_ffmpeg
def test_real_extract_audio_max_seconds(tmp_path):
    src = _make_video(tmp_path / "talk.mp4", seconds=4)
    out = extract_audio(src, tmp_path / "out.mp3", "video/mp4", max_seconds=2)
    assert float(_probe(out)["format"]["duration"]) == pytest.approx(2.0, abs=0.2)


@needs_ffmpeg
def test_real_video_without_audio(tmp_path):
    src = _make_video(tmp_path / "silent.mp4", with_audio=False)
    with pytest.raises(NoAudioStream):
        extract_audio(src, tmp_path / "out.mp3", "video/mp4")


@needs_ffmpeg
def test_real_corrupted_file(tmp_path):
    src = tmp_path / "broken.mp4"
    src.write_bytes(b"\x00not a video at all" * 100)
    with pytest.raises(MediaError):
        extract_audio(src, tmp_path / "out.mp3", "video/mp4")
    with pytest.raises(MediaError):
        sample_frames(src, tmp_path / "f", "video/mp4")


@needs_ffmpeg
def test_real_wrong_declared_type(tmp_path):
    # mp4를 webm이라고 신고하면 matroska demuxer가 거부한다
    src = _make_video(tmp_path / "talk.mp4")
    with pytest.raises(MediaError):
        extract_audio(src, tmp_path / "out.mp3", "video/webm")


@needs_ffmpeg
@pytest.mark.parametrize(
    ("fps", "expected"), [(1, [0.0, 1.0, 2.0]), (2, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])]
)
def test_real_sample_frames(tmp_path, fps, expected):
    src = _make_video(tmp_path / "talk.mp4")
    frames = sample_frames(src, tmp_path / "f", "video/mp4", fps=fps)
    assert [f.timestamp for f in frames] == pytest.approx(expected)
    for f in frames:
        assert f.path.parent == tmp_path / "f"
        assert f.path.read_bytes()[:2] == b"\xff\xd8"  # JPEG


@needs_ffmpeg
def test_real_sample_frames_max_seconds(tmp_path):
    src = _make_video(tmp_path / "talk.mp4", seconds=4)
    frames = sample_frames(src, tmp_path / "f", "video/mp4", fps=1, max_seconds=2)
    assert [f.timestamp for f in frames] == pytest.approx([0.0, 1.0])


@needs_ffmpeg
def test_real_frames_are_downscaled(tmp_path):
    # 세로 영상(휴대폰)도 긴 변이 MAX_FRAME_SIDE를 넘지 않게 줄이고 비율은 유지한다
    src = _make_video(tmp_path / "portrait.mp4", seconds=1, size="720x1280")
    (frame,) = sample_frames(src, tmp_path / "f", "video/mp4", fps=1)
    (stream,) = _probe(frame.path)["streams"]
    assert max(stream["width"], stream["height"]) == audio.MAX_FRAME_SIDE
    assert stream["height"] / stream["width"] == pytest.approx(1280 / 720, rel=0.02)


@needs_ffmpeg
def test_real_small_frames_not_upscaled(tmp_path):
    src = _make_video(tmp_path / "small.mp4", seconds=1)
    (frame,) = sample_frames(src, tmp_path / "f", "video/mp4", fps=1)
    (stream,) = _probe(frame.path)["streams"]
    assert (stream["width"], stream["height"]) == (320, 240)


@needs_ffmpeg
def test_real_audio_only_file_has_no_video(tmp_path):
    src = tmp_path / "voice.webm"
    args = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=sample_rate=48000", "-t", "1"]
    args += ["-c:a", "libopus", str(src)]
    subprocess.run(  # noqa: S603
        args,
        check=True,
        capture_output=True,
    )
    with pytest.raises(NoVideoStream):
        sample_frames(src, tmp_path / "f", "video/webm")


def _ffmpeg_make(args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)  # noqa: S603, S607


@needs_ffmpeg
def test_real_truncated_file_detected(tmp_path):
    # moov가 앞에 있는 mp4를 앞부분만 남기면 ffmpeg는 성공 코드로 앞부분만 만들어 낸다
    full = tmp_path / "full.mp4"
    _ffmpeg_make(
        ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-f", "lavfi",
         "-i", "sine=sample_rate=44100", "-t", "6", "-c:v", "mpeg4", "-c:a", "aac",
         "-movflags", "+faststart", str(full)]
    )  # fmt: skip
    cut = tmp_path / "cut.mp4"
    data = full.read_bytes()
    cut.write_bytes(data[: len(data) // 4])
    with pytest.raises(MediaError):
        extract_audio(cut, tmp_path / "out.mp3", "video/mp4", expected_duration=6.0)
    with pytest.raises(MediaError):
        sample_frames(cut, tmp_path / "f", "video/mp4", expected_duration=6.0)
    # 원본은 같은 검사를 통과한다
    extract_audio(full, tmp_path / "ok.mp3", "video/mp4", expected_duration=6.0)
    assert len(sample_frames(full, tmp_path / "g", "video/mp4", expected_duration=6.0)) == 6


@needs_ffmpeg
def test_real_late_audio_is_padded(tmp_path):
    # 영상은 0초, 음성은 1.5초에 시작 → mp3도 3초여야 대본 시각이 영상과 맞는다
    src = tmp_path / "late.mp4"
    _ffmpeg_make(
        ["-f", "lavfi", "-t", "3", "-i", "testsrc=size=320x240:rate=10",
         "-itsoffset", "1.5", "-f", "lavfi", "-t", "1.5", "-i", "sine=sample_rate=44100",
         "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "aac", str(src)]
    )  # fmt: skip
    out = extract_audio(src, tmp_path / "out.mp3", "video/mp4", expected_duration=3.0)
    assert float(_probe(out)["format"]["duration"]) == pytest.approx(3.0, abs=0.2)


@needs_ffmpeg
def test_real_fps_1_5(tmp_path):
    src = _make_video(tmp_path / "talk.mp4", seconds=2)
    frames = sample_frames(src, tmp_path / "f", "video/mp4", fps=1.5)
    assert [f.timestamp for f in frames] == pytest.approx([0.0, 2 / 3, 4 / 3])


@needs_ffmpeg
def test_real_nonzero_start_time(tmp_path):
    # 타임스탬프가 5초부터 시작하는 webm도 프레임 시각은 0부터 센다
    src = tmp_path / "offset.webm"
    _ffmpeg_make(
        ["-f", "lavfi", "-t", "3", "-i", "testsrc=size=320x240:rate=10",
         "-output_ts_offset", "5", "-c:v", "libvpx", str(src)]
    )  # fmt: skip
    frames = sample_frames(src, tmp_path / "f", "video/webm", fps=1, expected_duration=3.0)
    assert [f.timestamp for f in frames] == pytest.approx([0.0, 1.0, 2.0])


@needs_ffmpeg
def test_real_rotated_phone_video(tmp_path):
    # 가로로 저장되고 90도 회전 정보가 붙은 휴대폰 영상 → 세로 프레임으로 나온다
    raw = _make_video(tmp_path / "raw.mp4", seconds=1, size="1280x720")
    src = tmp_path / "rotated.mp4"
    _ffmpeg_make(["-display_rotation", "90", "-i", str(raw), "-c", "copy", str(src)])
    (frame,) = sample_frames(src, tmp_path / "f", "video/mp4", fps=1)
    (stream,) = _probe(frame.path)["streams"]
    assert (stream["width"], stream["height"]) == (360, 640)


# ---------- iter_frames: 원본 해상도 프레임을 메모리로 ----------


@needs_ffmpeg
def test_real_iter_frames_full_resolution_rgb(tmp_path):
    src = _make_video(tmp_path / "v.mp4", seconds=3, size="1280x720")
    frames = list(audio.iter_frames(src, "video/mp4", fps=1))
    assert [f.timestamp for f in frames] == [0.0, 1.0, 2.0]
    assert frames[0].rgb.shape == (720, 1280, 3)  # 줄이지 않은 원본
    assert str(frames[0].rgb.dtype) == "uint8"


@needs_ffmpeg
def test_real_iter_frames_rotated_video(tmp_path):
    raw = _make_video(tmp_path / "raw.mp4", seconds=1, size="1280x720")
    src = tmp_path / "rotated.mp4"
    _ffmpeg_make(["-display_rotation", "90", "-i", str(raw), "-c", "copy", str(src)])
    (frame,) = list(audio.iter_frames(src, "video/mp4", fps=1))
    assert frame.rgb.shape == (1280, 720, 3)  # 회전 반영(세로)


@needs_ffmpeg
def test_real_iter_frames_audio_only_is_no_video(tmp_path):
    src = tmp_path / "a.mp4"
    _ffmpeg_make(["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(src)])
    with pytest.raises(NoVideoStream):
        list(audio.iter_frames(src, "video/mp4"))


@needs_ffmpeg
def test_real_iter_frames_corrupted(tmp_path):
    src = tmp_path / "bad.mp4"
    src.write_bytes(b"\x00" * 4096)
    with pytest.raises(MediaError):
        list(audio.iter_frames(src, "video/mp4"))


@needs_ffmpeg
def test_real_iter_frames_shorter_than_expected(tmp_path):
    src = _make_video(tmp_path / "v.mp4", seconds=2)
    with pytest.raises(MediaError):
        list(audio.iter_frames(src, "video/mp4", expected_duration=30))


@needs_ffmpeg
def test_real_iter_frames_max_seconds(tmp_path):
    src = _make_video(tmp_path / "v.mp4", seconds=5)
    frames = list(audio.iter_frames(src, "video/mp4", fps=1, max_seconds=2))
    assert len(frames) == 2


@needs_ffmpeg
def test_real_iter_frames_consumer_stops_early(tmp_path):
    src = _make_video(tmp_path / "v.mp4", seconds=5)
    gen = audio.iter_frames(src, "video/mp4", fps=1)
    next(gen)
    gen.close()  # ffmpeg를 정리하고 예외 없이 끝나야 한다


def test_iter_frames_rejects_oversized_video(monkeypatch, video):
    res = subprocess.CompletedProcess(
        [], 0, stdout='{"streams":[{"width":20000,"height":20000}]}', stderr=""
    )
    monkeypatch.setattr(audio, "_run", lambda args, timeout: res)
    with pytest.raises(MediaError):
        list(audio.iter_frames(video, "video/mp4"))


def test_iter_frames_rejects_bad_fps(video):
    with pytest.raises(ValueError):
        list(audio.iter_frames(video, "video/mp4", fps=5))


@needs_ffmpeg
def test_real_iter_frames_timeout_kills_ffmpeg(tmp_path):
    src = _make_video(tmp_path / "v.mp4", seconds=3, size="1280x720")
    gen = audio.iter_frames(src, "video/mp4", fps=1, timeout=0.001)
    with pytest.raises(audio.FFmpegUnavailable):
        list(gen)
