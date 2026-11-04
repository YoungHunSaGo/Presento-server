"""FFmpeg로 영상에서 음성 추출(모노 16kHz mp3)과 프레임 샘플링(초당 1~2장).

입력은 워커가 스토리지에서 내려받은 로컬 파일이다.

보안(media.py와 같은 방침):
- 신고된 Content-Type에 맞는 demuxer를 강제해, 재생목록 같은 형식이 다른 주소를
  따라가지 못하게 한다.
- 로컬 파일만 읽으므로 허용 프로토콜은 file 하나다. 경로는 "file:" 접두사를 붙인 절대 경로로 넘겨
  "-"로 시작하는 이름이 옵션으로 해석되거나 "http:..." 같은 이름이 프로토콜로 해석되지 않게 한다.
- 에러 메시지에는 서버 경로를 남기지 않는다.

에러 구분(파이프라인이 분석 실패 코드로 바꾼다):
- NoAudioStream: 음성 트랙이 없다 → NO_SPEECH_DETECTED
- NoVideoStream: 영상 트랙이 없다 → NO_FACE_DETECTED
- MediaError: 파일이 깨졌거나 신고한 형식과 다르다 → CORRUPTED_VIDEO
- FFmpegUnavailable: ffmpeg 없음·시간 초과(서버 문제) → INTERNAL_ERROR
"""

import json
import logging
import re
import subprocess  # noqa: S404 - 고정된 인자 목록으로만 실행한다
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.media import DEMUXERS, MediaError

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 16_000
AUDIO_BITRATE = "48k"  # 모노 음성은 이 정도로 충분하다. 10분 ≈ 3.6MB
WHISPER_MAX_BYTES = 25 * 1024 * 1024  # whisper-1 업로드 제한

MIN_FPS = 1.0
MAX_FPS = 2.0
MAX_FRAME_SIDE = 640  # MediaPipe에 넣기 전 긴 변을 이 크기 이하로 줄인다(키우지는 않음)
JPEG_QUALITY = 3  # ffmpeg -q:v (2~31, 낮을수록 고화질)
# iter_frames가 한 장씩 메모리로 받는 원본 프레임의 최대 픽셀 수(8K 정도). 넘으면 깨진 파일로 본다.
MAX_DECODE_PIXELS = 7680 * 4320

# 워커 동시 실행(--concurrency)끼리 코어를 다투지 않도록 ffmpeg 스레드 수를 제한한다.
FFMPEG_THREADS = 2
# timeout = 기본 + 영상 길이 × 배수.
# 4K HEVC 휴대폰 영상을 적은 코어로 디코딩해도 남도록 넉넉히 잡는다.
TIMEOUT_BASE_SEC = 120.0
TIMEOUT_PER_VIDEO_SEC = 3.0
FALLBACK_VIDEO_SEC = 600.0  # 길이를 모를 때 가정하는 영상 길이(명세 상한 10분)

# 결과물이 원본 길이의 이 비율 - 여유 초보다 짧으면 파일이 중간에 잘린 것으로 본다.
# (잘린 mp4도 ffmpeg는 성공 코드로 끝나고 앞부분만 만들어 낸다.)
MIN_COVERAGE_RATIO = 0.9
COVERAGE_SLACK_SEC = 1.0

_MISSING_STREAM = re.compile(r"matches no streams|does not contain any stream", re.IGNORECASE)
_STDERR_MAX = 300


class NoAudioStream(Exception):
    """영상에 음성 트랙이 없다."""


class NoVideoStream(Exception):
    """파일에 영상 트랙이 없다(음성만 있음)."""


class FFmpegUnavailable(Exception):
    """ffmpeg를 실행하지 못했다(설치 안 됨·시간 초과). 파일 탓이 아니다."""


@dataclass(frozen=True)
class SampledFrame:
    path: Path
    timestamp: float  # 영상 시작부터의 초


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - 셸 없이 고정 인자로 실행
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def _input_args(source: Path, content_type: str) -> tuple[list[str], str]:
    demuxer = DEMUXERS.get(content_type)
    if demuxer is None:
        raise MediaError(f"unsupported content type: {content_type}")
    if not source.is_file():
        raise MediaError("input file does not exist")
    src = str(source.resolve())
    args = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-protocol_whitelist", "file",
        "-threads", str(FFMPEG_THREADS),
        "-f", demuxer,
        "-i", f"file:{src}",
    ]  # fmt: skip
    return args, src


def _exec(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    tool = args[0]
    try:
        return _run(args, timeout)
    except FileNotFoundError:
        logger.warning("%s is not installed", tool)
        raise FFmpegUnavailable(f"{tool} not installed") from None
    except subprocess.TimeoutExpired:
        # 예외 객체에 전체 명령(경로)이 들어 있어 exc_info로 남기지 않는다.
        logger.warning("%s timed out after %.0fs", tool, timeout)
        raise FFmpegUnavailable(f"{tool} timed out") from None


def _ffmpeg(args: list[str], timeout: float, missing: type[Exception], *paths: str) -> None:
    res = _exec(args, timeout)
    if res.returncode == 0:
        return
    stderr = res.stderr.strip()
    for p in paths:
        stderr = stderr.replace(p, "<path>")
    # 앞쪽 경고가 길어도 놓치지 않도록 자르기 전에 판정한다.
    if _MISSING_STREAM.search(stderr):
        raise missing(stderr[:_STDERR_MAX])
    raise MediaError(stderr[:_STDERR_MAX])


def _target_seconds(expected: float | None, max_seconds: float | None) -> float | None:
    """결과물이 덮어야 하는 길이(초). 원본 길이를 모르면 None."""
    if expected is None:
        return None
    if expected <= 0:
        raise ValueError("expected_duration must be positive")
    return min(expected, max_seconds) if max_seconds is not None else expected


def _timeout_for(timeout: float | None, target: float | None, max_seconds: float | None) -> float:
    if timeout is not None:
        return timeout
    seconds = target or max_seconds or FALLBACK_VIDEO_SEC
    return TIMEOUT_BASE_SEC + TIMEOUT_PER_VIDEO_SEC * seconds


def _check_coverage(kind: str, covered: float, target: float | None) -> None:
    if target is None:
        return
    if covered < target * MIN_COVERAGE_RATIO - COVERAGE_SLACK_SEC:
        raise MediaError(f"{kind} covers only {covered:.1f}s of {target:.1f}s (truncated file?)")


def _audio_seconds(path: Path, timeout: float) -> float:
    args = [
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-f", "mp3",
        "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
        f"file:{path.resolve()}",
    ]  # fmt: skip
    res = _exec(args, timeout)
    try:
        return float(res.stdout.strip()) if res.returncode == 0 else 0.0
    except ValueError:
        return 0.0


def _limit_args(max_seconds: float | None) -> list[str]:
    if max_seconds is None:
        return []
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    return ["-t", f"{max_seconds:g}"]


def extract_audio(
    source: Path,
    dest: Path,
    content_type: str,
    *,
    max_seconds: float | None = None,
    expected_duration: float | None = None,
    timeout: float | None = None,
) -> Path:
    """음성 트랙을 모노 16kHz mp3로 dest에 저장하고 dest를 돌려준다.

    max_seconds가 있으면 앞부분 그만큼만 쓴다(길이 제한을 넘는 영상의 비용 보호).
    expected_duration(원본 길이, media.probe_duration)을 주면 결과가 그보다 많이 짧을 때
    잘린 파일로 보고 MediaError를 낸다.
    음성이 영상보다 늦게 시작하면 앞을 무음으로 채워, 받아쓰기 시각이 영상 시각과 맞게 한다.
    """
    base, src = _input_args(source, content_type)
    limit = _limit_args(max_seconds)
    target = _target_seconds(expected_duration, max_seconds)
    timeout = _timeout_for(timeout, target, max_seconds)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = str(dest.resolve())
    args = [
        *base, *limit,
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
        "-af", "aresample=async=1:first_pts=0",
        "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        "-c:a", "libmp3lame", "-b:a", AUDIO_BITRATE, "-threads", str(FFMPEG_THREADS),
        "-f", "mp3", f"file:{out}",
    ]  # fmt: skip
    _ffmpeg(args, timeout, NoAudioStream, src, out)

    size = dest.stat().st_size if dest.exists() else 0
    if size == 0:
        raise NoAudioStream("audio output is empty")
    if size > WHISPER_MAX_BYTES:
        # 10분 제한 안에서는 생기지 않는다. 생기면 설정이 잘못된 것이라 파일을 지우고 알린다.
        dest.unlink(missing_ok=True)
        raise MediaError(f"extracted audio is too large for whisper: {size} bytes")
    if target is not None:
        _check_coverage("audio", _audio_seconds(dest, timeout), target)
    return dest


def sample_frames(
    source: Path,
    out_dir: Path,
    content_type: str,
    *,
    fps: float = MIN_FPS,
    max_seconds: float | None = None,
    expected_duration: float | None = None,
    timeout: float | None = None,
) -> list[SampledFrame]:
    """초당 fps장(1~2)씩 JPEG로 out_dir에 저장하고 시각 순으로 돌려준다.

    out_dir은 비어 있어야 한다(이전 결과와 섞이지 않게). 없으면 만든다.
    휴대폰 영상의 회전 정보는 ffmpeg가 자동으로 반영한다.
    프레임이 한 장도 안 나오거나, expected_duration보다 많이 짧으면 MediaError.
    """
    if not MIN_FPS <= fps <= MAX_FPS:
        raise ValueError(f"fps must be between {MIN_FPS:g} and {MAX_FPS:g}")
    # image2 muxer가 %를 번호 패턴으로 해석한다.
    if "%" in str(out_dir):
        raise ValueError("out_dir must not contain '%'")
    base, src = _input_args(source, content_type)
    limit = _limit_args(max_seconds)
    target = _target_seconds(expected_duration, max_seconds)
    timeout = _timeout_for(timeout, target, max_seconds)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise ValueError("out_dir must be empty")
    out = str(out_dir.resolve())

    side = MAX_FRAME_SIDE
    vf = (
        f"fps={fps:g},"
        f"scale=w='min({side},iw)':h='min({side},ih)'"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    # -frame_pts 1: 파일 이름 번호 = 출력 pts(시간 단위 1/fps) → 번호/fps가 그 프레임의 시각
    args = [
        *base, *limit,
        "-map", "0:V:0", "-an", "-sn", "-dn",
        "-vf", vf,
        "-c:v", "mjpeg", "-q:v", str(JPEG_QUALITY), "-threads", str(FFMPEG_THREADS),
        "-f", "image2", "-frame_pts", "1",
        f"file:{out}/%08d.jpg",
    ]  # fmt: skip
    _ffmpeg(args, timeout, NoVideoStream, src, out)

    # 시작 시각보다 앞선(음수 pts) 프레임은 "-0000001.jpg"처럼 저장되는데, 영상 밖이라 버린다.
    frames = [
        SampledFrame(path=p, timestamp=int(p.stem) / fps)
        for p in out_dir.glob("*.jpg")
        if p.stem.isdigit()
    ]
    frames.sort(key=lambda f: f.timestamp)
    if not frames:
        raise MediaError("no frames could be decoded")
    _check_coverage("frames", frames[-1].timestamp + 1 / fps, target)
    return frames


@dataclass(frozen=True)
class DecodedFrame:
    timestamp: float  # 영상 시작부터의 초
    rgb: Any  # numpy uint8 배열 (높이, 너비, 3). 원본 해상도(회전 반영)


def _video_size(src: str, demuxer: str, timeout: float) -> tuple[int, int]:
    """회전을 반영한 영상 크기(너비, 높이)."""
    args = [
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-f", demuxer,
        "-select_streams", "V:0",
        "-show_entries", "stream=width,height:stream_side_data=rotation",
        "-of", "json", f"file:{src}",
    ]  # fmt: skip
    res = _exec(args, timeout)
    if res.returncode != 0:
        raise MediaError(res.stderr.replace(src, "<path>").strip()[:_STDERR_MAX])
    try:
        streams = json.loads(res.stdout or "{}").get("streams") or []
    except json.JSONDecodeError:
        raise MediaError("ffprobe output is not json") from None
    if not streams:
        raise NoVideoStream("no video stream")
    try:
        w, h = int(streams[0]["width"]), int(streams[0]["height"])
    except (KeyError, TypeError, ValueError):
        raise MediaError("video size unknown") from None
    rotation = 0
    for sd in streams[0].get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"]))
            except (TypeError, ValueError):
                rotation = 0
    if abs(rotation) % 180 == 90:  # ffmpeg가 출력 때 회전을 반영하므로 가로·세로가 바뀐다
        w, h = h, w
    if w <= 0 or h <= 0 or w * h > MAX_DECODE_PIXELS:
        raise MediaError(f"unsupported video size {w}x{h}")
    return w, h


def _read_exact(stream, n: int) -> bytes | None:
    """n바이트를 다 읽으면 그 바이트, 스트림이 끝나 모자라면 None."""
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _selected_times(
    src: str, demuxer: str, interval: float, max_seconds: float | None, timeout: float
) -> list[float]:
    """select 필터가 고를 프레임들의 시각(영상 시작 기준 초). 디코딩 없이 패킷 시각으로 구한다."""
    args = [
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-f", demuxer,
        "-select_streams", "V:0",
        "-show_entries", "stream=time_base:packet=pts,flags:format=start_time",
        "-of", "json", f"file:{src}",
    ]  # fmt: skip
    res = _exec(args, timeout)
    if res.returncode != 0:
        raise MediaError(res.stderr.replace(src, "<path>").strip()[:_STDERR_MAX])
    try:
        data = json.loads(res.stdout or "{}")
        num, den = (int(x) for x in data["streams"][0]["time_base"].split("/"))
        tb = num / den  # ffmpeg av_q2d와 같은 값
        raw_start = data.get("format", {}).get("start_time")
        start = float(raw_start) if raw_start not in (None, "N/A") else 0.0
        # ffmpeg는 입력 시작 시각을 타임베이스 단위 정수로 바꿔 pts에서 뺀다(-copyts가 없을 때)
        offset = round(start / tb)
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, ZeroDivisionError):
        raise MediaError("video timing unknown") from None
    pts = sorted(
        {
            int(p["pts"])
            for p in data.get("packets") or []
            if p.get("pts") not in (None, "N/A") and "D" not in (p.get("flags") or "")
        }  # D: 디코더가 버리는 패킷(편집 목록 앞부분 등) → 프레임으로 나오지 않는다
    )
    selected: list[float] = []
    prev: float | None = None
    for p in pts:
        t = float(p - offset) * tb  # select 필터의 t와 같은 계산
        if prev is None or t - prev >= interval:
            if max_seconds is not None and t >= max_seconds:
                break
            selected.append(round(max(0.0, t), 3))
            prev = t
    return selected


def _label(times: list[float], index: int, interval: float) -> float:
    """index번째로 받은 장의 시각. 목록이 모자라면(예상 밖) 마지막 시각에서 간격만큼 이어 붙인다."""
    if index < len(times):
        return times[index]
    last = times[-1] if times else -interval
    return round(last + (index - len(times) + 1) * interval, 3)


def iter_frames(
    source: Path,
    content_type: str,
    *,
    fps: float = MIN_FPS,
    max_seconds: float | None = None,
    expected_duration: float | None = None,
    timeout: float | None = None,
) -> Iterator[DecodedFrame]:
    """초당 fps장을 원본 해상도 RGB 배열로 한 장씩 돌려준다(디스크에 저장하지 않음).

    얼굴처럼 작은 대상을 원본에서 잘라 보려고 쓴다. ffmpeg가 rawvideo를 파이프로 내보내고,
    한 번에 한 장만 메모리에 둔다(1080p 약 6MB). 마지막 장을 돌려준 뒤 ffmpeg 오류·잘린 파일을
    확인해 예외를 낸다(sample_frames와 같은 구분: NoVideoStream / MediaError / FFmpegUnavailable).

    타임스탬프는 실제로 뽑은 원본 프레임의 시각(영상 시작 기준 초)이다. fps 필터는 시각을
    출력 순번으로 바꿔 버려 "0초"라고 표시한 프레임이 실제로는 0.47초 프레임이 되는 문제가 있었다.
    그래서 원본 시각을 유지하는 select 필터로 "첫 프레임, 그다음은 직전에 고른 프레임보다
    1/fps초 이상 뒤의 첫 프레임"을 고르고, 같은 규칙을 ffprobe가 읽은 패킷 시각(디코딩 없음)에
    적용해 각 장의 시각을 구한다(ffmpeg select와 같은 double 계산).
    """
    if not MIN_FPS <= fps <= MAX_FPS:
        raise ValueError(f"fps must be between {MIN_FPS:g} and {MAX_FPS:g}")
    import numpy as np

    base, src = _input_args(source, content_type)
    limit = _limit_args(max_seconds)
    target = _target_seconds(expected_duration, max_seconds)
    timeout = _timeout_for(timeout, target, max_seconds)
    demuxer = DEMUXERS[content_type]
    probe_timeout = min(timeout, TIMEOUT_BASE_SEC)
    width, height = _video_size(src, demuxer, probe_timeout)
    interval = 1.0 / fps
    times = _selected_times(src, demuxer, interval, max_seconds, probe_timeout)
    frame_bytes = width * height * 3
    select = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval!r})'"
    args = [
        *base, *limit,
        "-map", "0:V:0", "-an", "-sn", "-dn",
        # 크기를 못 박아 둔다. ffprobe가 크기·회전을 잘못 알려도 읽는 단위가 어긋나지 않는다.
        "-vf", f"{select},scale={width}:{height}",
        "-fps_mode", "passthrough",  # 고른 프레임을 복제·누락 없이 그대로 내보낸다
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]  # fmt: skip
    try:
        proc = subprocess.Popen(  # noqa: S603 - 셸 없이 고정 인자로 실행
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        logger.warning("ffmpeg is not installed")
        raise FFmpegUnavailable("ffmpeg not installed") from None
    stdout, stderr_pipe = proc.stdout, proc.stderr
    if stdout is None or stderr_pipe is None:  # PIPE로 열었으므로 생기지 않는다
        proc.kill()
        raise FFmpegUnavailable("ffmpeg pipes unavailable")

    err_parts: list[bytes] = []

    def drain_stderr() -> None:  # 파이프가 차서 ffmpeg가 멈추지 않게 계속 비운다
        for line in stderr_pipe:
            if sum(map(len, err_parts)) < 64_000:
                err_parts.append(line)

    state = {"timed_out": False}

    def kill() -> None:
        state["timed_out"] = True
        proc.kill()

    drainer = threading.Thread(target=drain_stderr, daemon=True)
    drainer.start()
    timer = threading.Timer(timeout, kill)
    timer.start()
    count = 0
    try:
        while (buf := _read_exact(stdout, frame_bytes)) is not None:
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            yield DecodedFrame(timestamp=_label(times, count, interval), rgb=rgb)
            count += 1
        proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:  # 소비자가 중간에 멈춘 경우
            proc.kill()
            proc.wait()
        drainer.join(timeout=5)
        stdout.close()
        stderr_pipe.close()

    # 정상 종료 직후 타이머가 울린 경우(경합)는 시간 초과로 보지 않는다.
    if state["timed_out"] and proc.returncode != 0:
        logger.warning("ffmpeg timed out after %.0fs", timeout)
        raise FFmpegUnavailable("ffmpeg timed out")
    if proc.returncode != 0:
        stderr = b"".join(err_parts).decode("utf-8", "replace").strip().replace(src, "<path>")
        if _MISSING_STREAM.search(stderr):
            raise NoVideoStream(stderr[:_STDERR_MAX])
        raise MediaError(stderr[:_STDERR_MAX])
    if count == 0:
        raise MediaError("no frames could be decoded")
    if count != len(times):
        logger.warning("frame time list mismatch (%d frames, %d times)", count, len(times))
    _check_coverage("frames", _label(times, count - 1, interval) + interval, target)
