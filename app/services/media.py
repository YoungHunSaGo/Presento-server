"""ffprobe로 영상 길이 확인.

보안:
- 업로드된 파일이 HLS 재생목록 같은 형식이면 ffprobe가 그 안의 다른 주소(내부망·로컬 파일)를
  따라갈 수 있다(SSRF). 그래서 신고된 Content-Type에 맞는 demuxer를 강제하고, 허용 프로토콜도
  제한한다.
- presigned URL에는 서명이 들어 있으므로 로그·예외 메시지에 남기지 않는다.

에러 구분:
- MediaError: 파일 자체가 깨졌거나 형식이 다르다(사용자 파일 문제 → CORRUPTED_VIDEO).
- ProbeUnavailable: ffprobe 없음·시간 초과·스토리지 연결 실패(서버 문제 → INTERNAL_ERROR).
"""

import logging
import math
import re
import subprocess  # noqa: S404 - 고정된 인자 목록으로만 실행한다
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Content-Type → ffprobe demuxer 이름
DEMUXERS = {
    "video/mp4": "mov",
    "video/quicktime": "mov",
    "video/webm": "matroska",
}
# 항상 presigned URL로만 읽으므로 로컬 파일(file:) 접근은 허용하지 않는다.
_PROTOCOLS = "http,https,tcp,tls"

# ffprobe stderr에서 "파일 문제"가 아니라 "가져오기 실패"를 뜻하는 표현
_NETWORK_ERRORS = re.compile(
    r"connection refused|connection reset|connection timed out|timed out|"
    r"failed to resolve|name or service not known|network is unreachable|"
    r"server returned [45]\d\d|http error [45]\d\d|end of file|i/o error",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://\S+")


class MediaError(Exception):
    """재생할 수 없는 파일(형식이 맞지 않거나 손상됨)."""


class ProbeUnavailable(Exception):
    """길이를 재지 못했지만 파일 탓은 아니다(서버·네트워크 문제)."""


ProbeDuration = Callable[[str, str, float, bool], float | None]


def _redact(text: str) -> str:
    return _URL.sub("<url>", text)[:300]


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - 셸 없이 고정 인자로 실행
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def _probe(args: list[str], timeout: float) -> str:
    try:
        res = _run(args, timeout)
    except FileNotFoundError:
        logger.warning("ffprobe is not installed")
        raise ProbeUnavailable("ffprobe not installed") from None
    except subprocess.TimeoutExpired:
        # 예외 객체에는 URL이 든 전체 명령이 들어 있어 exc_info로 남기지 않는다.
        logger.warning("ffprobe timed out after %.0fs", timeout)
        raise ProbeUnavailable("ffprobe timed out") from None
    if res.returncode != 0:
        stderr = _redact(res.stderr.strip())
        if _NETWORK_ERRORS.search(stderr):
            logger.warning("ffprobe could not read the source: %s", stderr)
            raise ProbeUnavailable(stderr)
        raise MediaError(stderr)
    return res.stdout


def _base_args(content_type: str) -> list[str]:
    demuxer = DEMUXERS.get(content_type)
    if demuxer is None:
        raise MediaError(f"unsupported content type: {content_type}")
    return ["ffprobe", "-v", "error", "-protocol_whitelist", _PROTOCOLS, "-f", demuxer]


def _parse_float(text: str) -> float | None:
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def probe_duration(
    source: str, content_type: str, timeout: float = 30.0, full_scan: bool = True
) -> float | None:
    """영상 길이(초). 길이를 알 수 없으면 None.

    헤더에 길이가 없으면(크롬 MediaRecorder webm 등) full_scan=True일 때 패킷을 끝까지 읽어 잰다.
    """
    base = _base_args(content_type)
    out = _probe(
        [*base, "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", source], timeout
    )
    duration = _parse_float(out)
    if duration is not None or not full_scan:
        return duration

    out = _probe(
        [*base, "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0", source],
        timeout,
    )
    end = 0.0
    for line in out.splitlines():
        parts = line.split(",")
        pts = _parse_float(parts[0]) if parts else None
        dur = _parse_float(parts[1]) if len(parts) > 1 else None
        if pts is not None:
            end = max(end, pts + (dur or 0.0))
    return end or None
