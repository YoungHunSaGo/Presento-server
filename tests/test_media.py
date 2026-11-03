import subprocess

import pytest

from app.services import media
from app.services.media import MediaError, ProbeUnavailable, probe_duration


class _Runner:
    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, args, timeout):
        self.calls.append(args)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        code, out, err = result
        return subprocess.CompletedProcess(args, code, out, err)


@pytest.fixture
def runner(monkeypatch):
    def install(*results):
        r = _Runner(*results)
        monkeypatch.setattr(media, "_run", r)
        return r

    return install


@pytest.mark.parametrize(
    ("content_type", "demuxer"),
    [("video/mp4", "mov"), ("video/quicktime", "mov"), ("video/webm", "matroska")],
)
def test_forces_demuxer_and_protocol_whitelist(runner, content_type, demuxer):
    r = runner((0, "12.5\n", ""))
    assert probe_duration("http://minio/x", content_type) == 12.5
    args = r.calls[0]
    assert args[args.index("-f") + 1] == demuxer
    assert args[args.index("-protocol_whitelist") + 1] == "http,https,tcp,tls"
    assert "file" not in args[args.index("-protocol_whitelist") + 1]


def test_webm_without_duration_scans_packets(runner):
    r = runner((0, "N/A\n", ""), (0, "0.000,0.033\n4.967,0.033\n2.000,\n", ""))
    assert probe_duration("http://minio/x", "video/webm") == pytest.approx(5.0)
    assert len(r.calls) == 2


def test_quick_mode_does_not_scan(runner):
    r = runner((0, "N/A\n", ""))
    assert probe_duration("http://minio/x", "video/webm", full_scan=False) is None
    assert len(r.calls) == 1


def test_corrupted_file_raises(runner):
    runner((1, "", "Invalid data found when processing input"))
    with pytest.raises(MediaError):
        probe_duration("http://minio/x", "video/mp4")


def test_unsupported_type_raises():
    with pytest.raises(MediaError):
        probe_duration("http://minio/x", "application/x-mpegurl")


@pytest.mark.parametrize(
    "exc", [FileNotFoundError("ffprobe"), subprocess.TimeoutExpired("ffprobe", 1)]
)
def test_missing_ffprobe_or_timeout_is_probe_unavailable(runner, exc):
    runner(exc)
    with pytest.raises(ProbeUnavailable):
        probe_duration("http://minio/x", "video/mp4")


@pytest.mark.parametrize("out", ["nan\n", "-3\n", "inf\n", "\n"])
def test_bad_numbers_are_unknown(runner, out):
    runner((0, out, ""))
    assert probe_duration("http://minio/x", "video/mp4", full_scan=False) is None
