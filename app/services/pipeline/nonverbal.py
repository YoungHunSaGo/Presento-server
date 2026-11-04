"""비언어 지표: 시선 정면 비율, 자세 안정성, 제스처 빈도 (MediaPipe, 로컬 실행).

두 단계로 나뉜다.
1. observe_video: audio.iter_frames가 원본 해상도로 한 장씩 주는 프레임(초당 1~2장)마다
   - 640px로 줄인 이미지에 Pose Landmarker(VIDEO 모드)를 돌리고,
   - Pose의 코·눈·귀 중심에 한 변이 어깨 너비의 FACE_CROP_SCALE배인 정사각형을 원본에서 잘라
     Face Landmarker에 넣는다. 강의실처럼 멀리서 찍어 얼굴이 작으면 전체 화면에서는 얼굴 검출기가
     얼굴을 못 찾기 때문이다(해상도를 올려도 같음). 잘라 낸 이미지는 메모리에서만 쓴다.
   - 어깨가 안 보여 자를 수 없으면(웹캠 근접 촬영 등) 640px 전체로 얼굴을 찾는다.
   - 얼굴은 IMAGE 모드로 찾는다. 초당 1~2장이고 잘라 내는 위치가 매번 달라 VIDEO 모드의
     추적(앞 프레임 위치 재사용)이 맞지 않는다.
   mediapipe·cv2는 이 함수 안에서만 불러온다(API 프로세스가 무거운 모듈을 싣지 않게).
   observe_frames는 이미 저장된 이미지 파일로 같은 일을 한다(테스트·도구용).
2. analyze_nonverbal: 관측값만으로 지표·구간을 계산한다(순수 함수, 테스트는 가짜 관측값으로).

규칙:
- 시선(gazeFrontRatio): 얼굴 변환 행렬에서 "머리 정면 방향"과 "머리→카메라 방향" 사이의
  yaw/pitch(도)를 구해 둘 다 기준각 이내인 프레임 비율. 화면 가장자리에 앉아 카메라를 봐도
  정면으로 잡힌다.
  얼굴을 못 찾았지만 자세가 잡힌 프레임은 Pose의 코·양쪽 귀 위치·visibility로 고개 방향을
  추정한다(estimate_head_turn). 확실히 돌린 경우(AWAY)는 "시선 벗어남", 정면(FRONT)은 정면으로
  분모에 넣고, 애매하면 측정 불가로 뺀다. FRONT는 좌우만 본 것이라 위아래 시선(원고 보기 등)은
  모르므로 약간 후하게 잡힐 수 있다. FRONT를 빼면 얼굴을 자주 놓치는 영상에서 비율이 크게
  낮아진다(돌린 프레임만 분모에 남음).
- 얼굴·자세가 둘 다 없는 프레임이 LOW_CONF_MIN_SEC 이상 이어진 구간은 lowConfidenceRanges.
  "얼굴도 못 찾고 고개 방향 추정도 안 되는" 프레임이 전체의 MAX_UNKNOWN_HEAD_RATIO 이상이면
  NoFaceDetected.
- 자세(postureStability): 어깨 중심·코 위치를 어깨 너비로 나눠 정규화하고,
  POSTURE_WINDOW_SEC 구간마다 직선 추세를 뺀 표준편차(흔들림)를 구한다.
  천천히 자리를 옮기는 것은 흔들림이 아니다. 프레임이 반 이상 찬 구간만 쓴다.
  흔들림 평균이 0이면 1점, POSTURE_SWAY_ZERO 이상이면 0점(선형).
- 제스처(gesturePerMin): 연속 두 프레임 사이 손목이 어깨 너비(영상 전체 중앙값)의
  GESTURE_MOVE_RATIO 이상 움직이면 움직임으로 본다. GESTURE_MERGE_SEC 안에 이어진 움직임은
  한 번으로 세되, 한 번이 GESTURE_MAX_SEC를 넘게 이어지면 새로 센다(계속 움직이면 많이 센다).
  손목이 보인 시간(분)으로 나눈다.
- 측정 불가: 자세가 잡힌 시간이 MIN_POSE_SEC 미만이면 posture_measured=False,
  손목이 보인 시간이 자세 시간의 MIN_WRIST_VISIBLE_RATIO 미만이면 gesture_measured=False
  (노트북 카메라는 보통 손이 안 보인다). 측정 불가 지표는 null로 돌려준다(명세 5장).
- 모든 구간의 끝은 영상 길이(duration_sec)를 넘지 않는다.

에러 구분(파이프라인이 분석 실패 코드로 바꾼다):
- NoFaceDetected: 얼굴이 거의 안 보인다 → NO_FACE_DETECTED
- NonverbalUnavailable: 모델 파일이 없거나 MediaPipe를 못 돌렸다(서버 문제) → INTERNAL_ERROR
"""

import contextlib
import logging
import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.schemas.report import NonverbalMetrics, TimeRange
from app.services.media import MediaError
from app.services.pipeline import audio
from app.services.pipeline.audio import SampledFrame

logger = logging.getLogger(__name__)

# ---- 시선 ----
GAZE_YAW_MAX_DEG = 20.0
GAZE_PITCH_MAX_DEG = 20.0
GAZE_AWAY_MIN_SEC = 3.0  # 이보다 길게 정면을 벗어난 구간만 GAZE_AWAY 구간으로 돌려준다

# ---- 얼굴 신뢰도 ----
LOW_CONF_MIN_SEC = 2.0
# 얼굴도 못 찾고 고개 방향도 추정 못 한 프레임이 이 비율 이상이면 NO_FACE_DETECTED
MAX_UNKNOWN_HEAD_RATIO = 0.8

# ---- 얼굴 찾기용 머리 영역 자르기 ----
POSE_FRAME_MAX_SIDE = 640  # 자세 인식용으로 줄이는 긴 변(audio.MAX_FRAME_SIDE와 같음)
FACE_CROP_SCALE = 1.2  # 자르는 정사각형 한 변 = 어깨 너비 × 이 값
FACE_CROP_MIN_PX = 48  # 화면 안으로 맞춘 뒤 이보다 작으면 자르지 않는다(원본 픽셀)
FACE_CROP_INPUT_PX = 256  # 잘라 낸 이미지의 긴 변을 이 크기로 맞춰 Face Landmarker에 넣는다
HEAD_POINT_VISIBILITY = 0.5  # 머리 중심 계산에 쓰는 코·눈·귀의 최소 visibility

# ---- 얼굴 없이 Pose로 고개 방향 추정 ----
HEAD_EAR_VISIBLE = 0.5  # 귀가 보인다고 보는 visibility
HEAD_EAR_HIDDEN = 0.2  # 귀가 가려졌다고 보는 visibility
HEAD_NOSE_HIDDEN = 0.2  # 코가 이보다 안 보이면 뒤돌아선 것으로 본다(어깨는 보일 때)
HEAD_FRONT_MAX_RATIO = 0.15  # |코 - 두 귀 가운데| / 두 귀 거리 ≤ 이 값이면 정면
HEAD_TURNED_MIN_RATIO = 0.35  # ≥ 이 값이면 확실히 돌림

# ---- 자세 ----
POSE_VISIBILITY_MIN = 0.5
POSTURE_WINDOW_SEC = 10.0
POSTURE_WINDOW_MIN_FILL = 0.5  # 구간에 기대 프레임 수의 이 비율 이상 있어야 쓴다
POSTURE_SWAY_ZERO = 0.25  # 어깨 너비 대비 흔들림이 이 값이면 0점
POSTURE_UNSTABLE_SWAY = 0.12  # 구간 흔들림이 이 값 이상이면 POSTURE_UNSTABLE 구간
MIN_POSE_SEC = 10.0
# 어깨 너비가 프레임 너비의 이 비율보다 좁으면(멀리 있거나 옆으로 돎) 정규화가 튀어 뺀다
MIN_SHOULDER_WIDTH_RATIO = 0.06

# ---- 제스처 ----
WRIST_VISIBILITY_MIN = 0.5
GESTURE_MOVE_RATIO = 0.25
GESTURE_MERGE_SEC = 1.0
GESTURE_MAX_SEC = 2.0  # 한 번의 제스처로 묶는 최대 길이
MIN_WRIST_VISIBLE_RATIO = 0.3
GESTURE_WINDOW_SEC = 30.0  # GESTURE_NONE/EXCESS 구간을 보는 단위
GESTURE_EXCESS_PER_MIN = 20.0
MIN_GESTURE_WINDOW_SEC = 15.0  # 구간 안에 손목이 이만큼 보여야 그 구간을 판단한다

# ---- MediaPipe ----
FACE_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5

# Pose 랜드마크 번호
_NOSE, _L_SHOULDER, _R_SHOULDER, _L_WRIST, _R_WRIST = 0, 11, 12, 15, 16
_L_EYE, _R_EYE, _L_EAR, _R_EAR = 2, 5, 7, 8

_DECIMALS = 2
_EPS = 1e-9


class NoFaceDetected(Exception):
    """얼굴이 잡힌 프레임이 너무 적다."""


class NonverbalUnavailable(Exception):
    """MediaPipe를 실행하지 못했다(모델 파일 없음 등). 영상 탓이 아니다."""


# 좌표 단위는 "프레임 너비"(x 오른쪽 0~1, y 아래 0~높이/너비). 가로세로 비율이 그대로 유지된다.
Point = tuple[float, float]


@dataclass(frozen=True)
class PoseObs:
    nose: Point
    left_shoulder: Point
    right_shoulder: Point
    left_wrist: Point | None  # 안 보이거나 화면 밖이면 None
    right_wrist: Point | None

    @property
    def shoulder_width(self) -> float:
        return _dist(self.left_shoulder, self.right_shoulder)

    @property
    def any_wrist(self) -> bool:
        return self.left_wrist is not None or self.right_wrist is not None


class HeadTurn(StrEnum):
    """얼굴 없이 Pose로 추정한 고개 방향."""

    FRONT = "FRONT"  # 정면 쪽(좌우로는 안 돌림, 위아래는 모름)
    AWAY = "AWAY"  # 확실히 옆이나 뒤로 돌림


@dataclass(frozen=True)
class FrameObs:
    t: float  # 영상 시작부터의 초
    yaw: float | None  # 얼굴이 안 잡히면 None (도)
    pitch: float | None
    pose: PoseObs | None  # 코·양어깨가 안 보이면 None
    head_turn: HeadTurn | None = None  # Pose로 추정한 고개 방향(애매하거나 Pose가 없으면 None)

    @property
    def has_face(self) -> bool:
        return self.yaw is not None and self.pitch is not None

    @property
    def observed(self) -> bool:
        """얼굴·자세·고개 방향 중 하나라도 알았다(lowConfidenceRanges 판정용).

        뒤돌아서 코가 안 보이면 pose는 None이지만 head_turn(AWAY)은 있다. 이 프레임은 시선
        벗어남으로 세므로 "평가에서 뺀" 저신뢰 구간에 넣지 않는다.
        """
        return self.has_face or self.pose is not None or self.head_turn is not None

    @property
    def head_known(self) -> bool:
        """얼굴을 찾았거나 고개 방향을 추정했다(NO_FACE_DETECTED 판정용)."""
        return self.has_face or self.head_turn is not None


@dataclass(frozen=True)
class NonverbalAnalysis:
    metrics: NonverbalMetrics
    frame_count: int
    face_frame_ratio: float
    posture_measured: bool
    gesture_measured: bool
    gesture_times: list[float]
    gaze_away_ranges: list[TimeRange]
    posture_unstable_ranges: list[TimeRange]
    gesture_none_ranges: list[TimeRange]
    gesture_excess_ranges: list[TimeRange]


# ---------------------------------------------------------------- 각도


def head_angles(matrix: Sequence[Sequence[float]]) -> tuple[float, float]:
    """MediaPipe 얼굴 변환 행렬(4x4, 마지막 열이 이동)에서 카메라 기준 (yaw, pitch)를 도로 구한다.

    머리 정면 방향 f(회전의 z축, 크기 성분 제거)와 머리→카메라 방향 c(-이동)를 비교한다.
    카메라를 똑바로 보면 화면 어디에 있든 둘 다 0 근처다. 이동이 0이면 c = (0, 0, 1).
    """
    f = [matrix[r][2] for r in range(3)]
    fn = math.sqrt(sum(v * v for v in f))
    if fn == 0 or not math.isfinite(fn):
        raise ValueError("degenerate transformation matrix")
    f = [v / fn for v in f]
    c = [-matrix[r][3] for r in range(3)]
    cn = math.sqrt(sum(v * v for v in c))
    if not math.isfinite(cn):
        raise ValueError("degenerate transformation matrix")
    c = [v / cn for v in c] if cn > 0 else [0.0, 0.0, 1.0]

    def yaw_of(v):
        return math.atan2(v[0], v[2])

    def pitch_of(v):
        return -math.atan2(v[1], math.hypot(v[0], v[2]))

    yaw = math.degrees(yaw_of(f) - yaw_of(c))
    yaw = (yaw + 180.0) % 360.0 - 180.0
    pitch = math.degrees(pitch_of(f) - pitch_of(c))
    return yaw, pitch


def _is_front(f: FrameObs) -> bool:
    if f.yaw is None or f.pitch is None:
        return False
    return abs(f.yaw) <= GAZE_YAW_MAX_DEG and abs(f.pitch) <= GAZE_PITCH_MAX_DEG


def _is_away(f: FrameObs) -> bool:
    """시선 벗어남: 얼굴이 기준각 밖이거나, 얼굴은 없고 Pose로 확실히 고개를 돌렸다."""
    if f.has_face:
        return not _is_front(f)
    return f.head_turn is HeadTurn.AWAY


def _gaze_measured(f: FrameObs) -> bool:
    """시선 비율 분모에 넣는 프레임: 얼굴을 찾았거나, Pose로 고개 방향을 알았다."""
    return f.has_face or f.head_turn is not None


def _gaze_front(f: FrameObs) -> bool:
    if f.has_face:
        return _is_front(f)
    return f.head_turn is HeadTurn.FRONT


# ---------------------------------------------------------------- 구간 도우미


def _frame_interval(frames: Sequence[FrameObs]) -> float:
    diffs = [b.t - a.t for a, b in zip(frames, frames[1:], strict=False) if b.t > a.t]
    return statistics.median(diffs) if diffs else 1.0


def _range(start: float, end: float, limit: float) -> TimeRange:
    return TimeRange(start=round(start, _DECIMALS), end=round(min(end, limit), _DECIMALS))


def _runs(
    frames: Sequence[FrameObs],
    pred: Callable[[FrameObs], bool],
    dt: float,
    min_sec: float,
    limit: float,
) -> list[TimeRange]:
    """pred가 참인 프레임이 이어진 구간 중 min_sec 이상인 것. 끝은 마지막 프레임 + 한 칸."""
    found: list[tuple[float, float]] = []
    start: float | None = None
    last = 0.0
    for f in frames:
        if pred(f):
            if start is None:
                start = f.t
            last = f.t
        elif start is not None:
            found.append((start, last + dt))
            start = None
    if start is not None:
        found.append((start, last + dt))
    return [_range(s, e, limit) for s, e in found if e - s >= min_sec - _EPS]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _windows(start: float, end: float, size: float) -> list[tuple[float, float]]:
    out = []
    t = start
    while t < end - _EPS:
        out.append((t, min(t + size, end)))
        t += size
    return out


def _merge(ranges: list[TimeRange]) -> list[TimeRange]:
    out: list[TimeRange] = []
    for r in ranges:
        if out and r.start <= out[-1].end:
            out[-1] = TimeRange(start=out[-1].start, end=max(out[-1].end, r.end))
        else:
            out.append(r)
    return out


# ---------------------------------------------------------------- 자세


def _detrended_var(ts: list[float], vs: list[float]) -> float:
    """직선 추세(최소제곱)를 뺀 분산."""
    n = len(ts)
    mt, mv = statistics.fmean(ts), statistics.fmean(vs)
    stt = sum((t - mt) ** 2 for t in ts)
    slope = sum((t - mt) * (v - mv) for t, v in zip(ts, vs, strict=True)) / stt if stt else 0.0
    return sum((v - mv - slope * (t - mt)) ** 2 for t, v in zip(ts, vs, strict=True)) / n


def _sway(frames: Sequence[FrameObs]) -> float:
    """프레임 묶음의 흔들림(어깨 너비 대비, 직선 추세 제외)."""
    poses = [f.pose for f in frames if f.pose is not None]
    ts = [f.t for f in frames if f.pose is not None]
    width = statistics.median(p.shoulder_width for p in poses)
    centers = [
        (
            (p.left_shoulder[0] + p.right_shoulder[0]) / 2,
            (p.left_shoulder[1] + p.right_shoulder[1]) / 2,
        )
        for p in poses
    ]
    noses = [p.nose for p in poses]

    def spread(points: list[Point]) -> float:
        var = _detrended_var(ts, [p[0] for p in points]) + _detrended_var(
            ts, [p[1] for p in points]
        )
        return math.sqrt(var) / width

    return (spread(centers) + spread(noses)) / 2


def _posture(
    pose_frames: Sequence[FrameObs], dt: float, start: float, end: float
) -> tuple[float, list[TimeRange]]:
    min_frames = max(3, math.ceil(POSTURE_WINDOW_SEC / dt * POSTURE_WINDOW_MIN_FILL - _EPS))
    sways: list[float] = []
    unstable: list[TimeRange] = []
    for w0, w1 in _windows(start, end, POSTURE_WINDOW_SEC):
        in_window = [f for f in pose_frames if w0 <= f.t < w1]
        if len(in_window) < min_frames:
            continue
        s = _sway(in_window)
        sways.append(s)
        if s >= POSTURE_UNSTABLE_SWAY:
            unstable.append(_range(w0, w1, end))
    if not sways:
        return 1.0, []
    score = 1.0 - statistics.fmean(sways) / POSTURE_SWAY_ZERO
    return round(min(1.0, max(0.0, score)), _DECIMALS), _merge(unstable)


# ---------------------------------------------------------------- 제스처


def _gesture_times(pose_frames: Sequence[FrameObs], dt: float, width: float) -> list[float]:
    """제스처 시각(움직임이 끝난 프레임 시각)."""
    moves: list[float] = []
    for a, b in zip(pose_frames, pose_frames[1:], strict=False):
        # 프레임이 빠져 간격이 벌어지면(얼굴·자세 놓침) 그 사이 움직임은 알 수 없다.
        if b.t - a.t > dt * 1.5 + _EPS or a.pose is None or b.pose is None:
            continue
        pairs = ((a.pose.left_wrist, b.pose.left_wrist), (a.pose.right_wrist, b.pose.right_wrist))
        if any(
            wa is not None and wb is not None and _dist(wa, wb) / width >= GESTURE_MOVE_RATIO - _EPS
            for wa, wb in pairs
        ):
            moves.append(b.t)
    gestures: list[float] = []
    last_move: float | None = None
    for t in moves:
        if (
            last_move is None
            or t - last_move > GESTURE_MERGE_SEC + _EPS
            or t - gestures[-1] >= GESTURE_MAX_SEC - _EPS
        ):
            gestures.append(t)
        last_move = t
    return gestures


def _gesture_windows(
    wrist_frames: Sequence[FrameObs],
    gestures: list[float],
    dt: float,
    start: float,
    end: float,
) -> tuple[list[TimeRange], list[TimeRange]]:
    none: list[TimeRange] = []
    excess: list[TimeRange] = []
    for w0, w1 in _windows(start, end, GESTURE_WINDOW_SEC):
        seen = sum(dt for f in wrist_frames if w0 <= f.t < w1)
        if seen < MIN_GESTURE_WINDOW_SEC - _EPS:
            continue
        n = sum(1 for g in gestures if w0 <= g < w1)
        if n == 0:
            none.append(_range(w0, w1, end))
        elif n / (seen / 60) >= GESTURE_EXCESS_PER_MIN - _EPS:
            excess.append(_range(w0, w1, end))
    return _merge(none), _merge(excess)


# ---------------------------------------------------------------- 분석


def _dedupe(frames: Sequence[FrameObs]) -> list[FrameObs]:
    """시각순 정렬, 같은 시각은 첫 프레임만."""
    out: list[FrameObs] = []
    for f in sorted(frames, key=lambda f: f.t):
        if not out or f.t > out[-1].t + _EPS:
            out.append(f)
    return out


def analyze_nonverbal(
    frames: Sequence[FrameObs], duration_sec: float | None = None
) -> NonverbalAnalysis:
    """프레임 관측값으로 비언어 지표와 피드백용 구간을 계산한다.

    duration_sec(영상 길이)를 주면 구간 끝을 그 안으로 자른다.
    """
    frames = _dedupe(frames)
    if not frames:
        raise NoFaceDetected("no frames")
    face_frames = [f for f in frames if f.has_face]
    face_ratio = len(face_frames) / len(frames)
    unknown_ratio = sum(1 for f in frames if not f.head_known) / len(frames)
    if unknown_ratio >= MAX_UNKNOWN_HEAD_RATIO - _EPS:
        raise NoFaceDetected(f"head unknown in {unknown_ratio:.0%} of frames")

    dt = _frame_interval(frames)
    start = frames[0].t
    end = frames[-1].t + dt
    if duration_sec is not None:
        if not math.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        end = min(end, duration_sec)
    low_conf = _runs(frames, lambda f: not f.observed, dt, LOW_CONF_MIN_SEC, end)

    measured = [f for f in frames if _gaze_measured(f)]
    front = sum(1 for f in measured if _gaze_front(f))
    gaze_ratio = round(front / len(measured), _DECIMALS) if measured else None
    gaze_away = _runs(frames, _is_away, dt, GAZE_AWAY_MIN_SEC, end)

    # 얼굴·자세가 둘 다 없는 프레임은 pose가 None이라 여기서 자연히 빠진다.
    pose_frames = [
        f
        for f in frames
        if f.pose is not None and f.pose.shoulder_width >= MIN_SHOULDER_WIDTH_RATIO
    ]
    pose_sec = len(pose_frames) * dt
    posture_measured = pose_sec >= MIN_POSE_SEC - _EPS
    stability: float | None = None
    unstable: list[TimeRange] = []
    if posture_measured:
        stability, unstable = _posture(pose_frames, dt, start, end)

    wrist_frames = [f for f in pose_frames if f.pose is not None and f.pose.any_wrist]
    wrist_sec = len(wrist_frames) * dt
    gesture_measured = posture_measured and wrist_sec >= pose_sec * MIN_WRIST_VISIBLE_RATIO - _EPS
    gestures: list[float] = []
    gesture_per_min: float | None = None
    g_none: list[TimeRange] = []
    g_excess: list[TimeRange] = []
    if gesture_measured:
        width = statistics.median(f.pose.shoulder_width for f in pose_frames if f.pose)
        gestures = _gesture_times(pose_frames, dt, width)
        gesture_per_min = round(len(gestures) / (wrist_sec / 60), _DECIMALS)
        g_none, g_excess = _gesture_windows(wrist_frames, gestures, dt, start, end)

    return NonverbalAnalysis(
        metrics=NonverbalMetrics(
            gaze_front_ratio=gaze_ratio,
            posture_stability=stability,
            gesture_per_min=gesture_per_min,
            low_confidence_ranges=low_conf,
        ),
        frame_count=len(frames),
        face_frame_ratio=round(face_ratio, _DECIMALS),
        posture_measured=posture_measured,
        gesture_measured=gesture_measured,
        gesture_times=gestures,
        gaze_away_ranges=gaze_away,
        posture_unstable_ranges=unstable,
        gesture_none_ranges=g_none,
        gesture_excess_ranges=g_excess,
    )


# ---------------------------------------------------------------- 머리 영역·고개 방향(순수 함수)

# Pose 랜드마크 하나: (x, y, visibility). x·y는 이미지 너비·높이에 대한 0~1 비율
# (화면 밖이면 범위를 벗어난다)
Landmark = tuple[float, float, float]


def head_crop_box(
    head_points: Sequence[Landmark],
    left_shoulder: Landmark,
    right_shoulder: Landmark,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """원본(width×height 픽셀)에서 얼굴을 찾을 정사각형 (x0, y0, x1, y1). 못 정하면 None.

    중심 = visibility가 HEAD_POINT_VISIBILITY 이상인 코·눈·귀의 평균, 한 변 = 어깨 너비 ×
    FACE_CROP_SCALE. 화면 밖으로 나가는 부분은 잘라 화면 안으로 맞춘다(정사각형이 아닐 수 있음).
    """
    pts = [p for p in head_points if p[2] >= HEAD_POINT_VISIBILITY]
    if not pts or min(left_shoulder[2], right_shoulder[2]) < POSE_VISIBILITY_MIN:
        return None
    cx = statistics.fmean(p[0] for p in pts) * width
    cy = statistics.fmean(p[1] for p in pts) * height
    shoulder = math.hypot(
        (left_shoulder[0] - right_shoulder[0]) * width,
        (left_shoulder[1] - right_shoulder[1]) * height,
    )
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(shoulder)):
        return None
    half = shoulder * FACE_CROP_SCALE / 2
    x0, y0 = max(0, round(cx - half)), max(0, round(cy - half))
    x1, y1 = min(width, round(cx + half)), min(height, round(cy + half))
    if x1 - x0 < FACE_CROP_MIN_PX or y1 - y0 < FACE_CROP_MIN_PX:
        return None
    return x0, y0, x1, y1


def _on_screen(p: Landmark) -> bool:
    return 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0


def estimate_head_turn(
    nose: Landmark, left_ear: Landmark, right_ear: Landmark, body_visible: bool
) -> HeadTurn | None:
    """얼굴을 못 찾았을 때 Pose의 코·양쪽 귀로 고개 방향을 추정한다. 애매하면 None.

    - 어깨는 보이는데 코가 거의 안 보임 → 뒤돌아섬(AWAY)
    - 두 귀가 다 보임 → 코가 두 귀 가운데에서 귀 거리 대비 얼마나 벗어났는지로 판단
      (HEAD_FRONT_MAX_RATIO 이하 FRONT, HEAD_TURNED_MIN_RATIO 이상 AWAY, 사이는 None)
    - 한쪽 귀만 보이고 다른 귀는 가려짐(옆모습) → AWAY
    - 코나 귀가 화면 밖이면(머리가 잘림) → None
    """
    # 화면 밖(카메라가 머리를 잘라 냄)은 가려진 것과 구별해 "모름"으로 본다.
    if not _on_screen(nose) or not _on_screen(left_ear) or not _on_screen(right_ear):
        return None
    if nose[2] < HEAD_NOSE_HIDDEN:
        return HeadTurn.AWAY if body_visible else None
    if nose[2] < HEAD_EAR_VISIBLE:
        return None
    le, re_ = left_ear[2], right_ear[2]
    if le >= HEAD_EAR_VISIBLE and re_ >= HEAD_EAR_VISIBLE:
        span = abs(left_ear[0] - right_ear[0])
        if span < _EPS:
            return None
        ratio = abs(nose[0] - (left_ear[0] + right_ear[0]) / 2) / span
        # 경계값은 포함한다(0.512-0.5=0.01200…0002 같은 부동소수점 오차 허용)
        if ratio <= HEAD_FRONT_MAX_RATIO + _EPS:
            return HeadTurn.FRONT
        if ratio >= HEAD_TURNED_MIN_RATIO - _EPS:
            return HeadTurn.AWAY
        return None
    if (le >= HEAD_EAR_VISIBLE and re_ < HEAD_EAR_HIDDEN) or (
        re_ >= HEAD_EAR_VISIBLE and le < HEAD_EAR_HIDDEN
    ):
        return HeadTurn.AWAY
    return None


# ---------------------------------------------------------------- MediaPipe 실행


def _pose_obs(landmarks, width: int, height: int) -> PoseObs | None:
    aspect = height / width

    def pt(i: int, min_vis: float) -> Point | None:
        lm = landmarks[i]
        vis = getattr(lm, "visibility", None)
        if vis is not None and vis < min_vis:
            return None
        # Pose는 화면 밖 랜드마크도 추정해 준다. 화면 밖은 안 보이는 것으로 본다.
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return None
        return (lm.x, lm.y * aspect)

    nose = pt(_NOSE, POSE_VISIBILITY_MIN)
    ls = pt(_L_SHOULDER, POSE_VISIBILITY_MIN)
    rs = pt(_R_SHOULDER, POSE_VISIBILITY_MIN)
    if nose is None or ls is None or rs is None:
        return None
    return PoseObs(
        nose=nose,
        left_shoulder=ls,
        right_shoulder=rs,
        left_wrist=pt(_L_WRIST, WRIST_VISIBILITY_MIN),
        right_wrist=pt(_R_WRIST, WRIST_VISIBILITY_MIN),
    )


def _landmark(lms, i: int) -> Landmark:
    lm = lms[i]
    vis = getattr(lm, "visibility", None)
    return (float(lm.x), float(lm.y), 1.0 if vis is None else float(vis))


def _check_models(face_model_path: Path, pose_model_path: Path) -> None:
    for p in (face_model_path, pose_model_path):
        if not p.is_file():
            raise NonverbalUnavailable(f"model file not found: {p.name}")


class _Observer:
    """Pose(VIDEO, 640px)와 Face(IMAGE, 원본에서 잘라 낸 머리 영역) 랜드마커를 함께 쓴다."""

    def __init__(self, face_model_path: Path, pose_model_path: Path) -> None:
        _check_models(face_model_path, pose_model_path)
        try:
            import cv2
            import mediapipe as mp
            import numpy as np
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:  # pragma: no cover - 설치 문제
            raise NonverbalUnavailable("mediapipe/opencv is not installed") from e
        self._cv2, self._mp, self._np = cv2, mp, np
        face_opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_MIN_PRESENCE_CONFIDENCE,
        )
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=POSE_MIN_PRESENCE_CONFIDENCE,
        )
        self._face = vision.FaceLandmarker.create_from_options(face_opts)
        self._pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self._last_ms = -1

    def close(self) -> None:
        for lm in (self._face, self._pose):
            with contextlib.suppress(Exception):  # 정리 단계
                lm.close()

    def _image(self, rgb):
        return self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=self._np.ascontiguousarray(rgb)
        )

    def _resize_long(self, rgb, target: int):
        h, w = rgb.shape[:2]
        scale = target / max(h, w)
        if scale >= 1 and target == POSE_FRAME_MAX_SIDE:
            return rgb  # 자세용은 키우지 않는다
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        interp = self._cv2.INTER_AREA if scale < 1 else self._cv2.INTER_LINEAR
        return self._cv2.resize(rgb, size, interpolation=interp)

    def observe(self, t: float, rgb) -> FrameObs:
        """원본 해상도 RGB 한 장 → 관측값."""
        ts = max(round(t * 1000), self._last_ms + 1)  # VIDEO 모드는 시각이 계속 커져야 한다
        self._last_ms = ts
        full_h, full_w = rgb.shape[:2]
        small = self._resize_long(rgb, POSE_FRAME_MAX_SIDE)
        pose_res = self._pose.detect_for_video(self._image(small), ts)
        lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

        pose_obs = _pose_obs(lms, small.shape[1], small.shape[0]) if lms else None
        box = None
        turn = None
        if lms:
            ls, rs = _landmark(lms, _L_SHOULDER), _landmark(lms, _R_SHOULDER)
            head = [_landmark(lms, i) for i in (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR)]
            box = head_crop_box(head, ls, rs, full_w, full_h)
            body_visible = min(ls[2], rs[2]) >= POSE_VISIBILITY_MIN
            turn = estimate_head_turn(
                _landmark(lms, _NOSE), _landmark(lms, _L_EAR), _landmark(lms, _R_EAR), body_visible
            )

        if box is not None:
            x0, y0, x1, y1 = box
            face_input = self._resize_long(rgb[y0:y1, x0:x1], FACE_CROP_INPUT_PX)
        else:
            face_input = small  # 어깨가 안 보이는 근접 촬영 등: 전체 화면에서 찾는다
        face_res = self._face.detect(self._image(face_input))

        # yaw/pitch는 잘라 낸 이미지 기준이다. head_angles는 "머리 정면 방향"과
        # "머리→카메라 방향"의 차이를 재므로, 얼굴이 화면 어디에 있든 카메라를 보면 0 근처가
        # 나온다는 점은 잘라 내도 같다.
        # 잘라 내서 생기는 오차는 다음과 같고, 지금은 보정하지 않는다(마일스톤 6 평가셋에서 기준각과
        # 함께 확인).
        # - 원근 왜곡: 화면 가장자리에서 잘라 낸 조각은 중심에서 찍은 사진과 모양이 조금 다르다
        #   (가장자리로 갈수록 대략 1/cos(벗어난 각도)만큼). 광각 렌즈 가장자리에서 몇 도 수준.
        # - 화각 가정: MediaPipe는 입력을 넓은 화각(세로 약 63도) 카메라 전체 화면으로 보지만
        #   잘라 낸 조각은 거의 평행 투영에 가깝다. 거리(깊이)는 틀리지만 회전 추정에는 영향이 작다.
        # - 확대 잡음: 50~80px 얼굴을 256px로 키우므로 랜드마크가 흔들리고, pitch가 특히 그렇다.
        yaw = pitch = None
        if face_res.face_landmarks and face_res.facial_transformation_matrixes:
            try:
                yaw, pitch = head_angles(face_res.facial_transformation_matrixes[0].tolist())
            except ValueError:
                yaw = pitch = None
        return FrameObs(t=t, yaw=yaw, pitch=pitch, pose=pose_obs, head_turn=turn)


_MEDIA_ERRORS = (MediaError, audio.NoVideoStream, audio.NoAudioStream, audio.FFmpegUnavailable)


def _run_observer(
    face_model_path: Path, pose_model_path: Path, items: Iterable[tuple[float, Any]]
) -> list[FrameObs]:
    """(시각, RGB 또는 None) 목록을 관측값으로. 영상 오류는 그대로, MediaPipe 오류는 바꿔 낸다."""
    observer: _Observer | None = None
    out: list[FrameObs] = []
    failure: str | None = None
    try:
        # 모델 파일이 깨졌으면 여기서 MediaPipe 오류가 난다(아래에서 서버 문제로 바꾼다).
        observer = _Observer(face_model_path, pose_model_path)
        for t, rgb in items:
            if rgb is None:
                out.append(FrameObs(t=t, yaw=None, pitch=None, pose=None))
                continue
            out.append(observer.observe(t, rgb))
    except (*_MEDIA_ERRORS, NonverbalUnavailable):
        raise  # 영상 문제(깨짐·영상 트랙 없음·ffmpeg)는 파이프라인이 알맞은 코드로 바꾼다
    except Exception as e:  # noqa: BLE001 - MediaPipe 경계: 그 밖의 오류는 서버 문제로 분류
        # 모델 파일이 깨졌거나 MediaPipe 내부 오류. 메시지·로그에 경로를 남기지 않는다.
        logger.debug("mediapipe failed", exc_info=True)
        failure = type(e).__name__
    finally:
        if observer is not None:
            observer.close()
    if failure is not None:
        logger.warning("mediapipe failed: %s", failure)
        raise NonverbalUnavailable(f"mediapipe failed: {failure}")
    return out


def observe_video(
    video: Path,
    content_type: str,
    face_model_path: Path,
    pose_model_path: Path,
    *,
    fps: float = audio.MIN_FPS,
    max_seconds: float | None = None,
    expected_duration: float | None = None,
    frames: Callable[..., Iterable[audio.DecodedFrame]] = audio.iter_frames,
) -> list[FrameObs]:
    """영상에서 초당 fps장을 원본 해상도로 받아 관측값을 만든다(이미지는 저장하지 않음)."""
    _check_models(face_model_path, pose_model_path)
    decoded = frames(
        video, content_type, fps=fps, max_seconds=max_seconds, expected_duration=expected_duration
    )
    # MediaPipe가 중간에 실패해도 ffmpeg·타이머를 바로 정리하도록 생성기를 닫는다.
    with contextlib.closing(decoded) if hasattr(decoded, "close") else contextlib.nullcontext():
        return _run_observer(
            face_model_path, pose_model_path, ((f.timestamp, f.rgb) for f in decoded)
        )


def observe_frames(
    frames: Sequence[SampledFrame], face_model_path: Path, pose_model_path: Path
) -> list[FrameObs]:
    """저장된 이미지 파일로 관측값을 만든다(그 이미지를 원본으로 보고 같은 방식으로 자른다)."""
    _check_models(face_model_path, pose_model_path)
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - 설치 문제
        raise NonverbalUnavailable("opencv is not installed") from e

    def load():
        for frame in sorted(frames, key=lambda f: f.timestamp):
            bgr = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning("frame at %.1fs could not be decoded", frame.timestamp)
                yield frame.timestamp, None
                continue
            yield frame.timestamp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return _run_observer(face_model_path, pose_model_path, load())
