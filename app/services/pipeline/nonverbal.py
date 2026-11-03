"""비언어 지표: 시선 정면 비율, 자세 안정성, 제스처 빈도 (MediaPipe, 로컬 실행).

두 단계로 나뉜다.
1. observe_frames: audio.sample_frames가 뽑은 프레임(초당 1~2장)에 MediaPipe Face/Pose Landmarker를
   VIDEO 모드로 돌려 프레임별 관측값(FrameObs)을 만든다. mediapipe는 이 함수 안에서만 불러온다
   (API 프로세스가 무거운 모듈을 싣지 않게).
2. analyze_nonverbal: 관측값만으로 지표·구간을 계산한다(순수 함수, 테스트는 가짜 관측값으로).

규칙:
- 시선(gazeFrontRatio): 얼굴 변환 행렬에서 "머리 정면 방향"과 "머리→카메라 방향" 사이의
  yaw/pitch(도)를 구해 둘 다 기준각 이내인 프레임 비율. 화면 가장자리에 앉아 카메라를 봐도
  정면으로 잡힌다. 얼굴이 잡힌 프레임만 센다.
- 얼굴이 안 잡힌 프레임이 LOW_CONF_MIN_SEC 이상 이어진 구간은 lowConfidenceRanges.
  얼굴이 잡힌 프레임이 전체의 MIN_FACE_FRAME_RATIO 미만이면 NoFaceDetected.
- 자세(postureStability): 어깨 중심·코 위치를 어깨 너비로 나눠 정규화하고,
  POSTURE_WINDOW_SEC 구간마다 직선 추세를 뺀 표준편차(흔들림)를 구한다.
  천천히 자리를 옮기는 것은 흔들림이 아니다. 프레임이 반 이상 찬 구간만 쓴다.
  흔들림 평균이 0이면 1점, POSTURE_SWAY_ZERO 이상이면 0점(선형).
- 제스처(gesturePerMin): 연속 두 프레임 사이 손목이 어깨 너비(영상 전체 중앙값)의
  GESTURE_MOVE_RATIO 이상 움직이면 움직임으로 본다. GESTURE_MERGE_SEC 안에 이어진 움직임은
  한 번으로 세되, 한 번이 GESTURE_MAX_SEC를 넘게 이어지면 새로 센다(계속 움직이면 많이 센다).
  손목이 보인 시간(분)으로 나눈다.
- 얼굴이 안 잡힌 구간(lowConfidenceRanges) 안의 프레임은 자세·제스처 계산에서도 뺀다.
- 측정 불가: 자세가 잡힌 시간이 MIN_POSE_SEC 미만이면 posture_measured=False,
  손목이 보인 시간이 자세 시간의 MIN_WRIST_VISIBLE_RATIO 미만이면 gesture_measured=False
  (노트북 카메라는 보통 손이 안 보인다). 이때 postureStability=1.0, gesturePerMin=0.0으로 두므로
  피드백·점수는 measured 플래그를 봐야 한다.
- 모든 구간의 끝은 영상 길이(duration_sec)를 넘지 않는다.

에러 구분(파이프라인이 분석 실패 코드로 바꾼다):
- NoFaceDetected: 얼굴이 거의 안 보인다 → NO_FACE_DETECTED
- NonverbalUnavailable: 모델 파일이 없거나 MediaPipe를 못 돌렸다(서버 문제) → INTERNAL_ERROR
"""

import logging
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.schemas.report import NonverbalMetrics, TimeRange
from app.services.pipeline.audio import SampledFrame

logger = logging.getLogger(__name__)

# ---- 시선 ----
GAZE_YAW_MAX_DEG = 20.0
GAZE_PITCH_MAX_DEG = 20.0
GAZE_AWAY_MIN_SEC = 3.0  # 이보다 길게 정면을 벗어난 구간만 GAZE_AWAY 구간으로 돌려준다

# ---- 얼굴 신뢰도 ----
LOW_CONF_MIN_SEC = 2.0
MIN_FACE_FRAME_RATIO = 0.2

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


@dataclass(frozen=True)
class FrameObs:
    t: float  # 영상 시작부터의 초
    yaw: float | None  # 얼굴이 안 잡히면 None (도)
    pitch: float | None
    pose: PoseObs | None  # 코·양어깨가 안 보이면 None

    @property
    def has_face(self) -> bool:
        return self.yaw is not None and self.pitch is not None


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


def _inside(t: float, ranges: Sequence[TimeRange]) -> bool:
    return any(r.start <= t < r.end for r in ranges)


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
    if face_ratio < MIN_FACE_FRAME_RATIO:
        raise NoFaceDetected(f"face found in {face_ratio:.0%} of frames")

    dt = _frame_interval(frames)
    start = frames[0].t
    end = frames[-1].t + dt
    if duration_sec is not None:
        if not math.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        end = min(end, duration_sec)
    low_conf = _runs(frames, lambda f: not f.has_face, dt, LOW_CONF_MIN_SEC, end)

    front = sum(1 for f in face_frames if _is_front(f))
    gaze_ratio = round(front / len(face_frames), _DECIMALS)
    gaze_away = _runs(frames, lambda f: f.has_face and not _is_front(f), dt, GAZE_AWAY_MIN_SEC, end)

    pose_frames = [
        f
        for f in frames
        if f.pose is not None
        and f.pose.shoulder_width >= MIN_SHOULDER_WIDTH_RATIO
        and not _inside(f.t, low_conf)
    ]
    pose_sec = len(pose_frames) * dt
    posture_measured = pose_sec >= MIN_POSE_SEC - _EPS
    if posture_measured:
        stability, unstable = _posture(pose_frames, dt, start, end)
    else:
        stability, unstable = 1.0, []

    wrist_frames = [f for f in pose_frames if f.pose is not None and f.pose.any_wrist]
    wrist_sec = len(wrist_frames) * dt
    gesture_measured = posture_measured and wrist_sec >= pose_sec * MIN_WRIST_VISIBLE_RATIO - _EPS
    if gesture_measured:
        width = statistics.median(f.pose.shoulder_width for f in pose_frames if f.pose)
        gestures = _gesture_times(pose_frames, dt, width)
        gesture_per_min = round(len(gestures) / (wrist_sec / 60), _DECIMALS)
        g_none, g_excess = _gesture_windows(wrist_frames, gestures, dt, start, end)
    else:
        gestures, gesture_per_min, g_none, g_excess = [], 0.0, [], []

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


def observe_frames(
    frames: Sequence[SampledFrame], face_model_path: Path, pose_model_path: Path
) -> list[FrameObs]:
    """프레임마다 Face/Pose Landmarker(VIDEO 모드)를 돌려 관측값을 만든다."""
    for p in (face_model_path, pose_model_path):
        if not p.is_file():
            raise NonverbalUnavailable(f"model file not found: {p.name}")
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as e:  # pragma: no cover - 설치 문제
        raise NonverbalUnavailable("mediapipe is not installed") from e

    face_opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=vision.RunningMode.VIDEO,
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

    out: list[FrameObs] = []
    last_ms = -1
    failure: str | None = None
    try:
        with (
            vision.FaceLandmarker.create_from_options(face_opts) as face_lm,
            vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm,
        ):
            for frame in sorted(frames, key=lambda f: f.timestamp):
                # VIDEO 모드는 타임스탬프(ms)가 계속 커져야 한다.
                ts = max(round(frame.timestamp * 1000), last_ms + 1)
                last_ms = ts
                try:
                    image = mp.Image.create_from_file(str(frame.path))
                except (RuntimeError, ValueError, OSError):
                    logger.warning("frame at %.1fs could not be decoded", frame.timestamp)
                    out.append(FrameObs(t=frame.timestamp, yaw=None, pitch=None, pose=None))
                    continue
                face = face_lm.detect_for_video(image, ts)
                pose = pose_lm.detect_for_video(image, ts)

                yaw = pitch = None
                if face.face_landmarks and face.facial_transformation_matrixes:
                    try:
                        yaw, pitch = head_angles(face.facial_transformation_matrixes[0].tolist())
                    except ValueError:
                        yaw = pitch = None
                pose_obs = (
                    _pose_obs(pose.pose_landmarks[0], image.width, image.height)
                    if pose.pose_landmarks
                    else None
                )
                out.append(FrameObs(t=frame.timestamp, yaw=yaw, pitch=pitch, pose=pose_obs))
    except Exception as e:  # noqa: BLE001 - MediaPipe 경계: 어떤 오류든 서버 문제로 분류
        # 모델 파일이 깨졌거나 MediaPipe 내부 오류. 메시지·로그에 경로를 남기지 않는다.
        logger.debug("mediapipe failed", exc_info=True)
        failure = type(e).__name__
    if failure is not None:
        logger.warning("mediapipe failed: %s", failure)
        raise NonverbalUnavailable(f"mediapipe failed: {failure}")
    return out
