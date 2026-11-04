"""nonverbal: 계산 로직은 가짜 랜드마크로, MediaPipe 실행은 짧은 이미지 몇 장으로만 확인한다.

실제 MediaPipe 테스트는 모델 파일이 있어야 돈다(없으면 건너뜀).
MEDIAPIPE_FACE_MODEL_PATH / MEDIAPIPE_POSE_MODEL_PATH 환경 변수로 경로를 준다.
얼굴 사진 픽스처(tests/fixtures/faces/*.jpg)가 있으면 얼굴 인식까지 확인한다.
"""

import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.pipeline import nonverbal as nv
from app.services.pipeline.audio import SampledFrame
from app.services.pipeline.nonverbal import (
    FrameObs,
    HeadTurn,
    NoFaceDetected,
    NonverbalUnavailable,
    PoseObs,
    analyze_nonverbal,
    head_angles,
    observe_frames,
)

# conftest가 테스트마다 설정 환경 변수를 지우므로 모듈을 불러올 때 읽어 둔다.
FACE_MODEL = os.environ.get("MEDIAPIPE_FACE_MODEL_PATH", "/opt/models/face_landmarker.task")
POSE_MODEL = os.environ.get("MEDIAPIPE_POSE_MODEL_PATH", "/opt/models/pose_landmarker_full.task")
FACES_DIR = Path(__file__).parent / "fixtures" / "faces"

W = 0.2  # 기본 어깨 너비(프레임 너비 대비)


def pose(
    dx: float = 0.0,
    dy: float = 0.0,
    scale: float = 1.0,
    lw: tuple[float, float] | None = (0.0, 0.0),
    rw: tuple[float, float] | None = (0.0, 0.0),
) -> PoseObs:
    """어깨 너비 W*scale인 자세. dx, dy는 어깨 너비 대비 몸 이동, lw/rw는 손목 이동."""
    s = W * scale
    cx, cy = 0.5 + dx * s, 0.45 + dy * s

    def wrist(base_x: float, off):
        return None if off is None else (base_x + off[0] * s, cy + s + off[1] * s)

    return PoseObs(
        nose=(cx, cy - 0.8 * s),
        left_shoulder=(cx + s / 2, cy),
        right_shoulder=(cx - s / 2, cy),
        left_wrist=wrist(cx + 0.6 * s, lw),
        right_wrist=wrist(cx - 0.6 * s, rw),
    )


def frame(t: float, yaw: float | None = 0.0, pitch: float | None = 0.0, p: PoseObs | None = None):
    if p is None and yaw is not None:
        p = pose()
    return FrameObs(t=t, yaw=yaw, pitch=pitch, pose=p)


def still(n: int, fps: float = 1.0, start: float = 0.0) -> list[FrameObs]:
    return [frame(start + i / fps) for i in range(n)]


def ranges(rs) -> list[tuple[float, float]]:
    return [(r.start, r.end) for r in rs]


# ---------- 머리 각도 ----------


def _rot(
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    scale: float = 1.0,
    t: tuple[float, float, float] = (0.0, 0.0, -50.0),
) -> list[list[float]]:
    y, x = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
    rx = [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]
    r = [[sum(ry[i][k] * rx[k][j] for k in range(3)) * scale for j in range(3)] for i in range(3)]
    return [[*r[0], t[0]], [*r[1], t[1]], [*r[2], t[2]], [0, 0, 0, 1]]


@pytest.mark.parametrize(
    ("yaw", "pitch"), [(0, 0), (30, 0), (-25, 0), (0, 15), (0, -40), (35, -10)]
)
def test_head_angles_recovers_yaw_and_pitch(yaw, pitch):
    y, p = head_angles(_rot(yaw, pitch))
    assert y == pytest.approx(yaw, abs=1e-6)
    assert p == pytest.approx(pitch, abs=1e-6)


def test_head_angles_ignores_scale():
    assert head_angles(_rot(20, 10, scale=3.7)) == pytest.approx((20, 10))


def test_off_center_face_looking_at_camera_is_front():
    # 카메라 오른쪽 30, 앞 65에 있는 얼굴이 카메라를 똑바로 본다
    t = (30.0, 0.0, -65.0)
    toward_camera = math.degrees(math.atan2(-30.0, 65.0))
    assert head_angles(_rot(toward_camera, 0, t=t)) == pytest.approx((0.0, 0.0), abs=1e-6)
    # 같은 자리에서 카메라 축과 평행하게(정면 벽) 보면 카메라 기준 약 25도 돌아간 것
    yaw, _ = head_angles(_rot(0, 0, t=t))
    assert yaw == pytest.approx(-toward_camera, abs=1e-6)
    assert abs(yaw) > nv.GAZE_YAW_MAX_DEG


def test_face_above_camera_pitch_is_relative():
    # 얼굴이 카메라보다 20 위(y 위쪽)에 있고 카메라를 내려다본다 → 카메라 기준 pitch 0
    t = (0.0, 20.0, -60.0)
    down = math.degrees(math.atan2(20.0, 60.0))
    assert head_angles(_rot(0, down, t=t))[1] == pytest.approx(0.0, abs=1e-6)
    # 같은 자리에서 고개를 들어 앞(카메라 축과 평행)을 보면 기준각을 벗어날 수 있다
    assert abs(head_angles(_rot(0, 0, t=t))[1]) == pytest.approx(down, abs=1e-6)


def test_head_angles_degenerate_matrix():
    with pytest.raises(ValueError):
        head_angles([[0, 0, 0, 0]] * 4)


# ---------- 시선 ----------


def test_gaze_front_ratio():
    frames = still(7) + [frame(7 + i, yaw=40) for i in range(3)]
    r = analyze_nonverbal(frames)
    assert r.metrics.gaze_front_ratio == 0.7


@pytest.mark.parametrize(
    ("yaw", "pitch", "front"),
    [(20.0, 0.0, True), (20.01, 0.0, False), (-20.0, 0.0, True), (0.0, 20.0, True),
     (0.0, -20.01, False), (19.0, 19.0, True), (21.0, 5.0, False)],
)  # fmt: skip
def test_gaze_threshold_inclusive(yaw, pitch, front):
    assert nv.GAZE_YAW_MAX_DEG == 20.0 and nv.GAZE_PITCH_MAX_DEG == 20.0
    frames = still(1) + [frame(1, yaw=yaw, pitch=pitch)]
    assert analyze_nonverbal(frames).metrics.gaze_front_ratio == (1.0 if front else 0.5)


def test_gaze_away_ranges_only_long_runs():
    frames = (
        still(5)
        + [frame(5 + i, yaw=45) for i in range(4)]  # 5~9초 4초 → 구간
        + still(3, start=9)
        + [frame(12 + i, pitch=-35) for i in range(2)]  # 12~14초 2초 → 짧아서 뺌
        + still(6, start=14)
    )
    r = analyze_nonverbal(frames)
    assert ranges(r.gaze_away_ranges) == [(5.0, 9.0)]


def test_gaze_away_run_at_end():
    frames = still(5) + [frame(5 + i, yaw=-50) for i in range(3)]
    assert ranges(analyze_nonverbal(frames).gaze_away_ranges) == [(5.0, 8.0)]


# ---------- 얼굴 없이 자세만 잡힌 프레임 ----------


def turned(t: float, turn: HeadTurn | None) -> FrameObs:
    """얼굴은 못 찾고 자세만 잡힌 프레임(Pose로 추정한 고개 방향)."""
    return FrameObs(t=t, yaw=None, pitch=None, pose=pose(), head_turn=turn)


def test_face_missing_but_clearly_turned_counts_as_away():
    # 정면 6장 + 얼굴 없이 Pose로 확실히 돌린 4장 → 분모 10, 정면 6
    frames = still(6) + [turned(6 + i, HeadTurn.AWAY) for i in range(4)]
    r = analyze_nonverbal(frames)
    assert r.metrics.gaze_front_ratio == 0.6
    assert ranges(r.gaze_away_ranges) == [(6.0, 10.0)]
    assert r.metrics.low_confidence_ranges == []


def test_face_missing_with_frontal_pose_counts_as_front():
    # 얼굴 각도로 벗어남 4장 + 얼굴 없이 Pose로는 정면 6장 → 분모 10, 정면 6
    frames = [frame(i, yaw=40) for i in range(4)] + [
        turned(4 + i, HeadTurn.FRONT) for i in range(6)
    ]
    r = analyze_nonverbal(frames)
    assert r.metrics.gaze_front_ratio == 0.6
    assert ranges(r.gaze_away_ranges) == [(0.0, 4.0)]


def test_pose_only_mixed_front_and_away_is_not_biased():
    # 얼굴을 한 번도 못 찾은 먼 화자: Pose 정면 7장 + 돌림 3장 → 0.7(정면을 빼면 0.0이 됨)
    frames = [turned(i, HeadTurn.FRONT) for i in range(7)]
    frames += [turned(7 + i, HeadTurn.AWAY) for i in range(3)]
    assert analyze_nonverbal(frames).metrics.gaze_front_ratio == 0.7


def test_face_missing_with_ambiguous_pose_is_unmeasured():
    frames = still(6) + [turned(6 + i, None) for i in range(4)]
    r = analyze_nonverbal(frames)
    assert r.metrics.gaze_front_ratio == 1.0
    assert r.gaze_away_ranges == []
    assert r.metrics.low_confidence_ranges == []  # 자세는 잡혔으므로 저신뢰 구간은 아님


def test_frames_without_face_and_pose_are_excluded():
    # 정면 6장 + 얼굴·자세 둘 다 없는 4장 → 분모 6, 저신뢰 구간
    frames = still(6) + [frame(6 + i, yaw=None, pitch=None) for i in range(4)]
    r = analyze_nonverbal(frames)
    assert r.metrics.gaze_front_ratio == 1.0
    assert r.gaze_away_ranges == []
    assert ranges(r.metrics.low_confidence_ranges) == [(6.0, 10.0)]


def test_gaze_away_run_mixes_turned_face_and_pose_only():
    # 각도로 벗어남 2초 + 얼굴 없이 Pose로 돌림 2초가 이어지면 하나의 4초 구간
    frames = still(5) + [frame(5 + i, yaw=40) for i in range(2)]
    frames += [turned(7 + i, HeadTurn.AWAY) for i in range(2)]
    frames += still(5, start=9)
    assert ranges(analyze_nonverbal(frames).gaze_away_ranges) == [(5.0, 9.0)]


def test_only_frontal_pose_frames_are_not_no_face():
    # 얼굴은 한 번도 못 찾았지만 Pose로 고개 방향은 다 알았다 → NO_FACE 아님
    r = analyze_nonverbal([turned(i, HeadTurn.FRONT) for i in range(20)])
    assert r.metrics.gaze_front_ratio == 1.0
    assert r.face_frame_ratio == 0.0


def test_turned_around_without_pose_obs_is_not_low_confidence():
    # 뒤돌아 코가 안 보여 pose는 None이지만 Pose로 AWAY → 시선 벗어남, 저신뢰 구간 아님
    back = [
        FrameObs(t=3 + i, yaw=None, pitch=None, pose=None, head_turn=HeadTurn.AWAY)
        for i in range(4)
    ]
    r = analyze_nonverbal(still(3) + back + still(3, start=7))
    assert r.metrics.low_confidence_ranges == []
    assert ranges(r.gaze_away_ranges) == [(3.0, 7.0)]


def test_no_face_when_head_unknown_in_80_percent():
    assert nv.MAX_UNKNOWN_HEAD_RATIO == 0.8
    # 얼굴 없음·Pose 애매 8장 + 정면 2장 → 80% → NO_FACE_DETECTED
    frames = still(2) + [turned(2 + i, None) for i in range(8)]
    with pytest.raises(NoFaceDetected):
        analyze_nonverbal(frames)
    # 7장이면 70% → 통과
    frames = still(3) + [turned(3 + i, None) for i in range(7)]
    assert analyze_nonverbal(frames).face_frame_ratio == 0.3


# ---------- 얼굴 신뢰도 ----------


def test_low_confidence_ranges_and_excluded_from_ratio():
    frames = (
        still(3)
        + [frame(3 + i, yaw=None, pitch=None) for i in range(3)]  # 3~6초 얼굴 없음 → 구간
        + still(2, start=6)
        + [frame(8, yaw=None, pitch=None)]  # 1초 → 짧아서 구간 아님
        + [frame(9, yaw=60)]
    )
    r = analyze_nonverbal(frames)
    assert ranges(r.metrics.low_confidence_ranges) == [(3.0, 6.0)]
    # 얼굴 잡힌 6장 중 정면 5장
    assert r.metrics.gaze_front_ratio == round(5 / 6, 2)
    assert r.face_frame_ratio == 0.6


def test_low_confidence_threshold_inclusive():
    assert nv.LOW_CONF_MIN_SEC == 2.0
    frames = still(2) + [frame(2 + i, yaw=None, pitch=None) for i in range(2)] + still(2, start=4)
    assert ranges(analyze_nonverbal(frames).metrics.low_confidence_ranges) == [(2.0, 4.0)]


@pytest.mark.parametrize(("faces", "ok"), [(2, False), (3, True)])
def test_no_face_detected_when_80_percent_unknown(faces, ok):
    # 얼굴·자세가 둘 다 없는 프레임도 "고개 방향을 모르는" 프레임이다
    frames = still(faces) + [frame(faces + i, yaw=None, pitch=None) for i in range(10 - faces)]
    if ok:
        assert analyze_nonverbal(frames).face_frame_ratio == 0.3
    else:
        with pytest.raises(NoFaceDetected):
            analyze_nonverbal(frames)


def test_no_frames_is_no_face():
    with pytest.raises(NoFaceDetected):
        analyze_nonverbal([])


def test_unsorted_frames_are_sorted():
    frames = still(10) + [frame(10 + i, yaw=45) for i in range(4)]
    frames.reverse()
    assert ranges(analyze_nonverbal(frames).gaze_away_ranges) == [(10.0, 14.0)]


# ---------- 자세 ----------


def test_still_posture_is_one():
    r = analyze_nonverbal(still(30))
    assert r.posture_measured
    assert r.metrics.posture_stability == 1.0
    assert r.posture_unstable_ranges == []


def test_swaying_lowers_score_and_marks_range():
    # 0~10초 가만히, 10~20초 좌우로 어깨 너비의 ±20% 흔들림, 20~30초 가만히
    frames = still(10)
    frames += [frame(10 + i, p=pose(dx=0.2 if i % 2 else -0.2)) for i in range(10)]
    frames += still(10, start=20)
    r = analyze_nonverbal(frames)
    # 흔들린 구간 흔들림 0.2 → 평균 0.2/3 → 1 - 0.0667/0.25 ≈ 0.73
    assert r.metrics.posture_stability == pytest.approx(1 - (0.2 / 3) / 0.25, abs=0.01)
    assert ranges(r.posture_unstable_ranges) == [(10.0, 20.0)]


def test_posture_is_scale_invariant():
    def sway(scale):
        fs = [frame(i, p=pose(dx=0.1 if i % 2 else -0.1, scale=scale)) for i in range(20)]
        return analyze_nonverbal(fs).metrics.posture_stability

    assert sway(1.0) == sway(2.5)


def test_heavy_sway_clamps_to_zero():
    frames = [frame(i, p=pose(dx=0.5 if i % 2 else -0.5)) for i in range(20)]
    assert analyze_nonverbal(frames).metrics.posture_stability == 0.0


def test_slow_drift_is_not_sway():
    # 초당 어깨 너비 10%씩 한 방향으로 걸어감 → 직선 추세를 빼므로 흔들림 아님
    frames = [frame(i, p=pose(dx=0.1 * i)) for i in range(30)]
    r = analyze_nonverbal(frames)
    assert r.metrics.posture_stability == 1.0
    assert r.posture_unstable_ranges == []


def test_short_tail_window_is_ignored():
    # 20초 가만히 + 끝 2프레임 흔들림 → 프레임이 반도 안 찬 마지막 구간은 쓰지 않는다
    frames = still(20) + [frame(20, p=pose(dx=0.4)), frame(21, p=pose(dx=-0.4))]
    r = analyze_nonverbal(frames)
    assert r.metrics.posture_stability == 1.0
    assert r.posture_unstable_ranges == []
    assert r.posture_unstable_ranges == []


def test_posture_not_measured_without_pose():
    frames = [FrameObs(t=i, yaw=0.0, pitch=0.0, pose=None) for i in range(30)]
    r = analyze_nonverbal(frames)
    assert not r.posture_measured and not r.gesture_measured
    assert r.metrics.posture_stability is None  # 측정 불가 → null
    assert r.metrics.gesture_per_min is None
    assert r.metrics.gaze_front_ratio == 1.0
    assert r.gesture_none_ranges == [] and r.gesture_excess_ranges == []
    dumped = r.metrics.model_dump(by_alias=True)
    assert dumped["postureStability"] is None and dumped["gesturePerMin"] is None


def test_posture_measured_threshold():
    assert nv.MIN_POSE_SEC == 10.0
    assert not analyze_nonverbal(still(9)).posture_measured
    assert analyze_nonverbal(still(10)).posture_measured


def test_pose_only_frames_are_used_for_posture():
    # 얼굴은 없고 자세만 잡힌(고개를 돌린) 프레임도 자세 계산에 들어간다
    frames = still(10)
    frames += [
        frame(10 + i, yaw=None, pitch=None, p=pose(dx=0.2 if i % 2 else -0.2)) for i in range(10)
    ]
    frames += still(10, start=20)
    r = analyze_nonverbal(frames)
    assert r.metrics.posture_stability < 1.0
    assert ranges(r.posture_unstable_ranges) == [(10.0, 20.0)]
    assert r.metrics.low_confidence_ranges == []


def test_tiny_shoulder_width_frames_are_skipped():
    frames = still(15) + [frame(15 + i, p=pose(scale=0.05, dx=5)) for i in range(5)]
    assert analyze_nonverbal(frames).metrics.posture_stability == 1.0


# ---------- 제스처 ----------


def _moves(offsets, fps: float = 1.0) -> list[FrameObs]:
    """offsets[i]: i번째 프레임의 왼손목 위치(어깨 너비 대비)."""
    return [frame(i / fps, p=pose(lw=(o, 0.0))) for i, o in enumerate(offsets)]


def _excursions(
    peaks: list[float], hold: float = 1.5, rest: float = 4.0, fps: float = 2.0, lead: float = 10.0
) -> list[float]:
    """lead초 쉼 → (0.5초 올림, hold초 유지, 0.5초 내림, rest초 쉼) × 동작 → lead초 쉼."""
    out = [0.0] * int(lead * fps)
    for peak in peaks:
        out += [peak / 2, peak]  # 0.5초에 걸쳐 올림(2fps)
        out += [peak] * int(hold * fps)
        out += [peak / 2]
        out += [0.0] * int(rest * fps)
    return out + [0.0] * int(lead * fps)  # 끝에도 쉬는 구간


@pytest.mark.parametrize(("peak", "counted"), [(0.3, 1), (0.2, 0)])
def test_gesture_start_amplitude(peak, counted):
    # 평소 위치에서 어깨 너비의 0.25 이상 벗어나야 제스처
    assert nv.GESTURE_START_AWAY == 0.25
    frames = _moves(_excursions([peak]), fps=2.0)
    assert len(analyze_nonverbal(frames).gesture_times) == counted


def test_hysteresis_dip_above_end_is_one_gesture():
    # 0.4까지 올렸다 0.2(끝 기준 0.15보다 큼)로 잠깐 내렸다 다시 0.4 → 한 번
    offsets = [0.0] * 20 + [0.2, 0.4, 0.4, 0.4, 0.2, 0.2, 0.4, 0.4, 0.4, 0.2] + [0.0] * 20
    assert len(analyze_nonverbal(_moves(offsets, fps=2.0)).gesture_times) == 1


def test_turnaround_without_stopping_is_one_gesture():
    # 손을 올렸다가 멈추지 않고 방향만 바꿔 내리면(반환점) 한 번
    offsets = [0.0] * 20 + [0.3, 0.6, 0.9, 0.6, 0.3] + [0.0] * 20
    assert len(analyze_nonverbal(_moves(offsets, fps=2.0)).gesture_times) == 1


@pytest.mark.parametrize("rest", [3.0, 1.0])
def test_rest_between_movements_separates_gestures(rest):
    # 1초만 멈췄다 다시 들어도 별개의 동작
    frames = _moves(_excursions([0.5, 0.5], rest=rest), fps=2.0)
    assert len(analyze_nonverbal(frames).gesture_times) == 2


@pytest.mark.parametrize("hold", [3.0, 15.0])
@pytest.mark.parametrize("which", ["both", "left"])
def test_one_frame_dropout_keeps_gesture(hold, which):
    # 손을 든 동안(또는 내리기 직전) 어느 한 프레임 손목을 놓쳐도 같은 동작.
    # 판정은 1초 전 프레임까지 있어야 해서 한 프레임만 놓쳐도 1~2초를 건너뛰므로 위치마다 확인한다.
    offsets = _excursions([0.6], hold=hold)
    up = [k for k, o in enumerate(offsets) if o == 0.6]
    for k in range(up[0], up[-1] + 2):
        frames = _moves(offsets, fps=2.0)
        lw = None
        rw = (0.0, 0.0) if which == "left" else None  # 오른손은 제자리에 보임
        frames[k] = frame(k / 2, p=pose(lw=lw, rw=rw))
        assert len(analyze_nonverbal(frames).gesture_times) == 1, k


@pytest.mark.parametrize(("period", "fps"), [(8.0, 2.0), (4.0, 2.0), (2.5, 2.0), (4.0, 1.0)])
def test_frequent_gestures_are_counted(period, fps):
    # period초마다 1초 올리고 1초 내리는 손동작(앞 10초는 쉼) → 거의 다 센다
    # (잦은 손동작을 놓치면 GESTURE_NONE으로 잘못 판단한다)
    def lift(t):
        ph = (t - 10) % period
        return 0.0 if t < 10 else 0.6 * (ph if ph < 1 else 2 - ph if ph < 2 else 0)

    n = int(120 * fps)
    r = analyze_nonverbal(_moves([lift(i / fps) for i in range(n)], fps=fps))
    truth = math.ceil(110 / period)
    assert abs(len(r.gesture_times) - truth) <= max(1, 0.1 * truth), len(r.gesture_times)
    assert r.gesture_none_ranges == [] or r.gesture_none_ranges[0].start == 0.0


def test_resting_jitter_is_not_a_gesture():
    # 가만히 있는 손목이 프레임마다 ±0.01(프레임 너비) 떨림 → 제스처 아님
    import random

    rng = random.Random(1)
    frames = [
        frame(i / 2, p=pose(lw=(rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))))
        for i in range(120)
    ]
    assert analyze_nonverbal(frames).gesture_times == []


@pytest.mark.parametrize("hold", [8.0, 15.0, 25.0, 50.0])
def test_holding_hand_up_is_one_gesture(hold):
    # 손을 들고 오래 가만히 있다가(슬라이드 가리키기) 내려도 한 번. 평소 위치가 들린 손 쪽으로
    # 옮겨 가도 제스처 시작 때의 기준점으로 재므로 내리는 것을 새 동작으로 세지 않는다.
    frames = _moves(_excursions([0.5], hold=hold), fps=2.0)
    assert len(analyze_nonverbal(frames).gesture_times) == 1


@pytest.mark.parametrize("second", [0.5, -0.5, 0.3])
def test_new_gesture_soon_after_long_hold_is_counted(second):
    # 15초 들고 있다가 내린 뒤 2초 만에 다시 든 손은 새 동작
    frames = _moves(_excursions([0.5, second], hold=15.0, rest=2.0), fps=2.0)
    assert len(analyze_nonverbal(frames).gesture_times) == 2


def test_hand_moved_to_new_rest_position_for_long_time():
    # 손을 옮겨 GESTURE_ANCHOR_MAX_SEC보다 오래 두면 그곳이 새 평소 위치 → 거기서 옛 평소 위치
    # 쪽으로(하지만 그 가까이까지는 아니게) 움직인 손도 움직인 순간 센다
    assert nv.GESTURE_ANCHOR_MAX_SEC == 60.0
    offsets = [0.0] * 20 + [0.25, 0.5] + [0.5] * 140 + [0.35, 0.2, 0.2, 0.35] + [0.5] * 20
    times = analyze_nonverbal(_moves(offsets, fps=2.0)).gesture_times
    assert times == [10.0, 81.5]


def test_waving_from_the_first_frame_is_counted():
    # 쉬는 프레임 없이 처음부터 계속 손을 움직여도(평소 위치 = 가장 느린 프레임들) 센다
    wave = [0.0, 0.6, 1.2, 0.6] * 30  # 60초
    r = analyze_nonverbal(_moves(wave, fps=2.0))
    assert len(r.gesture_times) >= 20
    assert r.gesture_none_ranges == []


def test_moving_hand_right_after_long_gap_is_not_still():
    # 자세를 오래 놓친 뒤 첫 프레임은 긴 간격으로 나눈 속도로 '멈춤'이 되면 안 된다
    frames = still(10) + [FrameObs(t=10 + i, yaw=0.0, pitch=0.0, pose=None) for i in range(30)]
    offsets = [0.0, 0.6, 1.2, 0.6] * 10
    frames += [frame(40 + i / 2, p=pose(lw=(o, 0.0))) for i, o in enumerate(offsets)]
    r = analyze_nonverbal(frames)
    assert len([g for g in r.gesture_times if g >= 40]) >= 5


def test_body_sway_is_not_a_gesture():
    # 몸 전체가 좌우로 흔들리면 손목도 같이 움직이지만, 어깨 기준으로는 제자리 → 제스처 아님
    frames = [frame(i / 2, p=pose(dx=0.4 * math.sin(i / 2))) for i in range(120)]
    assert analyze_nonverbal(frames).gesture_times == []


def test_long_continuous_waving_counts_repeatedly():
    # 30초 동안 크게 계속 휘저으면 GESTURE_MAX_SEC(3초)마다 다시 센다
    assert nv.GESTURE_MAX_SEC == 3.0
    wave = [0.0, 0.6, 1.2, 0.6] * 15  # 2초 주기, 2fps, 30초
    frames = _moves([0.0] * 20 + wave + [0.0] * 20, fps=2.0)
    # 적어도 3초마다 한 번(10), 많아도 왕복 한 번에 한 번(15). 2fps에서 1초 변위가 왕복 주기와
    # 맞물리면 반환점마다 멈춘 것처럼 보여 왕복마다 센다.
    r = analyze_nonverbal(frames)
    assert 10 <= len(r.gesture_times) <= 15
    assert r.gesture_excess_ranges  # 분당 20회 초과


def test_separate_movements_are_counted_separately():
    frames = _moves(_excursions([0.5, 0.6, 0.5]), fps=2.0)
    assert len(analyze_nonverbal(frames).gesture_times) == 3


def test_separate_movements_one_fps():
    frames = _moves(_excursions([0.5, 0.5], hold=2.0, fps=1.0), fps=1.0)
    assert len(analyze_nonverbal(frames).gesture_times) == 2


def test_either_wrist_counts_once():
    offsets = _excursions([0.5])
    frames = [frame(i / 2, p=pose(lw=(o, 0.0), rw=(-o, 0.0))) for i, o in enumerate(offsets)]
    assert len(analyze_nonverbal(frames).gesture_times) == 1


def test_invisible_wrist_is_not_a_gesture():
    frames = still(10) + [frame(10 + i, p=pose(lw=None)) for i in range(10)]
    frames[12] = frame(12, p=pose(lw=(1.0, 1.0)))  # 앞뒤가 None: 한 프레임 오검출
    assert analyze_nonverbal(frames).gesture_times == []


def test_gap_in_pose_frames_is_not_a_gesture():
    # 10초 뒤 자세를 놓쳤다가 13초에 손이 다른 곳 → 사이 움직임은 모른다
    frames = still(10) + [FrameObs(t=10 + i, yaw=0.0, pitch=0.0, pose=None) for i in range(3)]
    frames += [frame(13 + i, p=pose(lw=(1.0, 0.0))) for i in range(10)]
    assert analyze_nonverbal(frames).gesture_times == []


def test_gesture_per_min():
    # 60초 동안 6번 → 분당 6
    offsets = _excursions([0.5] * 6, hold=1.5, rest=6.0, lead=2.0)  # 1번당 8.5초
    offsets = offsets[:120] + [0.0] * max(0, 120 - len(offsets))
    r = analyze_nonverbal(_moves(offsets, fps=2.0))
    assert len(r.gesture_times) == 6
    assert r.metrics.gesture_per_min == 6.0


def test_gesture_none_and_excess_windows():
    assert nv.GESTURE_WINDOW_SEC == 30.0 and nv.GESTURE_EXCESS_PER_MIN == 20.0
    offsets = [0.0] * 60  # 0~30초: 제스처 없음
    # 30~60초: 2.5초마다 한 번(분당 24) → 과다
    offsets += _excursions([0.6] * 12, hold=0.5, rest=1.0, lead=0.0)[:60]
    offsets += [0.0] * (180 - len(offsets))  # 60~90초: 없음
    r = analyze_nonverbal(_moves(offsets, fps=2.0))
    assert ranges(r.gesture_excess_ranges) == [(30.0, 60.0)]
    assert ranges(r.gesture_none_ranges) == [(0.0, 30.0), (60.0, 90.0)]


def test_wrists_not_visible_means_gesture_not_measured():
    # 노트북 카메라: 머리·어깨만 보인다 → GESTURE_NONE을 만들면 안 된다
    frames = [frame(i, p=pose(lw=None, rw=None)) for i in range(60)]
    r = analyze_nonverbal(frames)
    assert r.posture_measured and not r.gesture_measured
    assert r.metrics.gesture_per_min is None
    assert r.metrics.posture_stability == 1.0
    assert r.gesture_none_ranges == [] and r.gesture_times == []


def test_wrist_visible_ratio_threshold():
    assert nv.MIN_WRIST_VISIBLE_RATIO == 0.3

    def run(visible):
        frames = still(visible) + [
            frame(visible + i, p=pose(lw=None, rw=None)) for i in range(100 - visible)
        ]
        return analyze_nonverbal(frames).gesture_measured

    assert run(30) and not run(29)


def test_gesture_rate_uses_wrist_visible_time():
    # 60초 중 손목은 앞 30초만 보이고 그 안에 3번 → 분당 6
    offsets = ([0.0] * 9 + [0.5]) * 3
    frames = _moves(offsets) + [frame(30 + i, p=pose(lw=None, rw=None)) for i in range(30)]
    r = analyze_nonverbal(frames)
    assert r.metrics.gesture_per_min == 6.0


def test_turning_sideways_jitter_is_not_gesture():
    # 앞 30초 정면(어깨 너비 W), 뒤 30초 옆으로 돎(너비 0.35W)에서 손목이 0.1W씩 떨린다.
    # 쌍마다의 어깨 너비로 나누면 0.29 → 제스처로 잘못 세지만, 전체 중앙값(W)으로 나누면 0.1
    frames = still(30)
    for i in range(30):
        p = pose()
        narrow = PoseObs(
            nose=p.nose,
            left_shoulder=(0.5 + 0.35 * W / 2, 0.45),
            right_shoulder=(0.5 - 0.35 * W / 2, 0.45),
            left_wrist=(0.6 + (0.1 * W if i % 2 else 0.0), 0.65),
            right_wrist=None,
        )
        frames.append(frame(30 + i, p=narrow))
    assert analyze_nonverbal(frames).gesture_times == []


def test_gesture_window_with_little_pose_is_skipped():
    # 30~60초에 손목이 10초만 보임(15초 미만) → 판단하지 않는다
    frames = still(30) + [FrameObs(t=30 + i, yaw=0.0, pitch=0.0, pose=None) for i in range(20)]
    frames += still(10, start=50)
    r = analyze_nonverbal(frames)
    assert ranges(r.gesture_none_ranges) == [(0.0, 30.0)]


def test_adjacent_none_windows_merge():
    r = analyze_nonverbal(still(90))
    assert ranges(r.gesture_none_ranges) == [(0.0, 90.0)]


# ---------- 영상 길이·중복 프레임 ----------


def test_ranges_are_clamped_to_duration():
    frames = still(5) + [frame(5 + i, yaw=45) for i in range(4)]  # 마지막 프레임 8초 → 끝 9초
    r = analyze_nonverbal(frames, duration_sec=8.4)
    assert ranges(r.gaze_away_ranges) == [(5.0, 8.4)]
    r = analyze_nonverbal(still(50), duration_sec=49.5)
    assert ranges(r.gesture_none_ranges) == [(0.0, 49.5)]


def test_low_confidence_range_clamped_to_duration():
    frames = still(5) + [frame(5 + i, yaw=None, pitch=None) for i in range(3)]
    assert ranges(analyze_nonverbal(frames, 7.5).metrics.low_confidence_ranges) == [(5.0, 7.5)]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_invalid_duration(bad):
    with pytest.raises(ValueError):
        analyze_nonverbal(still(5), duration_sec=bad)


def test_duplicate_timestamps_are_counted_once():
    frames = still(10) + still(10)  # 같은 시각이 두 번씩
    r = analyze_nonverbal(frames)
    assert r.frame_count == 10
    assert r.posture_measured  # 10초(중복 빼고)


# ---------- 랜드마크 → 관측값 ----------


def _landmarks(**over):
    from types import SimpleNamespace

    lms = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
    for i, (x, y, v) in over.items():
        lms[int(i[1:])] = SimpleNamespace(x=x, y=y, visibility=v)
    return lms


def test_pose_obs_uses_frame_width_units():
    lms = _landmarks(i0=(0.5, 0.2, 0.9), i11=(0.6, 0.5, 0.9), i12=(0.4, 0.5, 0.9))
    p = nv._pose_obs(lms, 640, 360)
    assert p is not None
    assert p.nose == pytest.approx((0.5, 0.2 * 360 / 640))
    assert p.shoulder_width == pytest.approx(0.2)


def test_pose_obs_rejects_offscreen_or_invisible_points():
    lms = _landmarks(i15=(1.2, 0.9, 0.95), i16=(0.3, 0.9, 0.2))
    p = nv._pose_obs(lms, 640, 480)
    assert p is not None and p.left_wrist is None and p.right_wrist is None
    assert nv._pose_obs(_landmarks(i11=(0.6, -0.1, 0.9)), 640, 480) is None
    assert nv._pose_obs(_landmarks(i0=(0.5, 0.2, 0.3)), 640, 480) is None


# ---------- 명세 형태 ----------


def test_metrics_match_spec_shape():
    r = analyze_nonverbal(still(5) + [frame(5 + i, yaw=None, pitch=None) for i in range(3)])
    dumped = r.metrics.model_dump(by_alias=True)
    assert set(dumped) == {
        "gazeFrontRatio",
        "postureStability",
        "gesturePerMin",
        "lowConfidenceRanges",
    }
    assert dumped["lowConfidenceRanges"] == [{"start": 5.0, "end": 8.0}]
    assert 0.0 <= dumped["gazeFrontRatio"] <= 1.0
    assert dumped["postureStability"] is None  # 자세 5초 < 10초 → 측정 불가
    r = analyze_nonverbal(still(30))
    for key, value in r.metrics.model_dump(by_alias=True).items():
        if key != "lowConfidenceRanges":
            assert value is not None


# ---------- MediaPipe 실행 ----------


def test_observe_frames_missing_model_is_unavailable(tmp_path):
    with pytest.raises(NonverbalUnavailable) as exc_info:
        observe_frames([], tmp_path / "nope.task", tmp_path / "nope2.task")
    assert str(tmp_path) not in str(exc_info.value)


_have_models = Path(FACE_MODEL).is_file() and Path(POSE_MODEL).is_file()
needs_models = pytest.mark.skipif(not _have_models, reason="MediaPipe 모델 파일 없음")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 없음")


def _blank_frames(tmp_path: Path, n: int) -> list[SampledFrame]:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    out = []
    for i in range(n):
        p = tmp_path / f"{i:03d}.jpg"
        subprocess.run(  # noqa: S603 - 고정 인자
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=gray:s=320x240:d=1",
                "-frames:v",
                "1",
                str(p),
            ],  # fmt: skip
            check=True,
        )
        out.append(SampledFrame(path=p, timestamp=i / 2))
    return out


@needs_models
@needs_ffmpeg
def test_mediapipe_on_blank_frames_finds_no_face(tmp_path):
    frames = _blank_frames(tmp_path, 3)
    # 같은 시각이 겹쳐도 VIDEO 모드 타임스탬프 오류가 나지 않아야 한다
    frames.append(SampledFrame(path=frames[-1].path, timestamp=frames[-1].timestamp))
    obs = observe_frames(frames, Path(FACE_MODEL), Path(POSE_MODEL))
    assert len(obs) == 4
    assert all(not o.has_face and o.pose is None for o in obs)
    with pytest.raises(NoFaceDetected):
        analyze_nonverbal(obs)


@needs_models
def test_mediapipe_undecodable_frame_is_skipped(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    obs = observe_frames(
        [SampledFrame(path=bad, timestamp=0.0)], Path(FACE_MODEL), Path(POSE_MODEL)
    )
    assert obs == [FrameObs(t=0.0, yaw=None, pitch=None, pose=None)]


@needs_models
def test_corrupted_model_file_is_unavailable(tmp_path):
    broken = tmp_path / "broken.task"
    broken.write_bytes(b"garbage")
    with pytest.raises(NonverbalUnavailable):
        observe_frames([], broken, Path(POSE_MODEL))


FRONT_FACE = FACES_DIR / "front.jpg"
SIDE_FACE = FACES_DIR / "side40.jpg"


def _observe_one(path: Path) -> FrameObs:
    return observe_frames(
        [SampledFrame(path=path, timestamp=0.0)], Path(FACE_MODEL), Path(POSE_MODEL)
    )[0]


@needs_models
@pytest.mark.skipif(not FRONT_FACE.is_file(), reason="tests/fixtures/faces/front.jpg 없음(README)")
def test_mediapipe_front_face_is_front():
    o = _observe_one(FRONT_FACE)
    assert o.has_face
    assert abs(o.yaw) <= nv.GAZE_YAW_MAX_DEG and abs(o.pitch) <= nv.GAZE_PITCH_MAX_DEG  # type: ignore[arg-type]


@needs_models
@pytest.mark.skipif(not SIDE_FACE.is_file(), reason="tests/fixtures/faces/side40.jpg 없음(README)")
def test_mediapipe_side_face_is_away():
    o = _observe_one(SIDE_FACE)
    # 40도 돌린 얼굴은 얼굴이 잡히면 기준각 밖, 안 잡혀도 자세가 잡히면 시선 벗어남
    assert o.observed
    assert nv._is_away(o)
    if o.has_face:
        assert 25 <= abs(o.yaw) <= 60  # type: ignore[arg-type]


# ---------- 머리 영역 자르기·Pose 고개 방향(순수 함수) ----------

VIS = 0.99


def lm(x, y, v=VIS):
    return (x, y, v)


def test_head_crop_box_small_face_far_away():
    # 1920×1080, 사람이 멀리 서 있음: 어깨 너비 0.08(154px) → 한 변 약 184px 정사각형
    head = [lm(0.50, 0.30), lm(0.49, 0.29), lm(0.51, 0.29), lm(0.48, 0.30), lm(0.52, 0.30)]
    box = nv.head_crop_box(head, lm(0.54, 0.40), lm(0.46, 0.40), 1920, 1080)
    assert box is not None
    x0, y0, x1, y1 = box
    side = 0.08 * 1920 * nv.FACE_CROP_SCALE
    assert x1 - x0 == pytest.approx(side, abs=2) and y1 - y0 == pytest.approx(side, abs=2)
    assert (x0 + x1) / 2 == pytest.approx(0.5 * 1920, abs=1)
    assert (y0 + y1) / 2 == pytest.approx(0.296 * 1080, abs=2)  # 코·눈·귀 평균


def test_head_crop_box_is_clipped_to_screen():
    # 머리가 화면 왼쪽 위 모서리에 걸침 → 화면 안으로 잘라 맞춘다(정사각형이 아니게 됨)
    head = [lm(0.02, 0.03)]
    x0, y0, x1, y1 = nv.head_crop_box(head, lm(0.12, 0.2), lm(-0.08, 0.2), 1000, 800)
    assert (x0, y0) == (0, 0)
    assert x1 == round(20 + 0.2 * 1000 * 1.2 / 2) and y1 == round(24 + 120)
    assert x1 <= 1000 and y1 <= 800


def test_head_crop_box_mostly_outside_is_none():
    head = [lm(1.05, 0.5)]  # 머리 중심이 화면 오른쪽 밖
    assert nv.head_crop_box(head, lm(1.08, 0.6), lm(1.02, 0.6), 1000, 800) is None


@pytest.mark.parametrize(
    ("head", "ls", "rs"),
    [
        ([lm(0.5, 0.3, 0.2)], lm(0.55, 0.4), lm(0.45, 0.4)),  # 머리 점이 안 보임
        ([lm(0.5, 0.3)], lm(0.55, 0.4, 0.1), lm(0.45, 0.4)),  # 어깨가 안 보임
        ([lm(0.5, 0.3)], lm(0.501, 0.4), lm(0.5, 0.4)),  # 어깨 너비가 너무 작음(48px 미만)
    ],
)
def test_head_crop_box_none_cases(head, ls, rs):
    assert nv.head_crop_box(head, ls, rs, 1920, 1080) is None


@pytest.mark.parametrize(
    ("nose", "le", "re_", "body", "expected"),
    [
        (lm(0.50, 0.3), lm(0.46, 0.3), lm(0.54, 0.3), True, HeadTurn.FRONT),  # 코가 가운데
        (lm(0.512, 0.3), lm(0.46, 0.3), lm(0.54, 0.3), True, HeadTurn.FRONT),  # 0.15
        (lm(0.52, 0.3), lm(0.46, 0.3), lm(0.54, 0.3), True, None),  # 0.25: 애매
        (lm(0.528, 0.3), lm(0.46, 0.3), lm(0.54, 0.3), True, HeadTurn.AWAY),  # 0.35
        (lm(0.58, 0.3), lm(0.46, 0.3), lm(0.54, 0.3), True, HeadTurn.AWAY),  # 귀 바깥
        (lm(0.55, 0.3), lm(0.50, 0.3), lm(0.50, 0.3, 0.1), True, HeadTurn.AWAY),  # 옆모습
        (lm(0.55, 0.3), lm(0.50, 0.3), lm(0.50, 0.3, 0.3), True, None),  # 귀가 반쯤: 애매
        (lm(0.5, 0.3, 0.1), lm(0.46, 0.3), lm(0.54, 0.3), True, HeadTurn.AWAY),  # 뒤돌아섬
        (lm(0.5, 0.3, 0.1), lm(0.46, 0.3), lm(0.54, 0.3), False, None),  # 몸도 안 보임
        (lm(0.5, 0.3, 0.3), lm(0.46, 0.3), lm(0.54, 0.3), True, None),  # 코가 애매
        (lm(0.5, 0.3), lm(0.5, 0.3), lm(0.5, 0.3), True, None),  # 귀 거리 0
        (lm(0.5, -0.05, 0.1), lm(0.46, 0.0), lm(0.54, 0.0), True, None),  # 머리가 화면 위로 잘림
        (lm(0.5, 0.3), lm(-0.02, 0.3), lm(0.04, 0.3, 0.1), True, None),  # 귀가 화면 밖
    ],
)
def test_estimate_head_turn(nose, le, re_, body, expected):
    assert nv.estimate_head_turn(nose, le, re_, body) is expected


# ---------- _Observer: 가짜 Pose/Face 랜드마커로 자르기 흐름 확인 ----------


def _pose_landmarks(points: dict[int, tuple[float, float, float]]):
    from types import SimpleNamespace

    base = [SimpleNamespace(x=0.5, y=0.9, visibility=0.0) for _ in range(33)]
    for i, (x, y, v) in points.items():
        base[i] = SimpleNamespace(x=x, y=y, visibility=v)
    return base


FAR_SPEAKER = {  # 1920×1080에서 멀리 선 사람(어깨 너비 0.08)
    0: lm(0.50, 0.30), 2: lm(0.49, 0.29), 5: lm(0.51, 0.29), 7: lm(0.48, 0.30),
    8: lm(0.52, 0.30), 11: lm(0.54, 0.40), 12: lm(0.46, 0.40),
    15: lm(0.56, 0.6), 16: lm(0.44, 0.6),
}  # fmt: skip


class FakeLandmarkers:
    def __init__(self, pose_points, face_found=True):
        from types import SimpleNamespace

        import numpy as np

        self.pose_inputs: list[tuple[int, int]] = []
        self.face_inputs = []
        self.pose_points = pose_points
        self.face_found = face_found
        ident = np.eye(4)
        ident[2, 3] = -50.0
        self.pose = SimpleNamespace(detect_for_video=self._pose)
        self.face = SimpleNamespace(detect=self._face, close=lambda: None)
        self.pose.close = lambda: None
        self._matrix = ident

    def _pose(self, img, ts):
        from types import SimpleNamespace

        self.pose_inputs.append(img.shape[:2])
        lms = [_pose_landmarks(self.pose_points)] if self.pose_points else []
        return SimpleNamespace(pose_landmarks=lms)

    def _face(self, img):
        from types import SimpleNamespace

        self.face_inputs.append(img)
        if not self.face_found:
            return SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[])
        return SimpleNamespace(face_landmarks=[[0]], facial_transformation_matrixes=[self._matrix])


def make_observer(fake: FakeLandmarkers):
    from types import SimpleNamespace

    import cv2
    import numpy as np

    obs = object.__new__(nv._Observer)
    obs._cv2, obs._np = cv2, np
    obs._mp = SimpleNamespace(
        Image=lambda image_format, data: data, ImageFormat=SimpleNamespace(SRGB=1)
    )
    obs._pose, obs._face, obs._last_ms = fake.pose, fake.face, -1
    return obs


def full_frame(w=1920, h=1080):
    import numpy as np

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(w, dtype=np.uint32)[None, :] % 256  # 위치를 알 수 있게 무늬
    rgb[:, :, 1] = np.arange(h, dtype=np.uint32)[:, None] % 256
    return rgb


def test_observer_small_face_uses_crop_from_full_resolution():
    fake = FakeLandmarkers(FAR_SPEAKER)
    o = make_observer(fake).observe(1.0, full_frame())
    assert fake.pose_inputs == [(360, 640)]  # 자세는 640px로 줄인 이미지
    (face_img,) = fake.face_inputs
    assert max(face_img.shape[:2]) == nv.FACE_CROP_INPUT_PX  # 머리 영역을 256px로
    assert face_img.shape[0] == face_img.shape[1]  # 정사각형
    assert o.has_face and o.yaw == pytest.approx(0.0) and o.pitch == pytest.approx(0.0)
    assert o.pose is not None and o.head_turn is HeadTurn.FRONT


def test_observer_face_missing_with_frontal_pose():
    fake = FakeLandmarkers(FAR_SPEAKER, face_found=False)
    o = make_observer(fake).observe(1.0, full_frame())
    assert not o.has_face and o.head_turn is HeadTurn.FRONT
    r = analyze_nonverbal([o] + still(4, start=2))
    assert r.metrics.gaze_front_ratio == 1.0  # 이 프레임은 분모에서 빠짐


def test_observer_face_missing_with_turned_pose():
    turned_points = {**FAR_SPEAKER, 0: lm(0.535, 0.30), 8: lm(0.52, 0.30, 0.05)}  # 옆모습
    fake = FakeLandmarkers(turned_points, face_found=False)
    o = make_observer(fake).observe(1.0, full_frame())
    assert not o.has_face and o.head_turn is HeadTurn.AWAY
    r = analyze_nonverbal([o] + still(4, start=2))
    assert r.metrics.gaze_front_ratio == 0.8  # 분모에 들어가 시선 벗어남으로 셈


def test_observer_crop_outside_screen_is_clipped():
    edge = {
        0: lm(0.02, 0.05), 7: lm(0.0, 0.05), 8: lm(0.04, 0.05),
        11: lm(0.12, 0.2), 12: lm(-0.06, 0.2),
    }  # fmt: skip
    fake = FakeLandmarkers(edge)
    make_observer(fake).observe(1.0, full_frame(1000, 800))
    (face_img,) = fake.face_inputs
    h, w = face_img.shape[:2]
    assert max(h, w) == nv.FACE_CROP_INPUT_PX and h != w  # 잘려서 정사각형이 아님
    assert int(face_img[0, 0, 0]) == 0 and int(face_img[0, 0, 1]) == 0  # 원본 왼쪽 위에서 시작


def test_observer_without_shoulders_searches_whole_small_frame():
    fake = FakeLandmarkers({0: lm(0.5, 0.4)})  # 근접 촬영: 어깨가 안 보임
    o = make_observer(fake).observe(1.0, full_frame())
    (face_img,) = fake.face_inputs
    assert face_img.shape[:2] == (360, 640)
    assert o.pose is None and o.has_face


def test_observer_no_pose_at_all():
    fake = FakeLandmarkers({}, face_found=False)
    o = make_observer(fake).observe(1.0, full_frame())
    assert o.pose is None and o.head_turn is None and not o.head_known


def test_observe_video_passes_media_errors_through(tmp_path, monkeypatch):
    from app.services.media import MediaError

    monkeypatch.setattr(nv, "_check_models", lambda *a: None)

    class Dummy:
        def __init__(self, *a):
            pass

        def observe(self, t, rgb):
            return FrameObs(t=t, yaw=0.0, pitch=0.0, pose=None)

        def close(self):
            pass

    monkeypatch.setattr(nv, "_Observer", Dummy)

    def broken(*a, **kw):
        yield nv.audio.DecodedFrame(0.0, full_frame(64, 48))
        raise MediaError("truncated")

    with pytest.raises(MediaError):
        nv.observe_video(
            tmp_path / "v.mp4", "video/mp4", tmp_path / "f", tmp_path / "p", frames=broken
        )

    def ok(*a, **kw):
        for i in range(3):
            yield nv.audio.DecodedFrame(float(i), full_frame(64, 48))

    obs = nv.observe_video(
        tmp_path / "v.mp4", "video/mp4", tmp_path / "f", tmp_path / "p", frames=ok
    )
    assert [o.t for o in obs] == [0.0, 1.0, 2.0]


@needs_models
@needs_ffmpeg
def test_observe_video_real_mediapipe_on_blank_video(tmp_path):
    video = tmp_path / "blank.mp4"
    ffmpeg = shutil.which("ffmpeg")
    subprocess.run(  # noqa: S603 - 고정 인자
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=640x360:d=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],  # fmt: skip
        check=True,
    )
    obs = nv.observe_video(video, "video/mp4", Path(FACE_MODEL), Path(POSE_MODEL), fps=1.0)
    assert len(obs) == 3 and all(not o.has_face and o.pose is None for o in obs)


def test_observe_video_closes_frame_generator_when_mediapipe_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(nv, "_check_models", lambda *a: None)
    closed = {"value": False}

    class Boom:
        def __init__(self, *a):
            pass

        def observe(self, t, rgb):
            raise RuntimeError("mediapipe crashed")

        def close(self):
            pass

    monkeypatch.setattr(nv, "_Observer", Boom)

    def frames(*a, **kw):
        try:
            for i in range(100):
                yield nv.audio.DecodedFrame(float(i), full_frame(64, 48))
        finally:
            closed["value"] = True  # iter_frames는 여기서 ffmpeg·타이머를 정리한다

    with pytest.raises(NonverbalUnavailable):
        nv.observe_video(
            tmp_path / "v.mp4", "video/mp4", tmp_path / "f", tmp_path / "p", frames=frames
        )
    assert closed["value"]
