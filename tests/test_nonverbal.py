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


def test_pose_only_frames_count_as_gaze_away():
    # 정면 6장 + 얼굴 없이 어깨만 잡힌 4장 → 분모 10, 정면 6
    frames = still(6) + [frame(6 + i, yaw=None, pitch=None, p=pose()) for i in range(4)]
    r = analyze_nonverbal(frames)
    assert r.metrics.gaze_front_ratio == 0.6
    assert ranges(r.gaze_away_ranges) == [(6.0, 10.0)]
    assert r.metrics.low_confidence_ranges == []


def test_frames_without_face_and_pose_are_excluded():
    # 정면 6장 + 얼굴·자세 둘 다 없는 4장 → 분모 6, 저신뢰 구간
    frames = still(6) + [frame(6 + i, yaw=None, pitch=None) for i in range(4)]
    r = analyze_nonverbal(frames)
    assert r.metrics.gaze_front_ratio == 1.0
    assert r.gaze_away_ranges == []
    assert ranges(r.metrics.low_confidence_ranges) == [(6.0, 10.0)]


def test_gaze_away_run_mixes_turned_face_and_pose_only():
    # 각도로 벗어남 2초 + 얼굴 없이 자세만 2초가 이어지면 하나의 4초 구간
    frames = still(5) + [frame(5 + i, yaw=40) for i in range(2)]
    frames += [frame(7 + i, yaw=None, pitch=None, p=pose()) for i in range(2)]
    frames += still(5, start=9)
    assert ranges(analyze_nonverbal(frames).gaze_away_ranges) == [(5.0, 9.0)]


def test_no_face_rule_still_counts_face_frames_only():
    # 자세만 잡힌 프레임이 많아도 얼굴 프레임이 20% 미만이면 NO_FACE_DETECTED
    frames = still(1) + [frame(1 + i, yaw=None, pitch=None, p=pose()) for i in range(9)]
    with pytest.raises(NoFaceDetected):
        analyze_nonverbal(frames)


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


@pytest.mark.parametrize(("faces", "ok"), [(1, False), (2, True)])
def test_no_face_detected_below_20_percent(faces, ok):
    frames = still(faces) + [frame(faces + i, yaw=None, pitch=None) for i in range(10 - faces)]
    if ok:
        assert analyze_nonverbal(frames).face_frame_ratio == 0.2
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


@pytest.mark.parametrize(("move", "counted"), [(0.25, True), (0.24, False)])
def test_gesture_threshold(move, counted):
    assert nv.GESTURE_MOVE_RATIO == 0.25
    offsets = [0.0] * 10 + [move] + [move] * 9
    r = analyze_nonverbal(_moves(offsets))
    assert r.gesture_times == ([10.0] if counted else [])


def test_short_continuous_movement_is_one_gesture():
    # 10, 11초 연달아 움직임(간격 1초 = GESTURE_MERGE_SEC) → 한 번
    offsets = [0.0] * 10 + [0.3, 0.6] + [0.6] * 8
    assert analyze_nonverbal(_moves(offsets)).gesture_times == [10.0]


def test_long_continuous_movement_counts_every_max_sec():
    assert nv.GESTURE_MAX_SEC == 2.0
    # 60초 내내 손을 흔든다 → 2초마다 한 번 = 분당 30, 모든 구간이 EXCESS
    offsets = [0.0 if i % 2 else 0.5 for i in range(60)]
    r = analyze_nonverbal(_moves(offsets))
    assert r.metrics.gesture_per_min == pytest.approx(30, abs=1.5)
    assert ranges(r.gesture_excess_ranges) == [(0.0, 60.0)]
    assert r.gesture_none_ranges == []


def test_separate_movements_are_counted_separately():
    offsets = [0.0] * 5 + [0.5, 0.5, 0.5, 0.0] + [0.0] * 11  # 5초, 8초에 움직임
    assert analyze_nonverbal(_moves(offsets)).gesture_times == [5.0, 8.0]


def test_merge_with_two_fps():
    # 0.5초 간격으로 연달아 움직이면 한 번, 1.5초 떨어지면 따로
    offsets = [0.0] * 20 + [0.3, 0.6, 0.9, 0.9, 0.9, 0.0] + [0.0] * 14
    r = analyze_nonverbal(_moves(offsets, fps=2.0))
    assert r.gesture_times == [10.0, 12.5]


def test_either_wrist_counts_once():
    frames = still(10)
    frames.append(frame(10, p=pose(lw=(0.5, 0.0), rw=(-0.5, 0.0))))
    frames += [frame(11 + i, p=pose(lw=(0.5, 0.0), rw=(-0.5, 0.0))) for i in range(9)]
    assert analyze_nonverbal(frames).gesture_times == [10.0]


def test_invisible_wrist_is_not_a_gesture():
    frames = still(10) + [frame(10 + i, p=pose(lw=None)) for i in range(10)]
    frames[12] = frame(12, p=pose(lw=(1.0, 1.0)))  # 앞뒤가 None이라 비교 불가
    assert analyze_nonverbal(frames).gesture_times == []


def test_gap_in_pose_frames_is_not_a_gesture():
    # 10초 뒤 자세를 놓쳤다가 13초에 손이 다른 곳 → 사이 움직임은 모른다
    frames = still(10) + [FrameObs(t=10 + i, yaw=0.0, pitch=0.0, pose=None) for i in range(3)]
    frames += [frame(13 + i, p=pose(lw=(1.0, 0.0))) for i in range(10)]
    assert analyze_nonverbal(frames).gesture_times == []


def test_gesture_per_min():
    # 60초, 10초마다 손을 뻗었다 바로 내림(연속 움직임 = 한 번) → 분당 6
    offsets = []
    for _ in range(6):
        offsets += [0.0] * 9 + [0.5]
    r = analyze_nonverbal(_moves(offsets))
    assert len(r.gesture_times) == 6
    assert r.metrics.gesture_per_min == 6.0


def test_gesture_none_and_excess_windows():
    assert nv.GESTURE_WINDOW_SEC == 30.0 and nv.GESTURE_EXCESS_PER_MIN == 20.0
    offsets = [0.0] * 30  # 0~30초: 제스처 없음
    offsets += [0.5 if (i // 2) % 2 else 0.0 for i in range(30)]  # 30~60초: 2초마다 움직임
    offsets += [0.0] * 10 + [0.5] * 20  # 60~90초: 1번
    r = analyze_nonverbal(_moves(offsets))
    assert ranges(r.gesture_none_ranges) == [(0.0, 30.0)]
    assert ranges(r.gesture_excess_ranges) == [(30.0, 60.0)]


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
