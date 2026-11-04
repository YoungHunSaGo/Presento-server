"""비언어 지표 안정성: 랜드마크에 작은 무작위 잡음을 섞어 여러 번 돌려도 결과가 거의 같아야 한다.

같은 영상이라도 플랫폼(ffmpeg 색 변환·MediaPipe 빌드)에 따라 랜드마크가 프레임 너비의 1~2% 정도
달라진다. 잡음 ±0.005·±0.01(프레임 너비)을 시드 10개로 섞어
- 실행끼리의 차이(최대-최소)/평균 ≤ 10%
- 제스처는 잡음 없는 값과의 차이도 ≤ 10%
를 확인한다. 시선은 각도 지표라 같은 비율의 각도 잡음(±1~2도)을 섞는다.
"""

import math
import random
import statistics

import pytest

from app.services.pipeline.nonverbal import FrameObs, PoseObs, analyze_nonverbal

SEEDS = range(10)


def session(seed=None, amp=0.0, *, fps=2.0, secs=120, width=0.1, period=8.0, big=0.6):
    """먼 화자(어깨 너비=프레임의 width): 몸이 좌우로 흔들리고, period초마다 손을 big×어깨 너비만큼
    1초 올렸다 1초 내린다. 시선 yaw는 20초 주기로 ±30도."""
    rng = random.Random(seed)

    def n():
        return rng.uniform(-amp, amp) if amp else 0.0

    frames = []
    for i in range(int(secs * fps)):
        t = i / fps
        cx, cy = 0.5 + 0.02 * math.sin(2 * math.pi * t / 4), 0.45
        ph = t % period
        lift = big * width * (ph if ph < 1 else (2 - ph) if ph < 2 else 0)
        p = PoseObs(
            nose=(cx + n(), cy - 0.8 * width + n()),
            left_shoulder=(cx + width / 2 + n(), cy + n()),
            right_shoulder=(cx - width / 2 + n(), cy + n()),
            left_wrist=(cx + 0.6 * width + n(), cy + width - lift + n()),
            right_wrist=(cx - 0.6 * width + n(), cy + width + n()),
        )
        yaw = 30 * math.sin(2 * math.pi * t / 20) + (n() / amp * 2 if amp else 0.0)
        frames.append(FrameObs(t=t, yaw=yaw, pitch=0.0, pose=p))
    return analyze_nonverbal(frames)


def spread(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return (max(values) - min(values)) / mean if mean else 0.0


METRICS = {
    "gesturePerMin": lambda r: r.metrics.gesture_per_min,
    "postureStability": lambda r: r.metrics.posture_stability,
    "gazeFrontRatio": lambda r: r.metrics.gaze_front_ratio,
}


@pytest.mark.parametrize("amp", [0.005, 0.01])
@pytest.mark.parametrize("width", [0.1, 0.2])  # 먼 화자, 가까운 화자
@pytest.mark.parametrize("metric", list(METRICS))
def test_metric_is_stable_under_landmark_noise(amp, width, metric):
    get = METRICS[metric]
    values = [get(session(seed, amp, width=width)) for seed in SEEDS]
    assert spread(values) <= 0.10, (metric, values)


@pytest.mark.parametrize("amp", [0.005, 0.01])
@pytest.mark.parametrize(
    "scenario",
    [
        {},  # 8초마다 보통 손동작
        {"period": 5.0},  # 잦은 손동작
        {"period": 4.0},  # 시간의 절반이 손동작
        {"big": 1.0},  # 큰 손동작
        {"width": 0.2},
    ],
)
def test_gesture_rate_close_to_noiseless(amp, scenario):
    base = session(**scenario).metrics.gesture_per_min
    values = [session(seed, amp, **scenario).metrics.gesture_per_min for seed in SEEDS]
    assert base > 0
    assert max(abs(v - base) for v in values) / base <= 0.10, (base, values)
    assert spread(values) <= 0.10, values
