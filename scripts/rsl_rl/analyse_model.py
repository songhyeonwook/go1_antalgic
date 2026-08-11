from pathlib import Path
import csv
import re

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_DIR = Path(
    "/home/unicon/wj/go1_antalgic/scripts/rsl_rl/logs/"
    "unitree_go1_antalgic/2026_08_03_13_44_58_phase1_s42_P1-000"
)

MIN_CHECKPOINT = 3000
WINDOW_SIZE = 100
TOP_K = 30

# True: 넘어짐 비율이 조금이라도 낮으면 무조건 상위
# False: 안전등급이 같으면 속도 추종부터 비교 — 추천
STRICT_PRIORITY = False

OUTPUT_CSV = RUN_DIR / "sorted_checkpoints_after_3000.csv"


TAGS = {
    # 기본 성능
    "reward": "Train/mean_reward",
    "episode_length": "Train/mean_episode_length",

    # 속도 추종
    "xy_raw": "Metrics/base_velocity/error_vel_xy",
    "yaw_raw": "Metrics/base_velocity/error_vel_yaw",

    # 종료 원인
    "timeout": "Episode_Termination/time_out",
    "bad_orientation": "Episode_Termination/bad_orientation",
    "base_contact": "Episode_Termination/base_contact",
    "root_too_low": "Episode_Termination/root_too_low",

    # 부드러움 및 에너지
    "action_rate": "Episode_Reward/action_rate_l2",
    "torque": "Episode_Reward/dof_torques_l2",
    "dof_acc": "Episode_Reward/dof_acc_l2",
}


def load_tensorboard(run_dir):
    event_accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    event_accumulator.Reload()

    available = set(event_accumulator.Tags()["scalars"])

    for tag in TAGS.values():
        if tag not in available:
            raise RuntimeError(f"TensorBoard에 태그가 없습니다: {tag}")

    return {
        name: {
            event.step: event.value
            for event in event_accumulator.Scalars(tag)
        }
        for name, tag in TAGS.items()
    }


def get_checkpoints(run_dir):
    result = []

    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)

        if match:
            checkpoint = int(match.group(1))

            if checkpoint >= MIN_CHECKPOINT:
                result.append(checkpoint)

    return sorted(result)


def window_mean(values, checkpoint):
    start = checkpoint - WINDOW_SIZE + 1

    selected = [
        value
        for step, value in values.items()
        if start <= step <= checkpoint
    ]

    if not selected:
        return float("nan")

    return float(np.mean(selected))


def get_stability_tier(failure_percent):
    """
    등급이 작을수록 안전한 모델.

    0: 비정상 종료율 0.02% 이하
    1: 비정상 종료율 0.05% 이하
    2: 비정상 종료율 0.10% 이하
    3: 비정상 종료율 0.50% 이하
    4: 그 이상
    """
    if failure_percent <= 0.02:
        return 0
    if failure_percent <= 0.05:
        return 1
    if failure_percent <= 0.10:
        return 2
    if failure_percent <= 0.50:
        return 3
    return 4


data = load_tensorboard(RUN_DIR)
checkpoints = get_checkpoints(RUN_DIR)

rows = []

for checkpoint in checkpoints:
    value = {
        name: window_mean(tag_data, checkpoint)
        for name, tag_data in data.items()
    }

    # 현재 metric은 500 step 기준으로 정규화되어 있지만
    # Phase1 episode는 1000 step이므로 약 0.5를 곱함
    xy_error = value["xy_raw"] * 0.5
    yaw_error = value["yaw_raw"] * 0.5

    timeout_percent = value["timeout"] * 100.0
    failure_percent = max(0.0, 100.0 - timeout_percent)

    bad_orientation_percent = value["bad_orientation"] * 100.0
    base_contact_percent = value["base_contact"] * 100.0
    root_too_low_percent = value["root_too_low"] * 100.0

    # 서로 단위가 다르기 때문에 허용 오차 기준으로 정규화
    # 값이 작을수록 속도 추종 성능이 좋음
    tracking_score = (
        xy_error / 0.05
        + yaw_error / 0.06
    )

    # 이 TensorBoard 항목들은 음수 reward penalty
    # 절댓값 합이 작을수록 부드럽고 에너지 사용이 적음
    smooth_energy_score = -(
        value["action_rate"]
        + value["torque"]
        + value["dof_acc"]
    )

    rows.append({
        "checkpoint": checkpoint,
        "stability_tier": get_stability_tier(failure_percent),

        "failure_percent": failure_percent,
        "timeout_percent": timeout_percent,
        "episode_length": value["episode_length"],

        "bad_orientation_percent": bad_orientation_percent,
        "base_contact_percent": base_contact_percent,
        "root_too_low_percent": root_too_low_percent,

        "xy_error": xy_error,
        "yaw_error": yaw_error,
        "tracking_score": tracking_score,

        "reward": value["reward"],

        "action_rate_penalty": value["action_rate"],
        "torque_penalty": value["torque"],
        "dof_acc_penalty": value["dof_acc"],
        "smooth_energy_score": smooth_energy_score,
    })


if STRICT_PRIORITY:
    # 완전한 사전식 우선순위
    #
    # 1. 비정상 종료율
    # 2. 평균 생존 길이
    # 3. 속도 추종
    # 4. Reward
    # 5. 부드러움과 에너지
    #
    # 주의: 비정상 종료율이 조금이라도 다르면
    # 속도 추종과 Reward는 사실상 비교되지 않음
    rows.sort(
        key=lambda row: (
            row["failure_percent"],
            -row["episode_length"],
            row["tracking_score"],
            -row["reward"],
            row["smooth_energy_score"],
        )
    )

else:
    # 추천 정렬
    #
    # 1. 안전등급
    # 2. 같은 안전등급 내 속도 추종
    # 3. Reward
    # 4. 부드러움과 에너지
    # 5. 최종 동률이면 비정상 종료율
    rows.sort(
        key=lambda row: (
            row["stability_tier"],
            row["tracking_score"],
            -row["reward"],
            row["smooth_energy_score"],
            row["failure_percent"],
        )
    )


# 순위 추가
for rank, row in enumerate(rows, start=1):
    row["rank"] = rank


# CSV 저장
fieldnames = [
    "rank",
    "checkpoint",
    "stability_tier",
    "failure_percent",
    "timeout_percent",
    "episode_length",
    "bad_orientation_percent",
    "base_contact_percent",
    "root_too_low_percent",
    "xy_error",
    "yaw_error",
    "tracking_score",
    "reward",
    "action_rate_penalty",
    "torque_penalty",
    "dof_acc_penalty",
    "smooth_energy_score",
]

with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


# 상위 모델 출력
print(
    "| 순위 | 모델 | 안전등급 | 실패율 | XY | Yaw | "
    "Reward | 부드러움·에너지 |"
)
print("|---:|---|---:|---:|---:|---:|---:|---:|")

for row in rows[:TOP_K]:
    print(
        f"| {row['rank']} "
        f"| `model_{row['checkpoint']}` "
        f"| {row['stability_tier']} "
        f"| {row['failure_percent']:.4f}% "
        f"| {row['xy_error']:.4f} "
        f"| {row['yaw_error']:.4f} "
        f"| {row['reward']:.3f} "
        f"| {row['smooth_energy_score']:.4f} |"
    )

print(f"\n결과 저장: {OUTPUT_CSV}")