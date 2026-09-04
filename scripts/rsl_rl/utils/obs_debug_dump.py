# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""학습 전 구간의 정책 입력/출력을 CSV 로 흘려쓰는 디버그 도구.

정규화가 실제로 어떤 값을 정책에 흘려보내는지, 그 분포가 학습 내내 어떻게
변하는지 보기 위한 것이다. 학습이 끝날 때까지 매 act() 호출마다 기록한다.

    <out_dir>/obs_raw.csv    정규화 '이전' 관측
    <out_dir>/obs_norm.csv   정규화 '이후' 관측 — 정책 첫 층이 실제로 받는 값
    <out_dir>/action.csv     그 입력으로 정책이 낸 action (탐색 노이즈 포함 샘플)

파일들은 같은 (step, env_id) 행 순서라 행 단위로 서로 대응된다.

yaml 의 normalize 가 false 면 정규화기 자리에 rsl_rl 의 nn.Identity 가 남고
훅의 output 이 입력 텐서 그 자체라 obs_norm.csv 가 obs_raw.csv 의 완전한 사본이
된다. 그래서 그 경우에는 obs_norm.csv 를 만들지 않는다 (obs_raw.csv 가 곧 정책
입력이다). 아래 용량표는 정규화가 켜진 3파일 기준이고, 꺼져 있으면 약 55% 다.

⚠️ 크기: 행 수 = (max_iterations x num_steps_per_env) x envs_per_step 이다.
   phase 1(5000 x 24 = 120,000 호출) 기준 대략:

       envs_per_step     총 행수      세 파일 합계
                   1     120,000          146 MB
                   4     480,000          584 MB
                  16   1,920,000          2.3 GB
                  64   7,680,000          9.3 GB
                2048 245,760,000        298.8 GB   ← 디스크가 감당 못 한다

   그래서 기본값은 4 이고, 시작할 때 예상 용량과 남은 디스크를 함께 찍는다.
   여유보다 크면 경고만 하고 계속 진행하니 필요하면 중단하면 된다.

원리: 관측 정규화기는 raw obs 를 받아 normalized obs 를 내놓는 nn.Module 이므로
forward hook 하나로 입력과 출력을 동시에 잡을 수 있다. action 은 policy.act 를
감싸서 잡고, 두 값을 같은 act() 호출 안에서 짝지어 기록한다. 메모리에 쌓지 않고
바로 파일에 쓰므로, 학습을 중간에 끊어도 그 시점까지가 그대로 남는다.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import torch

# 관측 레이아웃 (mdp/obs_normalizer.py 의 span 과 동일)
_LEGS = ("FL", "FR", "RL", "RR")
# 관절은 per-TYPE 순서 (hip x4, thigh x4, calf x4) — mdp/mirror.py 와 동일
_JOINTS = tuple(f"{t}_{leg}" for t in ("hip", "thigh", "calf") for leg in _LEGS)

_POLICY_COLUMNS = (
    ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
    + ["gravity_x", "gravity_y", "gravity_z"]
    + ["cmd_vx", "cmd_vy", "cmd_wz"]
    + [f"jpos_{j}" for j in _JOINTS]
    + [f"jvel_{j}" for j in _JOINTS]
    + [f"act_{j}" for j in _JOINTS]
    + [f"onehot_{leg}" for leg in _LEGS]
)  # 3+3+3+12+12+12+4 = 49

# privileged 그룹 [L, lin_vel(3)]. phase 1/2 는 obs_groups 가
# "policy": ["policy", "privileged_obs"] 라 actor 입력이 49+4=53 이 된다
# (teacher PPO 라 privileged 를 본다). phase 3 student 는 policy 49 만 받는다.
_PRIVILEGED_COLUMNS = ["splint_L", "lin_vel_x", "lin_vel_y", "lin_vel_z"]

_ACTION_COLUMNS = [f"a_{j}" for j in _JOINTS]  # 12

# 관측의 jpos_calf_* 는 default_joint_pos 기준 '상대각'이라 부목으로 잠긴
# calf 의 실제 접힌 각도를 알 수 없다. 시뮬레이터에서 절대각을 직접 읽는다.
_CALF_COLUMNS = [f"calf_q_{leg}" for leg in _LEGS]
_CALF_JOINT_NAMES = [f"{leg}_calf_joint" for leg in _LEGS]

_NORMALIZER_ATTRS = ("actor_obs_normalizer", "student_obs_normalizer")


def _obs_columns(width: int) -> list[str]:
    """관측 폭에 맞는 채널 이름표. 모르는 폭이면 일반 인덱스 이름을 쓴다."""
    if width == len(_POLICY_COLUMNS):
        return list(_POLICY_COLUMNS)
    if width == len(_POLICY_COLUMNS) + len(_PRIVILEGED_COLUMNS):
        return list(_POLICY_COLUMNS) + list(_PRIVILEGED_COLUMNS)
    return [f"obs{i}" for i in range(width)]


def _action_columns(width: int) -> list[str]:
    if width == len(_ACTION_COLUMNS):
        return list(_ACTION_COLUMNS)
    return [f"a{i}" for i in range(width)]


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


class ObsDebugDumper:
    """학습 내내 정책 입력(raw/normalized)과 출력 action 을 CSV 로 흘려쓴다.

    Args:
        policy: runner.alg.policy.
        out_dir: CSV 를 쓸 디렉토리 (없으면 만든다).
        envs_per_step: 매 act() 호출에서 기록할 env 수. 이게 곧 용량 손잡이다.
            0 이나 음수면 전체 env 를 쓴다 (용량 주의).
        total_calls: 학습 전체의 예상 act() 호출 수
            (= max_iterations * num_steps_per_env). 시작 시 예상 용량을 찍는
            용도로만 쓰며, 없으면 용량 예측을 건너뛴다.
        flush_every: 몇 번의 act() 호출마다 파일을 flush 할지. 학습이 중간에
            끊겨도 여기까지는 디스크에 남는다.
        logger: 있으면 진행 상황을 여기로 남긴다.
    """

    def __init__(self, policy, out_dir, env=None, envs_per_step: int = 4,
                 total_calls: int | None = None, flush_every: int = 2000,
                 logger=None):
        self.policy = policy
        self.env = env
        self.out_dir = Path(out_dir)
        self.envs_per_step = int(envs_per_step)   # <= 0 이면 전체 env
        self.flush_every = max(1, int(flush_every))
        self.logger = logger

        self._call = 0
        self._rows = 0
        self._pending = None
        self._closed = False
        self._files: dict[str, object] = {}
        self._writers: dict[str, object] = {}
        self._headers_written = False

        # 기록할 env 인덱스 (첫 호출 때 결정)
        self._env_sel = None
        self._env_ids = None

        # calf 절대각 열. env 가 없거나 관절을 못 찾으면 조용히 생략한다.
        self._calf_ids = None
        if env is not None:
            try:
                names = list(env.unwrapped.scene["robot"].data.joint_names)
                self._calf_ids = [names.index(n) for n in _CALF_JOINT_NAMES]
            except (KeyError, ValueError, AttributeError) as err:
                self._log(f"[obs-debug] calf 관절을 찾지 못해 각도 열을 생략합니다: {err}")

        # 정규화기 훅 — raw 입력과 normalized 출력을 한 번에 잡는다.
        self._norm_module = None
        for attr in _NORMALIZER_ATTRS:
            module = getattr(policy, attr, None)
            if module is not None:
                self._norm_module = module
                self._norm_name = attr
                break
        if self._norm_module is None:
            raise RuntimeError(
                f"관측 정규화기를 찾지 못했습니다. 후보: {_NORMALIZER_ATTRS}"
            )

        # 정규화가 꺼져 있으면(yaml normalize: false → install_obs_normalizer 가
        # 아무것도 끼우지 않아 rsl_rl 의 nn.Identity 가 그대로 남는다) 훅이 잡는
        # output 이 입력 텐서 그 자체라 obs_norm.csv 가 obs_raw.csv 의 완전한
        # 사본이 된다. 그 경우 파일을 아예 만들지 않는다.
        self._has_norm = not isinstance(self._norm_module, torch.nn.Identity)
        self._names = (
            ("obs_raw", "obs_norm", "action") if self._has_norm
            else ("obs_raw", "action")
        )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for name in self._names:
            handle = (self.out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8")
            self._files[name] = handle
            self._writers[name] = csv.writer(handle)

        self._handle = self._norm_module.register_forward_hook(self._on_normalize)
        self._orig_act = policy.act
        policy.act = self._wrapped_act

        env_desc = "전체" if self.envs_per_step <= 0 else str(self.envs_per_step)
        mode = ("raw + norm" if self._has_norm
                else "raw 만 — 정규화가 꺼져 있어 obs_norm.csv 는 만들지 않음")
        self._log(
            f"[obs-debug] {self._norm_name} 훅 등록 — 학습 전 구간 기록 "
            f"(act() 호출마다 env {env_desc}개, {mode}) → {self.out_dir}"
        )
        if total_calls:
            self._log_size_estimate(int(total_calls))

    # ── 내부 ────────────────────────────────────────────────────────────
    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message, flush=True)

    def _log_size_estimate(self, total_calls: int) -> None:
        per_step = self.envs_per_step if self.envs_per_step > 0 else 2048  # 대략치
        rows = total_calls * per_step
        # 값 하나가 대략 10바이트 + step/env_id 12바이트
        obs_files = 2 if self._has_norm else 1
        approx = rows * ((53 * 10 + 12) * obs_files + (12 * 10 + 12))
        free = shutil.disk_usage(self.out_dir).free
        note = "" if approx < free * 0.8 else "   ⚠️ 남은 디스크에 비해 큽니다"
        self._log(
            f"[obs-debug] 예상 {rows:,}행 / 약 {_human(approx)} "
            f"(남은 디스크 {_human(free)}){note}"
        )

    def _on_normalize(self, module, inputs, output):
        # act() 이외의 경로(critic 평가 등)는 다른 모듈이라 여기 들어오지 않는다.
        self._pending = (inputs[0], output)

    def _ensure_headers(self, obs_width: int, act_width: int) -> None:
        if self._headers_written:
            return
        self._headers_written = True
        base = ["step", "env_id"] + _obs_columns(obs_width)
        extra = _CALF_COLUMNS if self._calf_ids is not None else []
        self._writers["obs_raw"].writerow(base + extra)
        if self._has_norm:
            self._writers["obs_norm"].writerow(base)
        self._writers["action"].writerow(["step", "env_id"] + _action_columns(act_width))

    def _select_envs(self, num_envs: int, device):
        """기록할 env 인덱스를 전 구간에 고르게 퍼뜨린다.

        peg_leg.leg_policy.mode = env_fixed 는 env_id 순서대로 조건 구간을
        나눈다 (mdp/events.py 의 _assign_leg_by_ratio 는
        pos = (env_id + 0.5) / num_envs 를 누적비율에 넣는다). 그래서 앞에서
        N 개만 뽑으면 normal 비율(기본 0.5)에 걸려 정상 env 만 잡히고 부상
        env 가 한 번도 기록되지 않는다. 구간 중앙을 균등 샘플링하면 조건별
        비율이 그대로 반영된다.
        """
        if self._env_sel is not None:
            return self._env_sel
        if self.envs_per_step <= 0 or self.envs_per_step >= num_envs:
            ids = list(range(num_envs))
        else:
            span = num_envs / self.envs_per_step
            ids = [
                min(num_envs - 1, int(i * span + span / 2))
                for i in range(self.envs_per_step)
            ]
        self._env_ids = ids
        self._env_sel = torch.tensor(ids, device=device, dtype=torch.long)
        self._log(f"[obs-debug] 기록 대상 env {ids}")
        return self._env_sel

    def _write_block(self, name: str, block, extra=None) -> None:
        writer = self._writers[name]
        step = self._call
        for i, row in enumerate(block):
            cells = [f"{v:.6g}" for v in row]
            if extra is not None:
                cells += [f"{v:.6g}" for v in extra[i]]
            writer.writerow([step, self._env_ids[i]] + cells)

    def _wrapped_act(self, obs, **kwargs):
        actions = self._orig_act(obs, **kwargs)

        # ── rollout 호출만 기록한다 ──────────────────────────────────────
        # PPO 는 update() 안에서도 policy.act 를 부른다
        # (rsl_rl/algorithms/ppo.py:250 — masks/hidden_state 를 넘긴다).
        # 그 호출의 obs 는 롤아웃 버퍼에서 셔플된 미니배치라 시간 순서가 없고,
        # 기록되는 action 도 실제 취한 행동이 아니라 재평가 시 새로 뽑은 샘플이다.
        # rollout 경로는 policy.act(obs) 로 인자 없이 부르므로(ppo.py:147)
        # 그것만 남긴다. 이 구분이 없으면 CSV 의 약 45% 가 오염되고
        # obs 의 act_* 열이 action.csv 를 한 스텝 민 값과 어긋난다.
        if kwargs:
            self._pending = None
            return actions

        if self._closed or self._pending is None:
            self._pending = None
            return actions

        raw, norm = self._pending
        self._pending = None

        sel = self._select_envs(raw.shape[0], raw.device)
        raw_b = raw[sel].detach().float().cpu().numpy()
        act_b = actions[sel].detach().float().cpu().numpy()

        calf_b = None
        if self._calf_ids is not None:
            joint_pos = self.env.unwrapped.scene["robot"].data.joint_pos
            calf_b = joint_pos[sel][:, self._calf_ids].detach().float().cpu().numpy()

        self._ensure_headers(raw_b.shape[1], act_b.shape[1])
        self._write_block("obs_raw", raw_b, calf_b)
        if self._has_norm:
            self._write_block("obs_norm", norm[sel].detach().float().cpu().numpy())
        self._write_block("action", act_b)

        self._rows += len(self._env_ids)
        self._call += 1
        if self._call % self.flush_every == 0:
            for handle in self._files.values():
                handle.flush()
            self._log(
                f"[obs-debug] {self._call:,} 호출 / {self._rows:,}행 기록 중…"
            )
        return actions

    # ── 공개 API ────────────────────────────────────────────────────────
    def close(self) -> None:
        """훅을 풀고 파일을 닫는다. 두 번 불러도 안전하다."""
        if self._closed:
            return
        self._closed = True
        self._handle.remove()
        self.policy.act = self._orig_act
        for handle in self._files.values():
            handle.flush()
            handle.close()
        total = sum(
            (self.out_dir / f"{n}.csv").stat().st_size
            for n in self._names
            if (self.out_dir / f"{n}.csv").exists()
        )
        self._log(
            f"[obs-debug] 종료 — {self._call:,} 호출 / {self._rows:,}행, "
            f"{_human(total)} → {self.out_dir}"
        )
