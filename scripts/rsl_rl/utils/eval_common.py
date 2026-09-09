# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""평가 스크립트 공통 루틴 (test.py / play_result.py / play_view.py).

세 스크립트는 같은 순서를 밟는다:
  1. phase yaml → ExperimentConfig                (train.py 와 같은 로더)
  2. Isaac Sim 부팅 후 레지스트리에서 env_cfg / agent_cfg 를 받아 yaml 설정 적용
  3. 평가 조건(--peg_leg 또는 yaml eval.peg_leg)을 리셋 이벤트의 leg_ratios 로 강제
  4. runner 생성 → obs normalizer 설치 → 체크포인트 로드 → inference policy

isaaclab / rsl_rl / go1_lab 은 Isaac Sim 이 뜬 뒤에만 import 할 수 있으므로,
그것들을 쓰는 함수는 전부 함수 안에서 lazy import 한다. 이 모듈 자체는
AppLauncher 이전에 import 해도 안전하다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config_builder import ExperimentConfig, load_experiment_config

SCRIPT_DIR = Path(__file__).resolve().parents[1]  # scripts/rsl_rl
LEGS = ("FL", "FR", "RL", "RR")
# _peg_leg_index (-1, 0, 1, 2, 3) → 조건 이름
CONDITION_LABELS = ("Normal", "FL", "FR", "RL", "RR")

# 평가 조건 → 리셋 이벤트 leg_ratios (normal, FL, FR, RL, RR).
# leg_deterministic=True(env_fixed) 와 함께 쓰면 env_id 순서대로 조건 블록이 잡힌다.
#   balanced + 5 env  → env0 Normal, env1 FL, env2 FR, env3 RL, env4 RR
#   balanced + 40 env → env 0-7 Normal, 8-15 FL, 16-23 FR, 24-31 RL, 32-39 RR
EVAL_LEG_RATIOS = {
    "normal": (1.0, 0.0, 0.0, 0.0, 0.0),
    "fl": (0.0, 1.0, 0.0, 0.0, 0.0),
    "fr": (0.0, 0.0, 1.0, 0.0, 0.0),
    "rl": (0.0, 0.0, 0.0, 1.0, 0.0),
    "rr": (0.0, 0.0, 0.0, 0.0, 1.0),
    "balanced": (0.2, 0.2, 0.2, 0.2, 0.2),
}


# ── 인자 / 설정 (Isaac Sim 부팅 전에 호출 가능) ─────────────────────────────

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """세 스크립트가 공유하는 인자."""
    parser.add_argument(
        "--phase_config_path", type=str, required=True,
        help="학습에 쓴 phase yaml 경로 (configs/phase/<1|2|3>/*.yaml)",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="재생할 model_*.pt 경로")
    parser.add_argument(
        "--common_config_path", type=str, default=str(SCRIPT_DIR / "configs" / "common.yaml"),
    )
    parser.add_argument("--num_envs", type=int, default=None, help="미지정 시 스크립트별 기본값")
    parser.add_argument("--seed", type=int, default=None, help="미지정 시 yaml train.seed")
    parser.add_argument(
        "--peg_leg", type=str, choices=tuple(EVAL_LEG_RATIOS), default=None,
        help="평가 조건. 미지정 시 yaml 의 eval.peg_leg (play_result 는 balanced)",
    )
    parser.add_argument(
        "--splint_length", type=float, nargs="+", default=None, metavar="L",
        help="부목 길이 [m]. 값 1개면 고정, 2개면 균등 범위. 미지정 시 yaml",
    )
    parser.add_argument(
        "--clean", action="store_true", help="마찰·질량 랜덤화, push, 관측 노이즈를 끄고 평가",
    )


def load_config(args) -> ExperimentConfig:
    return load_experiment_config(
        phase_path=Path(args.phase_config_path).expanduser().resolve(),
        common_path=args.common_config_path,
    )


def resolve_checkpoint(args) -> Path:
    path = Path(args.checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {path}")
    return path


def peg_leg_enabled(config: ExperimentConfig) -> bool:
    return bool(config.environment.values["peg_leg"]["enabled"])


def resolve_eval_mode(args, config: ExperimentConfig, default: str | None = None) -> str:
    """--peg_leg > default > yaml eval.peg_leg 순으로 평가 조건을 정한다."""
    if args.peg_leg is not None:
        mode = args.peg_leg
    elif default is not None:
        mode = default
    else:
        mode = config.evaluation.peg_leg
    mode = mode.strip().lower()
    if mode not in EVAL_LEG_RATIOS:
        raise ValueError(f"peg_leg 는 {sorted(EVAL_LEG_RATIOS)} 중 하나여야 합니다: {mode!r}")
    if mode != "normal" and not peg_leg_enabled(config):
        raise ValueError(
            f"peg_leg.enabled=false 인 env({config.environment.path.name})에서는 "
            f"--peg_leg {mode} 를 쓸 수 없습니다. phase 2/3 yaml 을 지정하세요."
        )
    return mode


def check_splint_length_arg(values) -> None:
    """개수만 미리 검사한다 (범위 검사는 go1_lab 을 import 할 수 있는 build_env_cfg 에서)."""
    if values is not None and len(values) not in (1, 2):
        raise ValueError("--splint_length 는 값 1개(고정) 또는 2개(범위)만 받습니다.")


# ── env / agent 구성 (Isaac Sim 부팅 후 호출) ─────────────────────────────────

def _register_tasks() -> None:
    """gym 레지스트리에 Isaac Lab / go1_lab 태스크를 등록한다 (import 부수효과).

    load_cfg_from_registry 와 gym.make 는 'Template-Go1-Lab-v0' 이 등록돼 있어야
    하므로 그것들을 쓰는 함수가 먼저 부른다. 여러 번 불러도 무해하다.
    """
    import isaaclab_tasks  # noqa: F401
    import go1_lab.tasks  # noqa: F401


def log(*args, **kwargs) -> None:
    """flush 를 기본으로 켠 print.

    stdout 이 파일로 리다이렉트되면 블록 버퍼링되는데 Isaac Sim 종료는 파이썬
    버퍼를 비우지 않아 마지막 출력이 통째로 사라진다.
    """
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def build_env_cfg(
    config: ExperimentConfig, *, num_envs: int, seed: int, device: str, eval_mode: str,
    splint_length=None, clean: bool = False,
):
    """레지스트리의 env_cfg 에 yaml 설정과 평가 조건을 적용한다."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    _register_tasks()
    env_cfg = load_cfg_from_registry(config.train.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.sim.device = device
    env_cfg.seed = int(seed)
    # steps_per_iteration 은 env 설정에서 쓰이지 않는다 (train.py 와 시그니처만 맞춤).
    env_cfg.apply_environment_settings(
        config.environment.values, int(config.train.num_steps_per_env), eval_peg_leg=eval_mode,
    )

    peg_event = env_cfg.events.randomize_peg_leg_actuation
    if peg_event is not None:
        # 학습 yaml 의 조건 비율 대신 평가 조건을 강제한다. env_fixed 배정이라
        # 리셋해도 조건이 바뀌지 않고 env_id 순서대로 블록이 잡힌다.
        peg_event.params["leg_ratios"] = EVAL_LEG_RATIOS[eval_mode]
        peg_event.params["leg_deterministic"] = True
        if splint_length is not None:
            from go1_lab.splint import SPLINT_MAX, SPLINT_MIN

            lo, hi = float(min(splint_length)), float(max(splint_length))
            if lo < SPLINT_MIN or hi > SPLINT_MAX:
                raise ValueError(
                    f"--splint_length {splint_length} 는 USD 설계 한계 "
                    f"[{SPLINT_MIN}, {SPLINT_MAX}] 안에 있어야 합니다."
                )
            peg_event.params["splint_length_range"] = (lo, hi)
    elif splint_length is not None:
        raise ValueError("--splint_length 는 peg_leg.enabled=true 인 env 에서만 쓸 수 있습니다.")

    if clean:
        for name in ("physics_material", "add_base_mass", "push_robot"):
            if getattr(env_cfg.events, name, None) is not None:
                setattr(env_cfg.events, name, None)
        env_cfg.observations.policy.enable_corruption = False
    return env_cfg


def set_viewer_follow(env_cfg, env_index: int = 0, eye=(-3.0, 2.0, 1.2), lookat=(0.0, 0.0, 0.35)):
    """GUI 카메라를 지정 env 의 로봇에 고정한다.

    IsaacLab 뷰어의 Viewer Settings > Environment Index 로 실행 중에도 바꿀 수 있다.
    """
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.env_index = int(env_index)
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = tuple(eye)
    env_cfg.viewer.lookat = tuple(lookat)


def build_agent_cfg(config: ExperimentConfig, *, seed: int, device: str):
    """레지스트리의 agent cfg 에 train.py 와 같은 정책 설정을 적용한다.

    noise_std_type 이 체크포인트와 다르면 파라미터 이름(std / log_std)이 어긋나
    load 가 실패하므로 반드시 yaml 값을 반영한다.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    _register_tasks()
    agent_cfg = load_cfg_from_registry(config.train.task, config.train.agent)
    agent_cfg.seed = int(seed)
    agent_cfg.device = device
    agent_cfg.policy.noise_std_type = config.exploration.noise_std_type
    agent_cfg.policy.init_noise_std = config.exploration.init_noise_std
    return agent_cfg


def make_gym_env(config: ExperimentConfig, env_cfg, render_mode=None):
    """gym env 생성 (RecordVideo 등 gym 래퍼는 이 위에, RSL-RL 래퍼는 wrap_rsl 로)."""
    import gymnasium as gym

    _register_tasks()
    return gym.make(config.train.task, cfg=env_cfg, render_mode=render_mode)


def wrap_rsl(env, agent_cfg):
    """RSL-RL VecEnv 래퍼. 생성 시 env.reset() 이 한 번 돌아 조건 배정이 끝난다."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def make_policy(env, agent_cfg, config: ExperimentConfig, checkpoint, device: str):
    """runner 생성 → obs normalizer 설치 → 체크포인트 로드.

    Returns:
        (runner, policy_fn, policy_module). policy_fn 은 act_inference (결정적),
        policy_module 은 runner.alg.policy (recurrent 여부 / aux head 접근용).

    normalizer 는 체크포인트에 running 통계 버퍼가 들어 있을 수 있어 load 전에
    train.py 와 같은 규약으로 끼워 넣는다 (normalize: false 면 아무것도 안 한다).
    runner 클래스는 agent cfg 의 class_name 으로 고른다 (OnPolicyRunner / Phase3DistillationRunner).
    """
    from go1_lab.tasks.manager_based.go1_lab.mdp.obs_normalizer import install_obs_scaler

    from .rsl_rl_compat import resolve_runner_class

    runner_cls = resolve_runner_class(agent_cfg.class_name)
    runner = runner_cls(env=env, train_cfg=agent_cfg.to_dict(), log_dir=None, device=device)
    nm = config.train.normalize
    replaced = install_obs_scaler(runner.alg.policy, env.get_observations(), nm.obs, nm.priv) if nm.enable else []
    runner.load(str(checkpoint), load_optimizer=False, map_location=device)
    log(f"[eval] Obs scale     : {', '.join(replaced) if replaced else 'disabled — raw obs'}")
    policy_fn = runner.get_inference_policy(device=env.unwrapped.device)
    return runner, policy_fn, runner.alg.policy


def step_env(env, policy_fn, policy_module, obs):
    """정책 1회 평가 + env 1 step.

    phase 3 student 는 recurrent 라 종료된 env 의 LSTM hidden/cell 을 같은 시점에
    초기화해야 새 에피소드에 이전 기억이 남지 않는다.
    """
    actions = policy_fn(obs)
    obs, rew, dones, extras = env.step(actions)
    if getattr(policy_module, "is_recurrent", False):
        policy_module.reset(dones)
    return obs, rew, dones, extras, actions


# ── 조건 조회 / 출력 ──────────────────────────────────────────────────────────

def conditions_of(base) -> list[str]:
    """env 별 현재 조건 이름 (_peg_leg_index 기준, healthy env 는 전부 Normal)."""
    idx = getattr(base, "_peg_leg_index", None)
    if idx is None:
        return ["Normal"] * base.num_envs
    return [CONDITION_LABELS[int(i) + 1] for i in idx.tolist()]


def print_conditions(base, max_list: int = 10) -> None:
    conds = conditions_of(base)
    counts = {c: conds.count(c) for c in CONDITION_LABELS if conds.count(c)}
    head = ", ".join(f"env{i}={c}" for i, c in enumerate(conds[:max_list]))
    more = f" … (+{len(conds) - max_list})" if len(conds) > max_list else ""
    log(f"[eval] 조건 분포      : {counts}")
    log(f"[eval] env 배정       : {head}{more}")


def print_header(config: ExperimentConfig, checkpoint, device: str, seed: int, num_envs: int, eval_mode: str, env_cfg) -> None:
    ranges = env_cfg.commands.base_velocity.ranges
    log(f"[eval] Phase          : {config.phase}  ({config.environment.path.name})")
    log(f"[eval] Checkpoint     : {checkpoint}")
    log(f"[eval] Device / Seed  : {device} / {seed}")
    log(f"[eval] Environments   : {num_envs}")
    log(f"[eval] Peg-leg eval   : {eval_mode}")
    peg_event = env_cfg.events.randomize_peg_leg_actuation
    if peg_event is not None:
        lo, hi = peg_event.params["splint_length_range"]
        log(f"[eval] Splint length  : {lo:.3f} ~ {hi:.3f} m")
    log(f"[eval] lin_vel_x / y  : {ranges.lin_vel_x} / {ranges.lin_vel_y}")
    log(f"[eval] ang_vel_z      : {ranges.ang_vel_z}")
