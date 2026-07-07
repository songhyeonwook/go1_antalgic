"""
커스텀 USD를 쓰는 Go1 러프 지형 속도 환경 등록.

- 베이스: Isaac-Velocity-Rough-Unitree-Go1-v0 (공식)
- 변경: robot USD 경로를 GO1_CUSTOM_CFG(환경변수 GO1_USD_PATH로 지정 가능)로 교체
- 필요시 num_envs를 GO1_NUM_ENVS 환경변수로 덮어쓰기
"""

import os

import gymnasium as gym

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from go1_lab.asset import GO1_CUSTOM_CFG


def _load_env_cfg():
    # 공식 env cfg 불러온 뒤 로봇만 교체
    env_cfg = load_cfg_from_registry("Isaac-Velocity-Rough-Unitree-Go1-v0", "env_cfg_entry_point")

    # 커스텀 USD 사용
    env_cfg.scene.robot = GO1_CUSTOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 환경 개수 환경변수로 간단히 덮어쓰기 (없으면 기존 값 유지)
    num_envs_env = os.getenv("GO1_NUM_ENVS")
    if num_envs_env:
        try:
            env_cfg.scene.num_envs = int(num_envs_env)
        except ValueError:
            pass

    return env_cfg


def _load_agent_cfg():
    # 공식 PPO 설정 그대로 사용
    return load_cfg_from_registry("Isaac-Velocity-Rough-Unitree-Go1-v0", "rsl_rl_cfg_entry_point")


# 새로운 Gym ID 등록
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-CustomUSD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:_load_env_cfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:_load_agent_cfg",
    },
)

