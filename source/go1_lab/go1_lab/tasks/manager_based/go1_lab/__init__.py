# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents # import go1_lab.tasks.manager_based.go1_lab.agents

##
# Register Gym environments.
##
# ~/wj/go1_antalgic/source/go1_lab/go1_lab/tasks/manager_based/go1_lab
# 패키지 시작기준점: go1_lab/tasks/manager_based/go1_lab

gym.register(
    id="Template-Go1-Lab-v0",
    entry_point=f"{__name__}.go1_lab_env:Go1LabEnv", # go1_lab.tasks.manager_based.go1_lab.go1_lab_env:Go1LabEnv -> class Go1LabEnv -> from go1_lab.tasks.manager_based.go1_lab.go1_lab_env import Go1LabEnv
    disable_env_checker=True,
    kwargs={ # 환경설정 클래스
        "env_cfg_entry_point": f"{__name__}.go1_lab_env_cfg:Go1LabEnvCfg", 
        
        # 학습 설정 클래스 from go1_lab.tasks.manager_based.go1_lab.agents.rsl_rl_ppo_cfg import PPORunnerCfg       
        #"rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeacherRunnerCfg", # phase 1 + phase2 사용
        #"rsl_rl_distill_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DistillRunnerCfg", # phase 3 사용

        # 수정할 놈임
        "rsl_rl_phase1_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Phase1HealthyRunnerCfg",
        "rsl_rl_phase2_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Phase2InjuryRunnerCfg",
        "rsl_rl_phase3_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DistillRunnerCfg"
    },
)


def _official_go1_rsl_rl_cfg():
    """Use Isaac Lab's published Go1 RSL-RL architecture for baseline evaluation."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    return load_cfg_from_registry(
        "Isaac-Velocity-Rough-Unitree-Go1-v0", "rsl_rl_cfg_entry_point"
    )


gym.register(
    id="Template-Go1-Lab-OfficialBaseline-v0",
    entry_point=f"{__name__}.go1_lab_env:Go1LabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_lab_env_cfg:Go1LabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:_official_go1_rsl_rl_cfg",
    },
)
