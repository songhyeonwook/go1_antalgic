# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Go1-Lab-v0",
    entry_point=f"{__name__}.go1_lab_env:Go1LabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_lab_env_cfg:Go1LabEnvCfg",
        # phase 별 runner 설정 (train.py 의 --phase 가 선택)
        "rsl_rl_phase1_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Phase1HealthyRunnerCfg",
        "rsl_rl_phase2_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Phase2InjuryRunnerCfg",
        "rsl_rl_phase3_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DistillRunnerCfg",
    },
)
