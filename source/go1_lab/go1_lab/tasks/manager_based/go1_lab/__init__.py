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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_symmetric_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HealthyPPOLstmSymmetryRunnerCfg",
        "rsl_rl_duty_refine_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HealthyPPOLstmDutyRefineRunnerCfg",
        "rsl_rl_trot_boost_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HealthyPPOLstmTrotBoostRunnerCfg",
        "rsl_rl_official_symmetric_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OfficialGo1SymmetryRunnerCfg",
        "rsl_rl_official_symmetric_refine_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OfficialGo1SymmetryRefineRunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeacherRunnerCfg",
        "rsl_rl_phase1_mirror_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HealthyPPOLstmMirrorRunnerCfg",
        "rsl_rl_teacher_mirror_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeacherMirrorRunnerCfg",
        "rsl_rl_phase1_mlp_symmetric_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HealthyMlpSymmetryRunnerCfg",
        "rsl_rl_phase1_mlp_symmetric_refine_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HealthyMlpSymmetryRefineRunnerCfg",
        "rsl_rl_teacher_mlp_symmetric_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeacherMlpSymmetryRunnerCfg",
        "rsl_rl_teacher_mlp_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeacherMlpRunnerCfg",
        "rsl_rl_distill_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DistillRunnerCfg",
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
