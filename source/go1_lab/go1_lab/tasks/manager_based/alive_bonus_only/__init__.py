# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Alive-bonus-only fault-tolerant Go1 baseline task."""

import gymnasium as gym

from . import agents


gym.register(
    id="Go1-Alive-Bonus-Only-v0",
    entry_point="go1_lab.tasks.manager_based.go1_lab.go1_lab_env:Go1LabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.alive_bonus_only_env_cfg:AliveBonusOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AliveBonusOnlyRunnerCfg",
    },
)
