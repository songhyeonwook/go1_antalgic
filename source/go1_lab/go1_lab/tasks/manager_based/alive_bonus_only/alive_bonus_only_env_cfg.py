# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fault-tolerant baseline with velocity tracking and alive bonus only."""

from __future__ import annotations

import os

from isaaclab.envs import mdp as mdp_base
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from go1_lab.tasks.manager_based.go1_lab.go1_lab_env_cfg import Go1LabEnvCfg


@configclass
class AliveBonusOnlyEnvCfg(Go1LabEnvCfg):
    """Peg-leg fault-tolerant baseline reward used for comparison.

    The task keeps the Go1 peg-leg randomization/action-mask mechanics from
    ``Go1LabEnvCfg`` but removes antalgic/pain, energy, symmetry, air-time, and
    posture shaping terms. The remaining reward is exactly:

    - linear velocity tracking
    - yaw angular velocity tracking
    - alive bonus

    This gives a standard fault-tolerant baseline in the style of prior
    velocity-tracking plus survival-bonus locomotion baselines.
    """

    def __post_init__(self):
        # This baseline is intrinsically a fault-tolerant peg-leg task. Make it
        # independent of any leftover GO1_PHASE value in the shell.
        old_phase = os.environ.get("GO1_PHASE")
        os.environ["GO1_PHASE"] = "teacher"
        try:
            super().__post_init__()
        finally:
            if old_phase is None:
                os.environ.pop("GO1_PHASE", None)
            else:
                os.environ["GO1_PHASE"] = old_phase

        self.use_peg_leg_action_mask = True
        self._keep_velocity_tracking_and_alive_only()

    def _keep_velocity_tracking_and_alive_only(self) -> None:
        allowed_reward_terms = {
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "survival_bonus",
        }

        for name in dir(self.rewards):
            if name.startswith("_") or name in allowed_reward_terms:
                continue
            try:
                value = getattr(self.rewards, name)
            except Exception:
                continue
            if isinstance(value, RewTerm):
                setattr(self.rewards, name, None)

        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = float(
                os.getenv("GO1_ALIVE_ONLY_TRACK_LIN_VEL_WEIGHT", "2.5")
            )
        if hasattr(self.rewards, "track_ang_vel_z_exp"):
            self.rewards.track_ang_vel_z_exp.weight = float(
                os.getenv("GO1_ALIVE_ONLY_TRACK_ANG_VEL_WEIGHT", "0.5")
            )

        self.rewards.survival_bonus = RewTerm(
            func=mdp_base.is_alive,
            weight=float(os.getenv("GO1_ALIVE_ONLY_ALIVE_WEIGHT", "1.0")),
        )
