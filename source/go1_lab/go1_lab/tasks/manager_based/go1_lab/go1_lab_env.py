# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go1 Lab 환경 - ActuatorNetMLP 호환 Peg-Leg Action Masking."""

from __future__ import annotations

import os
import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from .go1_lab_env_cfg import Go1LabEnvCfg


class Go1LabEnv(ManagerBasedRLEnv):
    """Go1 Lab 환경 (ManagerBasedRLEnv 확장).

    ⚠️ ActuatorNetMLP 호환을 위한 핵심 오버라이드:
      Go1은 Explicit Actuator(ActuatorNetMLP)를 사용하므로,
      joint_stiffness로는 관절을 고정할 수 없습니다.
      
      대신 step()을 오버라이드하여:
        (1) process_action() 전에 부상 calf joint의 action을 0으로 마스킹
        (2) physics sub-step마다 joint_pos/vel을 lock angle로 강제
      이 두 가지를 통해 관절을 물리적으로 고정합니다.
    """

    cfg: Go1LabEnvCfg

    def step(self, action: torch.Tensor):
        """환경 step - 부상 다리 action masking 후 물리 시뮬레이션 수행."""

        # ━━━ (1) Action Masking: physics loop 전에 부상 calf action을 0으로 강제 ━━━
        action = self._mask_peg_leg_action(action)

        # process actions (masked)
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # ━━━ (2) Physics loop with joint enforcement ━━━
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # ⭐ 부상 calf joint의 target을 lock angle로 강제 덮어쓰기
            self._enforce_peg_leg_joint_targets()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            # ⭐ physics 후에도 joint_pos/vel을 강제하여 다음 sub-step에서 올바른 상태로 시작
            self._enforce_peg_leg_joint_state()

        # post-step: 나머지는 부모 클래스와 동일
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # ⭐ Grace Period: 부상 환경은 처음 10스텝 동안 높이 종료 비활성화
        # 다리가 짧아진 직후 정책이 적응할 시간을 줍니다.
        grace_steps = int(os.getenv("GO1_PEG_GRACE_STEPS", "10"))
        if hasattr(self, "_peg_leg_index"):
            is_injured = self._peg_leg_index >= 0
            in_grace = self.episode_length_buf <= grace_steps
            grace_mask = is_injured & in_grace
            if grace_mask.any():
                # time_out은 유지하되, terminated(높이/접촉 등)만 억제
                self.reset_terminated[grace_mask] = False
                self.reset_buf[grace_mask] = self.reset_time_outs[grace_mask]

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Private helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _mask_peg_leg_action(self, action: torch.Tensor) -> torch.Tensor:
        """부상 calf joint의 action을 0으로 마스킹합니다.

        action=0이면 target = default_pos + 0 * scale = lock_angle이 되어,
        ActuatorNetMLP가 관절을 현재 위치(lock_angle)에 유지하는 토크를 출력합니다.
        """
        if not hasattr(self, "_peg_leg_index"):
            return action

        peg_idx = self._peg_leg_index  # (num_envs,) -1=정상, 0~3=부상
        is_injured = peg_idx >= 0
        if not is_injured.any():
            return action

        action = action.clone()  # 원본 수정 방지
        injured_envs = torch.where(is_injured)[0]
        # ⚠️ Go1 joint order is PER-TYPE ([..hips.., ..thighs.., ..calves..]),
        # NOT per-leg, so the calf action index is NOT leg_idx*3+2 — that froze a
        # DIFFERENT leg's joint (e.g. an RL injury masked FL_calf, killing a front
        # leg → the robot could not walk). Use the real calf joint index (resolved
        # by name in events.py), matching _enforce_peg_leg_joint_targets/_state.
        if hasattr(self, "_peg_leg_calf_joint_index"):
            calf_action_idx = self._peg_leg_calf_joint_index[injured_envs]
            valid = calf_action_idx >= 0
            action[injured_envs[valid], calf_action_idx[valid]] = 0.0
        else:
            action[injured_envs, peg_idx[injured_envs] * 3 + 2] = 0.0
        return action

    def _enforce_peg_leg_joint_targets(self) -> None:
        """apply_action() 후, write_data_to_sim() 전에 joint target을 강제합니다.

        action masking으로 target ≈ lock_angle이지만,
        혹시라도 action_manager가 다른 값을 설정했을 경우를 대비합니다.
        """
        if not hasattr(self, "_peg_leg_index"):
            return

        peg_idx = self._peg_leg_index
        is_injured = peg_idx >= 0
        if not is_injured.any():
            return

        robot = self.scene["robot"]
        injured_envs = torch.where(is_injured)[0]
        calf_joints = self._peg_leg_calf_joint_index[injured_envs]
        lock_angles = self._peg_leg_calf_lock_angle[injured_envs]

        # joint_pos_target을 lock angle로 강제
        if hasattr(robot.data, "joint_pos_target") and robot.data.joint_pos_target.ndim >= 2:
            robot.data.joint_pos_target[injured_envs, calf_joints] = lock_angles

    def _enforce_peg_leg_joint_state(self) -> None:
        """physics step 후, joint_pos/vel을 lock angle/0으로 강제합니다.

        ActuatorNetMLP는 완벽한 위치 추종을 보장하지 않으므로 (특히 외력이 작용할 때),
        매 sub-step 후 관절 상태를 직접 강제하여 rigid lock을 시뮬레이션합니다.
        """
        if not hasattr(self, "_peg_leg_index"):
            return

        peg_idx = self._peg_leg_index
        is_injured = peg_idx >= 0
        if not is_injured.any():
            return

        robot = self.scene["robot"]
        injured_envs = torch.where(is_injured)[0]
        calf_joints = self._peg_leg_calf_joint_index[injured_envs]
        lock_angles = self._peg_leg_calf_lock_angle[injured_envs]

        if hasattr(robot.data, "joint_pos") and robot.data.joint_pos.ndim >= 2:
            robot.data.joint_pos[injured_envs, calf_joints] = lock_angles
        if hasattr(robot.data, "joint_vel") and robot.data.joint_vel.ndim >= 2:
            robot.data.joint_vel[injured_envs, calf_joints] = 0.0
