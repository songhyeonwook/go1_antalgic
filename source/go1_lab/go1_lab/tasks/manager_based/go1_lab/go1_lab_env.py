# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go1 Lab 환경 - explicit actuator 호환 Peg-Leg Action Masking."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .go1_lab_env_cfg import Go1LabEnvCfg


class Go1LabEnv(ManagerBasedRLEnv):
    """Go1 Lab 환경 (ManagerBasedRLEnv 확장).

    step()을 오버라이드하여 부목(peg-leg) 고정을 구현합니다:
      (1) process_action() 전에 부상 calf joint의 action을 0으로 마스킹
          (last_action 관측 일관성 확보용 — 고정 자체는 (2)가 담당)
      (2) 매 physics sub-step에서 apply_action() 직후 부상 calf의
          joint_pos_target 을 lock angle 로 덮어쓰기 → PD 가 오차를 보고 붙잡음.
    리셋 시 초기 배치는 randomize_peg_leg_actuation 이 write_joint_state_to_sim
    으로 수행합니다.
    """

    cfg: Go1LabEnvCfg

    def step(self, action: torch.Tensor):
        """환경 step - 부상 다리 action masking 후 물리 시뮬레이션 수행."""

        # ━━━ (1) Action Masking: physics loop 전에 부상 calf action을 0으로 강제 ━━━
        if self.cfg.use_peg_leg_action_mask:
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

        # post-step: 나머지는 부모 클래스와 동일
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # ⭐ Grace Period: 부상 환경은 처음 10스텝 동안 높이 종료 비활성화
        # 다리가 짧아진 직후 정책이 적응할 시간을 줍니다.
        if hasattr(self, "_peg_leg_index"):
            is_injured = self._peg_leg_index >= 0
            in_grace = self.episode_length_buf <= self.cfg.grace_steps
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
        """잠긴 calf joint의 action을 0으로 마스킹합니다.

        주 목적은 last_action 관측 일관성입니다 (고정 관절의 action 은 항상 0).
        주의: action offset 은 액션 항 init 시 default_joint_pos 를 clone 한 값이라
        action=0 이 lock 목표를 만들지는 않습니다 — 실제 고정은
        _enforce_peg_leg_joint_targets 의 target 덮어쓰기 + PD 가 담당합니다.

        ⚠️ 인덱스 공간: 부목 관절 4개가 추가되어 num_joints(16) ≠ action_dim(12)
        이므로 반드시 action-index 버퍼(_peg_leg_calf_action_index)를 씁니다.
        joint index 를 action 버퍼에 그대로 쓰면 엉뚱한 관절이 마스킹됩니다.
        """
        if not hasattr(self, "_peg_leg_calf_action_index"):
            return action

        locked = self._peg_leg_lock_active
        if not locked.any():
            return action

        action = action.clone()  # 원본 수정 방지
        locked_envs = torch.where(locked)[0]
        calf_action_idx = self._peg_leg_calf_action_index[locked_envs]
        valid = (calf_action_idx >= 0) & (calf_action_idx < action.shape[1])
        if valid.any():
            action[locked_envs[valid], calf_action_idx[valid]] = 0.0
        return action

    def _enforce_peg_leg_joint_targets(self) -> None:
        """apply_action() 후, write_data_to_sim() 전에 부상 calf 의 target 을 lock angle 로 강제합니다.

        이것이 부목 고정의 주 경로입니다: PD 가 target 오차를 보고 관절을 붙잡습니다.
        ⚠️ 측정값(robot.data.joint_pos/joint_vel)에 대입하는 방식은 금지 — 그 버퍼는
        PhysX 읽기 캐시라 sim 에 전달되지 않고, 액추에이터가 바로 그 캐시로 PD 오차를
        계산하므로 스푸핑되어 홀딩 토크가 0 이 됩니다 (실측: 부상 calf 0.00 Nm vs
        정상 4.74 Nm, 관절이 lock 에서 평균 0.81 rad 이탈해 무릎이 접혔음).
        """
        if not hasattr(self, "_peg_leg_lock_active"):
            return

        locked = self._peg_leg_lock_active & (self._peg_leg_calf_joint_index >= 0)
        if not locked.any():
            return

        robot = self.scene["robot"]
        locked_envs = torch.where(locked)[0]
        calf_joints = self._peg_leg_calf_joint_index[locked_envs]
        lock_angles = self._peg_leg_calf_lock_angle[locked_envs]

        # joint_pos_target을 lock angle로 강제
        if hasattr(robot.data, "joint_pos_target") and robot.data.joint_pos_target.ndim >= 2:
            robot.data.joint_pos_target[locked_envs, calf_joints] = lock_angles
