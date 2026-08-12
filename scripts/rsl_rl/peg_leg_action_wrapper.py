from __future__ import annotations

import gymnasium as gym
import torch


class PegLegActionMaskWrapper(gym.Wrapper):
    """잠긴 calf action을 0으로 강제하는 래퍼 (play/export 스크립트용).

    학습 경로에서는 Go1LabEnv.step() 이 내부에서 같은 마스킹을 수행하므로 이
    래퍼는 필요 없지만, 평가/내보내기 스크립트가 명시적으로 감쌀 때 사용합니다.

    ⚠️ 부목 관절 4개가 추가되어 num_joints(16) ≠ action_dim(12) 이므로 반드시
    action-index 버퍼(_peg_leg_calf_action_index)를 씁니다. joint index 를 action
    버퍼 인덱스로 쓰면 엉뚱한 관절이 마스킹됩니다.
    """

    def _mask_actions(self, actions: torch.Tensor) -> torch.Tensor:
        base_env = self.unwrapped
        if not isinstance(actions, torch.Tensor) or actions.ndim != 2:
            return actions
        action_ids = getattr(base_env, "_peg_leg_calf_action_index", None)
        lock_active = getattr(base_env, "_peg_leg_lock_active", None)
        if action_ids is None or lock_active is None:
            return actions

        masked = actions.clone()
        num_actions = masked.shape[1]
        valid_env = lock_active & (action_ids >= 0) & (action_ids < num_actions)
        if not torch.any(valid_env):
            return masked

        env_rows = torch.nonzero(valid_env, as_tuple=False).squeeze(-1)
        calf_cols = action_ids[env_rows].long()
        masked[env_rows, calf_cols] = 0.0
        return masked

    def step(self, action):
        return self.env.step(self._mask_actions(action))
