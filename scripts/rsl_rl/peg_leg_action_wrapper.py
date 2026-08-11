from __future__ import annotations

import gymnasium as gym
import torch


class PegLegActionMaskWrapper(gym.Wrapper):
    """고장 다리 calf action을 0으로 강제하는 래퍼."""

    def _mask_actions(self, actions: torch.Tensor) -> torch.Tensor:
        base_env = self.unwrapped
        if not isinstance(actions, torch.Tensor) or actions.ndim != 2:
            return actions
        if not hasattr(base_env, "_peg_leg_calf_joint_index"):
            return actions
        joint_ids = base_env._peg_leg_calf_joint_index
        if joint_ids is None:
            return actions

        masked = actions.clone()
        num_actions = masked.shape[1]
        valid_env = (joint_ids >= 0) & (joint_ids < num_actions)
        if not torch.any(valid_env):
            return masked

        env_rows = torch.nonzero(valid_env, as_tuple=False).squeeze(-1)
        calf_cols = joint_ids[env_rows].long()
        masked[env_rows, calf_cols] = 0.0
        return masked

    def step(self, action):
        return self.env.step(self._mask_actions(action))
