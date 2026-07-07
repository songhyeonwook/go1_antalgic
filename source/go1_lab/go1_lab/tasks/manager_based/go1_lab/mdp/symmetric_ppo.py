# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LSTM-compatible left/right mirror augmentation for RSL-RL PPO.

Why a custom algorithm
-----------------------
RSL-RL's built-in ``symmetry_cfg`` data augmentation doubles the minibatch along
its first dimension. For a *recurrent* policy the recurrent minibatch is laid out
as ``[time, trajectory, feature]`` and is accompanied by ``masks`` and per-chunk
``hidden_states`` that are NOT doubled — so the augmentation corrupts the
sequence/hidden bookkeeping and feeds a malformed tensor into the LSTM. That is
why the recurrent runs disabled it.

This module instead applies the augmentation at the **rollout-storage** level,
*before* the update. The storage is doubled along the environment dimension: the
appended half holds the exact left/right mirror of every stored transition. The
update then runs through RSL-RL's own (correct) recurrent minibatch generator,
which produces perfectly consistent obs / masks / hidden / action batches for the
mirrored trajectories. No symmetry_cfg is used, so none of the broken recurrent
symmetry code is touched.

Faithfulness
------------
This is a *structural* augmentation only — the reward (paper eq.3: task − energy
− pain) is never modified. It enforces the paper's stated
``subject to normal-gait mirror symmetry`` constraint by making the policy
left/right equivariant, so the antalgic response still *emerges* from the
pain/energy objective while left- and right-injury cases become mirror images.

What is mirrored
----------------
- observations["policy"]      : proprioception (48) + height-scan grid (187)
- observations["privileged_obs"]: injury index FL↔FR / RL↔RR (splint, friction kept)
- actions, mu                 : 12-joint L/R swap + hip-abduction sign flip
- sigma                       : 12-joint L/R swap (no sign flip)
- rewards, dones, values, returns, advantages, log-probs : mirror-invariant → copied
- LSTM hidden states          : repeated (mirror of the latent is undefined; the
                                per-step obs→action equivariance constraint holds
                                given identical memory, exact at episode/zero starts)
"""

from __future__ import annotations

import copy
import os

import torch
from tensordict import TensorDict

try:
    from rsl_rl.algorithms import PPO

    _HAS_RSL = True
except Exception:  # pragma: no cover - rsl_rl absent outside training (e.g. list_envs)
    PPO = object  # type: ignore[assignment, misc]
    _HAS_RSL = False

from . import mirror


def _mirror_enabled() -> bool:
    return os.getenv("GO1_MIRROR_AUG", "1").strip().lower() in {"1", "true", "yes", "on"}


class SymmetricPPO(PPO):
    """PPO that left/right mirror-augments the rollout storage before each update.

    Drop-in for ``PPO``: select it by setting ``class_name="SymmetricPPO"`` on the
    RSL-RL algorithm config. Disable at runtime with ``GO1_MIRROR_AUG=0`` (then it
    behaves exactly like stock PPO).
    """

    def update(self):  # noqa: D401
        storage = getattr(self, "storage", None)
        if not _mirror_enabled() or storage is None or storage.step == 0:
            return super().update()

        original = storage
        self.storage = self._build_mirrored_storage(original)
        if not getattr(self, "_mirror_logged", False):
            self._mirror_logged = True
            keys = list(original.observations.keys())
            print(
                f"[SymmetricPPO] mirror augmentation ACTIVE: "
                f"num_envs {original.num_envs} -> {self.storage.num_envs}; obs groups {keys}",
                flush=True,
            )
        try:
            out = super().update()  # trains on the doubled storage
        finally:
            self.storage = original
            original.clear()  # reset step=0 so the next rollout fills fresh
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_mirrored_storage(self, st):
        """Return a shallow clone of ``st`` with every field doubled [orig | mirror]."""
        d = copy.copy(st)
        d.num_envs = st.num_envs * 2

        # observations (TensorDict, batch_size [T, N])
        d.observations = self._mirror_obs_td(st.observations)

        # action-like fields -> mirror
        d.actions = torch.cat([st.actions, mirror.mirror_action(st.actions)], dim=1)
        if getattr(st, "mu", None) is not None:
            d.mu = torch.cat([st.mu, mirror.mirror_action(st.mu)], dim=1)
        if getattr(st, "sigma", None) is not None:
            d.sigma = torch.cat([st.sigma, mirror.mirror_sigma(st.sigma)], dim=1)

        # mirror-invariant scalar fields -> copy
        for field in ("rewards", "dones", "values", "returns", "advantages", "actions_log_prob"):
            t = getattr(st, field, None)
            if t is not None:
                setattr(d, field, torch.cat([t, t.clone()], dim=1))

        # recurrent hidden states: list of [T, num_layers, N, H] -> env dim = 2
        if getattr(st, "saved_hidden_states_a", None) is not None:
            d.saved_hidden_states_a = [torch.cat([h, h.clone()], dim=2) for h in st.saved_hidden_states_a]
            d.saved_hidden_states_c = [torch.cat([h, h.clone()], dim=2) for h in st.saved_hidden_states_c]

        return d

    @staticmethod
    def _mirror_obs_td(obs: TensorDict) -> TensorDict:
        T, N = int(obs.shape[0]), int(obs.shape[1])
        parts = {}
        for key in obs.keys():
            x = obs[key]
            if key == "policy":
                mx = mirror.mirror_policy_obs(x)
            elif key in ("privileged_obs", "privileged"):
                mx = mirror.mirror_privileged_obs(x)
            else:
                # Unknown group (e.g. critic-only); copy verbatim to stay safe.
                mx = x.clone()
            parts[key] = torch.cat([x, mx], dim=1)
        return TensorDict(parts, batch_size=[T, N * 2], device=obs.device)


def _install() -> None:
    """Expose SymmetricPPO in the runner namespace so eval(class_name) resolves it."""
    try:
        import rsl_rl.runners.on_policy_runner as _opr

        _opr.SymmetricPPO = SymmetricPPO
    except Exception:  # pragma: no cover - runner not importable outside training
        pass


if _HAS_RSL:
    _install()
