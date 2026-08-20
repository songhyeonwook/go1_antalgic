# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 distillation + 보조 예측 헤드 (부목 길이 L̂ 단독).

왜 보조 헤드인가
----------------
P3-latent-001 분석(test/analyze_student.py)에서 student LSTM latent 는 L 을
~9 mm 수준으로만 인코딩했다. 원인은 능력이 아니라 유인이다: distillation
loss 는 teacher action 모사뿐이라 latent 가 L 을 정밀하게 인코딩할 이유가
없다. 여기서는 latent(256) 위에 선형 헤드를 얹어 privileged GT L 을 직접
지도해 표현을 개선한다.

손실
----
  total = behavior + λ · Σ_k MSE(head_k(latent), target_k_norm)   (부상 env 만)

target 은 rollout 에 저장된 privileged 관측에서 읽는다.
healthy env 는 L=0 (더미) 이므로 injured_flag 로 마스킹한다.

사용: agents/rsl_rl_ppo_cfg.py 의 DistillRunnerCfg 가
StudentTeacherRecurrentAux / DistillationAux 를 class_name 으로 지정한다.
"""

from __future__ import annotations

from .rls import RLS_L_PRIOR, RLS_L_SCALE

try:
    import torch.nn as nn

    from rsl_rl.algorithms import Distillation
    from rsl_rl.modules import StudentTeacherRecurrent

    _HAS_RSL = True
except Exception:  # pragma: no cover - rsl_rl absent outside training (e.g. list_envs)
    nn = None  # type: ignore[assignment]
    Distillation = object  # type: ignore[assignment, misc]
    StudentTeacherRecurrent = object  # type: ignore[assignment, misc]
    _HAS_RSL = False


class StudentTeacherRecurrentAux(StudentTeacherRecurrent):
    """StudentTeacherRecurrent + latent 선형 보조 헤드.

    act_inference() 가 마지막 memory 출력(latent)을 캐싱하고, aux_predict() 가
    그 latent 에서 aux_num_targets 차원(기본 1 = [L̂_norm])을 예측한다.
    """

    def __init__(self, obs, obs_groups, num_actions, aux_num_targets: int = 1, **kwargs):
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        rnn_hidden_dim = kwargs.get("rnn_hidden_dim", 256)
        self.aux_head = nn.Linear(rnn_hidden_dim, aux_num_targets)
        self._last_latent = None

    def act_inference(self, obs):
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        out_mem = self.memory_s(obs).squeeze(0)
        self._last_latent = out_mem
        return self.student(out_mem)

    def aux_predict(self):
        """직전 act_inference() 의 latent 에서 aux 타깃 예측 (정규화 단위)."""
        if self._last_latent is None:
            raise RuntimeError("aux_predict() must be called after act_inference().")
        return self.aux_head(self._last_latent)

    def load_state_dict(self, state_dict, strict=True):
        """구(2출력 [L̂, μ̂]) 체크포인트 호환: aux_head 를 현재 폭으로 잘라 로드.

        구 순서가 [L, μ] 였으므로 앞 행 슬라이스가 정확히 L 헤드를 보존한다.
        (μ 추정 제거 이후에도 P3-aux-001/002 등 과거 결과 재현 가능하게 유지)
        """
        w = state_dict.get("aux_head.weight")
        n = self.aux_head.out_features
        if w is not None and w.shape[0] > n:
            state_dict = dict(state_dict)
            state_dict["aux_head.weight"] = w[:n]
            state_dict["aux_head.bias"] = state_dict["aux_head.bias"][:n]
            print(f"[aux] 구 체크포인트 aux_head {w.shape[0]}→{n}출력으로 절단 로드 "
                  "(L 헤드 보존, μ 헤드 폐기)", flush=True)
        return super().load_state_dict(state_dict, strict=strict)


class DistillationAux(Distillation):
    """Distillation + 부상 env 마스킹된 보조 지도 손실.

    parent 의 update() 흐름(스텝 순회, gradient_length 누적, hidden 관리)을
    그대로 유지하고 스텝 손실에 λ·aux 만 더한다. gradient clip 은 parent 와
    동일하게 student MLP 에만 적용한다 (baseline 과의 비교 조건 유지).
    """

    def __init__(
        self,
        policy,
        aux_loss_coef: float = 0.5,
        aux_mask: dict | None = None,
        aux_targets: list[dict] | None = None,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.aux_loss_coef = float(aux_loss_coef)
        self.aux_mask = aux_mask or {"group": "privileged_obs", "index": 4}
        # 기본은 L 단독 — μ 는 antalgic 보행에서 비식별(연구 주장: μ 강건성).
        # 정규화 상수는 rls_estimate 관측 채널과 단일 소스(mdp/rls.py) 공유.
        self.aux_targets = aux_targets or [
            {"name": "splint_length", "group": "privileged_obs", "index": 5,
             "shift": RLS_L_PRIOR, "scale": RLS_L_SCALE},
        ]
        n_head = self.policy.aux_head.out_features
        if n_head != len(self.aux_targets):
            raise ValueError(
                f"aux_head({n_head}) 와 aux_targets({len(self.aux_targets)}) 차원 불일치 — "
                "policy.aux_num_targets 와 algorithm.aux_targets 를 맞추세요."
            )

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_aux = [0.0] * len(self.aux_targets)
        aux_cnt = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():

                # inference the student for gradient computation
                actions = self.policy.act_inference(obs)

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)
                mean_behavior_loss += behavior_loss.item()
                step_loss = behavior_loss

                # 보조 지도 손실 (부상 env 만 — healthy 는 L=0 더미)
                mask = obs[self.aux_mask["group"]][:, self.aux_mask["index"]] > 0.5
                if bool(mask.any()):
                    aux_pred = self.policy.aux_predict()
                    for j, spec in enumerate(self.aux_targets):
                        target = (
                            obs[spec["group"]][:, spec["index"]] - spec["shift"]
                        ) / spec["scale"]
                        aux_loss = nn.functional.mse_loss(
                            aux_pred[mask, j], target[mask]
                        )
                        step_loss = step_loss + self.aux_loss_coef * aux_loss
                        mean_aux[j] += aux_loss.item()
                    aux_cnt += 1

                # total loss
                loss = loss + step_loss
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.policy.student.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}
        for j, spec in enumerate(self.aux_targets):
            loss_dict[f"aux_{spec['name']}"] = mean_aux[j] / max(aux_cnt, 1)

        return loss_dict


def _install() -> None:
    """Expose classes in the runner namespace so eval(class_name) resolves them."""
    try:
        import rsl_rl.runners.distillation_runner as _drn

        _drn.StudentTeacherRecurrentAux = StudentTeacherRecurrentAux
        _drn.DistillationAux = DistillationAux
    except Exception:  # pragma: no cover - runner not importable outside training
        pass


if _HAS_RSL:
    _install()
