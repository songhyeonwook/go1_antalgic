# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 distillation + 보조 예측 헤드 (splint 파라미터 [L̂, μ̂]).

왜 보조 헤드인가
----------------
P3-latent-001 분석(test/analyze_student.py)에서 student LSTM latent 는
L 을 ~9 mm 수준으로만, μ 는 전혀(R²=0.05) 인코딩하지 않았다. 원인은 능력이
아니라 유인이다: distillation loss 는 teacher action 모사뿐이고, teacher 의
μ 의존 성분은 student 입장에서 예측 불가능한 노이즈라 평균화하는 것이 loss
최적이다. 여기서는 latent(256) 위에 선형 헤드를 얹어 privileged GT [L, μ] 를
직접 지도해 latent 가 두 파라미터를 인코딩할 '이유'를 만든다.

  - 관측 차원 불변 → phase 1/2 체크포인트 호환 유지
  - 배포 시 aux_predict() 로 [L̂_norm, μ̂_norm] 을 바로 읽음 (별도 추정기 불필요)
  - 헤드 gradient 가 LSTM 까지 흐르므로 latent 표현 자체가 개선됨

손실
----
  total = behavior + λ · Σ_k MSE(head_k(latent), target_k_norm)   (부상 env 만)

target 은 rollout 에 저장된 privileged 관측에서 읽는다. μ 채널은 학습 환경의
mu_noise_std 노이즈가 얹힌 값이라 (unbiased) label noise 로 작용한다 — student
가 도달 가능한 추정 정확도 이상으로 과신하지 않게 하는 의도된 설계.
healthy env 는 L=μ=0 (더미) 이므로 injured_flag 로 마스킹한다.

사용: agents/rsl_rl_ppo_cfg.py 의 DistillRunnerCfg 가
StudentTeacherRecurrentAux / DistillationAux 를 class_name 으로 지정한다.
"""

from __future__ import annotations

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
    그 latent 에서 [L̂_norm, μ̂_norm] 등 aux_num_targets 차원을 예측한다.
    """

    def __init__(self, obs, obs_groups, num_actions, aux_num_targets: int = 2, **kwargs):
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
        self.aux_targets = aux_targets or [
            {"name": "splint_length", "group": "privileged_obs", "index": 5,
             "shift": 0.39, "scale": 0.06},
            {"name": "foot_friction", "group": "privileged_obs", "index": 6,
             "shift": 1.0, "scale": 0.5},
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

                # 보조 지도 손실 (부상 env 만 — healthy 는 L=μ=0 더미)
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
