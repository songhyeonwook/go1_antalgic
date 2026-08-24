# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 distillation + 보조 예측 헤드 (부목 길이 L̂ + 부상 다리 분류).


손실
----
  total = behavior
        + λ_reg · Σ_k MSE(head_k(latent), target_k_norm)      (부상 env 만)
        + λ_cls · CE(cls_head(latent), 부상 다리 클래스)        (전 env)


"""

from __future__ import annotations

from .rls import RLS_L_PRIOR, RLS_L_SCALE

try:
    import torch
    import torch.nn as nn

    from rsl_rl.algorithms import Distillation
    from rsl_rl.modules import StudentTeacherRecurrent

    _HAS_RSL = True
except Exception:  # pragma: no cover - rsl_rl absent outside training (e.g. list_envs)
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Distillation = object  # type: ignore[assignment, misc]
    StudentTeacherRecurrent = object  # type: ignore[assignment, misc]
    _HAS_RSL = False


class StudentTeacherRecurrentAux(StudentTeacherRecurrent):
    """StudentTeacherRecurrent + latent 선형 보조 헤드.
    """

    def __init__(self, obs, obs_groups, num_actions, aux_num_targets: int = 1,
                 aux_num_classes: int = 5, **kwargs):
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        rnn_hidden_dim = kwargs.get("rnn_hidden_dim", 256)
        self.aux_head = nn.Linear(rnn_hidden_dim, aux_num_targets)
        self.aux_cls_head = (
            nn.Linear(rnn_hidden_dim, aux_num_classes) if aux_num_classes > 0 else None
        )
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

    def aux_predict_cls(self):
        """직전 act_inference() latent 에서 부상 다리 분류 logits (softmax 이전).

        확률이 필요하면 호출부에서 softmax 를 취한다 — 학습 손실은
        cross_entropy 가 logits 를 직접 받으므로 여기서 취하지 않는다.
        """
        if self.aux_cls_head is None:
            raise RuntimeError("aux_cls_head 비활성 상태입니다 (aux_num_classes=0).")
        if self._last_latent is None:
            raise RuntimeError("aux_predict_cls() must be called after act_inference().")
        return self.aux_cls_head(self._last_latent)

    def load_state_dict(self, state_dict, strict=True):
  
        w = state_dict.get("aux_head.weight")
        n = self.aux_head.out_features
        if w is not None and w.shape[0] > n:
            state_dict = dict(state_dict)
            state_dict["aux_head.weight"] = w[:n]
            state_dict["aux_head.bias"] = state_dict["aux_head.bias"][:n]
            print(f"[aux] 구 체크포인트 aux_head {w.shape[0]}→{n}출력으로 절단 로드 "
                  "(L 헤드 보존, μ 헤드 폐기)", flush=True)
        # 분류 헤드가 없던 구 체크포인트: 현재 초기값을 채워 strict 로드를 통과시킨다
        if self.aux_cls_head is not None and "aux_cls_head.weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["aux_cls_head.weight"] = self.aux_cls_head.weight.detach().clone()
            state_dict["aux_cls_head.bias"] = self.aux_cls_head.bias.detach().clone()
            print("[aux] 구 체크포인트에 aux_cls_head 없음 — 무작위 초기값으로 시작",
                  flush=True)
        return super().load_state_dict(state_dict, strict=strict)


class DistillationAux(Distillation):
    """Distillation + 보조 지도 손실 (L 회귀 + 부상 다리 분류).

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
        aux_cls_loss_coef: float = 0.5,
        aux_cls: dict | None = None,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.aux_loss_coef = float(aux_loss_coef)
        self.aux_mask = aux_mask or {"group": "privileged_obs", "index": 4}
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

        # ── 부상 다리 분류 헤드 ──────────────────────────────────────────
        # privileged one-hot 은 [FL, FR, RL, RR, injured_flag] 이므로
        # 라벨 = flag 면 argmax(one_hot[0:4]), 아니면 normal_class.
        self.aux_cls_loss_coef = float(aux_cls_loss_coef)
        self.aux_cls = aux_cls if aux_cls is not None else {
            "group": "privileged_obs", "leg_start": 0, "num_legs": 4,
            "flag_index": 4, "normal_class": 4, "masked": False,
        }
        cls_head = getattr(self.policy, "aux_cls_head", None)
        self._cls_on = cls_head is not None and bool(self.aux_cls)
        if self._cls_on:
            need = (int(self.aux_cls["num_legs"]) if self.aux_cls.get("masked")
                    else int(self.aux_cls["normal_class"]) + 1)
            if cls_head.out_features != need:
                raise ValueError(
                    f"aux_cls_head({cls_head.out_features}) 와 aux_cls 설정이 요구하는 "
                    f"클래스 수({need}) 불일치 — policy.aux_num_classes 를 맞추세요 "
                    f"(masked={bool(self.aux_cls.get('masked'))})."
                )

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_aux = [0.0] * len(self.aux_targets)
        aux_cnt = 0
        mean_cls = 0.0
        mean_cls_acc = 0.0
        cls_cnt = 0
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

                # 부상 다리 분류 손실 (기본: 정상 클래스를 포함해 전 env 사용)
                if self._cls_on:
                    spec = self.aux_cls
                    g = obs[spec["group"]]
                    ls, nl = int(spec["leg_start"]), int(spec["num_legs"])
                    flag = g[:, int(spec["flag_index"])] > 0.5
                    leg = g[:, ls:ls + nl].argmax(dim=-1)
                    if spec.get("masked"):
                        sel, labels = flag, leg
                    else:
                        sel = torch.ones_like(flag)
                        labels = torch.where(
                            flag, leg, torch.full_like(leg, int(spec["normal_class"]))
                        )
                    if bool(sel.any()):
                        logits = self.policy.aux_predict_cls()
                        cls_loss = nn.functional.cross_entropy(logits[sel], labels[sel])
                        step_loss = step_loss + self.aux_cls_loss_coef * cls_loss
                        mean_cls += cls_loss.item()
                        mean_cls_acc += (
                            logits[sel].argmax(dim=-1) == labels[sel]
                        ).float().mean().item()
                        cls_cnt += 1

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
        if self._cls_on and cls_cnt:
            loss_dict["aux_injured_leg"] = mean_cls / cls_cnt
            loss_dict["aux_injured_leg_acc"] = mean_cls_acc / cls_cnt

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
