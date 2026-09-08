

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3: Teacher(Phase-2 MLP, 동결) -> Student(LSTM) distillation + 보조 예측 헤드.

Student
  obs(49) -> LSTM(256) -> h_t -+- policy MLP [512,256,128] -> action(12)   MSE vs teacher action
                               +- splint_head Linear(256->1) -> L_hat       MSE vs GT L (부상 env 만)
                               +- vel_head    Linear(256->3) -> v_hat       MSE vs GT base_lin_vel (전 env)
Teacher
  [obs(49) | privileged(4)] = 53 -> MLP [512,256,128] -> action(12)   (rollout 시 no_grad 로 1회 평가)

privileged_obs 레이아웃은 go1_lab_env_cfg.Go1LabPrivilegedObsCfg 의 term 순서와 같다:
  [0] 부목 길이 L [m] (정상 = 0 sentinel)    [1:4] base_lin_vel [m/s]



"""


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.modules import StudentTeacherRecurrent
from rsl_rl.runners import DistillationRunner
from rsl_rl.networks import MLP

PRIV_GROUP = "privileged_obs"
PRIV_DIM = 4
PRIV_L_IDX = 0
PRIV_VEL = slice(1, 4)


def _as_vec(value, n: int, name: str) -> torch.Tensor:
    """스칼라(broadcast) 또는 길이 n 리스트 → float32 텐서 (n,)."""
    if isinstance(value, (int, float)):
        return torch.full((n,), float(value))
    t = torch.as_tensor([float(v) for v in value], dtype=torch.float32)
    if t.numel() != n:
        raise ValueError(f"mse_norm.{name} 길이 {t.numel()} != {n}")
    return t


class Phase3StudentTeacher(StudentTeacherRecurrent):
    """StudentTeacherRecurrent + LSTM latent 위의 선형 보조 헤드 2개."""

    def __init__(self, obs: TensorDict, obs_groups: dict, num_actions: int,
                 rnn_hidden_dim: int = 256, teacher_hidden_dims=(256, 256, 256), activation: str = "elu",
                 mse_norm_enable: bool = False,
                 action_mean=0.0, action_pstd: float = 1.0,
                 splint_mean: float = 0.0, splint_std: float = 1.0,
                 vel_mean=0.0, vel_pstd: float = 1.0,
                 **kwargs):

        
        super().__init__(obs, obs_groups, num_actions, rnn_hidden_dim=rnn_hidden_dim,
                         teacher_hidden_dims=teacher_hidden_dims, activation=activation, **kwargs)
        
        # rsl_rl 3.1.2 의 StudentTeacherRecurrent 는 teacher MLP 입력을 항상 rnn_hidden_dim 으로 만든다
        # (teacher_recurrent=False 여도). MLP teacher 는 관측 53-dim 을 직접 받아야 하므로 다시 만든다.
        if not self.teacher_recurrent:
            num_teacher_obs = sum(int(obs[g].shape[-1]) for g in obs_groups["teacher"])
            self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
            self.teacher.eval()
            
        priv_dim = int(obs[PRIV_GROUP].shape[-1])
        if priv_dim != PRIV_DIM:
            raise ValueError(
                f"{PRIV_GROUP} 차원 {priv_dim} != {PRIV_DIM}. "
                "Go1LabPrivilegedObsCfg 를 바꿨다면 PRIV_* 상수를 함께 갱신하세요."
            )
        self.splint_head = nn.Linear(rnn_hidden_dim, 1)
        self.vel_head = nn.Linear(rnn_hidden_dim, 3)
        self._latent: torch.Tensor | None = None


         # ── 출력 정규화 상수 ────────────────────────────────────────────────────
        self.mse_norm = bool(mse_norm_enable)
        if self.mse_norm:
            for name, s in (("action_pstd", action_pstd), ("splint_std", splint_std), ("vel_pstd", vel_pstd)):
                if not float(s) > 0.0:
                    raise ValueError(f"mse_norm.{name} 는 0 보다 커야 합니다: {s}")
            a_mean = _as_vec(action_mean, num_actions, "action_mean")
            v_mean = _as_vec(vel_mean, 3, "vel_mean")
            a_std, L_mean, L_std, v_std = float(action_pstd), float(splint_mean), float(splint_std), float(vel_pstd)
        else:
            a_mean, v_mean = torch.zeros(num_actions), torch.zeros(3)
            a_std, L_mean, L_std, v_std = 1.0, 0.0, 1.0, 1.0

        # 정규화값 모델에 저장
        self.register_buffer("norm_action_mean", a_mean)                 # (12,)
        self.register_buffer("norm_action_std", torch.tensor(a_std))     # 스칼라 (12관절 pooled)
        self.register_buffer("norm_splint_mean", torch.tensor(L_mean))
        self.register_buffer("norm_splint_std", torch.tensor(L_std))
        self.register_buffer("norm_vel_mean", v_mean)                     # (3,)
        self.register_buffer("norm_vel_std", torch.tensor(v_std))         # 스칼라 (3축 pooled)

    def output_norm_summary(self) -> str:
        if not self.mse_norm:
            return "disabled (항등: mean 0, std 1)"
        r4 = lambda t: [round(x, 4) for x in t.tolist()]
        return (f"enabled | action mean {r4(self.norm_action_mean)} pstd {self.norm_action_std.item():.4f}"
                f" | splint mean {self.norm_splint_mean.item():.4f} std {self.norm_splint_std.item():.4f}"
                f" | vel mean {r4(self.norm_vel_mean)} pstd {self.norm_vel_std.item():.4f}")

    def _student_latent(self, obs: TensorDict) -> torch.Tensor:
        x = self.student_obs_normalizer(self.get_student_obs(obs))
        self._latent = self.memory_s(x).squeeze(0)          # (N, rnn_hidden_dim) LSTM
        return self._latent

    # forward
    def act(self, obs: TensorDict) -> torch.Tensor:          # rollout: 노이즈 샘플
        self.update_distribution(self._student_latent(obs))
        return self.distribution.sample() # 실제 환경에 넣을 action을 샘플링

    def act_inference(self, obs: TensorDict) -> torch.Tensor:  # 배포 / 평가
        return self.student(self._student_latent(obs)) # MLP

      # ── 출력 정규화: 학습은 target 을 z 공간으로, 평가는 헤드 출력을 물리 단위로 ──
    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.norm_action_mean) / self.norm_action_std
    
    def normalize_splint(self, L: torch.Tensor) -> torch.Tensor:
        return (L - self.norm_splint_mean) / self.norm_splint_std

    def normalize_vel(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self.norm_vel_mean) / self.norm_vel_std

    def act_with_aux(self, obs: TensorDict):
        """학습용. forward 한 번으로 (action, L_hat[정규화], v_hat[m/s]) 를 모두 반환."""
        h = self._student_latent(obs)
        return self.student(h), self.splint_head(h).squeeze(-1), self.vel_head(h)

    def aux_inference(self):
        """분석용. 직전 latent 의 헤드 출력을 역정규화해 (L_hat [m], v_hat [m/s]) 반환. mse_norm 비활성이면 항등."""
        if self._latent is None:
            raise RuntimeError("act()/act_inference() 이후에 호출하세요.")
        L_hat = self.splint_head(self._latent).squeeze(-1) * self.norm_splint_std + self.norm_splint_mean
        v_hat = self.vel_head(self._latent) * self.norm_vel_std + self.norm_vel_mean
        return L_hat, v_hat



class Phase3Distillation(Distillation):
    """Distillation(behavior cloning) + splint/velocity 헤드 MSE."""

    def __init__(self, policy, splint_loss_coef: float = 0.5, vel_loss_coef: float = 1.0,
                 splint_length_range=(0.33, 0.45), **kwargs):
        super().__init__(policy, **kwargs)
        lo, hi = (float(v) for v in splint_length_range)
        if not hi > lo:
            raise ValueError(f"splint_length_range 가 잘못됨: {splint_length_range}")
        self.splint_loss_coef = float(splint_loss_coef)
        self.vel_loss_coef = float(vel_loss_coef)

        # #min-max normalization
        # # L 을 [-1, 1] 로 정규화: (L - mid) / half
        # self.L_mid = 0.5 * (lo + hi) # 0.39
        # self.L_half = 0.5 * (hi - lo) # 0.06

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        keys = ("behavior", "splint_length", "base_lin_vel")
        stats = {k: torch.zeros((), device=self.device) for k in keys}
        loss, cnt = 0, 0

        for _ in range(self.num_learning_epochs):
            """
            24-step rollout 전체의 계산 그래프를 유지하는 것은 무거우므로 rollout에서는 그래프를 버리고,
            학습할 때 필요한 데이터를 다시 forward한다. 이때 gradient_length만큼만 시간축 계산 그래프를 연결하여
            truncated BPTT를 수행한다. 따라서 full 24-step BPTT와 완전히 같지는 않지만, 
            메모리를 줄이면서 일정 길이의 temporal dependency를 학습하는 절충 방식이다.
            """
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states() # gradient length만 backpropagation하기 위함(gradient length=4라면 h1~h3 -- h4 이 둘의 계산 그래프를 끊는다.)
            for obs, _, privileged_actions, dones in self.storage.generator(): # obs, student action, teacher action, dones
                """
                저장된 Student action을 써도 숫자 비교 자체는 가능하지만, gradient 계산을 위해 어차피 Student를 다시 forward해야 하므로 그때 나온 최신 action을 쓰자
                """

                actions, splint_z, vel_z = self.policy.act_with_aux(obs)
                priv = obs[PRIV_GROUP]
                pol = self.policy
                

                # (1) behavior cloning: (a - μ)/σ 공간에서 비교. μ 는 차이에서 상쇄되고 σ 로 나눈 효과만 남는다.
                #     loss_fn 에 정규화한 입력을 넣으므로 loss_type 이 huber 여도 같은 의미다.
                behavior_loss = self.loss_fn(pol.normalize_action(actions), pol.normalize_action(privileged_actions))

                # (2) 속도 헤드: 전 env. target 을 같은 z 공간으로
                vel_loss = nn.functional.mse_loss(vel_z, pol.normalize_vel(priv[:, PRIV_VEL]))


                # (3) 부목 길이 헤드: 마스크는 반드시 raw L 로 판정한다.
                #     정규화 후에는 정상 env 의 sentinel 0 이 (0-μ_L)/σ_L ≈ -11 로 가서 "> 0" 판정이 깨진다.
                L_raw = priv[:, PRIV_L_IDX]
                injured = (L_raw > 0).float()               # 부상 env 만 1
                n_inj = injured.sum().clamp(min=1.0)        # 분모 0 방지
                err = splint_z - pol.normalize_splint(L_raw)
                splint_loss = (err.square() * injured).sum() / n_inj

                # # (1) behavior cloning
                # behavior_loss = self.loss_fn(actions, privileged_actions)

                # # (2) 속도 헤드: 전 env
                # priv = obs[PRIV_GROUP]
                # vel_loss = nn.functional.mse_loss(vel_pred, priv[:, PRIV_VEL])

                # # (3) 부목 길이 헤드: 부상 env 만 (정상은 L=0 sentinel). 마스크 평균이라 분기/동기화 없음
                # injured = (priv[:, PRIV_L_IDX] > 0).float() # privilged information를 통해 각 환경의 부목길이가 0인지 이상인지 확인 
                # n_inj = injured.sum().clamp(min=1.0) # 몇 개의 env가 부목길이가 있는지 (분모에해당 함으로 0이되지않도록 최소 1) 
                # err = splint_pred - (priv[:, PRIV_L_IDX])
                # splint_loss = (err.square() * injured).sum() / n_inj # 부상 env에 대해서만 계산하기 위함 + 환경에 대한 부상 로스를 평균냄

                step_loss = (behavior_loss
                             + self.splint_loss_coef * splint_loss
                             + self.vel_loss_coef * vel_loss)
                loss = loss + step_loss # gradient_length 를 채우기 위함
                cnt += 1

                stats["behavior"] += behavior_loss.detach()
                stats["base_lin_vel"] += vel_loss.detach()
                stats["splint_length"] += splint_loss.detach()

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad() # gradient 제거
                    loss.backward() #  backpropa
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm: # g
                        # teacher 는 grad 가 None 이라 자동 제외. LSTM/MLP/헤드 전부 clip.
                        """
                        Gradient clipping은 여러 step의 loss가 누적되면서 gradient가 과도하게 커질 경우, 
                        한 번의 update에서 weight가 지나치게 크게 변하는 것을 막기 위해 gradient의 최대 크기를 제한하는 역할
                        """
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step() # weight 갱신
                    # 50 timestep의 backward가 끝났으므로 현재 LSTM state와 이전 50-step graph의 연결을 끊는다.
                    self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1)) # 끝난 에피소드에 대하여 h, c를 0으로
                self.policy.detach_hidden_states(dones.view(-1)) # done된 env만 detach

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states() # 현재 LSTM state 저장
        self.policy.detach_hidden_states() # 마지막 계산그래프 연결제거
        return {k: (v / cnt).item() for k, v in stats.items()} # 학습 전체 과정의 평균 loss


    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
            if num_transitions_per_env % self.gradient_length != 0:
                raise ValueError(
                    f"num_steps_per_env({num_transitions_per_env}) 는 "
                    f"gradient_length({self.gradient_length}) 의 배수여야 합니다. "
                    "아니면 잔여 step 의 loss 가 버려지고 backward 창이 epoch 경계를 넘습니다."
                )
            super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
    
# loss / optimizer / update 방법
class Phase3DistillationRunner(DistillationRunner):
    """DistillationRunner 는 class_name 을 rsl_rl 모듈 네임스페이스에서 eval() 하므로
    이 파일의 클래스를 찾지 못한다. _construct_algorithm 만 덮어써 직접 조립한다."""

    def _construct_algorithm(self, obs: TensorDict) -> Phase3Distillation:
        policy_cfg = {k: v for k, v in self.policy_cfg.items() if k != "class_name"}
        alg_cfg = {k: v for k, v in self.alg_cfg.items() if k != "class_name"}

        # Student/Teacher Network를 생성
        policy = Phase3StudentTeacher(
            obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg
        ).to(self.device) # Model

        # 학습 코드 신경망을 어떻게 학습할지
        alg = Phase3Distillation(
            policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **alg_cfg
        )
        alg.init_storage(
            "distillation", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions]
        )
        return alg

    