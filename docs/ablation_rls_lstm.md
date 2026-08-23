# RLS + LSTM Ablation — 부목 길이 추정 기여도 분해

- 날짜: 2026-08-23
- 체크포인트: `logs/unitree_go1_antalgic/2026_08_17_16_00_07_phase3_s42_P3-splint-003-0816/model_3999.pt`
- 데이터: `test/dumps/p3_splint003_final_balanced.npz` (T=2500, N=40 env, balanced 5조건)
- 도구: `test/analyze_student.py`의 재생/probe 기계를 재사용한 occlusion ablation

## 배경

Phase 3 student는 policy 관측 51차원 중 2차원으로 live RLS 부목 길이 추정
채널 `[L̂_norm, √P_norm]`을 받고 (`mdp/rls.py`, eb26293), LSTM latent에서
aux head가 L̂을 출력한다 (`mdp/aux_distillation.py`). 이 문서는 "L 추정
정확도에서 RLS와 LSTM이 각각 얼마나 기여하는가"를 재학습 없이 분해한다.

## 방법

기존 rollout 덤프를 LSTM에 오프라인 재생 (재생 검증 [S0] |Δaction|=0 확인됨).
세 조건을 동일 프로토콜(env 단위 held-out split, 첫 done 이전 구간 제외,
stride-2 서브샘플, 부상 env만)로 비교:

| 조건 | 정의 |
|---|---|
| A. Full | 덤프 그대로 (RLS live + LSTM + aux) |
| B. RLS 차단 | obs[49]=0.0, obs[50]=1.0 (prior 상수) 로 고정 후 재생 — "RLS가 한 번도 갱신되지 않은" 입력 |
| C. RLS 단독 | live RLS 채널 값 자체 (`obs[49]·0.06+0.39`), LSTM 미사용 |

B에서는 두 가지를 따로 본다: 기존 aux head 그대로 평가(B-aux)와,
차단된 latent 위에 probe를 재학습한 것(B-probe = 순수 고유수용성 정보량).

## 결과 — L 추정 (부상 env, test 3204 샘플)

| 조건 | MAE median | MAE 90% | R² |
|---|---|---|---|
| **A. RLS + LSTM (aux)** | **0.9 mm** | 3.4 mm | 0.993 |
| C. RLS 단독 | 2.1 mm | 5.2 mm | – |
| B-probe. 고유수용성만 (latent probe 재학습) | 6.4 mm | 14.3 mm | 0.918 |
| B-aux. RLS 차단 + 기존 aux head | 38.3 mm | 63.9 mm | −0.68 (붕괴) |
| 참고: 관측 이력 MLP (`analyze_dump [D]`) | ≈ 9 mm | – | – |
| 참고: 오프라인 RLS (`analyze_dump [C]`) | ≈ 0.5 mm | – | – |

리셋 후 수렴 (median mm):

| | 0–0.5s | 0.5–1s | 1–2s | 2–5s | >5s |
|---|---|---|---|---|---|
| A. RLS+LSTM | 2.1 | 0.9 | 0.9 | 0.8 | 0.9 |
| C. RLS 단독 | 2.7 | 2.3 | 2.4 | 2.2 | 2.0 |
| B-aux. RLS 차단 | 15.0 | 21.1 | 25.0 | 30.6 | 47.2 (시간에 따라 악화) |

부수 결과:

- 부상 다리 5-way 분류 (latent probe): A 92.5% / B 99.8% — 다리 식별은
  RLS 없이도 충분 (고유수용성 신호 기반).
- 행동 민감도: RLS 차단 시 |Δaction| 평균 — 정상 env 0.005 rad,
  부상 env 0.087–0.130 rad (최대 1.08 rad). 보행 정책 자체가 RLS 값에
  조건화되어 있음.

## 해석

1. **융합 이득은 실재**: RLS 단독 2.1 mm → LSTM 결합 0.9 mm (~2.3×).
   리셋 직후 0–0.5 s 구간에서도 A(2.1 mm)가 C(2.7 mm)보다 빠르다 —
   RLS 갱신 전 공백을 고유수용성 이력이 메꾼다.
2. **고유수용성만의 한계는 6–9 mm**: B-probe 6.4 mm는 이력-MLP(≈9 mm)와
   정합. RLS 채널 없이 처음부터 학습한 student도 이 수준일 가능성이 높다.
   즉 mm급 정확도의 주역은 RLS 채널이다 (~7×).
3. **배포 리스크**: 학습된 aux head는 RLS 입력에 강하게 의존한다.
   차단 시 38 mm로 붕괴하고 시간이 갈수록 악화(15→47 mm) — head가
   "시간이 지나면 RLS가 수렴해 있다"는 전제로 가중치를 잡았다는 뜻.
   실기에서 게이트(부목 접지 토크 / 기준 발 스탠스)가 장기간 안 걸리면
   추정 열화 가능 → √P_norm 채널이 그 신뢰도 신호이므로, 배포 시
   L̂ 소비 로직은 √P 게이팅을 권장.

## 한계

- **Occlusion ≠ 재학습 ablation**: B는 RLS와 함께 학습된 모델의 입력
  차단이므로, "RLS 없이 학습한 student"의 성능 하한 근사다. 상한 추정에는
  `use_rls_estimate: false` 로 phase 2→3 재학습이 필요하다 (teacher 관측
  차원이 바뀌므로 phase 2부터 다시 — `docs` 참고: RLS 채널은 teacher와
  공유되는 policy 그룹에 있어 phase 간 차원·분포 일치가 요구됨).
- **오픈루프 재생**: 관측 시퀀스는 원본(Full) 정책의 rollout이다. 차단
  상태로 폐루프를 돌리면 상태 분포가 달라져 실제 열화는 더 클 수 있다.
- 단일 체크포인트·단일 덤프(seed 42) 기준.

## 재현

```bash
# 오프라인 분석 원본 (Isaac 불필요) — [S1]~[S6]
cd test && PYTHONPATH= python3 analyze_student.py dumps/p3_splint003_final_balanced.npz
```

Ablation은 `analyze_student.py`의 `build_student`/`replay_lstm`/`ridge_probe`를
그대로 import 해 위 3조건으로 재생·평가한 것이다. 조건 B는 재생 전에
`obs_policy[..., 49] = 0.0`, `obs_policy[..., 50] = 1.0` (prior 인코딩,
`mdp/rls.py`의 `RLS_L_PRIOR`/`RLS_P0` 참조)으로 고정하고, 조건 C는
`obs_policy[..., 49] * 0.06 + 0.39` 를 gt_L과 직접 비교한다. 나머지
split/유효 구간/서브샘플 프로토콜은 원본 스크립트와 동일하다.
