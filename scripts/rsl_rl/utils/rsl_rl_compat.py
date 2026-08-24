"""RSL-RL 버전 호환 패치 유틸.

train / test / play_result / rollout_dump 가 공유한다 (이전에는 네 파일에
같은 함수가 복제돼 있었다).
"""

from __future__ import annotations

# PPO 생성자가 받지 않는 키 — agent cfg 를 dict 로 편 뒤 제거해야 한다.
_UNSUPPORTED_ALGORITHM_KEYS = ("optimizer", "config_class", "share_cnn_encoders")
# class_name 이 비어 있으면 RSL-RL 이 해석하지 못하는 정책 하위 설정들
_POLICY_COMPONENTS = ("actor", "critic", "student", "teacher")


def patch_rsl_rl_agent_cfg(agent_cfg_dict: dict) -> dict:
    """RSL-RL 3.0.1+ 호환을 위해 agent config dict 를 제자리 수정합니다.

    1. policy 하위 actor/critic/student/teacher 에 class_name 이 없으면 "MLP" 를 채운다.
    2. PPO 생성자가 지원하지 않는 algorithm 키를 제거한다.
    """
    policy_cfg = agent_cfg_dict.get("policy")
    if isinstance(policy_cfg, dict):
        for component_name in _POLICY_COMPONENTS:
            component_cfg = policy_cfg.get(component_name)
            if isinstance(component_cfg, dict):
                component_cfg.setdefault("class_name", "MLP")

    algorithm_cfg = agent_cfg_dict.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        for key in _UNSUPPORTED_ALGORITHM_KEYS:
            algorithm_cfg.pop(key, None)

    return agent_cfg_dict
