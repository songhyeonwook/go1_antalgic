from __future__ import annotations

def resolve_runner_class(class_name: str):
    """agent cfg 의 class_name 으로 runner 클래스를 고른다.
    go1_lab / rsl_rl 은 Isaac Sim 부팅 뒤에만 import 가능하므로 함수 안에서 지연 import 한다."""
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    if class_name == "OnPolicyRunner":
        return OnPolicyRunner
    if class_name == "DistillationRunner":
        return DistillationRunner
    if class_name == "Phase3DistillationRunner":
        from go1_lab.tasks.manager_based.go1_lab.agents.phase3_student import Phase3DistillationRunner
        return Phase3DistillationRunner
    raise ValueError(f"Unsupported runner class: {class_name!r}")