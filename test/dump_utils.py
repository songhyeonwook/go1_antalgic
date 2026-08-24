"""롤아웃 덤프 분석 스크립트가 공유하는 기하 헬퍼.

analyze_dump / analyze_dump_perf / l_sensitivity_report / mu_robustness_report
가 같은 쿼터니언 변환을 각자 복제하고 있어 여기로 모았다.
"""

from __future__ import annotations

import numpy as np


def quat_to_rot(q):
    """(..., 4) wxyz → (..., 3, 3) 회전행렬."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def quat_rot_inv_x(quat, v):
    """world 벡터 v 를 body x 축으로 투영 (Rᵀv 의 x 성분)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    # body x축의 world 표현 = R @ [1, 0, 0]
    bx = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    return (bx * v).sum(-1)
