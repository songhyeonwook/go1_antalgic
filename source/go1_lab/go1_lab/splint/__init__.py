"""부목(HKAFO형) 자산 패키지.

test/ 스파이크(view_reset_random.py 등)로 검증된 부목 모델을 학습 환경이 쓰는
정식 모듈로 승격한 것입니다.

  - usd_builder: 원본 Go1 USD 를 reference 로 물고 다리별 부목 링크 + prismatic
    관절을 추가한 USD 변형본을 생성 (원본은 절대 수정하지 않음)
  - presence: 부목을 부상 다리에만 "존재"하게 만드는 런타임 토글
    (렌더 visibility + 질량/관성 + 콜라이더). replicate_physics=False 필수.

⚠️ pxr 의존 코드는 전부 함수 내부에서 lazy import 하므로 이 패키지 자체는
SimulationApp 인스턴스화 이전에 import 해도 안전합니다. 단, 함수 호출은
반드시 앱 실행 이후에 해야 합니다.
"""

from __future__ import annotations

import os

from .presence import LEGS, TINY_MASS, set_splint_presence  # noqa: F401
from .usd_builder import (  # noqa: F401
    NOMINAL_LEG_REACH,
    SPLINT_LATERAL,
    SPLINT_MAX,
    SPLINT_MIN,
    SPLINT_PITCH,
    SPLINT_TIP_RADIUS,
    build_splint_usd,
)

# 생성된 USD 캐시 위치 (git 추적 제외)
GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")


def build_cached_splint_usd(src_usd: str, attach: str = "thigh") -> str:
    """부목 USD 를 생성해 캐시 경로를 반환합니다.

    builder 상수가 바뀌어도 stale 캐시가 남지 않도록 매 호출마다 재생성합니다
    (생성 비용 < 1 s). 동시 실행 대비 임시 파일에 만든 뒤 os.replace 로 원자적
    교체합니다.
    """
    os.makedirs(GENERATED_DIR, exist_ok=True)
    dst = os.path.join(GENERATED_DIR, f"go1_splint_{attach}.usd")
    tmp = os.path.join(GENERATED_DIR, f"go1_splint_{attach}.tmp-{os.getpid()}.usd")
    if os.path.exists(tmp):
        os.remove(tmp)
    build_splint_usd(src_usd, tmp, attach=attach)
    os.replace(tmp, dst)
    return dst
