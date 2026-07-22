"""
파이프라인 단계 간 입출력 데이터 구조 (dataclass)
- mm 절대측정 안 함 → 픽셀 상대값 기준
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Stage(str, Enum):
    """에이전트 진행 단계. (COLLECT·DONE=멀티턴은 PR-B 예정)"""
    AWAIT_IMAGE = "await_image"
    NEEDS_RETAKE = "needs_retake"   # 트리아지: 원거리·흐림 → 재촬영 요청(YOLO 안 돌림)
    REJECTED = "rejected"           # 트리아지: 균열 점검 대상 아님(비콘크리트/무관 사진)
    ANALYZED = "analyzed"


@dataclass
class TriageResult:
    """[1차 게이트] 사진이 판정 가치가 있는지 + 비전이 읽어낸 메타데이터.
    같은 비전 호출 1회로 게이트 판정과 보고서용 메타를 함께 받는다(추가 호출 없음).
    """
    verdict: str = "ok"        # ok | retake_far | retake_blur | not_crack
    ok: bool = True            # True면 분석 진행, False면 게이트에서 멈춤
    message: str = ""          # 사용자 안내(재촬영 사유 등)
    provider: str = "heuristic"  # claude | heuristic | mock
    blur_score: float = 0.0    # 라플라시안 분산(낮을수록 흐림)
    # 비전이 읽어낸 보고서용 메타(값이 없으면 빈 문자열/None) — report 기본현황·점검결과 보강
    meta: dict = field(default_factory=dict)  # {structure_part, material, orientation,
    #                                            branching, efflorescence, spalling, notes}


@dataclass
class Detection:
    box: List[int]          # [x1, y1, x2, y2] 원본 좌표
    conf: float
    cls: int = 0            # 클래스 id (다중클래스 detection용; 균열 전용 모델은 0)
    label: str = "crack"    # 클래스명(crack/spalling/efflorescence/rebar_exposure/steel_defect/paint_damage)


@dataclass
class DetectResult:
    detections: List[Detection] = field(default_factory=list)
    image_size: List[int] = field(default_factory=lambda: [0, 0])  # [W, H]


@dataclass
class CrackFeatures:
    crack_count: int = 0
    max_length_ratio: float = 0.0   # 최장 균열 길이 / 이미지 대각선
    avg_width_px: float = 0.0       # 평균 폭(픽셀)
    max_confidence: float = 0.0     # 최고 탐지 신뢰도(균열 채널)
    # [2차 MVP] 복합 결함 요약 — 균열 외 면적 결함(bbox). 균열 전용 모델이면 빈 dict → 기존과 동일.
    #   {label: {"count": int, "max_conf": float}}  예: {"rebar_exposure": {"count":2,"max_conf":0.71}}
    defects: dict = field(default_factory=dict)


@dataclass
class RiskResult:
    score: int = 0
    grade: str = "정상"
    contributions: List[dict] = field(default_factory=list)  # {rule, detail, points}


@dataclass
class Evidence:
    text: str
    source: str
    score: float = 0.0


@dataclass
class RagResult:
    evidences: List[Evidence] = field(default_factory=list)


@dataclass
class Report:
    """현업(FMS·국토안전관리원) 정기안전점검 결과보고서 6섹션 서식.
    1 기본현황 · 2 점검결과 · 3 안전등급 · 4 종합의견 · 5 판단근거(RAG) · 6 유의사항
    """
    basic_info: str = ""          # 1. 시설물 기본현황
    inspection_result: str = ""   # 2. 점검 결과
    safety_grade: str = ""        # 3. 안전등급 평가
    overall_opinion: str = ""     # 4. 종합의견
    evidence_basis: str = ""      # 5. 판단 근거 (안전기준 RAG)
    caveats: str = ""             # 6. 유의사항

    SECTIONS = [
        ("1. 시설물 기본현황", "basic_info"),
        ("2. 점검 결과", "inspection_result"),
        ("3. 안전등급 평가", "safety_grade"),
        ("4. 종합의견", "overall_opinion"),
        ("5. 판단 근거 (안전기준 RAG)", "evidence_basis"),
        ("6. 유의사항", "caveats"),
    ]

    def to_markdown(self) -> str:
        head = (
            "# 균열 안전점검 결과보고서 (AI 자가진단 보조)\n\n"
            "> 본 보고서는 AI 균열 자가진단 시스템이 생성한 **초안**이며, "
            "비전문가의 초기 스크리닝을 돕기 위한 참고 자료입니다. "
            "정확한 판단은 전문가의 정밀 점검을 따릅니다."
        )
        parts = [head]
        for title, attr in self.SECTIONS:
            parts.append(f"## {title}\n{getattr(self, attr).strip()}")
        return "\n\n---\n\n".join(parts)


@dataclass
class AgentState:
    """세션 전체 상태. 무거운 결과(detect/features)를 캐시해 Streamlit 재실행 시 재계산 방지.
    각 노드가 state를 받아 갱신·반환하는 구조 → 추후 LangGraph 이식 용이.
    """
    stage: str = Stage.AWAIT_IMAGE
    image_hash: str = ""
    triage: Optional[TriageResult] = None
    detect: Optional[DetectResult] = None
    features: Optional[CrackFeatures] = None
    risk: Optional[RiskResult] = None
    rag: Optional[RagResult] = None
    report: Optional[Report] = None
