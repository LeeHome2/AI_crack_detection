"""
파이프라인 단계 간 입출력 데이터 구조 (dataclass)
- mm 절대측정 안 함 → 픽셀 상대값 기준
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Detection:
    box: List[int]          # [x1, y1, x2, y2] 원본 좌표
    conf: float


@dataclass
class DetectResult:
    detections: List[Detection] = field(default_factory=list)
    image_size: List[int] = field(default_factory=lambda: [0, 0])  # [W, H]


@dataclass
class CrackFeatures:
    crack_count: int = 0
    max_length_ratio: float = 0.0   # 최장 균열 길이 / 이미지 대각선
    avg_width_px: float = 0.0       # 평균 폭(픽셀)
    max_confidence: float = 0.0     # 최고 탐지 신뢰도


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
