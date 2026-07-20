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
    summary: str = ""
    risk_explain: str = ""
    actions: str = ""
    inspection_advice: str = ""
