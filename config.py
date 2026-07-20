"""
전역 설정 상수 (경로·임계값·모델명 집약)
- 핸드오프 컨벤션: 경로는 상수로 상단 분리
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 모델 ----
# 타일 학습본 (train_tiled_full). 없으면 앱이 안내 메시지 표시
YOLO_WEIGHTS = os.path.join(
    BASE_DIR, "runs", "detect", "runs", "crack", "train_tiled_full", "weights", "best.pt"
)

# ---- 타일 슬라이스 추론 ----
TILE = 640
OVERLAP = 0.2
CONF = 0.15          # 자신감 낮은 모델 기준. 오탐 많으면 0.25로
IOU_MERGE = 0.5      # 타일 경계 중복 박스 병합(NMS)

# ---- Rule Engine 임계값 ----
# ※ 재보정: 이 모델은 신뢰도가 낮게 압축돼 있음(폰 100장 테스트 max 0.36).
#   기획안의 0.80은 잘 보정된 0~1 신뢰도 가정 → 도달 불가라 실제 분포에 맞춰 계단식으로.
RULE_CONF_STRONG = 0.28    # 강한 탐지 (상위권 신뢰도)
RULE_CONF_MODERATE = 0.20  # 약한 탐지
RULE_COUNT_MANY = 3        # 균열 다수
RULE_COUNT_SEVERE = 5      # 균열 매우 다수
RULE_LENGTH_HIGH = 0.15    # 최장 균열 길이비(대각선 대비) 큰 편
GRADE_BINS = [(0, 39, "정상"), (40, 69, "주의"), (70, 89, "위험"), (90, 999, "긴급")]

# ---- RAG ----
CHROMA_DIR = os.path.join(BASE_DIR, "knowledge", "chroma")
EMBED_MODEL = "BAAI/bge-m3"   # 한국어 지원 오픈소스 임베딩
RAG_TOP_K = 3
# RAG는 관련도와 무관하게 항상 top-k를 반환하므로, Rule 가점은 유사도 임계값 이상일 때만.
# (임계값 미설정 시 모든 사진이 +20을 받아 폰 테스트 계단식 보정이 깨짐)
# BGE-m3 코사인 유사도(=1-거리) 기준. build_index 후 관측된 분포로 재보정 가능.
RAG_MATCH_MIN_SCORE = 0.55

# ---- 보고서 생성 (Claude API) ----
# ※ 실제 사용 가능한 모델 ID로 교체 필요 (예: claude-sonnet-4-5-20250929 등).
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# 자가진단 등급(정상/주의/위험/긴급) → 현업 상태평가등급(A~E) 참고 매핑
STATE_GRADE_MAP = {
    "정상": "A~B등급 (양호)",
    "주의": "C등급 (보통)",
    "위험": "D등급 (미흡)",
    "긴급": "E등급 (불량)",
}

# ---- 모델 성능 지표 (학습 결과, 화면 표시용) ----
MODEL_METRICS = {"mAP50": 0.179, "mAP50_95": 0.062, "precision": 0.285, "recall": 0.242}
