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
RAG_TOP_K = 3

# 임베딩 제공자: "solar"(기본, Upstage API) | "bge"(로컬 BGE-m3) | "mock"(오프라인 테스트)
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "solar")

# Solar(Upstage) 임베딩 — 비대칭(문서=passage / 쿼리=query, 동일 공간), 4096차원.
# 배포에 유리: 로컬 모델·torch 불필요, 인스턴스·이미지 경량화. (런타임에 Upstage API 필요)
# ※ 모델 ID는 Upstage 콘솔에서 최신값 확인 권장(변경될 수 있음).
SOLAR_EMBED_ENDPOINT = os.environ.get(
    "SOLAR_EMBED_ENDPOINT", "https://api.upstage.ai/v1/solar/embeddings")
SOLAR_EMBED_PASSAGE = os.environ.get("SOLAR_EMBED_PASSAGE", "solar-embedding-1-large-passage")
SOLAR_EMBED_QUERY = os.environ.get("SOLAR_EMBED_QUERY", "solar-embedding-1-large-query")
UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY", "")

# 로컬 폴백(BGE-m3) — EMBED_PROVIDER=bge 일 때만 사용
EMBED_MODEL = "BAAI/bge-m3"   # 한국어 지원 오픈소스 임베딩

# RAG는 관련도와 무관하게 항상 top-k를 반환하므로, Rule 가점은 유사도 임계값 이상일 때만.
# (임계값 미설정 시 모든 사진이 +20을 받아 폰 테스트 계단식 보정이 깨짐)
# 코사인 유사도(=1-거리) 기준. 실제 임베더(Solar/BGE)로 인덱스 만든 뒤 분포 보고 재보정.
# ※ Solar와 BGE는 유사도 스케일이 다르므로 임베더 바꾸면 이 값도 재확인 필요.
RAG_MATCH_MIN_SCORE = float(os.environ.get("RAG_MATCH_MIN_SCORE", "0.55"))

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
