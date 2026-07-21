"""
전역 설정 상수 (경로·임계값·모델명 집약)
- 핸드오프 컨벤션: 경로는 상수로 상단 분리
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env(name, default=""):
    """환경변수를 읽되 인라인 주석(' #...')·양끝 공백을 제거해 방어.
    docker compose env_file 은 값 뒤 주석을 값에 포함시킬 수 있어(KEY=값 # 설명 → '값 # 설명'),
    이를 정리해 헤더 인코딩 오류·잘못된 파싱을 막는다.
    """
    raw = os.environ.get(name, default) or ""
    # 공백으로 구분된 인라인 주석 제거 (예: 'solar   # 설명' → 'solar')
    if " #" in raw:
        raw = raw.split(" #", 1)[0]
    return raw.strip()

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
EMBED_PROVIDER = _env("EMBED_PROVIDER", "solar") or "solar"

# Solar(Upstage) 임베딩 — 비대칭(문서=passage / 쿼리=query, 동일 공간), 4096차원.
# 배포에 유리: 로컬 모델·torch 불필요, 인스턴스·이미지 경량화. (런타임에 Upstage API 필요)
# ※ 모델 ID는 Upstage 콘솔에서 최신값 확인 권장(변경될 수 있음).
SOLAR_EMBED_ENDPOINT = _env(
    "SOLAR_EMBED_ENDPOINT", "https://api.upstage.ai/v1/solar/embeddings")
SOLAR_EMBED_PASSAGE = _env("SOLAR_EMBED_PASSAGE", "solar-embedding-1-large-passage")
SOLAR_EMBED_QUERY = _env("SOLAR_EMBED_QUERY", "solar-embedding-1-large-query")
UPSTAGE_API_KEY = _env("UPSTAGE_API_KEY")

# 로컬 폴백(BGE-m3) — EMBED_PROVIDER=bge 일 때만 사용
EMBED_MODEL = "BAAI/bge-m3"   # 한국어 지원 오픈소스 임베딩

# RAG는 관련도와 무관하게 항상 top-k를 반환하므로, Rule 가점은 유사도 임계값 이상일 때만.
# (임계값 미설정 시 모든 사진이 +20을 받아 폰 테스트 계단식 보정이 깨짐)
# 점수(=1-거리) 기준.
# ※ Solar 실측 재보정: 관련 근거가 ~0.20~0.25 로 분포 → 0.55면 대부분 필터링돼 RAG가
#    사실상 꺼짐. 그래서 Solar 분포에 맞춰 기본값 0.20 으로 낮춤.
#    (임베더/문서 데이터가 바뀌면 분포 재관측 후 env RAG_MATCH_MIN_SCORE 로 조정)
try:
    RAG_MATCH_MIN_SCORE = float(_env("RAG_MATCH_MIN_SCORE", "0.20") or "0.20")
except ValueError:
    RAG_MATCH_MIN_SCORE = 0.20

# ---- 보고서 생성 (LLM) ----
# 제공자 체인: claude(ANTHROPIC 키) → solar(UPSTAGE 키) → mock. auto면 사용 가능한 걸 자동 선택.
REPORT_PROVIDER = _env("REPORT_PROVIDER", "auto")   # auto | claude | solar | mock

# Claude (Anthropic). ※ 실제 사용 가능한 모델 ID로 교체 (예: claude-sonnet-4-5-20250929 등).
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-sonnet-4-5") or "claude-sonnet-4-5"
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")

# Solar (Upstage) 채팅 — 보고서 LLM 폴백. OpenAI 호환. UPSTAGE_API_KEY 재사용(임베딩과 동일 키).
# ※ 엔드포인트·모델명은 Upstage 콘솔 기준으로 env 조정 가능 (모델 예: solar-pro2 / solar-pro).
SOLAR_CHAT_ENDPOINT = _env("SOLAR_CHAT_ENDPOINT", "https://api.upstage.ai/v1/solar/chat/completions")
SOLAR_CHAT_MODEL = _env("SOLAR_CHAT_MODEL", "solar-pro2")

# 자가진단 등급(정상/주의/위험/긴급) → 현업 상태평가등급(A~E) 참고 매핑
STATE_GRADE_MAP = {
    "정상": "A~B등급 (양호)",
    "주의": "C등급 (보통)",
    "위험": "D등급 (미흡)",
    "긴급": "E등급 (불량)",
}

# ---- 모델 성능 지표 (학습 결과, 화면 표시용) ----
MODEL_METRICS = {"mAP50": 0.179, "mAP50_95": 0.062, "precision": 0.285, "recall": 0.242}
