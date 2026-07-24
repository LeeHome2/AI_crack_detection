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
# 가중치 자동 선택: env → 6종 최종 → 6종 임시 → defect6_v2 → 크랙 폴백 순으로 '실제 파일'을 선택.
# ※ 'real'=존재 + 1KB 초과. git-lfs 포인터(수백 B)나 손상 파일은 건너뛰어 크랙으로 안전 폴백
#   → LFS 미설정 서버·모델 미커밋 상태에서도 앱이 죽지 않고 크랙 모델로 동작.
_YOLO_CRACK = os.path.join(
    BASE_DIR, "runs", "detect", "runs", "crack", "train_tiled_full", "weights", "best.pt")


def _real_weights(p):
    try:
        return bool(p) and os.path.exists(p) and os.path.getsize(p) > 1024
    except OSError:
        return False


_YOLO_CANDIDATES = [
    _env("YOLO_WEIGHTS", ""),                                        # 명시 지정 최우선
    os.path.join(BASE_DIR, "models", "yolov8s_defect6_final.pt"),   # 6종 최종(학습완료)
    os.path.join(BASE_DIR, "models", "yolov8s_defect6_epoch43.pt"),  # 6종 임시
    os.path.join(BASE_DIR, "runs", "detect", "defect6_v2", "weights", "best.pt"),
    _YOLO_CRACK,                                                     # 균열 안전판
]
YOLO_WEIGHTS = next((p for p in _YOLO_CANDIDATES if _real_weights(p)), _YOLO_CRACK)

# ---- 타일 슬라이스 추론 ----
TILE = 640
OVERLAP = 0.2
CONF = 0.15          # 자신감 낮은 모델 기준. 오탐 많으면 0.25로
IOU_MERGE = 0.5      # 타일 경계 중복 박스 병합(NMS)

# ---- 탐지 후처리 (재학습 없이 이음새 오탐·개수 보정) ----
# 이음새 필터: 균열 중심선이 '매우 곧은 직선'이면 이음새로 보고 제외 (구불한 균열은 보존).
#  직선성 = 주축대비 수직편차/길이. 이 값 미만이면 이음새로 간주. 실사진으로 튜닝 필요.
SEAM_FILTER_ENABLED = _env("SEAM_FILTER_ENABLED", "1") not in ("0", "false", "no", "off")
try:
    SEAM_STRAIGHTNESS_MAX = float(_env("SEAM_STRAIGHTNESS_MAX", "0.030") or "0.030")
except ValueError:
    SEAM_STRAIGHTNESS_MAX = 0.030

# ---- Rule Engine 임계값 ----
# ※ 재보정: 이 모델은 신뢰도가 낮게 압축돼 있음(폰 100장 테스트 max 0.36).
#   기획안의 0.80은 잘 보정된 0~1 신뢰도 가정 → 도달 불가라 실제 분포에 맞춰 계단식으로.
RULE_CONF_STRONG = 0.28    # 강한 탐지 (상위권 신뢰도)
RULE_CONF_MODERATE = 0.20  # 약한 탐지
RULE_COUNT_MANY = 3        # 균열 다수
RULE_COUNT_SEVERE = 5      # 균열 매우 다수
RULE_LENGTH_HIGH = 0.15    # 최장 균열 길이비(대각선 대비) 큰 편
GRADE_BINS = [(0, 39, "정상"), (40, 69, "주의"), (70, 89, "위험"), (90, 999, "긴급")]

# ---- [2차 MVP] 복합 결함 위험 가중 ----
# 결함별 기본 가점 — 구조적 심각도 순. 균열(crack)은 위 RULE_CONF/COUNT/LENGTH 채널이 담당(중복 방지).
# label 은 변환기 CLASS_NAMES / Detection.label 과 정합.
#   철근노출: 피복 탈락·활성 부식·단면 손실 → 최고 위험(균열 강탐지 +30과 동급)
#   강재손상: 구조 강재 손상 · 박리/박락: 콘크리트 피복 손실(철근노출 전조)
#   백태/누수: 수분 침투 지표(2차 열화 유발) · 도장손상: 방식 코팅·미관(경미)
DEFECT_WEIGHTS = {
    "rebar_exposure": 30,
    "steel_defect": 25,
    "spalling": 20,
    "efflorescence": 10,
    "paint_damage": 5,
    "crack": 0,            # crack 채널이 계산 → 복합 합산에서 중복 가점 방지
}
# 결함별 신뢰도 하한 — 이 값 미만 탐지는 위험 산정에서 무시(면적 결함 bbox는 신뢰도 높게 나옴).
# ※ [잠정] 타일링 defect6_tiled-3 클래스별 AP50 순으로 조정(실측 conf 분포 아닌 성능 프록시).
#   강한 클래스(철근노출 28%·균열 19%)는 낮춰 데모에서 뜨게, 약한 클래스(박리 4.7%·도장 7%)는
#   높여 오탐 억제. analyze_model.py 의 실측 confidence 분포 나오면 최종 확정.
DEFECT_CONF_MIN = {
    "rebar_exposure": 0.25,   # AP50 27.95% (최강) → 낮춰서 확실히 탐지
    "steel_defect": 0.35,     # AP50 10.61% (중)
    "spalling": 0.45,         # AP50 4.70% (최약) → 높여서 오탐 억제
    "efflorescence": 0.35,    # AP50 9.56% (중약)
    "paint_damage": 0.45,     # AP50 6.97% (약) → 높여서 오탐 억제
    "crack": 0.20,            # AP50 18.75%, crack 채널이 별도 담당 → 유지
}
DEFECT_CONF_MIN_DEFAULT = 0.35
# 동일 결함 다수 인스턴스 소폭 가산.
DEFECT_MULTI_COUNT = 3
DEFECT_MULTI_BONUS = 5
# 복합(서로 다른 유의 결함 ≥2종 동시) 가점 — 복합 열화 상호작용(예: 균열+백태=누수성 균열, 박락+철근노출=진행성 열화).
COMPOSITE_MULTI_TYPE_BONUS = 10
# 한글 표기(설명가능성·보고서용)
DEFECT_KO = {
    "crack": "균열", "spalling": "박리/박락", "efflorescence": "백태/누수",
    "rebar_exposure": "철근노출", "steel_defect": "강재손상", "paint_damage": "도장손상",
}

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
# 점수(=1-코사인거리) 기준. (build_index.py 가 컬렉션을 hnsw:space=cosine 으로 생성)
# ※ Solar+코사인 실측(2026-07, 17문서·98청크):
#     관련 근거 top-1 = 0.47~0.59, top-3 최저 ≈ 0.38 / 완전 off-topic 쿼리 최고 ≈ 0.18.
#     → 두 밴드 사이 0.30 을 기본값으로. off-topic(+0.12)·관련(+0.08) 양쪽 마진 확보.
#   (구 0.20 은 L2 기본거리에서 1-거리가 음수로 나오던 시절 값 — 코사인 전환으로 상향)
#   (임베더/문서 데이터가 바뀌면 rag_smoke_test.py 로 분포 재관측 후 env 로 조정)
try:
    RAG_MATCH_MIN_SCORE = float(_env("RAG_MATCH_MIN_SCORE", "0.30") or "0.30")
except ValueError:
    RAG_MATCH_MIN_SCORE = 0.30

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

# ---- [1차] 비전 트리아지 (triage) ----
# 사진 업로드 시 YOLO 전에 1차 게이트: 근접/원거리/흐림/비균열 판정 + 보고서용 메타 추출.
# Claude 비전 1회 호출(키 있으면). 키 없거나 실패하면 휴리스틱(흐림만)으로 폴백 → 앱 안 죽음.
TRIAGE_ENABLED = _env("TRIAGE_ENABLED", "1") not in ("0", "false", "no", "off")
# 라플라시안 분산(선명도). 이 값 미만이면 '흐림'으로 보고 API 전에 무료로 재촬영 요청.
#  ※ 해상도·피사체에 따라 분포가 달라 실사진으로 튜닝 필요(기본 보수적으로 낮게).
try:
    TRIAGE_BLUR_MIN = float(_env("TRIAGE_BLUR_MIN", "60") or "60")
except ValueError:
    TRIAGE_BLUR_MIN = 60.0
# 비전 호출 전 이미지 긴 변 축소(px). 토큰·비용 절약(판정엔 충분). 0이면 원본.
try:
    VISION_MAX_SIDE = int(_env("VISION_MAX_SIDE", "1024") or "1024")
except ValueError:
    VISION_MAX_SIDE = 1024
# 비전 모델은 보고서 LLM과 동일 키·모델 재사용(ANTHROPIC_*). 필요시 별도 지정 가능.
VISION_MODEL = _env("VISION_MODEL", "") or ANTHROPIC_MODEL

# 자가진단 등급(정상/주의/위험/긴급) → 현업 상태평가등급(A~E) 참고 매핑
STATE_GRADE_MAP = {
    "정상": "A~B등급 (양호)",
    "주의": "C등급 (보통)",
    "위험": "D등급 (미흡)",
    "긴급": "E등급 (불량)",
}

# ---- 모델 성능 지표 (학습 결과, 화면 표시용) ----
MODEL_METRICS = {"mAP50": 0.179, "mAP50_95": 0.062, "precision": 0.285, "recall": 0.242}

# ---- 배포 빌드 식별 (화면 표시용) ----
# 8501(복합·CD 자동)과 8502(크랙 안전판·동결)을 화면에서 눈으로 구분하기 위한 라벨.
# 원칙: '실제 로드된 가중치'로 자동 판별 → 라벨이 실제 모델과 어긋나지 않음(정직).
#   경로에 'defect6' 있으면 복합(2차), 아니면 크랙 폴백 포함 균열(1차).
# 필요 시 APP_VARIANT/APP_VARIANT_DESC env 로 강제 지정 가능(자동판별보다 우선).
def _auto_variant():
    w = (YOLO_WEIGHTS or "").replace("\\", "/").lower()
    if "defect6" in w:
        return ("2차 MVP", "복합 6종 (균열·박리·백태·철근노출·강재·도장)")
    return ("1차 MVP", "균열 전용 · 크랙 안전판")   # 크랙 폴백 포함 기본


_AUTO_VARIANT = _auto_variant()
APP_VARIANT = _env("APP_VARIANT", "") or _AUTO_VARIANT[0]
APP_VARIANT_DESC = _env("APP_VARIANT_DESC", "") or _AUTO_VARIANT[1]
