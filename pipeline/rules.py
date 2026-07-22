"""
[5] Rule 위험도 산정 (rules.py)
- 신뢰도·개수 기반 점수 -> 등급
- 점수는 항상 이 코드가 매김 (YOLO/RAG/LLM은 재료만 제공)
- 각 가점의 근거를 contributions에 기록 (설명가능성)
- [2차 MVP] 복합 결함: 균열 채널 + 결함별 가점(철근노출+30·박락+20·백태/누수+10 등) 합산.
  feat.defects 가 비면(균열 전용 모델) 균열 전용 채점과 완전히 동일(하위호환).
"""
import config
from schemas import CrackFeatures, RiskResult, RagResult


def _grade(score):
    for lo, hi, name in config.GRADE_BINS:
        if lo <= score <= hi:
            return name
    return "정상"


def _score_defects(feat: CrackFeatures, contribs: list) -> int:
    """[2차 MVP] 균열 외 면적 결함(feat.defects)의 복합 위험 가점.
    - 결함별 신뢰도 하한 통과 시에만 가점(면적 결함 오탐 억제).
    - 동일 결함 다수 → 소폭 가산. 서로 다른 결함 ≥2종 동시 → 복합 가점.
    """
    score = 0
    present = []
    for label, stat in (feat.defects or {}).items():
        if label == "crack":
            continue  # 균열은 crack 채널에서 계산(중복 방지)
        weight = config.DEFECT_WEIGHTS.get(label, 0)
        if weight <= 0:
            continue
        conf = float(stat.get("max_conf", 0.0))
        count = int(stat.get("count", 0))
        thr = config.DEFECT_CONF_MIN.get(label, config.DEFECT_CONF_MIN_DEFAULT)
        if count <= 0 or conf < thr:
            continue
        ko = config.DEFECT_KO.get(label, label)
        score += weight
        contribs.append({"rule": f"{ko} 탐지",
                         "detail": f"신뢰도 {conf:.2f} ≥ {thr}, {count}개",
                         "points": weight})
        present.append(label)
        if count >= config.DEFECT_MULTI_COUNT:
            score += config.DEFECT_MULTI_BONUS
            contribs.append({"rule": f"{ko} 다수",
                             "detail": f"{count}개 ≥ {config.DEFECT_MULTI_COUNT}",
                             "points": config.DEFECT_MULTI_BONUS})

    # 복합 결함 동시 발생(유의 균열 포함, 서로 다른 결함 ≥2종)
    distinct = set(present)
    if feat.crack_count and feat.max_confidence >= config.RULE_CONF_MODERATE:
        distinct.add("crack")
    if len(distinct) >= 2:
        score += config.COMPOSITE_MULTI_TYPE_BONUS
        kos = ", ".join(config.DEFECT_KO.get(l, l) for l in sorted(distinct))
        contribs.append({"rule": "복합 결함 동시 발생",
                         "detail": f"서로 다른 결함 {len(distinct)}종({kos}) 동시 탐지",
                         "points": config.COMPOSITE_MULTI_TYPE_BONUS})
    return score


def evaluate(feat: CrackFeatures, rag: RagResult = None) -> RiskResult:
    contribs = []
    score = 0

    # 균열 탐지 — 계단식(신뢰도가 낮게 압축된 모델 특성 반영)
    if feat.max_confidence >= config.RULE_CONF_STRONG:
        score += 30
        contribs.append({"rule": "균열 탐지(강)",
                         "detail": f"최고 신뢰도 {feat.max_confidence:.2f} ≥ {config.RULE_CONF_STRONG}",
                         "points": 30})
    elif feat.max_confidence >= config.RULE_CONF_MODERATE:
        score += 15
        contribs.append({"rule": "균열 탐지(약)",
                         "detail": f"최고 신뢰도 {feat.max_confidence:.2f} ≥ {config.RULE_CONF_MODERATE}",
                         "points": 15})

    # 균열 개수 — 계단식
    if feat.crack_count >= config.RULE_COUNT_SEVERE:
        score += 25
        contribs.append({"rule": "균열 매우 다수",
                         "detail": f"균열 {feat.crack_count}개 ≥ {config.RULE_COUNT_SEVERE}",
                         "points": 25})
    elif feat.crack_count >= config.RULE_COUNT_MANY:
        score += 15
        contribs.append({"rule": "균열 다수",
                         "detail": f"균열 {feat.crack_count}개 ≥ {config.RULE_COUNT_MANY}",
                         "points": 15})

    # 최장 균열 길이비 (기하 특징)
    if feat.max_length_ratio >= config.RULE_LENGTH_HIGH:
        score += 15
        contribs.append({"rule": "긴 균열",
                         "detail": f"최장 길이비 {feat.max_length_ratio:.2f} ≥ {config.RULE_LENGTH_HIGH}",
                         "points": 15})

    # RAG 긴급기준 매칭 — ChromaDB는 관련도와 무관하게 항상 top-k를 반환하므로
    # '근거가 있으면 무조건 +20'이면 모든 사진이 가점을 받아 계단식 보정이 깨진다.
    # → 유사도(RAG_MATCH_MIN_SCORE) 이상인 '관련성 높은' 근거가 있을 때만 가점.
    if rag and rag.evidences:
        strong = [e for e in rag.evidences if e.score >= config.RAG_MATCH_MIN_SCORE]
        if strong:
            score += 20
            contribs.append({"rule": "RAG 긴급기준 매칭",
                             "detail": f"유사도 {config.RAG_MATCH_MIN_SCORE}+ 근거 {len(strong)}건 검색됨",
                             "points": 20})

    # [2차 MVP] 복합 결함 가점 — 균열 외 면적 결함(철근노출·박락·백태/누수 등)
    score += _score_defects(feat, contribs)

    return RiskResult(score=score, grade=_grade(score), contributions=contribs)
