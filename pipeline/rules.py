"""
[5] Rule 위험도 산정 (rules.py)
- 신뢰도·개수 기반 점수 -> 등급
- 점수는 항상 이 코드가 매김 (YOLO/RAG/LLM은 재료만 제공)
- 각 가점의 근거를 contributions에 기록 (설명가능성)
"""
import config
from schemas import CrackFeatures, RiskResult, RagResult


def _grade(score):
    for lo, hi, name in config.GRADE_BINS:
        if lo <= score <= hi:
            return name
    return "정상"


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

    # RAG 긴급기준 매칭 (근거 문서가 검색되면 신호로만 반영)
    if rag and rag.evidences:
        score += 20
        contribs.append({"rule": "RAG 긴급기준 매칭",
                         "detail": f"안전기준 근거 {len(rag.evidences)}건 검색됨",
                         "points": 20})

    # (확장 예정) 철근노출 +30 / 누수 +10 — 범위 외

    return RiskResult(score=score, grade=_grade(score), contributions=contribs)
