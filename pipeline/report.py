"""
[6] LLM 보고서 생성 (report.py)
- Claude API로 비전문가용 보고서 초안 생성
- 점수/등급은 Rule 결과를 그대로 전달 (LLM이 재산정 금지)
- API 키 없으면 템플릿 목업 반환 (오프라인/개발 시연용)
"""
import config
from schemas import CrackFeatures, RiskResult, RagResult, Report


def _has_api():
    if not config.ANTHROPIC_API_KEY:
        return False
    try:
        import anthropic  # noqa
        return True
    except Exception:
        return False


def _prompt(feat, risk, rag):
    ev = "\n".join(f"- {e.text} (출처: {e.source})" for e in rag.evidences) or "- (검색된 근거 없음)"
    contribs = "\n".join(f"- {c['rule']}: {c['detail']} (+{c['points']})"
                         for c in risk.contributions) or "- (해당 없음)"
    return f"""당신은 시설물 안전점검 보조 AI입니다. 비전문가(건물주·일반인)가 이해할 수 있게 쉽게 서술하세요.
점수와 등급은 아래 값을 그대로 사용하고 재계산하지 마세요. 과장·단정을 피하고, 마지막에 전문가 정밀점검을 권고하세요.

[분석 결과]
- 균열 개수: {feat.crack_count}
- 최고 탐지 신뢰도: {feat.max_confidence}
- 최장 균열 길이 비율: {feat.max_length_ratio}
- 평균 균열 폭(px): {feat.avg_width_px}

[위험도(코드 산정)]
- 점수: {risk.score} / 등급: {risk.grade}
- 근거:
{contribs}

[안전기준 근거(RAG)]
{ev}

다음 4개 항목으로 나눠 한국어로 작성하세요:
1. 요약  2. 위험도 설명  3. 권고 조치  4. 전문점검 권고"""


def _mock(feat, risk, rag):
    """API 없을 때 템플릿 보고서."""
    ev = rag.evidences[0].text if rag.evidences else "안전기준 근거는 지식베이스 구축 후 표시됩니다."
    return Report(
        summary=f"업로드한 사진에서 균열 {feat.crack_count}개가 탐지되었습니다. "
                f"산정된 위험도는 '{risk.grade}'(점수 {risk.score})입니다.",
        risk_explain=f"최고 탐지 신뢰도 {feat.max_confidence}, 평균 균열 폭 {feat.avg_width_px}px 기준으로 "
                     f"규칙에 따라 등급이 산정되었습니다.",
        actions="균열 진행 여부를 주기적으로 관찰하고, 폭이 커지거나 수가 늘면 조치가 필요합니다.",
        inspection_advice=f"정확한 판단을 위해 전문가의 정밀점검을 권고합니다. 참고 기준: {ev}",
    )


def _parse(text):
    """LLM 텍스트를 4블록으로 대략 분리."""
    import re
    blocks = re.split(r'\n?\s*\d[\.\)]\s*', text)
    blocks = [b.strip() for b in blocks if b.strip()]
    while len(blocks) < 4:
        blocks.append("")
    return Report(summary=blocks[0], risk_explain=blocks[1],
                  actions=blocks[2], inspection_advice=blocks[3])


def generate(feat: CrackFeatures, risk: RiskResult, rag: RagResult) -> Report:
    if not _has_api():
        return _mock(feat, risk, rag)
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": _prompt(feat, risk, rag)}],
    )
    return _parse(msg.content[0].text)
