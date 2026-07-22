"""
[6] LLM 보고서 생성 (report.py)
- 현업(FMS·국토안전관리원) 정기안전점검 결과보고서 6섹션 서식으로 초안 생성
  1 기본현황 · 2 점검결과 · 3 안전등급 · 4 종합의견 · 5 판단근거(RAG) · 6 유의사항
- 점수/등급은 Rule 결과를 그대로 사용 (LLM이 재산정 금지)
- 결정적 섹션(1 기본현황·5 판단근거·6 유의사항)은 코드가 조립하고,
  서술 섹션(2 점검결과·3 안전등급·4 종합의견)만 LLM/목업이 작성 → 환각·출처 오염 방지
- API 키 없으면 템플릿 목업 반환 (오프라인/개발 시연용)
"""
import re
import datetime

import config
from schemas import CrackFeatures, RiskResult, RagResult, Report


# ────────────────────────────── 준비 판단 ──────────────────────────────
def _has_api():
    key = (config.ANTHROPIC_API_KEY or "").strip()
    # 빈 값·비ASCII(주석/한글 유입 등)면 목업 사용 (HTTP 헤더 인코딩 오류 방지)
    if not key or not key.isascii():
        return False
    try:
        import anthropic  # noqa
        return True
    except Exception:
        return False


# ─────────────────────── 결정적 섹션 (코드 조립) ───────────────────────
def _meta_val(meta, key, default="미상"):
    v = (meta or {}).get(key, "") if isinstance(meta, dict) else ""
    v = (v or "").strip()
    return v if v and v != "미상" else default


def _basic_info(meta=None) -> str:
    """1. 시설물 기본현황 — 사용자 입력 자리표시자 + (트리아지 비전 메타) + 점검 메타."""
    today = datetime.date.today().isoformat()
    part = _meta_val(meta, "structure_part")     # 벽/바닥/기둥/천장/외벽…
    material = _meta_val(meta, "material")        # 콘크리트/벽돌/타일…
    struct = "철근콘크리트" if material in ("콘크리트", "미상") else material
    return (
        "| 항목 | 내용 |\n"
        "|---|---|\n"
        "| 시설물명 | (사용자 입력) 예: ○○빌딩 외벽 |\n"
        "| 위치 | (사용자 입력) 예: 서울시 ○○구 |\n"
        f"| 점검 부위 | {part} (AI 비전 추정) |\n"
        f"| 구조형식 | {struct} |\n"
        f"| 점검일자 | {today} |\n"
        "| 점검방식 | 사진 기반 AI 자가진단 (비전 트리아지 + YOLO 타일 탐지 + OpenCV 형태분석) |"
    )


def _meta_observation(meta) -> str:
    """트리아지 비전이 읽어낸 균열 양상 소견 한 줄 (점검결과 보강). 없으면 빈 문자열."""
    if not isinstance(meta, dict) or not meta:
        return ""
    bits = []
    o = _meta_val(meta, "orientation", "")
    if o:
        bits.append(f"방향 {o}")
    b = _meta_val(meta, "branching", "")
    if b and b != "없음":
        bits.append(f"분기 {b}")
    if _meta_val(meta, "efflorescence", "") == "있음":
        bits.append("백태(누수 흔적) 동반")
    if _meta_val(meta, "spalling", "") == "있음":
        bits.append("콘크리트 박리 동반")
    notes = _meta_val(meta, "notes", "")
    line = ", ".join(bits)
    if notes:
        line = (line + " — " + notes) if line else notes
    return f"\n**AI 비전 육안 소견:** {line}" if line else ""


def _evidence_basis(rag: RagResult) -> str:
    """5. 판단 근거 (RAG) — 검색된 근거 문장을 출처와 함께 그대로 인용."""
    if not rag.evidences:
        return "- 안전기준 근거는 지식베이스(RAG) 구축 후 표시됩니다. (knowledge/build_index.py 실행)"
    return "\n".join(
        f"- {e.text}\n  (근거 출처: {e.cite()}, 유사도 {e.score})" for e in rag.evidences
    )


def _caveats() -> str:
    """6. 유의사항 — 고정 문구 (한계·오탐 고지)."""
    return (
        "본 결과는 AI 기반 초기 스크리닝 자료입니다. 사진 촬영 각도·조명·해상도에 따라 "
        "결과가 달라질 수 있으며, 규칙적 이음새(타일 줄눈·패널 경계)를 균열로 오인할 수 있습니다. "
        "또한 사진만으로는 실제 mm 단위 폭을 확정할 수 없어 상대값으로 표기합니다. "
        "최종 안전 판단과 조치는 반드시 전문가의 정밀 점검을 통해 확정하시기 바랍니다."
    )


def _contrib_line(risk: RiskResult) -> str:
    if not risk.contributions:
        return "가점 항목 없음 (정상 범위)"
    return " · ".join(f"{c['rule']}(+{c['points']})" for c in risk.contributions)


# ─────────────────────── 서술 섹션 목업 (API 없을 때) ───────────────────────
def _mock_inspection_result(feat: CrackFeatures, meta=None) -> str:
    length_pct = round(feat.max_length_ratio * 100, 1)
    return (
        "업로드된 사진에서 다음과 같은 균열이 탐지되었습니다.\n\n"
        "| 항목 | 결과 |\n"
        "|---|---|\n"
        f"| 탐지된 균열 개수 | {feat.crack_count}개소 |\n"
        f"| 최고 탐지 신뢰도 | {feat.max_confidence} |\n"
        f"| 최장 균열 길이 (상대) | 이미지 대각선의 약 {length_pct}% |\n"
        f"| 평균 균열 폭 (상대) | 픽셀 기준 {feat.avg_width_px}px |\n\n"
        "※ 사진만으로는 실제 mm 단위 폭을 확정할 수 없어 상대값으로 표기합니다. "
        "탐지 이미지(박스 표시)는 첨부 참조."
        + _meta_observation(meta)
    )


def _mock_safety_grade(risk: RiskResult) -> str:
    state = config.STATE_GRADE_MAP.get(risk.grade, "-")
    return (
        "| 구분 | 결과 |\n"
        "|---|---|\n"
        f"| 위험도 점수 (Rule) | {risk.score} / 100 |\n"
        f"| 자가진단 등급 | **{risk.grade}** |\n"
        f"| 참고 상태평가등급 | {state} 수준 |\n\n"
        f"산정 근거(Rule 기여): {_contrib_line(risk)}"
    )


def _mock_overall_opinion(feat: CrackFeatures, risk: RiskResult) -> str:
    if risk.grade in ("위험", "긴급"):
        head = "탐지된 균열의 규모·수량으로 볼 때 **즉시 전문가 정밀 점검**이 필요합니다."
    elif risk.grade == "주의":
        head = "균열이 확인되어 **유지관찰 및 추가 점검**이 필요합니다."
    else:
        head = "현재 뚜렷한 위험 신호는 낮으나 **주기적 관찰**을 권장합니다."
    return (
        f"- {head}\n"
        "- 균열폭이 0.3mm를 초과하거나 시간이 지나며 진행(확장)될 경우 "
        "적극적 보수(충전·주입)가 필요합니다.\n"
        "- 구조적 원인(하중·부등침하) 가능성을 배제할 수 없으므로, "
        "**전문가의 정밀 점검**을 권고합니다."
    )


def _mock_narrative(feat, risk, rag, meta=None) -> dict:
    return {
        "inspection_result": _mock_inspection_result(feat, meta),
        "safety_grade": _mock_safety_grade(risk),
        "overall_opinion": _mock_overall_opinion(feat, risk),
    }


# ─────────────────────── 서술 섹션 LLM (API 있을 때) ───────────────────────
_SEC_TITLES = {
    "inspection_result": "2. 점검 결과",
    "safety_grade": "3. 안전등급 평가",
    "overall_opinion": "4. 종합의견",
}


def _meta_prompt_block(meta) -> str:
    """트리아지 비전 메타를 프롬프트에 주입(보이는 사실만, 지어내지 말 것)."""
    if not isinstance(meta, dict) or not any((v or "").strip() for v in meta.values()):
        return "- (비전 메타 없음)"
    keymap = {"structure_part": "부위", "material": "재질", "orientation": "방향",
              "branching": "분기", "efflorescence": "백태", "spalling": "박리",
              "notes": "소견"}
    lines = []
    for k, label in keymap.items():
        v = (meta.get(k, "") or "").strip()
        if v and v != "미상":
            lines.append(f"- {label}: {v}")
    return "\n".join(lines) or "- (비전 메타 없음)"


def _prompt(feat, risk, rag, meta=None) -> str:
    ev = "\n".join(f"- {e.text} (출처: {e.source})" for e in rag.evidences) \
        or "- (검색된 근거 없음)"
    contribs = "\n".join(
        f"- {c['rule']}: {c['detail']} (+{c['points']})" for c in risk.contributions
    ) or "- (해당 없음)"
    state = config.STATE_GRADE_MAP.get(risk.grade, "-")
    return f"""당신은 시설물 안전점검 보조 AI입니다. 비전문가(건물주·일반인)가 이해할 수 있게 쉽게,
그러나 정기안전점검 결과보고서(FMS·국토안전관리원) 서식의 어조로 서술하세요.
점수와 등급은 아래 값을 그대로 사용하고 재계산하지 마세요. 과장·단정을 피하고, 정밀점검을 권고하세요.

[분석 결과]
- 균열 개수: {feat.crack_count}
- 최고 탐지 신뢰도: {feat.max_confidence}
- 최장 균열 길이 비율(대각선 대비): {feat.max_length_ratio}
- 평균 균열 폭(px): {feat.avg_width_px}

[AI 비전 육안 관찰(트리아지) — 보이는 사실만, 새로 지어내지 말 것]
{_meta_prompt_block(meta)}

[위험도(코드 산정, 재계산 금지)]
- 점수: {risk.score} / 등급: {risk.grade} / 참고 상태평가등급: {state}
- 근거:
{contribs}

[안전기준 근거(RAG) — 이 사실만 인용, 새로운 기준을 지어내지 말 것]
{ev}

아래 3개 섹션만, 정확히 이 마크다운 제목을 그대로 사용해 한국어로 작성하세요.
표가 자연스러운 곳(점검 결과·안전등급)은 마크다운 표를 쓰세요.

## 2. 점검 결과
(탐지된 균열 개수·신뢰도·길이·폭을 정리)

## 3. 안전등급 평가
(위험도 점수/자가진단 등급/참고 상태평가등급과 산정 근거)

## 4. 종합의견
(즉시 조치/추가 점검/유지관찰 방향을 짧고 명확하게 — 3줄 이내)"""


def _parse_narrative(text: str) -> dict:
    """LLM 출력에서 '## 2./3./4.' 섹션을 분리."""
    # 각 섹션 헤더 위치로 분할
    pattern = re.compile(r"##\s*([234])\.\s*[^\n]*\n", re.MULTILINE)
    matches = list(pattern.finditer(text))
    by_num = {}
    for i, m in enumerate(matches):
        num = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        by_num[num] = text[start:end].strip()
    return {
        "inspection_result": by_num.get("2", ""),
        "safety_grade": by_num.get("3", ""),
        "overall_opinion": by_num.get("4", ""),
    }


def _fill_empty(narr: dict, feat, risk, rag, meta=None) -> dict:
    """LLM이 특정 섹션을 비우면 목업으로 보충 (강건성)."""
    fallback = _mock_narrative(feat, risk, rag, meta)
    for k, v in narr.items():
        if not (v or "").strip():
            narr[k] = fallback[k]
    return narr


def _llm_narrative(feat, risk, rag, meta=None) -> dict:
    """Claude (Anthropic) 로 서술 섹션 생성."""
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1800,
        messages=[{"role": "user", "content": _prompt(feat, risk, rag, meta)}],
    )
    return _fill_empty(_parse_narrative(msg.content[0].text), feat, risk, rag, meta)


def _solar_available():
    key = (config.UPSTAGE_API_KEY or "").strip()
    return bool(key) and key.isascii()


def _solar_narrative(feat, risk, rag, meta=None) -> dict:
    """Solar (Upstage) 채팅으로 서술 섹션 생성. OpenAI 호환 chat/completions."""
    import requests
    resp = requests.post(
        config.SOLAR_CHAT_ENDPOINT,
        headers={"Authorization": f"Bearer {config.UPSTAGE_API_KEY}",
                 "Content-Type": "application/json"},
        json={
            "model": config.SOLAR_CHAT_MODEL,
            "messages": [{"role": "user", "content": _prompt(feat, risk, rag, meta)}],
            "max_tokens": 1800,
            "temperature": 0.3,
        },
        timeout=40,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return _fill_empty(_parse_narrative(text), feat, risk, rag, meta)


def active_provider() -> str:
    """실제로 쓸 보고서 LLM 제공자 결정: claude → solar → mock."""
    p = config.REPORT_PROVIDER
    if p in ("claude", "solar", "mock"):
        return p
    if _has_api():
        return "claude"
    if _solar_available():
        return "solar"
    return "mock"


def provider_label() -> str:
    return {"claude": "Claude", "solar": "Solar", "mock": "목업"}.get(active_provider(), "목업")


# ────────────────────────────── 진입점 ──────────────────────────────
def generate(feat: CrackFeatures, risk: RiskResult, rag: RagResult, meta=None) -> Report:
    """6섹션 보고서 Report 생성. 결정적 섹션은 코드, 서술 섹션은 LLM/목업.
    제공자 체인(claude→solar→mock). LLM 호출 실패해도 목업으로 폴백해 앱이 죽지 않음.
    meta: 트리아지 비전이 읽어낸 메타(구조부위·재질·양상) — 기본현황·점검결과 보강용(없어도 됨).
    """
    provider = active_provider()
    narr = None
    try:
        if provider == "claude":
            narr = _llm_narrative(feat, risk, rag, meta)
        elif provider == "solar":
            narr = _solar_narrative(feat, risk, rag, meta)
    except Exception:
        narr = None                     # LLM 오류 → 목업 폴백
    if narr is None:
        narr = _mock_narrative(feat, risk, rag, meta)
    return Report(
        basic_info=_basic_info(meta),
        inspection_result=narr["inspection_result"],
        safety_grade=narr["safety_grade"],
        overall_opinion=narr["overall_opinion"],
        evidence_basis=_evidence_basis(rag),
        caveats=_caveats(),
    )
