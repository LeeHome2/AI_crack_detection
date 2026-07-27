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


def _basic_info(meta=None, user_info=None) -> str:
    """1. 시설물 기본현황 — 사용자 입력 + (트리아지 비전 메타) + 점검 메타."""
    today = datetime.date.today().isoformat()
    part = _meta_val(meta, "structure_part")     # 벽/바닥/기둥/천장/외벽…
    material = _meta_val(meta, "material")        # 콘크리트/벽돌/타일…
    struct = "철근콘크리트" if material in ("콘크리트", "미상") else material

    # 사용자 입력 우선, 없으면 플레이스홀더
    ui = user_info or {}
    facility = ui.get("facility_name") or "(미입력)"
    location = ui.get("location") or "(미입력)"
    inspector = ui.get("inspector") or "(미입력)"
    part_detail = ui.get("part_detail") or ""

    # 점검 부위: 사용자 상세입력 > AI 추정 > 미상
    if part_detail:
        part_str = part_detail
    elif part and part != "미상":
        part_str = f"{part} (AI 추정)"
    else:
        part_str = "(미입력)"

    return (
        "| 항목 | 내용 |\n"
        "|---|---|\n"
        f"| 시설물명 | {facility} |\n"
        f"| 위치 | {location} |\n"
        f"| 점검 부위 | {part_str} |\n"
        f"| 구조형식 | {struct} |\n"
        f"| 점검일자 | {today} |\n"
        f"| 점검자 | {inspector} |\n"
        "| 점검방식 | 사진 기반 AI 자가진단 (비전 트리아지 + YOLO 탐지 + OpenCV 형태분석) |"
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


def _model_defect_set(feat) -> set:
    """탐지 모델이 잡은 결함 집합(균열 채널 + 면적 결함)."""
    s = set(getattr(feat, "defects", None) or {})
    if getattr(feat, "crack_count", 0):
        s.add("crack")
    return s


def _vision_crosscheck(meta, feat) -> str:
    """[교차검증] 비전이 관찰했으나 탐지 모델이 놓친 결함을 육안 소견으로 보강.
    탐지 정확도가 낮아도 근거 있는 결과가 나오게 하는 안전장치.
    """
    if not isinstance(meta, dict):
        return ""
    vision = meta.get("defects_observed") or []
    if not vision:
        return ""
    model = _model_defect_set(feat)
    ko = config.DEFECT_KO
    missed = [ko.get(d, d) for d in vision if d not in model]
    confirmed = [ko.get(d, d) for d in vision if d in model]
    out = []
    if confirmed:
        out.append(f"비전·탐지모델 일치: {', '.join(confirmed)}")
    if missed:
        out.append(f"**모델 미검출·비전 관찰: {', '.join(missed)}** (탐지 정확도 초기 단계 — 해당 부위 재확인 권장)")
    return ("\n**AI 비전 교차검증:** " + " · ".join(out)) if out else ""


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


def _defect_rows(feat: CrackFeatures) -> str:
    """[2차 MVP] 균열 외 탐지 결함(feat.defects)을 표 행으로. 없으면 빈 문자열."""
    defects = getattr(feat, "defects", None) or {}
    if not defects:
        return ""
    ko = config.DEFECT_KO
    rows = []
    for label, stat in defects.items():
        name = ko.get(label, label)
        rows.append(f"| {name} | {stat.get('count', 0)}개소 (최고 신뢰도 {stat.get('max_conf', 0):.2f}) |")
    return "\n".join(rows)


def _defect_prompt_block(feat: CrackFeatures) -> str:
    """LLM 프롬프트용 복합 결함 목록(보이는 사실만)."""
    defects = getattr(feat, "defects", None) or {}
    if not defects:
        return "- (균열 외 탐지 결함 없음)"
    ko = config.DEFECT_KO
    return "\n".join(
        f"- {ko.get(l, l)}: {s.get('count',0)}개소 (신뢰도 {s.get('max_conf',0):.2f})"
        for l, s in defects.items())


# ─────────────────────── 서술 섹션 목업 (API 없을 때) ───────────────────────
def _mock_inspection_result(feat: CrackFeatures, meta=None) -> str:
    length_pct = round(feat.max_length_ratio * 100, 1)
    defect_rows = _defect_rows(feat)

    # 신뢰도 해석
    conf = feat.max_confidence
    if conf >= 0.8:
        conf_desc = f"{conf} (높음)"
    elif conf >= 0.5:
        conf_desc = f"{conf} (보통)"
    else:
        conf_desc = f"{conf} (낮음)"

    crack_block = (
        "업로드된 사진을 AI 모델로 분석한 결과, 다음과 같은 결함이 탐지되었습니다.\n\n"
        "**균열 탐지 결과**\n\n"
        "| 항목 | 측정값 | 비고 |\n"
        "|---|---|---|\n"
        f"| 탐지된 균열 | **{feat.crack_count}개소** | YOLO 객체탐지 |\n"
        f"| 탐지 신뢰도 | {conf_desc} | 최고값 기준 |\n"
        f"| 균열 길이 | 대각선 대비 약 **{length_pct}%** | 최장 기준 |\n"
        f"| 균열 폭 | **{feat.avg_width_px}px** | 픽셀 평균 |\n"
    )
    composite = ""
    if defect_rows:
        composite = (
            "\n**복합 결함 (균열 외)**\n\n"
            "| 결함 유형 | 탐지 결과 |\n"
            "|---|---|\n"
            f"{defect_rows}\n"
        )
    tail = (
        "\n> 📌 **참고:** 사진만으로는 실제 mm 단위 폭을 확정할 수 없어 "
        "상대값(픽셀/비율)으로 표기합니다. 정확한 폭 측정은 현장 정밀점검이 필요합니다."
    )
    return crack_block + composite + tail + _meta_observation(meta) + _vision_crosscheck(meta, feat)


def _mock_safety_grade(risk: RiskResult) -> str:
    state = config.STATE_GRADE_MAP.get(risk.grade, "-")
    # 등급별 아이콘과 색상 설명
    grade_icon = {"정상": "🟢", "주의": "🟡", "위험": "🟠", "긴급": "🔴"}.get(risk.grade, "⚪")
    grade_action = {
        "정상": "현재 뚜렷한 결함이 없거나 경미한 수준입니다.",
        "주의": "결함이 확인되어 정기적 관찰이 필요합니다.",
        "위험": "결함이 상당하여 전문가 점검이 필요합니다.",
        "긴급": "심각한 결함으로 즉시 조치가 필요합니다.",
    }.get(risk.grade, "")

    return (
        f"{grade_icon} **자가진단 등급: {risk.grade}**\n\n"
        "| 구분 | 결과 |\n"
        "|---|---|\n"
        f"| 위험도 점수 | **{risk.score}점** / 100점 |\n"
        f"| 자가진단 등급 | **{risk.grade}** |\n"
        f"| 참고 상태평가등급 | {state} 수준 |\n\n"
        f"> {grade_action}\n\n"
        f"**산정 근거 (규칙 기반 평가)**\n\n{_contrib_line(risk)}"
    )


def _mock_overall_opinion(feat: CrackFeatures, risk: RiskResult) -> str:
    defects = getattr(feat, "defects", None) or {}

    # 등급별 종합 의견 헤더
    if risk.grade == "긴급":
        head = "🔴 **즉시 조치 필요** — 탐지된 결함이 심각한 수준으로, 즉시 전문가 정밀진단 및 긴급 보수가 필요합니다."
    elif risk.grade == "위험":
        head = "🟠 **전문가 점검 권고** — 결함의 종류와 규모로 볼 때 빠른 시일 내 전문가 정밀점검을 받으시기 바랍니다."
    elif risk.grade == "주의":
        head = "🟡 **정기 관찰 필요** — 결함이 확인되었으나 당장 위험한 수준은 아닙니다. 정기적으로 상태를 확인하세요."
    else:
        head = "🟢 **양호** — 현재 뚜렷한 위험 신호는 없습니다. 주기적으로 상태를 관찰하시기 바랍니다."

    lines = [head, ""]

    # 조치 권고사항
    lines.append("**권고 조치사항**")

    # 복합 결함별 조치 방향(고위험 결함 우선 고지)
    if "rebar_exposure" in defects:
        lines.append("- ⚠️ **철근노출** 확인 → 부식·단면손실 우려. 방청처리 및 단면복구 필요")
    if "spalling" in defects:
        lines.append("- ⚠️ **박리/박락** 확인 → 들뜬 부위 제거 후 단면복구 권고")
    if "efflorescence" in defects:
        lines.append("- 💧 **백태(누수흔적)** 확인 → 누수 원인 규명 및 방수 보수 검토")

    if feat.crack_count > 0:
        lines.append("- 균열폭 0.3mm 초과 또는 시간에 따른 진행(확장) 시 적극적 보수(충전/주입) 필요")

    lines.append("- 구조적 원인(하중, 부등침하 등) 가능성 배제 불가 → **전문가 정밀점검 권고**")

    return "\n".join(lines)


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
    if not isinstance(meta, dict) or not meta:
        return "- (비전 메타 없음)"
    keymap = {"structure_part": "부위", "material": "재질", "orientation": "방향",
              "branching": "분기", "efflorescence": "백태", "spalling": "박리",
              "notes": "소견"}
    lines = []
    for k, label in keymap.items():
        v = (meta.get(k, "") or "").strip()   # 문자열 값만
        if v and v != "미상":
            lines.append(f"- {label}: {v}")
    obs = meta.get("defects_observed") or []   # 리스트: 비전이 육안으로 본 결함
    if obs:
        ko = config.DEFECT_KO
        lines.append("- 비전 관찰 결함(육안): " + ", ".join(ko.get(d, d) for d in obs))
    return "\n".join(lines) or "- (비전 메타 없음)"


def _prompt(feat, risk, rag, meta=None) -> str:
    ev = "\n".join(f"- {e.text} (출처: {e.source})" for e in rag.evidences) \
        or "- (검색된 근거 없음)"
    contribs = "\n".join(
        f"- {c['rule']}: {c['detail']} (+{c['points']})" for c in risk.contributions
    ) or "- (해당 없음)"
    state = config.STATE_GRADE_MAP.get(risk.grade, "-")
    length_pct = round(feat.max_length_ratio * 100, 1)

    # 등급별 어조 가이드
    tone_guide = {
        "정상": "안심시키되 지속 관찰 권유",
        "주의": "경각심을 주되 과장하지 않음",
        "위험": "심각성을 전달하되 당장 무너진다는 식 과장 금지",
        "긴급": "즉각 조치 필요성을 명확히 전달",
    }.get(risk.grade, "")

    return f"""당신은 시설물 안전점검 전문 AI입니다.

**목표:** 비전문가(건물주, 일반인)도 쉽게 이해할 수 있으면서, 정기안전점검 보고서 수준의 전문성을 갖춘 문서 작성

**어조:** {tone_guide}

---
## 입력 데이터 (변경/재계산 금지, 그대로 인용)

**균열 탐지 결과**
| 항목 | 값 |
|---|---|
| 균열 개수 | {feat.crack_count}개소 |
| 최고 신뢰도 | {feat.max_confidence} |
| 균열 길이 | 대각선 대비 {length_pct}% |
| 균열 폭 | {feat.avg_width_px}px (상대값) |

**복합 결함 (YOLO 탐지)**
{_defect_prompt_block(feat)}

**AI 비전 육안 관찰 (트리아지)**
{_meta_prompt_block(meta)}

**위험도 평가 (코드 산정 — 재계산 금지)**
- 점수: **{risk.score}점** / 등급: **{risk.grade}** / 참고 상태평가등급: {state}
- 산정 근거:
{contribs}

**안전기준 근거 (RAG 검색 — 새로 지어내지 말 것)**
{ev}

---
## 작성 규칙 (필수)

1. **데이터 정확성**: 위 수치를 그대로 사용. 점수/등급 재계산 금지.
2. **균열폭 표현**: mm 단위로 단정 금지. "{feat.avg_width_px}px"는 상대값임을 명시. "0.3mm 기준"은 참고로만 인용.
3. **결함 구분**: YOLO 탐지 결함과 비전 관찰 결함 명확히 구분. 비전 관찰은 "(육안)"으로 표시.
4. **과장 금지**: "붕괴 위험", "매우 심각" 등 과도한 표현 자제. 사실 기반으로 서술.
5. **전문가 권고**: 반드시 "정밀점검 권고" 문구 포함.

---
## 출력 형식 (마크다운, 정확히 이 제목 사용)

## 2. 점검 결과
- 탐지된 결함을 마크다운 표로 정리
- 항목: 균열 개수, 신뢰도, 길이(비율), 폭(px)
- 복합 결함 있으면 별도 표로 추가
- 비전 관찰 결함은 "(육안)" 표시로 구분

## 3. 안전등급 평가
- 위험도 점수, 자가진단 등급, 참고 등급을 표로 정리
- 등급 산정 근거 간략 설명
- 등급별 의미 한 줄 설명 (🟢정상/🟡주의/🟠위험/🔴긴급)

## 4. 종합의견
- 3줄 이내로 핵심만 (즉시조치/추가점검/유지관찰 중 해당 방향)
- 구체적 권고 조치 1~2개
- 마지막에 "전문가 정밀점검 권고" 포함"""


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
    # 비전↔탐지 교차검증은 LLM이 빠뜨려도 항상 붙게(데모 안전장치·중복 방지)
    cc = _vision_crosscheck(meta, feat)
    if cc and "AI 비전 교차검증" not in (narr.get("inspection_result") or ""):
        narr["inspection_result"] = (narr.get("inspection_result") or "") + cc
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
def generate(feat: CrackFeatures, risk: RiskResult, rag: RagResult,
             meta=None, user_info=None) -> Report:
    """6섹션 보고서 Report 생성. 결정적 섹션은 코드, 서술 섹션은 LLM/목업.
    제공자 체인(claude→solar→mock). LLM 호출 실패해도 목업으로 폴백해 앱이 죽지 않음.
    meta: 트리아지 비전이 읽어낸 메타(구조부위·재질·양상) — 기본현황·점검결과 보강용(없어도 됨).
    user_info: 사용자가 입력한 시설물 정보 (시설물명, 위치, 점검자 등).
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

    # 사용자 비고가 있으면 종합의견 끝에 추가
    remarks = (user_info or {}).get("remarks", "")
    if remarks:
        narr["overall_opinion"] += f"\n\n**점검자 메모:** {remarks}"

    return Report(
        basic_info=_basic_info(meta, user_info),
        inspection_result=narr["inspection_result"],
        safety_grade=narr["safety_grade"],
        overall_opinion=narr["overall_opinion"],
        evidence_basis=_evidence_basis(rag),
        caveats=_caveats(),
    )
