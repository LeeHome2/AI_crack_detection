"""
[1] 비전 트리아지 (triage.py) — YOLO 전에 돌리는 1차 게이트 + 메타데이터 추출
================================================================================
목적: 사진이 '판정할 가치가 있는지' 먼저 거르고, 같은 비전 호출 1회로 보고서용
메타데이터(구조 부위·재질·균열 양상)도 함께 받아온다(추가 API 호출 없음).

흐름(방어적, 앱이 절대 안 죽게):
  1) 무료 휴리스틱 선검사 — 라플라시안 분산으로 '흐림'이면 API 전에 즉시 재촬영 요청.
  2) Claude 비전 1회 호출(키 있으면) — verdict(ok/retake_far/retake_blur/not_crack) +
     meta(JSON) 반환. 이미지는 긴 변 축소해 토큰/비용 절약.
  3) 키 없음/에러/파싱 실패 → 휴리스틱 결과로 폴백(흐림 아니면 ok, meta 비움).

verdict 의미:
  ok         : 분석 진행
  retake_far : 너무 멀리/작게 찍음 → 균열 부위 근접 재촬영
  retake_blur: 초점 흐림/흔들림 → 다시 선명하게
  not_crack  : 콘크리트 구조물 균열 사진이 아님(무관 사진) → 반려
"""
import base64
import json
import re

import numpy as np
import cv2

import config


# ────────────────────────────── 준비 판단 ──────────────────────────────
def _has_vision() -> bool:
    key = (config.ANTHROPIC_API_KEY or "").strip()
    if not key or not key.isascii():   # 빈 값·비ASCII → 헤더 인코딩 오류 방지
        return False
    try:
        import anthropic  # noqa
        return True
    except Exception:
        return False


def active_provider() -> str:
    if not config.TRIAGE_ENABLED:
        return "off"
    return "claude" if _has_vision() else "heuristic"


def provider_label() -> str:
    return {"claude": "Claude 비전", "heuristic": "휴리스틱", "off": "꺼짐"}.get(
        active_provider(), "휴리스틱")


# ────────────────────────────── 휴리스틱 ──────────────────────────────
def _blur_score(img_bgr) -> float:
    """라플라시안 분산 — 선명할수록 큼, 흐릴수록 0에 가까움."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ────────────────────────────── 비전 인코딩 ──────────────────────────────
def _encode_jpeg(img_bgr) -> str:
    """긴 변을 VISION_MAX_SIDE로 축소 후 JPEG base64 (토큰/비용 절약)."""
    h, w = img_bgr.shape[:2]
    m = config.VISION_MAX_SIDE
    if m and max(h, w) > m:
        s = m / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


_VISION_PROMPT = """당신은 시설물 안전점검 보조 AI의 '1차 게이트'이자 '육안 소견' 담당입니다.
아래 사진이 콘크리트/강재 시설물의 '결함'을 자동 분석하기에 적합한지 판정하고,
사진에서 육안으로 보이는 결함과 정보를 메타데이터로 정리하세요.
과장·추측을 피하고 '실제로 보이는 것'만 적으세요. 정밀 위치·수치 판정은 하지 마세요.

대상 결함(6종): 균열 · 박리/박락 · 백태/누수 · 철근노출 · 강재손상 · 도장손상

반드시 아래 JSON 형식으로만 답하세요(코드블록·설명 없이 JSON만):
{
  "verdict": "ok | retake_far | retake_blur | not_crack",
  "message": "사용자에게 보여줄 한 줄 안내(한국어). ok면 빈 문자열",
  "structure_part": "벽|바닥|기둥|천장|외벽|계단|기타|미상",
  "material": "콘크리트|벽돌|타일|석재|강재|기타|미상",
  "orientation": "수직|수평|대각|불규칙|망상|미상",
  "branching": "없음|일부|많음|미상",
  "efflorescence": "있음|없음|미상",
  "spalling": "있음|없음|미상",
  "defects_observed": ["균열","철근노출"],
  "notes": "육안 소견 한 줄(예: 백태 동반 대각 균열, 철근노출 의심)"
}

- defects_observed: 위 6종 중 사진에서 '실제로 보이는' 결함만 한국어로 배열에 담으세요.
  확실치 않으면 넣지 마세요. 아무 결함도 안 보이면 빈 배열 [].

판정 기준:
- retake_far : 결함이 너무 작게/멀리 찍혀 형태 판별이 어려움 → 근접 재촬영 필요
- retake_blur: 초점 흐림·흔들림으로 결함 경계가 뭉개짐
- not_crack  : 시설물 결함 점검 대상 사진이 아님(사람·풍경·문서 등)
- ok         : 결함(또는 결함 의심)이 충분히 보임 → 분석 진행"""


def _vision_triage(img_bgr) -> dict:
    """Claude 비전 1회 호출 → verdict + meta dict. 실패 시 예외 → 상위에서 폴백."""
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    b64 = _encode_jpeg(img_bgr)
    msg = client.messages.create(
        model=config.VISION_MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": _VISION_PROMPT},
            ],
        }],
    )
    return _parse_vision(msg.content[0].text)


def _parse_vision(text: str) -> dict:
    """LLM 출력에서 JSON 블록 추출(코드펜스·잡텍스트 방어)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no json in vision response")
    return json.loads(m.group(0))


# 비전이 말한 결함명(한국어) → 정규 라벨(detector/Rule과 정합)
_VISION_DEFECT_MAP = {
    "균열": "crack", "박리": "spalling", "박락": "spalling", "박리/박락": "spalling",
    "백태": "efflorescence", "누수": "efflorescence", "백태/누수": "efflorescence",
    "철근노출": "rebar_exposure", "노출철근": "rebar_exposure",
    "강재손상": "steel_defect", "강재": "steel_defect", "부식": "steel_defect",
    "도장손상": "paint_damage", "도장": "paint_damage",
}


def _canon_defects(items) -> list:
    """비전이 준 결함명 리스트 → 정규 라벨(중복 제거, 매핑 안 되면 무시)."""
    out = []
    if isinstance(items, list):
        for it in items:
            lab = _VISION_DEFECT_MAP.get(str(it).strip())
            if lab and lab not in out:
                out.append(lab)
    return out


_VALID = {"ok", "retake_far", "retake_blur", "not_crack"}
_DEFAULT_MSG = {
    "retake_far": "균열 부위가 작게 찍혔어요. 균열에 더 가까이, 화면을 채우도록 다시 촬영해 주세요.",
    "retake_blur": "사진이 흐릿해요. 초점을 맞추고 흔들리지 않게 다시 촬영해 주세요.",
    "not_crack": "콘크리트 구조물의 균열 사진으로 보이지 않아요. 점검할 벽·바닥의 균열을 촬영해 주세요.",
}
_META_KEYS = ("structure_part", "material", "orientation",
              "branching", "efflorescence", "spalling", "notes")


# ────────────────────────────── 진입점 ──────────────────────────────
def triage(img_bgr) -> "object":
    """이미지 1장 → TriageResult. (schemas 순환참조 피하려 함수 안에서 import)"""
    from schemas import TriageResult

    if not config.TRIAGE_ENABLED:
        return TriageResult(verdict="ok", ok=True, provider="off")

    blur = _blur_score(img_bgr)
    # 1) 무료 흐림 선검사 — API 전에 즉시 컷 (비용 절약)
    if blur < config.TRIAGE_BLUR_MIN:
        return TriageResult(verdict="retake_blur", ok=False,
                            message=_DEFAULT_MSG["retake_blur"],
                            provider="heuristic", blur_score=round(blur, 1))

    # 2) 비전 호출 (키 있으면)
    if _has_vision():
        try:
            raw = _vision_triage(img_bgr)
            verdict = str(raw.get("verdict", "ok")).strip()
            if verdict not in _VALID:
                verdict = "ok"
            meta = {k: str(raw.get(k, "") or "").strip() for k in _META_KEYS}
            meta["defects_observed"] = _canon_defects(raw.get("defects_observed"))
            message = str(raw.get("message", "") or "").strip()
            if verdict != "ok" and not message:
                message = _DEFAULT_MSG.get(verdict, "")
            return TriageResult(verdict=verdict, ok=(verdict == "ok"),
                                message=message, provider="claude",
                                blur_score=round(blur, 1), meta=meta)
        except Exception:
            pass   # 비전 에러 → 휴리스틱 폴백(아래)

    # 3) 휴리스틱 폴백 — 흐림 아니면 통과(원거리·비균열은 판별 불가라 보수적으로 진행)
    return TriageResult(verdict="ok", ok=True, provider="heuristic",
                        blur_score=round(blur, 1))
