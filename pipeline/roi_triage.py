"""
[ROI 기반 2단계 탐지] roi_triage.py
====================================
1단계: Claude Vision이 결함 의심 영역(ROI) 좌표 반환
2단계: ROI별로 YOLO 실행 → 좌표 변환 → 결과 병합

config.ROI_TRIAGE_ENABLED = True 일 때 활성화
"""
import base64
import json
import re
import cv2
import numpy as np

import config
from schemas import DetectResult, Detection


# ─────────────────────────────────────────────────────────────────────────────
# ROI 추출 프롬프트
# ─────────────────────────────────────────────────────────────────────────────
_ROI_PROMPT = """당신은 시설물 안전점검 AI의 '관심 영역(ROI) 식별' 담당입니다.

## 점검 대상 판별 (중요!)

**점검 대상 O (콘크리트 구조물)**:
- 콘크리트 벽면, 기둥, 보, 슬래브, 바닥
- 시멘트/모르타르 마감면
- 콘크리트 옹벽, 교각, 터널 내벽

**점검 대상 X (제외 영역)**:
- 천장 텍스 타일(acoustic ceiling tiles) - 격자 패턴이나 이음새가 있음
- 석고보드 천장 - 타일 이음새가 균열처럼 보일 수 있음
- 나무/금속 마감재
- 창문, 문, 가구 등 비구조재
- 하늘, 건물 외관, 나무, 차량 등 배경

## 판단 기준

**full_image_ok = true (ROI 분리 불필요)**:
- 콘크리트/시멘트 벽면/바닥이 화면 대부분(70%+)을 채움
- 결함(균열 등)이 화면 전체에 분포
- 배경이나 비점검 대상이 거의 없음

**full_image_ok = false (ROI 분리 필요)**:
- 원거리/전경 촬영: 건물 전체, 하늘, 나무, 차량 등 배경이 화면 30%+ 차지
- 천장 텍스 타일과 콘크리트 벽면이 함께 찍힘 → 벽면만 ROI로
- 점검 대상이 아닌 영역이 명확히 존재

## ROI 추출 규칙 (full_image_ok=false일 때만)
- 콘크리트/시멘트 표면만 포함 (천장 타일, 비구조재 제외)
- 결함 주변 여유 20%+ 넉넉하게
- ROI 하나가 화면의 최소 30% 이상 차지하도록

반드시 아래 JSON 형식으로만 답하세요:
{
  "rois": [
    {"x1": 0.0, "y1": 0.3, "x2": 1.0, "y2": 1.0, "description": "하단 콘크리트 벽면", "defect_type": "crack"}
  ],
  "full_image_ok": false,
  "reason": "상단에 천장 타일 포함 - 벽면만 점검 대상"
}

- 좌표: 0~1 비율값. ROI는 넓게 잡을 것 (최소 30% 면적)
- 천장 타일의 격자 이음새/패턴은 균열이 아님! ROI에서 제외할 것
- defect_type: crack | spalling | efflorescence | rebar_exposure | unknown"""


def _has_vision() -> bool:
    """Claude Vision 사용 가능 여부"""
    key = (config.ANTHROPIC_API_KEY or "").strip()
    if not key or not key.isascii():
        return False
    try:
        import anthropic
        return True
    except Exception:
        return False


def _encode_jpeg(img_bgr, max_side=1024) -> str:
    """이미지를 base64로 인코딩"""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def get_rois(img_bgr) -> dict:
    """Claude Vision으로 ROI 좌표 추출"""
    if not _has_vision():
        return {"rois": [], "full_image_ok": True, "reason": "Vision API 없음"}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        b64 = _encode_jpeg(img_bgr)

        msg = client.messages.create(
            model=config.VISION_MODEL,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": _ROI_PROMPT},
                ],
            }],
        )

        text = msg.content[0].text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"rois": [], "full_image_ok": True, "reason": "JSON 파싱 실패"}
        return json.loads(m.group(0))

    except Exception as e:
        print(f"[roi_triage] Vision 오류: {e}")
        return {"rois": [], "full_image_ok": True, "reason": str(e)}


def crop_roi(img_bgr, roi: dict) -> tuple:
    """ROI 좌표로 이미지 크롭
    Returns: (cropped_img, offset_x, offset_y, (width, height))
    """
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(roi["x1"] * w))
    y1 = max(0, int(roi["y1"] * h))
    x2 = min(w, int(roi["x2"] * w))
    y2 = min(h, int(roi["y2"] * h))

    cropped = img_bgr[y1:y2, x1:x2]
    return cropped, x1, y1, (x2-x1, y2-y1)


def transform_detections(detections: list, offset_x: int, offset_y: int) -> list:
    """크롭 영역의 탐지 결과를 원본 좌표로 변환"""
    transformed = []
    for d in detections:
        x1, y1, x2, y2 = d.box
        new_box = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
        transformed.append(Detection(
            box=new_box,
            conf=d.conf,
            label=getattr(d, "label", "crack")
        ))
    return transformed


def merge_detections(all_detections: list, img_size: tuple, iou_threshold=0.5) -> DetectResult:
    """여러 ROI 탐지 결과 병합 (NMS로 중복 제거)"""
    if not all_detections:
        return DetectResult(image_size=img_size, detections=[])

    boxes = np.array([d.box for d in all_detections])
    scores = np.array([d.conf for d in all_detections])
    labels = [getattr(d, "label", "crack") for d in all_detections]

    keep = []
    order = scores.argsort()[::-1]

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # IoU 계산
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])

        iou = inter / (area_i + area_j - inter + 1e-6)
        mask = iou < iou_threshold
        order = order[1:][mask]

    merged = [Detection(
        box=tuple(boxes[i].astype(int)),
        conf=float(scores[i]),
        label=labels[i]
    ) for i in keep]

    return DetectResult(image_size=img_size, detections=merged)


def detect_with_roi(img_bgr, detector_fn) -> tuple:
    """ROI 기반 2단계 탐지

    Args:
        img_bgr: 원본 이미지
        detector_fn: YOLO 탐지 함수 (img -> DetectResult)

    Returns:
        (DetectResult, rois_list or None)
    """
    h, w = img_bgr.shape[:2]

    # ROI가 비활성화면 전체 이미지 탐지
    if not getattr(config, "ROI_TRIAGE_ENABLED", False):
        return detector_fn(img_bgr), None

    # 1) Vision으로 ROI 추출
    roi_result = get_rois(img_bgr)
    rois = roi_result.get("rois", [])
    full_ok = roi_result.get("full_image_ok", False)

    # ROI 없거나 전체 OK면 기존 방식
    if full_ok or not rois:
        return detector_fn(img_bgr), None

    # 2) 각 ROI에서 탐지
    all_detections = []
    valid_rois = []

    for roi in rois:
        crop, ox, oy, (cw, ch) = crop_roi(img_bgr, roi)
        if cw < 50 or ch < 50:
            continue  # 너무 작은 ROI 스킵

        det = detector_fn(crop)
        transformed = transform_detections(det.detections, ox, oy)
        all_detections.extend(transformed)
        valid_rois.append(roi)

    # 3) 결과 병합
    merged = merge_detections(all_detections, (w, h))
    return merged, valid_rois


def is_enabled() -> bool:
    """ROI 트리아지 활성화 여부"""
    return getattr(config, "ROI_TRIAGE_ENABLED", False) and _has_vision()
