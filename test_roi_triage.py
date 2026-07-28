"""
ROI 기반 2단계 탐지 테스트
==========================
1) Claude 비전이 결함 의심 영역(ROI) 좌표를 반환
2) 해당 영역만 크롭하여 YOLO 실행
3) 좌표 변환 후 결과 병합

실행: python test_roi_triage.py [이미지경로]
"""
import sys
import os
import base64
import json
import re
import cv2
import numpy as np

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline import detector
from schemas import DetectResult, Detection

# ─────────────────────────────────────────────────────────────────────────────
# ROI 추출 프롬프트 (Claude Vision)
# ─────────────────────────────────────────────────────────────────────────────
_ROI_PROMPT = """당신은 시설물 안전점검 AI의 '관심 영역(ROI) 식별' 담당입니다.
이 사진에서 결함(균열, 박리, 철근노출, 백태 등)이 있거나 의심되는 영역의 좌표를 찾아주세요.

**중요**:
- 멀리 있는 배경 건물, 하늘, 나무 등은 제외하세요
- 가까이 있는 콘크리트/벽면/바닥 중 결함이 보이는 부분만 선택하세요
- 너무 작거나 멀리 보이는 영역은 제외하세요 (화면의 10% 이상 차지하는 영역만)

반드시 아래 JSON 형식으로만 답하세요:
{
  "rois": [
    {
      "x1": 0.1,
      "y1": 0.2,
      "x2": 0.6,
      "y2": 0.8,
      "description": "벽면 균열 영역",
      "defect_type": "crack"
    }
  ],
  "full_image_ok": false,
  "reason": "배경에 멀리 있는 건물이 포함되어 ROI 분리 필요"
}

좌표 규칙:
- x1, y1, x2, y2는 0~1 사이 비율값 (이미지 너비/높이 대비)
- x1 < x2, y1 < y2
- 영역이 겹쳐도 괜찮음 (나중에 병합)
- 결함이 전체 이미지에 고르게 분포하면 full_image_ok: true, rois: []

defect_type: crack | spalling | efflorescence | rebar_exposure | steel_defect | paint_damage | unknown
"""


def encode_image(img_bgr, max_side=1024):
    """이미지를 base64로 인코딩 (크기 제한)"""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def get_rois_from_vision(img_bgr) -> dict:
    """Claude Vision으로 ROI 좌표 추출"""
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    b64 = encode_image(img_bgr)

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
    print(f"[Vision 응답]\n{text}\n")

    # JSON 파싱
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("JSON을 찾을 수 없음")
    return json.loads(m.group(0))


def crop_roi(img_bgr, roi: dict) -> tuple:
    """ROI 좌표로 이미지 크롭, (cropped_img, offset_x, offset_y) 반환"""
    h, w = img_bgr.shape[:2]
    x1 = int(roi["x1"] * w)
    y1 = int(roi["y1"] * h)
    x2 = int(roi["x2"] * w)
    y2 = int(roi["y2"] * h)

    # 범위 보정
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cropped = img_bgr[y1:y2, x1:x2]
    return cropped, x1, y1, (x2-x1, y2-y1)


def detect_on_roi(img_crop, offset_x, offset_y) -> list:
    """크롭된 이미지에서 YOLO 실행, 좌표를 원본 기준으로 변환"""
    det = detector.detect(img_crop)

    transformed = []
    for d in det.detections:
        x1, y1, x2, y2 = d.box
        # 오프셋 적용
        new_box = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
        transformed.append(Detection(
            box=new_box,
            conf=d.conf,
            label=getattr(d, "label", "crack")
        ))
    return transformed


def merge_detections(all_detections: list, img_size: tuple, iou_threshold=0.5) -> DetectResult:
    """여러 ROI에서 나온 탐지 결과 병합 (NMS로 중복 제거)"""
    if not all_detections:
        return DetectResult(image_size=img_size, detections=[])

    # 단순 NMS: IoU가 높으면 신뢰도 높은 것만 유지
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

        # IoU가 낮은 것만 유지
        mask = iou < iou_threshold
        order = order[1:][mask]

    merged = [Detection(
        box=tuple(boxes[i].astype(int)),
        conf=float(scores[i]),
        label=labels[i]
    ) for i in keep]

    return DetectResult(image_size=img_size, detections=merged)


def draw_results(img_bgr, det_result, rois=None):
    """결과 시각화"""
    vis = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    # ROI 영역 표시 (파란색 점선)
    if rois:
        for roi in rois:
            x1 = int(roi["x1"] * w)
            y1 = int(roi["y1"] * h)
            x2 = int(roi["x2"] * w)
            y2 = int(roi["y2"] * h)
            # 점선 효과
            for i in range(x1, x2, 20):
                cv2.line(vis, (i, y1), (min(i+10, x2), y1), (255, 150, 0), 2)
                cv2.line(vis, (i, y2), (min(i+10, x2), y2), (255, 150, 0), 2)
            for i in range(y1, y2, 20):
                cv2.line(vis, (x1, i), (x1, min(i+10, y2)), (255, 150, 0), 2)
                cv2.line(vis, (x2, i), (x2, min(i+10, y2)), (255, 150, 0), 2)
            # 라벨
            cv2.putText(vis, f"ROI: {roi.get('description', '')[:20]}",
                       (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)

    # 탐지 결과 (빨간색 박스)
    for d in det_result.detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{getattr(d, 'label', 'crack')} {d.conf:.2f}"
        cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return vis


def roi_based_detect(img_bgr, debug=True):
    """ROI 기반 2단계 탐지 메인 함수"""
    h, w = img_bgr.shape[:2]

    # 1) Vision으로 ROI 추출
    print("=" * 60)
    print("[1단계] Claude Vision으로 ROI 추출 중...")
    try:
        roi_result = get_rois_from_vision(img_bgr)
    except Exception as e:
        print(f"Vision 오류: {e}")
        print("전체 이미지로 폴백")
        return detector.detect(img_bgr), None

    rois = roi_result.get("rois", [])
    full_ok = roi_result.get("full_image_ok", False)

    if full_ok or not rois:
        print("→ 전체 이미지 분석 (ROI 분리 불필요)")
        return detector.detect(img_bgr), None

    print(f"→ {len(rois)}개 ROI 식별됨")
    for i, roi in enumerate(rois):
        print(f"   ROI {i+1}: {roi.get('description', 'N/A')} "
              f"({roi['x1']:.2f},{roi['y1']:.2f})-({roi['x2']:.2f},{roi['y2']:.2f})")

    # 2) 각 ROI에서 YOLO 실행
    print("\n[2단계] ROI별 YOLO 탐지 중...")
    all_detections = []

    for i, roi in enumerate(rois):
        crop, ox, oy, (cw, ch) = crop_roi(img_bgr, roi)
        if cw < 50 or ch < 50:
            print(f"   ROI {i+1}: 너무 작음 ({cw}x{ch}), 건너뜀")
            continue

        print(f"   ROI {i+1}: {cw}x{ch} 크롭 → YOLO 실행")
        detections = detect_on_roi(crop, ox, oy)
        print(f"      → {len(detections)}개 탐지")
        all_detections.extend(detections)

    # 3) 결과 병합
    print(f"\n[3단계] 결과 병합 (NMS)")
    print(f"   총 {len(all_detections)}개 → ", end="")
    merged = merge_detections(all_detections, (w, h))
    print(f"{len(merged.detections)}개 (중복 제거)")

    return merged, rois


def compare_methods(img_path):
    """기존 방식 vs ROI 방식 비교"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지 로드 실패: {img_path}")
        return

    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n{'='*60}")
    print(f"이미지: {name}")
    print(f"크기: {img.shape[1]}x{img.shape[0]}")
    print("="*60)

    # 기존 방식 (전체 이미지)
    print("\n[기존 방식] 전체 이미지 YOLO")
    det_full = detector.detect(img)
    print(f"→ {len(det_full.detections)}개 탐지")

    # ROI 기반 방식
    print("\n" + "-"*60)
    det_roi, rois = roi_based_detect(img)

    # 결과 저장
    out_dir = os.path.join(os.path.dirname(img_path), "roi_test_results")
    os.makedirs(out_dir, exist_ok=True)

    # 기존 방식 결과
    vis_full = draw_results(img, det_full, None)
    cv2.imwrite(os.path.join(out_dir, f"{name}_full.jpg"), vis_full)

    # ROI 방식 결과
    vis_roi = draw_results(img, det_roi, rois)
    cv2.imwrite(os.path.join(out_dir, f"{name}_roi.jpg"), vis_roi)

    # 비교 출력
    print("\n" + "="*60)
    print(f"[비교 결과]")
    print(f"  기존 방식: {len(det_full.detections)}개 탐지")
    print(f"  ROI 방식:  {len(det_roi.detections)}개 탐지")
    print(f"  결과 저장: {out_dir}")
    print("="*60)

    return det_full, det_roi


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 기본 테스트 이미지
        test_dir = r"C:\Users\user\Desktop\crack_test"
        test_images = ["test1_crack.jpg", "test3_spalling.jpg"]

        print("ROI 기반 2단계 탐지 테스트")
        print("="*60)

        for img_name in test_images:
            img_path = os.path.join(test_dir, img_name)
            if os.path.exists(img_path):
                compare_methods(img_path)
            else:
                print(f"파일 없음: {img_path}")
    else:
        compare_methods(sys.argv[1])
