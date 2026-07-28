"""
[3] Feature 추출 (features.py)
- 각 탐지 박스 내부에서 균열선을 정량화
- 이진화(adaptive threshold) -> 스켈레톤화 -> 중심선 길이/폭
- mm 절대측정 안 함 -> 픽셀 상대값
"""
import numpy as np
import cv2

from schemas import DetectResult, CrackFeatures

try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def _crack_mask(gray_crop):
    """균열 픽셀 마스크 추출 + 텍스처 노이즈 제거.

    하이브리드 접근:
    1. Adaptive threshold (거친 텍스처 표면에 효과적)
    2. Canny edge (깨끗한 벽면의 얇은 균열에 효과적)
    두 결과를 합쳐서 최종 마스크 생성.
    """
    h, w = gray_crop.shape[:2]
    min_area = max(20, int(0.0002 * h * w))
    keep = np.zeros((h, w), np.uint8)

    # 방법 1: Adaptive threshold (기존 방식, 거친 표면용)
    binv = cv2.adaptiveThreshold(
        gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=10)
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    keep = _filter_crack_components(binv, keep, min_area)

    # 방법 2: Canny edge (깨끗한 벽면용) - 결과가 부족할 때 보완
    if keep.sum() < 50:  # 마스크 픽셀이 부족하면 Canny 시도
        canny = cv2.Canny(gray_crop, 50, 150)
        keep = _filter_crack_components(canny, keep, min_area=30)

    return keep


def _filter_crack_components(binv, keep, min_area=20):
    """연결 요소 필터링: 균열 형태(가늘고 긴)만 남김."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        elong = max(bw, bh) / (min(bw, bh) + 1e-6)
        fill = area / (bw * bh + 1e-6)
        # 균열 조건: 충분히 크고, 가늘고, bbox를 성기게 채움
        if area >= min_area and elong >= 3.0 and fill <= 0.6:
            keep[lbl == i] = 255
        elif area >= 10 and elong >= 5.0 and fill <= 0.3:
            keep[lbl == i] = 255
    return keep


def _analyze_box(gray_crop):
    """박스 내부 crop(grayscale) -> (길이px, 평균폭px)."""
    if gray_crop.size == 0:
        return 0.0, 0.0
    mask = _crack_mask(gray_crop)
    crack_pixels = int((mask > 0).sum())
    if crack_pixels < 10:
        return 0.0, 0.0

    if _HAS_SKIMAGE:
        length = float(skeletonize(mask > 0).sum())   # 중심선 픽셀 수 ~= 길이
    else:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        length = float(sum(cv2.arcLength(c, False) for c in cnts)) / 2

    # 길이는 crop 대각선을 넘을 수 없음 -> 상한 클리핑 (노이즈 방어)
    diag = (gray_crop.shape[0] ** 2 + gray_crop.shape[1] ** 2) ** 0.5
    length = min(length, diag)
    width = crack_pixels / length if length > 0 else 0.0
    return length, width


def skeleton_mask(img_bgr, det: DetectResult):
    """탐지 박스별 균열 중심선(스켈레톤)을 원본 크기 마스크로 반환 (시각화용).
    - 재학습 없이 OpenCV 스켈레톤으로 '균열을 정밀하게 따라 그린' 오버레이를 만든다.
    - 반환: (H, W) uint8, 균열 중심선=255. 탐지 없으면 전부 0.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    full = np.zeros((H, W), np.uint8)
    for d in det.detections:
        if getattr(d, "label", "crack") != "crack":
            continue   # 중심선(스켈레톤)은 균열에만 — 면적 결함 박스엔 그리지 않음
        x1, y1, x2, y2 = d.box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        mask = _crack_mask(crop)
        if _HAS_SKIMAGE and (mask > 0).sum() >= 10:
            sk = (skeletonize(mask > 0).astype(np.uint8) * 255)
        else:
            sk = mask   # skimage 없으면 마스크 자체로 폴백
        region = full[y1:y2, x1:x2]
        full[y1:y2, x1:x2] = np.maximum(region, sk)
    return full


def extract(img_bgr, det: DetectResult) -> CrackFeatures:
    """[2차 MVP] 균열 채널(OpenCV 형태분석) + 면적 결함 요약(feat.defects) 분리 추출.
    - label=='crack' 탐지: crack_count/폭/길이/최고신뢰도(균열 채널) 계산.
    - 그 외 결함(철근노출·박락·백태 등): {label:{count,max_conf}} 로 집계(폭/길이 미측정).
    - 균열 전용 모델(모든 라벨 crack)이면 기존 동작과 동일.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    diag = (W ** 2 + H ** 2) ** 0.5

    feat = CrackFeatures()
    if not det.detections:
        return feat

    lengths, widths, crack_confs = [], [], []
    defects = {}
    for d in det.detections:
        label = getattr(d, "label", "crack")
        if label == "crack":
            x1, y1, x2, y2 = d.box
            length, width = _analyze_box(gray[y1:y2, x1:x2])
            lengths.append(length)
            if width > 0:
                widths.append(width)
            crack_confs.append(d.conf)
        else:
            slot = defects.setdefault(label, {"count": 0, "max_conf": 0.0})
            slot["count"] += 1
            slot["max_conf"] = max(slot["max_conf"], d.conf)

    feat.crack_count = len(crack_confs)
    feat.max_length_ratio = round(max(lengths) / diag, 4) if lengths else 0.0
    feat.avg_width_px = round(float(np.mean(widths)), 2) if widths else 0.0
    feat.max_confidence = round(max(crack_confs), 3) if crack_confs else 0.0
    feat.defects = defects
    return feat
