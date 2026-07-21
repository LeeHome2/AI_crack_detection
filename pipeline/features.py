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
    - adaptive threshold로 어두운 선 추출
    - 연결요소 중 '균열다운' 것(충분히 크고 가늘고 긴)만 남김 → 화강암/거친 표면 speckle 제거
    """
    binv = cv2.adaptiveThreshold(
        gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=10)
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    h, w = gray_crop.shape[:2]
    min_area = max(30, int(0.002 * h * w))     # 너무 작은 speckle 제거
    keep = np.zeros_like(binv)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        elong = max(bw, bh) / (min(bw, bh) + 1e-6)   # 균열은 가늘고 긺
        fill = area / (bw * bh + 1e-6)                # 균열은 bbox를 성기게 채움
        if area >= min_area and elong >= 3.0 and fill <= 0.6:
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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    diag = (W ** 2 + H ** 2) ** 0.5

    feat = CrackFeatures(crack_count=len(det.detections))
    if not det.detections:
        return feat

    lengths, widths, confs = [], [], []
    for d in det.detections:
        x1, y1, x2, y2 = d.box
        length, width = _analyze_box(gray[y1:y2, x1:x2])
        lengths.append(length)
        if width > 0:
            widths.append(width)
        confs.append(d.conf)

    feat.max_length_ratio = round(max(lengths) / diag, 4) if lengths else 0.0
    feat.avg_width_px = round(float(np.mean(widths)), 2) if widths else 0.0
    feat.max_confidence = round(max(confs), 3) if confs else 0.0
    return feat
