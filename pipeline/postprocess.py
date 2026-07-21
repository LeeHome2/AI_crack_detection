"""
탐지 후처리 (postprocess.py) — 재학습 없이 두 문제 완화:
1) 이음새(타일 줄눈 등) 오탐: 균열은 불규칙, 이음새는 곧은 직선 → '직선성'으로 매우 반듯한
   박스를 제외 (보수적: 애매하면 보존, 전부 지워지면 원복).
2) 물리적 균열 개수: 박스 수가 아니라, 탐지 박스들의 균열 중심선을 원본에 모아 연결요소 수로
   실제 이어진 균열 덩어리 개수를 센다 (한 줄 균열이 여러 박스로 쪼개져도 1개로).
※ 임계값은 config에서 조정 (실사진으로 튜닝 필요). 근본 해결은 자체데이터 재학습.
"""
import numpy as np
import cv2

import config
from schemas import DetectResult
from pipeline.features import skeleton_mask

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None


def _line_mask(gray_crop):
    """이음새 판별용 마스크 — 균열모양 필터(가늘고 성긴) 없이, 박스 내 '가장 큰 어두운 선'만 추출.
    features._crack_mask 는 곧고 꽉 찬 선(이음새)을 일부러 버리므로 이음새 탐지엔 부적합 →
    여기선 임계화 후 최대 연결요소만 남겨(직선·곡선 무관) 그 형태로 직선성을 잰다.
    """
    binv = cv2.adaptiveThreshold(
        gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=10)
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    if n <= 1:
        return None
    # 배경(0) 제외 최대 면적 성분
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    if stats[i, cv2.CC_STAT_AREA] < 10:
        return None
    return (lbl == i)


def _box_skeleton(gray, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = gray[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    m = _line_mask(crop)
    if m is None:
        return None
    return skeletonize(m) if skeletonize is not None else m


def _straightness(skel_bool):
    """주축 대비 수직 편차 / 주축 길이. 작을수록 곧은 직선(이음새 의심). 짧으면 판단 보류."""
    ys, xs = np.nonzero(skel_bool)
    if len(xs) < 15:
        return 1.0
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ vt[0]
    perp = pts @ vt[1]
    span = float(np.ptp(proj))
    if span < 1:
        return 1.0
    return float(perp.std() / span)


def filter_seams(img_bgr, det: DetectResult) -> DetectResult:
    """매우 반듯한 직선(이음새 의심) 박스 제외. 보수적."""
    if not config.SEAM_FILTER_ENABLED or not det.detections:
        return det
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kept = []
    for d in det.detections:
        sk = _box_skeleton(gray, d.box)
        if sk is None or _straightness(sk) >= config.SEAM_STRAIGHTNESS_MAX:
            kept.append(d)      # 균열(구불) 또는 판단보류 → 보존
        # else: 매우 곧음 → 이음새로 보고 제외
    out = DetectResult(image_size=det.image_size)
    out.detections = kept if kept else det.detections   # 전부 지워지면 원복(과필터 방지)
    return out


def physical_crack_count(img_bgr, det: DetectResult) -> int:
    """탐지 박스들의 균열 중심선 → dilate로 인접 조각 연결 → 연결요소 수 = 물리적 개수."""
    if not det.detections:
        return 0
    sk = skeleton_mask(img_bgr, det)
    if not sk.any():
        return len(det.detections)
    sk = cv2.dilate(sk, np.ones((5, 5), np.uint8), iterations=2)
    n, _ = cv2.connectedComponents(sk)
    return max(1, n - 1)
