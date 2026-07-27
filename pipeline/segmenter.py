"""
[고도화] 균열 세그멘테이션 (segmenter.py) — 하이브리드 탐지의 '균열=seg' 채널.
================================================================================
YOLOv8s-seg(단일 클래스 crack) 타일 추론 → 균열 마스크(픽셀 정밀) + 마스크에서 뽑은
crack Detection(연결요소 bbox). 면적 결함(철근노출·박리 등)은 기존 bbox detector가 담당하고,
orchestrator 가 둘을 합친다(균열=seg / 면적=bbox).

설계 원칙:
- 기본 OFF(config.SEG_HYBRID_ENABLED). seg best.pt 를 models/ 에 두고 env 로 켜면 활성.
- 방어적: 모델 없음/로드 실패/비활성/추론 예외 → 빈 결과 → 앱은 기존 bbox 경로로 계속.
- 출력은 기존 파이프라인과 호환되는 Detection(label='crack') 이라 features·rules·렌더 무변경으로 흐름.
  (마스크 직접 특징은 mask_features()로 별도 제공 — 검증 후 features 경로에 연결 예정.)

검증 후 활성 순서:
  1) 데스크탑 seg 학습 best.pt → models/yolov8s_seg_crack_tiled_best.pt 로 커밋(git-lfs) ✅완료(72.2% mask)
  2) SEG_HYBRID_ENABLED=1 env 로 켜고 실사진으로 마스크 품질 확인
  3) 좋으면 features 를 mask_features() 기반으로 승격(균열 길이·폭을 마스크에서 직접)
"""
import os

import numpy as np
import cv2

import config
from schemas import Detection, DetectResult

try:
    from skimage.morphology import skeletonize
    _HAS_SK = True
except Exception:
    _HAS_SK = False

_model = None   # 1회 로드 캐시


def _valid(path):
    try:
        return bool(path) and os.path.exists(path) and os.path.getsize(path) > 1024
    except OSError:
        return False


def is_ready() -> bool:
    """하이브리드가 켜져 있고 seg 가중치가 실제 로드 가능한 상태인지."""
    return bool(config.SEG_HYBRID_ENABLED) and _valid(config.SEG_WEIGHTS)


def load_model():
    global _model
    if _model is not None:
        return _model
    if not is_ready():
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(config.SEG_WEIGHTS)
    except Exception as e:
        print(f"[segmenter] 로드 실패({config.SEG_WEIGHTS}) → seg 비활성: {e}")
        _model = None
    return _model


def _positions(total, tile, stride):
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def crack_mask(img_bgr):
    """타일 추론으로 원본 크기 균열 마스크(uint8, 255=균열) + 인스턴스 최고 신뢰도 반환."""
    model = load_model()
    H, W = img_bgr.shape[:2]
    full = np.zeros((H, W), np.uint8)
    max_conf = 0.0
    if model is None:
        return full, 0.0

    tile = config.TILE
    stride = int(tile * (1 - config.OVERLAP))
    for ty in _positions(H, tile, stride):
        for tx in _positions(W, tile, stride):
            crop = img_bgr[ty:ty + tile, tx:tx + tile]
            ch, cw = crop.shape[:2]
            try:
                r = model.predict(crop, conf=config.SEG_CONF, verbose=False)[0]
            except Exception:
                continue
            if getattr(r, "masks", None) is None:
                continue
            data = r.masks.data                          # (n, mh, mw) 0/1 텐서
            confs = (r.boxes.conf.tolist()
                     if getattr(r, "boxes", None) is not None else [])
            for i in range(int(data.shape[0])):
                m = data[i].cpu().numpy().astype(np.uint8)
                if m.shape[:2] != (ch, cw):
                    m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)
                region = full[ty:ty + ch, tx:tx + cw]
                full[ty:ty + ch, tx:tx + cw] = np.maximum(region, m * 255)
                if i < len(confs):
                    max_conf = max(max_conf, float(confs[i]))
    return full, round(max_conf, 3)


def segment(img_bgr):
    """이미지 → (DetectResult[crack만], 필터된 균열 마스크).
    마스크 연결요소 중 '가늘고 긴'(균열다운) 성분만 채택 → 도메인 밖 텍스처 덩어리 오탐 제거.
    각 채택 성분을 crack Detection(box)으로 → 기존 파이프라인과 호환.
    비활성/모델없음이면 빈 DetectResult + 빈 마스크(폴백 안전)."""
    H, W = img_bgr.shape[:2]
    res = DetectResult(image_size=[W, H])
    raw, mconf = crack_mask(img_bgr)
    if not raw.any():
        return res, raw

    n, lbl, stats, _c = cv2.connectedComponentsWithStats(
        (raw > 0).astype(np.uint8), connectivity=8)
    min_area = max(20, int(0.0003 * H * W))   # 미세 노이즈 성분 제거
    max_width = max(12.0, config.SEG_MAX_WIDTH_FRAC * min(H, W))   # 평균두께 px 상한
    mask = np.zeros_like(raw)                  # 형태 필터 통과분만 다시 채움
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            continue
        fill = area / (w * h + 1e-6)               # bbox 채움비 — 균열은 성기게(낮음), 덩어리는 높음
        thickness = area / (max(w, h) + 1e-6)      # 평균두께(면적/장축) — 균열은 얇음
        # 덩어리(꽉 참) 또는 두꺼운 성분 = 텍스처 오탐 → 제거. 대각 균열도 fill·두께 낮아 통과.
        if fill > config.SEG_MAX_FILL or thickness > max_width:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        mask[lbl == i] = 255
        res.detections.append(
            Detection(box=[x, y, x + w, y + h],
                      conf=mconf if mconf > 0 else 0.5, cls=0, label="crack"))
    return res, mask


def mask_features(mask):
    """균열 마스크 → (개수, 중심선 길이px, 평균 폭px). features 승격용(검증 후 연결).
    - 길이 = 스켈레톤(중심선) 픽셀 수, 폭 = 균열 면적 / 길이."""
    if mask is None or not np.any(mask):
        return 0, 0.0, 0.0
    binm = (mask > 0).astype(np.uint8)
    n, _ = cv2.connectedComponents(binm)
    count = max(0, n - 1)
    if _HAS_SK:
        length = float(skeletonize(binm > 0).sum())
    else:
        length = float(binm.sum()) / 3.0     # skimage 없으면 근사
    area = float(binm.sum())
    width = area / length if length > 0 else 0.0
    return count, round(length, 1), round(width, 2)
