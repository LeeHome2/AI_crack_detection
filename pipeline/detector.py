"""
[2] Vision AI — 균열 탐지 (detector.py)
- YOLOv8s 타일 학습본 + 타일 슬라이스 추론(overlap) + NMS 병합
- predict_tiled.py 로직을 함수화
"""
import os
import numpy as np

import config
from schemas import Detection, DetectResult

_model = None   # 모델 1회 로드 캐시

# 모델 class name → 우리 정규 라벨. 균열 전용 모델(ConcreteCrack 등)도 crack으로 흡수.
_LABEL_ALIAS = {
    "concretecrack": "crack", "crack": "crack",
    "spalling": "spalling",
    "efflorescene": "efflorescence", "efflorescence": "efflorescence",
    "exposure": "rebar_exposure", "rebar_exposure": "rebar_exposure",
    "steeldefect": "steel_defect", "steel_defect": "steel_defect",
    "paintdamage": "paint_damage", "paint_damage": "paint_damage",
}


def _canon_label(name: str) -> str:
    """모델이 보고한 클래스명을 정규 라벨로. 미상은 crack(균열 전용 모델 하위호환)."""
    return _LABEL_ALIAS.get(str(name).strip().lower().replace(" ", ""), "crack")


def _positions(total, tile, stride):
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def _nms(boxes, iou_thr):
    """[x1,y1,x2,y2,conf,cls] -> 클래스별 NMS (다른 클래스 박스는 서로 억제하지 않음)."""
    if not boxes:
        return []
    keep = []
    classes = set(b[5] for b in boxes)
    for c in classes:
        cb = sorted([b for b in boxes if b[5] == c], key=lambda b: b[4], reverse=True)
        while cb:
            b = cb.pop(0)
            keep.append(b)
            rest = []
            for o in cb:
                xx1, yy1 = max(b[0], o[0]), max(b[1], o[1])
                xx2, yy2 = min(b[2], o[2]), min(b[3], o[3])
                iw, ih = max(0, xx2 - xx1), max(0, yy2 - yy1)
                inter = iw * ih
                ab = (b[2] - b[0]) * (b[3] - b[1])
                ao = (o[2] - o[0]) * (o[3] - o[1])
                iou = inter / (ab + ao - inter + 1e-9)
                if iou < iou_thr:
                    rest.append(o)
            cb = rest
    return keep


def _valid_weights(path):
    """실제 로드 가능한 가중치인지(존재 + 1KB 초과). git-lfs 포인터·빈 파일 방어."""
    try:
        return bool(path) and os.path.exists(path) and os.path.getsize(path) > 1024
    except OSError:
        return False


def load_model():
    """YOLO 모델 로드 (없거나 로드 실패면 None → 앱은 RAG/규칙만으로 계속)."""
    global _model
    if _model is not None:
        return _model
    if not _valid_weights(config.YOLO_WEIGHTS):
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(config.YOLO_WEIGHTS)
    except Exception as e:
        # 손상·버전불일치·LFS 포인터 등 → 크래시 대신 안내(모델 없음 상태로 동작)
        print(f"[detector] 모델 로드 실패({config.YOLO_WEIGHTS}) → 탐지 없이 동작: {e}")
        _model = None
    return _model


def is_ready():
    return _valid_weights(config.YOLO_WEIGHTS)


def detect(img_bgr) -> DetectResult:
    """고해상도 이미지 1장 -> 타일 슬라이스 추론 -> 병합된 박스."""
    model = load_model()
    H, W = img_bgr.shape[:2]
    result = DetectResult(image_size=[W, H])
    if model is None:
        return result   # 모델 없으면 빈 결과 (앱에서 안내)

    names = getattr(model, "names", {}) or {}
    stride = int(config.TILE * (1 - config.OVERLAP))
    raw = []
    for ty in _positions(H, config.TILE, stride):
        for tx in _positions(W, config.TILE, stride):
            tile = img_bgr[ty:ty + config.TILE, tx:tx + config.TILE]
            r = model.predict(tile, conf=config.CONF, verbose=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cid = int(b.cls[0]) if b.cls is not None else 0
                raw.append([x1 + tx, y1 + ty, x2 + tx, y2 + ty, float(b.conf[0]), cid])

    for (x1, y1, x2, y2, c, cid) in _nms(raw, config.IOU_MERGE):
        label = _canon_label(names.get(cid, "crack"))
        result.detections.append(
            Detection(box=[int(x1), int(y1), int(x2), int(y2)],
                      conf=round(c, 3), cls=cid, label=label)
        )
    return result
