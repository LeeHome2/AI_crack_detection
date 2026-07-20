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


def _positions(total, tile, stride):
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def _nms(boxes, iou_thr):
    """[x1,y1,x2,y2,conf] -> 간단 NMS."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        b = boxes.pop(0)
        keep.append(b)
        rest = []
        for o in boxes:
            xx1, yy1 = max(b[0], o[0]), max(b[1], o[1])
            xx2, yy2 = min(b[2], o[2]), min(b[3], o[3])
            iw, ih = max(0, xx2 - xx1), max(0, yy2 - yy1)
            inter = iw * ih
            ab = (b[2] - b[0]) * (b[3] - b[1])
            ao = (o[2] - o[0]) * (o[3] - o[1])
            iou = inter / (ab + ao - inter + 1e-9)
            if iou < iou_thr:
                rest.append(o)
        boxes = rest
    return keep


def load_model():
    """YOLO 모델 로드 (없으면 None)."""
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(config.YOLO_WEIGHTS):
        return None
    from ultralytics import YOLO
    _model = YOLO(config.YOLO_WEIGHTS)
    return _model


def is_ready():
    return os.path.exists(config.YOLO_WEIGHTS)


def detect(img_bgr) -> DetectResult:
    """고해상도 이미지 1장 -> 타일 슬라이스 추론 -> 병합된 박스."""
    model = load_model()
    H, W = img_bgr.shape[:2]
    result = DetectResult(image_size=[W, H])
    if model is None:
        return result   # 모델 없으면 빈 결과 (앱에서 안내)

    stride = int(config.TILE * (1 - config.OVERLAP))
    raw = []
    for ty in _positions(H, config.TILE, stride):
        for tx in _positions(W, config.TILE, stride):
            tile = img_bgr[ty:ty + config.TILE, tx:tx + config.TILE]
            r = model.predict(tile, conf=config.CONF, verbose=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                raw.append([x1 + tx, y1 + ty, x2 + tx, y2 + ty, float(b.conf[0])])

    for (x1, y1, x2, y2, c) in _nms(raw, config.IOU_MERGE):
        result.detections.append(
            Detection(box=[int(x1), int(y1), int(x2), int(y2)], conf=round(c, 3))
        )
    return result
