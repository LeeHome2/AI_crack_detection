"""
[2] Vision AI — 균열 탐지 (detector.py)
- YOLOv8s 타일 학습본 + 타일 슬라이스 추론(overlap) + NMS 병합
- predict_tiled.py 로직을 함수화
- [하이브리드] crack 모델 + defect6 모델 조합으로 균열 검출 정확도 향상
"""
import os
import numpy as np

import config
from schemas import Detection, DetectResult

_model = None          # 기존 단일 모델 캐시 (비하이브리드용)
_crack_model = None    # crack 전용 모델 캐시
_defect6_model = None  # defect6 모델 캐시

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


def load_crack_model():
    """Crack 전용 모델 로드 (하이브리드용)."""
    global _crack_model
    if _crack_model is not None:
        return _crack_model
    if not _valid_weights(config.CRACK_MODEL_WEIGHTS):
        return None
    try:
        from ultralytics import YOLO
        _crack_model = YOLO(config.CRACK_MODEL_WEIGHTS)
    except Exception as e:
        print(f"[detector] crack 모델 로드 실패: {e}")
        _crack_model = None
    return _crack_model


def load_defect6_model():
    """Defect6 모델 로드 (하이브리드용)."""
    global _defect6_model
    if _defect6_model is not None:
        return _defect6_model
    if not _valid_weights(config.DEFECT6_MODEL_WEIGHTS):
        return None
    try:
        from ultralytics import YOLO
        _defect6_model = YOLO(config.DEFECT6_MODEL_WEIGHTS)
    except Exception as e:
        print(f"[detector] defect6 모델 로드 실패: {e}")
        _defect6_model = None
    return _defect6_model


def is_hybrid_ready():
    """하이브리드 모드 사용 가능 여부."""
    return (config.HYBRID_DETECT_ENABLED and
            _valid_weights(config.CRACK_MODEL_WEIGHTS) and
            _valid_weights(config.DEFECT6_MODEL_WEIGHTS))


def _detect_with_model(img_bgr, model, exclude_labels=None):
    """단일 모델로 타일 추론. exclude_labels에 해당하는 클래스는 제외."""
    H, W = img_bgr.shape[:2]
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
                label = _canon_label(names.get(cid, "crack"))

                # 제외 라벨 필터링
                if exclude_labels and label in exclude_labels:
                    continue

                raw.append([x1 + tx, y1 + ty, x2 + tx, y2 + ty, float(b.conf[0]), cid, label])

    return raw


def detect_hybrid(img_bgr) -> DetectResult:
    """하이브리드 검출: crack 모델 + defect6 모델(crack 제외)."""
    crack_model = load_crack_model()
    defect6_model = load_defect6_model()

    H, W = img_bgr.shape[:2]
    result = DetectResult(image_size=[W, H])

    if crack_model is None and defect6_model is None:
        return result

    raw = []

    # 1. Crack 모델로 균열 검출
    if crack_model is not None:
        crack_raw = _detect_with_model(img_bgr, crack_model, exclude_labels=None)
        raw.extend(crack_raw)

    # 2. Defect6 모델로 기타 결함 검출 (crack 제외)
    if defect6_model is not None:
        defect6_raw = _detect_with_model(img_bgr, defect6_model, exclude_labels={"crack"})
        raw.extend(defect6_raw)

    # NMS (라벨별로)
    boxes_for_nms = [[r[0], r[1], r[2], r[3], r[4], r[6]] for r in raw]  # label을 cls 대신 사용
    # 라벨별 NMS를 위해 라벨을 숫자로 매핑
    label_to_idx = {}
    for r in raw:
        if r[6] not in label_to_idx:
            label_to_idx[r[6]] = len(label_to_idx)

    nms_input = [[r[0], r[1], r[2], r[3], r[4], label_to_idx[r[6]]] for r in raw]
    kept = _nms(nms_input, config.IOU_MERGE)

    # 결과 생성
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    for (x1, y1, x2, y2, c, lid) in kept:
        label = idx_to_label[lid]
        result.detections.append(
            Detection(box=[int(x1), int(y1), int(x2), int(y2)],
                      conf=round(c, 3), cls=lid, label=label)
        )

    return result


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
