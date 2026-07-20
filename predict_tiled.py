"""
타일 슬라이스 추론 (predict_tiled.py)
- 고해상도 사진을 640 타일로 쪼개 각 타일에서 탐지 후, 원본 좌표로 병합
- 학습과 동일 원리(다운스케일 방지)로 내 사진에서도 균열을 잡게 함
- before/after 발표: 원본 통짜 추론 vs 타일 추론 비교에 사용

실행:
  (venv) D:\crack_detection> python predict_tiled.py
"""
import os
import glob
import cv2
from ultralytics import YOLO

WEIGHTS = r"D:\crack_detection\runs\detect\runs\crack\train_tiled_full\weights\best.pt"
SRC_DIR = r"D:\crack_detection\test_images"      # 내가 찍은 사진들
OUT_DIR = r"D:\crack_detection\runs\crack\predict_tiled"
TILE = 640
OVERLAP = 0.2
CONF = 0.15     # 자신감 낮은 모델이라 0.15로. 오탐 많으면 0.25로 올리기
IOU_MERGE = 0.5     # 타일 경계 중복 박스 병합 기준

STRIDE = int(TILE * (1 - OVERLAP))
os.makedirs(OUT_DIR, exist_ok=True)


def positions(total, tile, stride):
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def nms(boxes, iou_thr):
    """[x1,y1,x2,y2,conf] 목록 -> 간단 NMS."""
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
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            area_o = (o[2] - o[0]) * (o[3] - o[1])
            iou = inter / (area_b + area_o - inter + 1e-9)
            if iou < iou_thr:
                rest.append(o)
        boxes = rest
    return keep


def main():
    model = YOLO(WEIGHTS)
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs += glob.glob(os.path.join(SRC_DIR, ext))

    for img_path in imgs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        all_boxes = []
        for ty in positions(H, TILE, STRIDE):
            for tx in positions(W, TILE, STRIDE):
                tile = img[ty:ty + TILE, tx:tx + TILE]
                r = model.predict(tile, conf=CONF, verbose=False)[0]
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    all_boxes.append([x1 + tx, y1 + ty, x2 + tx, y2 + ty,
                                      float(b.conf[0])])
        merged = nms(all_boxes, IOU_MERGE)
        for (x1, y1, x2, y2, c) in merged:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(img, f"{c:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out, img)
        print(f"{os.path.basename(img_path)}: 균열 {len(merged)}개 -> {out}")


if __name__ == "__main__":
    main()
