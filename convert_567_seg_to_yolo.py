"""
567(서울시 노후 주택 균열 데이터, 폴더 189) → YOLOv8-seg 변환 (convert_567_seg_to_yolo.py)
================================================================================
목적: 567의 폴리곤 라벨을 YOLOv8-seg 포맷으로 변환.
특이점: 클래스가 곧 '심각도'다 — Class_ID 1/2/3 = 미세/보통/심한 균열.
       → 위험도 등급(정상/주의/위험)과 직결되는 데이터 기반 심각도 학습 가능.

로컬 확인 라벨 구조:
  {
    "Raw_Data_Info":    { "Resolution": [1440, 1080] },
    "Source_Data_Info": { "File_Extension": "jpg", "Source_Data_ID": "S-..." },
    "Learning_Data_Info": { "Annotations": [
        { "Class_ID": "2", "Type": "polygon",
          "polygon": [x1,y1, x2,y2, x3,y3, ...] }   # flat 배열
    ]}
  }
이미지 1440x1080 jpg.

YOLOv8-seg 라벨: <cls> x1 y1 ... xn yn (정규화 폴리곤).
심각도 클래스: 1→0(미세) / 2→1(보통) / 3→2(심한).

실행(데스크탑):
  (venv) D:\...> python convert_567_seg_to_yolo.py
"""
import os
import glob
import json
import random
import shutil

# ================== 설정 ==================
SRC = r"D:\AIHub_dataset\567"            # 189.서울시_노후_주택_균열 루트
OUT = r"D:\crack_detection\dataset_seg_567"
VAL_RATIO = 0.2
SEED = 0

# Class_ID(문자/숫자) → YOLO 클래스. 심각도 3단계.
SEVERITY_MAP = {"1": 0, "2": 1, "3": 2}          # 미세/보통/심한
CLASS_NAMES = ["crack_fine", "crack_moderate", "crack_severe"]
MIN_POLY_POINTS = 3
# ==========================================

random.seed(SEED)
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _resolution(obj):
    r = (obj.get("Raw_Data_Info", {}) or {}).get("Resolution")
    if isinstance(r, list) and len(r) == 2:
        return int(r[0]), int(r[1])
    # 방어: 다른 키
    return int(obj.get("width", 0) or 0), int(obj.get("height", 0) or 0)


def _annotations(obj):
    ld = obj.get("Learning_Data_Info", {}) or {}
    return ld.get("Annotations") or ld.get("annotations") or obj.get("annotations") or []


def _to_yolo_polygon(ann, W, H):
    cid = str(ann.get("Class_ID", ann.get("class_id", ""))).strip()
    if cid not in SEVERITY_MAP:
        return None
    poly = ann.get("polygon") or ann.get("points")
    if not poly:
        return None
    # flat [x1,y1,x2,y2,...] 또는 [[x,y],...] 모두 지원
    if isinstance(poly[0], (list, tuple)):
        flat = [c for p in poly for c in p]
    else:
        flat = list(poly)
    if len(flat) < MIN_POLY_POINTS * 2:
        return None
    norm = []
    for i in range(0, len(flat) - 1, 2):
        norm.append(min(max(flat[i] / W, 0.0), 1.0))
        norm.append(min(max(flat[i + 1] / H, 0.0), 1.0))
    return SEVERITY_MAP[cid], norm


def _find_image(json_path, obj, index):
    base = os.path.splitext(os.path.basename(json_path))[0]
    sid = (obj.get("Source_Data_Info", {}) or {}).get("Source_Data_ID")
    for cand_base in (base, sid):
        if not cand_base:
            continue
        d = os.path.dirname(json_path)
        for ext in IMG_EXTS:
            cand = os.path.join(d, cand_base + ext)
            if os.path.exists(cand):
                return cand
        if cand_base in index:
            return index[cand_base]
    return None


def main():
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

    index = {}
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(SRC, "**", "*" + ext), recursive=True):
            index[os.path.splitext(os.path.basename(p))[0]] = p

    jsons = glob.glob(os.path.join(SRC, "**", "*.json"), recursive=True)
    print(f"발견 json {len(jsons)}개 / 이미지 {len(index)}개")

    n_ok = n_skip = 0
    for jp in jsons:
        try:
            obj = json.load(open(jp, encoding="utf-8"))
        except Exception:
            n_skip += 1
            continue
        W, H = _resolution(obj)
        img_path = _find_image(jp, obj, index)
        if not img_path or not W or not H:
            n_skip += 1
            continue
        lines = []
        for a in _annotations(obj):
            r = _to_yolo_polygon(a, W, H)
            if r:
                cls, norm = r
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in norm))
        if not lines:
            n_skip += 1
            continue
        split = "val" if random.random() < VAL_RATIO else "train"
        shutil.copy2(img_path, os.path.join(OUT, "images", split, os.path.basename(img_path)))
        lbl = os.path.join(OUT, "labels", split,
                           os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        with open(lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        n_ok += 1

    yaml = os.path.join(OUT, "data_seg.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    print(f"완료 — 라벨 {n_ok} · 스킵 {n_skip} · data: {yaml}")
    print("※ 심각도 클래스(미세/보통/심한)는 위험도 등급 데이터 기반 산정에 활용 가능.")


if __name__ == "__main__":
    main()
