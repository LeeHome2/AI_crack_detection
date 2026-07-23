"""
AI HUB 162번(건물 균열 탐지드론) → YOLO 다중클래스(6종) 변환 (convert_aihub_6class_to_yolo.py)
================================================================================
목적: 복합 결함 진단(2차 MVP)용. 기존 변환기는 ConcreteCrack만 뽑았지만, 162엔 6종
결함이 bbox로 라벨돼 있음 → 전부 추출해 다중클래스 detection 학습 입력 생성.

162 결함 6종(attributes.class, 코드):
  ConcreteCrack(101)·Spalling(102)·Efflorescene(103)·Exposure(104)·SteelDefect(201)·PaintDamage(202)

우리 클래스 매핑(제품 서술과 정합):
  0 crack(균열) · 1 spalling(박리/박락) · 2 efflorescence(백태/누수) ·
  3 rebar_exposure(철근노출) · 4 steel_defect(강재손상) · 5 paint_damage(도장손상)

※ 하이브리드: 균열은 seg 모델(71769/567)이 담당하므로, 이 bbox 모델에서 균열을 빼려면
  EXCLUDE_CLASSES에 "crack" 추가(클래스 id는 고정 유지 — YOLO는 인스턴스 없는 클래스 무시).

변환: COCO bbox [x,y,w,h] → YOLO 정규화 [xc,yc,nw,nh]. tiff→jpg.

실행(데스크탑):
  (venv) D:\...> python convert_aihub_6class_to_yolo.py
"""
import json
import glob
import os
import shutil
from pathlib import Path
from PIL import Image

# ================== 설정 (실제 경로) ==================
BASE = r"D:\AIHub_dataset\112.건물_균열_탐지드론_개발을_위한_이미지\01.데이터"
IMAGE_DIR = os.path.join(BASE, r"1.Training\원천데이터")
LABEL_DIR = BASE
OUTPUT_DIR = r"D:\crack_detection\dataset_6class"

VAL_RATIO = 0.2
CONVERT_TO_JPG = True

# attributes.class(소문자 비교) → YOLO 클래스 id. 철자 변형(Efflorescene/Efflorescence) 모두 대응.
CLASS_MAP = {
    "concretecrack": 0,     # 균열
    "spalling": 1,          # 박리/박락
    "efflorescene": 2,      # 백태/누수 (가이드 표기)
    "efflorescence": 2,     # 철자 변형
    "exposure": 3,          # 철근노출
    "steeldefect": 4,       # 강재손상
    "paintdamage": 5,       # 도장손상
}
CLASS_NAMES = ["crack", "spalling", "efflorescence", "rebar_exposure",
               "steel_defect", "paint_damage"]

# 하이브리드: 균열을 bbox 모델에서 제외하려면 {"crack"} (id는 고정). 기본=전부 포함.
EXCLUDE_CLASSES = set()      # 예: {"crack"}
# =====================================================

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

_excluded_ids = {i for i, n in enumerate(CLASS_NAMES) if n in EXCLUDE_CLASSES}

json_files = glob.glob(os.path.join(LABEL_DIR, "**", "*.json"), recursive=True)
print(f"발견된 json: {len(json_files)}개")

image_index = {}
for ext in ["*.tiff", "*.tif", "*.jpg", "*.png"]:
    for p in glob.glob(os.path.join(IMAGE_DIR, "**", ext), recursive=True):
        image_index[os.path.basename(p)] = p
print(f"발견된 이미지: {len(image_index)}개")

converted = skipped_no_img = skipped_no_ann = skipped_error = 0
class_counts = {n: 0 for n in CLASS_NAMES}
unknown_classes = {}

for i, jf in enumerate(json_files):
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        skipped_error += 1
        continue
    if not data.get("images"):
        skipped_error += 1
        continue

    img_info = data["images"][0]
    file_name = img_info["file_name"]
    W, H = img_info["width"], img_info["height"]
    if file_name not in image_index:
        skipped_no_img += 1
        continue
    src_img_path = image_index[file_name]

    yolo_lines = []
    for ann in data.get("annotations", []):
        cls_raw = str(ann.get("attributes", {}).get("class", "")).strip()
        cid = CLASS_MAP.get(cls_raw.lower())
        if cid is None:
            unknown_classes[cls_raw] = unknown_classes.get(cls_raw, 0) + 1
            continue
        if cid in _excluded_ids:
            continue
        x, y, w, h = ann["bbox"]
        xc, yc = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        class_counts[CLASS_NAMES[cid]] += 1

    if not yolo_lines:
        skipped_no_ann += 1
        continue

    split = "val" if (i % int(1 / VAL_RATIO) == 0) else "train"
    stem = Path(file_name).stem
    if CONVERT_TO_JPG:
        out_img_path = os.path.join(OUTPUT_DIR, "images", split, stem + ".jpg")
        try:
            Image.open(src_img_path).convert("RGB").save(out_img_path, quality=95)
        except Exception as e:
            print(f"이미지 변환 실패 {file_name}: {e}")
            skipped_error += 1
            continue
    else:
        out_img_path = os.path.join(OUTPUT_DIR, "images", split, file_name)
        shutil.copy(src_img_path, out_img_path)

    with open(os.path.join(OUTPUT_DIR, "labels", split, stem + ".txt"), "w") as f:
        f.write("\n".join(yolo_lines))
    converted += 1
    if converted % 200 == 0:
        print(f"진행: {converted}개...")

# data.yaml (다중클래스)
names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(f"path: {OUTPUT_DIR}\ntrain: images/train\nval: images/val\n\nnames:\n{names_block}\n")

print(f"\n===== 6종 변환 완료 =====")
print(f"변환 성공: {converted}개 · 이미지없음 {skipped_no_img} · 결함없음 {skipped_no_ann} · 오류 {skipped_error}")
print("클래스별 객체 수(라벨 인스턴스):")
for n in CLASS_NAMES:
    print(f"  {n}: {class_counts[n]}")
if unknown_classes:
    print(f"※ 매핑 안 된 class 문자열(확인 필요): {unknown_classes}")
if EXCLUDE_CLASSES:
    print(f"※ 제외된 클래스(seg가 담당): {EXCLUDE_CLASSES}")
print(f"data.yaml: {os.path.join(OUTPUT_DIR, 'data.yaml')}")
