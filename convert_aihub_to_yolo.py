"""
AI HUB 건물 균열(162번) -> YOLO 학습 포맷 변환
- COCO bbox [x,y,w,h] -> YOLO [x_center,y_center,w,h] 정규화
- ConcreteCrack 클래스만 추출 (균열만)
- 이미지(.tiff)와 라벨(.json)을 파일명으로 짝 맞춤
- tiff -> jpg 변환 (YOLO 호환)
"""
import json
import glob
import os
import shutil
from pathlib import Path
from PIL import Image

# ================== 설정 (실제 경로) ==================
BASE = r"D:\AIHub_dataset\112.건물_균열_탐지드론_개발을_위한_이미지\01.데이터"

# 이미지(.tiff) 있는 폴더
IMAGE_DIR = os.path.join(BASE, r"1.Training\원천데이터")
# 라벨(.json) 있는 폴더 - 01.데이터 전체를 훑어서 짝 맞는 것 모두 찾음
LABEL_DIR = BASE
# 출력 폴더 (YOLO 학습용)
OUTPUT_DIR = r"D:\crack_detection\dataset"

VAL_RATIO = 0.2            # 학습:검증 = 8:2
CONVERT_TO_JPG = True      # tiff -> jpg 변환
TARGET_CLASS = "ConcreteCrack"   # 균열만 추출
# =====================================================

# 출력 폴더 구조 생성
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# json 목록 (하위 폴더 전체)
json_files = glob.glob(os.path.join(LABEL_DIR, "**", "*.json"), recursive=True)
print(f"발견된 json: {len(json_files)}개")

# 이미지 인덱스 (파일명 -> 실제 경로)
image_index = {}
for ext in ["*.tiff", "*.tif", "*.jpg", "*.png"]:
    for p in glob.glob(os.path.join(IMAGE_DIR, "**", ext), recursive=True):
        image_index[os.path.basename(p)] = p
print(f"발견된 이미지: {len(image_index)}개")

converted = 0
skipped_no_img = 0
skipped_no_crack = 0
skipped_error = 0

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
    W = img_info["width"]
    H = img_info["height"]

    # 짝 이미지 찾기
    if file_name not in image_index:
        skipped_no_img += 1
        continue
    src_img_path = image_index[file_name]

    # 균열(ConcreteCrack)만 -> YOLO 라인
    yolo_lines = []
    for ann in data.get("annotations", []):
        cls = ann.get("attributes", {}).get("class", "")
        if cls != TARGET_CLASS:
            continue
        x, y, w, h = ann["bbox"]           # COCO: 좌상단 x,y + 너비,높이
        xc = (x + w / 2) / W               # YOLO: 중심점 정규화
        yc = (y + h / 2) / H
        nw = w / W
        nh = h / H
        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    if not yolo_lines:
        skipped_no_crack += 1
        continue

    # train/val 분배
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

    # 라벨 저장
    with open(os.path.join(OUTPUT_DIR, "labels", split, stem + ".txt"), "w") as f:
        f.write("\n".join(yolo_lines))

    converted += 1
    if converted % 200 == 0:
        print(f"진행: {converted}개 변환 완료...")

print(f"\n===== 변환 완료 =====")
print(f"변환 성공: {converted}개")
print(f"건너뜀 - 짝 이미지 없음: {skipped_no_img}개")
print(f"건너뜀 - 균열 없음: {skipped_no_crack}개")
print(f"건너뜀 - 오류: {skipped_error}개")

# data.yaml 생성
yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

names:
  0: crack
"""
with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content)
print(f"data.yaml 생성: {os.path.join(OUTPUT_DIR, 'data.yaml')}")
