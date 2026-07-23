"""
Seg Step 2: AI Hub segmentation 데이터셋 -> YOLO seg 포맷 변환 (I/O 제한 버전)

HDD 100% 점유 방지를 위해 처리 간격 추가.

사용법 (PowerShell에서):
  cd D:\crack_detection
  .\venv\Scripts\Activate.ps1
  python seg_step2_convert_slow.py --71769
"""
import json
import glob
import os
import random
import shutil
import argparse
import time
from pathlib import Path
from PIL import Image
from collections import Counter
from tqdm import tqdm

# ================== 설정 ==================
BASE_DIR = r"D:\AIHub_dataset"
OUTPUT_DIR = r"D:\crack_detection\dataset_seg"

VAL_RATIO = 0.15
CONVERT_TO_JPG = True

# I/O 제한 설정
IO_DELAY = 0.02          # 파일당 20ms 대기
BATCH_SIZE = 100         # 100개마다 추가 대기
BATCH_DELAY = 1.0        # 배치당 1초 대기

# 71769 클래스 매핑 (균열 타입)
CLASS_MAP_71769 = {
    "crack": 0,
    "Crack": 0,
    "균열": 0,
    "ConcreteCrack": 0,
}

CLASS_NAMES = ["crack"]
# =====================================================

def build_image_index(image_dirs):
    """이미지 경로 인덱싱"""
    image_index = {}
    extensions = ["*.jpg", "*.png", "*.JPG", "*.PNG"]

    print("  이미지 폴더 스캔 중...")
    for img_dir in image_dirs:
        if not os.path.isdir(img_dir):
            continue
        for ext in extensions:
            for p in glob.glob(os.path.join(img_dir, ext)):
                image_index[os.path.basename(p)] = p
                image_index[Path(p).stem] = p
        time.sleep(0.1)  # 폴더당 대기

    return image_index

def collect_json_files(label_dirs):
    """JSON 라벨 파일 수집"""
    json_files = []
    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        json_files.extend(glob.glob(os.path.join(label_dir, "*.json")))
        time.sleep(0.1)  # 폴더당 대기

    random.shuffle(json_files)
    return json_files

def polygon_to_yolo(polygon, img_w, img_h):
    """Polygon 좌표를 YOLO segmentation 포맷으로 변환"""
    if not polygon or len(polygon) < 3:
        return None

    normalized = []
    for point in polygon:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        elif isinstance(point, dict):
            x, y = point.get("x", 0), point.get("y", 0)
        else:
            continue

        nx = max(0, min(1, x / img_w))
        ny = max(0, min(1, y / img_h))
        normalized.extend([nx, ny])

    if len(normalized) < 6:
        return None

    return normalized

def polyline_to_thin_polygon(points, thickness=3):
    """Polyline을 얇은 polygon으로 변환 (균열 라인용)"""
    if len(points) < 2:
        return None

    polygon = []
    # 정방향
    for i, pt in enumerate(points):
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            polygon.append([pt[0], pt[1] - thickness])
    # 역방향
    for pt in reversed(points):
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            polygon.append([pt[0], pt[1] + thickness])

    return polygon

def process_json_71769(jf, image_index, class_count):
    """71769 JSON 처리 - 다양한 포맷 지원"""
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, "error"

    # 이미지 정보 추출 (다양한 포맷 지원)
    img_info = data.get("images", [{}])[0] if data.get("images") else {}
    if not img_info:
        img_info = data.get("image", {})

    # file_name 또는 name
    file_name = img_info.get("file_name", "") or img_info.get("name", "")
    W = img_info.get("width", 0)
    H = img_info.get("height", 0)

    if not file_name or W <= 0 or H <= 0:
        return None, "error"

    stem = Path(file_name).stem
    src_img_path = image_index.get(file_name) or image_index.get(stem)
    if not src_img_path:
        return None, "no_image"

    yolo_lines = []

    # annotations 위치: 최상위 또는 image 내부
    annotations = data.get("annotations", [])
    if not annotations:
        annotations = img_info.get("annotations", [])

    for ann in annotations:
        # 클래스 이름 추출
        cls_str = ann.get("category_name", "") or ann.get("class", "") or \
                  ann.get("label", "") or ann.get("attributes", {}).get("class", "")

        cls_idx = CLASS_MAP_71769.get(cls_str, 0)

        # segmentation 또는 points
        segmentation = ann.get("segmentation", [])
        points = ann.get("points", [])
        shape = ann.get("shape", "").lower()

        # Polyline인 경우 얇은 polygon으로 변환
        if shape == "polyline" and points and len(points) >= 2:
            polygon = polyline_to_thin_polygon(points)
            if polygon:
                segmentation = [polygon]
        elif points and not segmentation:
            segmentation = [points]

        for seg in segmentation:
            if isinstance(seg, list) and len(seg) >= 3:
                # 포인트 리스트 형식 [[x,y], [x,y], ...]
                if all(isinstance(p, (list, tuple)) for p in seg):
                    normalized = polygon_to_yolo(seg, W, H)
                # 플랫 리스트 형식 [x1,y1,x2,y2,...]
                elif all(isinstance(x, (int, float)) for x in seg) and len(seg) >= 6:
                    pts = [(seg[i], seg[i+1]) for i in range(0, len(seg)-1, 2)]
                    normalized = polygon_to_yolo(pts, W, H)
                else:
                    continue

                if normalized:
                    coords = " ".join(f"{v:.6f}" for v in normalized)
                    yolo_lines.append(f"{cls_idx} {coords}")
                    class_count["crack"] += 1

    if not yolo_lines:
        return None, "no_annotation"

    return {
        "src_img_path": src_img_path,
        "file_name": file_name,
        "yolo_lines": yolo_lines,
    }, "ok"

def save_sample(result, split, output_dir, stats):
    """이미지와 라벨 저장"""
    stem = Path(result["file_name"]).stem
    src_img_path = result["src_img_path"]

    if CONVERT_TO_JPG:
        out_img_path = os.path.join(output_dir, "images", split, stem + ".jpg")
        try:
            Image.open(src_img_path).convert("RGB").save(out_img_path, quality=95)
        except Exception:
            stats["error"] += 1
            return False
    else:
        ext = Path(src_img_path).suffix
        out_img_path = os.path.join(output_dir, "images", split, stem + ext)
        shutil.copy(src_img_path, out_img_path)

    with open(os.path.join(output_dir, "labels", split, stem + ".txt"), "w") as f:
        f.write("\n".join(result["yolo_lines"]))

    return True

def convert_dataset(dataset_key, image_dirs, label_dirs, output_dir):
    """데이터셋 변환 (I/O 제한)"""
    print(f"\n{'='*60}")
    print(f"[{dataset_key}] 변환 시작 (I/O 제한 모드)")
    print("="*60)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    print("\n이미지 인덱싱 중...")
    image_index = build_image_index(image_dirs)
    print(f"  총 이미지: {len(image_index):,}개")

    if not image_index:
        print("  [X] 이미지를 찾을 수 없습니다.")
        return 0

    print("\nJSON 라벨 수집 중...")
    json_files = collect_json_files(label_dirs)
    print(f"  총 JSON: {len(json_files):,}개")

    if not json_files:
        print("  [X] JSON 파일을 찾을 수 없습니다.")
        return 0

    stats = {"converted": 0, "no_image": 0, "no_annotation": 0, "error": 0}
    class_count = Counter()
    val_count = int(len(json_files) * VAL_RATIO)

    print(f"\n변환 중... (Train: {len(json_files)-val_count:,}, Val: {val_count:,})")
    print(f"  I/O 제한: {IO_DELAY*1000:.0f}ms/파일, {BATCH_DELAY}s/{BATCH_SIZE}파일")

    for i, jf in enumerate(tqdm(json_files, desc="변환", unit="파일")):
        result, status = process_json_71769(jf, image_index, class_count)

        if status != "ok":
            stats[status] = stats.get(status, 0) + 1
            continue

        split = "val" if i < val_count else "train"
        if save_sample(result, split, output_dir, stats):
            stats["converted"] += 1

        # I/O 제한
        time.sleep(IO_DELAY)
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)

    print(f"\n변환 결과:")
    print(f"  성공: {stats['converted']:,}개")
    print(f"  이미지없음: {stats.get('no_image', 0):,}개")
    print(f"  어노테이션없음: {stats.get('no_annotation', 0):,}개")
    print(f"  오류: {stats.get('error', 0):,}개")

    print(f"\n클래스별 객체 수:")
    for cls, cnt in class_count.most_common():
        print(f"  {cls}: {cnt:,}개")

    return stats["converted"]

def main():
    parser = argparse.ArgumentParser(description="Seg 데이터셋 YOLO 변환 (I/O 제한)")
    parser.add_argument("--71769", dest="ds_71769", action="store_true")
    args = parser.parse_args()

    if not args.ds_71769:
        args.ds_71769 = True

    print("="*60)
    print("Seg Step 2: YOLO Segmentation 포맷 변환 (I/O 제한)")
    print("="*60)

    total_converted = 0

    if args.ds_71769:
        print("\n[71769] 폴더 탐색 중...")

        base_71769 = os.path.join(BASE_DIR, "075.건물_균열_탐지_이미지_고도화_SOC_시설물_균열패턴_이미지_데이터")

        # 이미지 폴더 (도로교량 + 도로터널만)
        image_dirs = [
            os.path.join(base_71769, "3.개방데이터", "1.데이터", "Training", "01.원천데이터"),
            os.path.join(base_71769, "3.개방데이터", "1.데이터", "Validation", "01.원천데이터"),
        ]

        # 라벨 폴더 (도로교량 + 도로터널)
        label_base_train = os.path.join(base_71769, "3.개방데이터", "1.데이터", "Training", "02.라벨링데이터")
        label_base_val = os.path.join(base_71769, "3.개방데이터", "1.데이터", "Validation", "02.라벨링데이터")

        label_dirs = []
        for base in [label_base_train, label_base_val]:
            if os.path.isdir(base):
                for d in os.listdir(base):
                    if d.startswith(("TL_", "VL_")) and ("도로교량" in d or "도로터널" in d):
                        label_dirs.append(os.path.join(base, d))

        # 디버그: 찾은 폴더 출력
        print(f"\n발견된 폴더:")
        print(f"  이미지 폴더: {len(image_dirs)}개")
        for d in image_dirs:
            exists = "✓" if os.path.isdir(d) else "✗"
            print(f"    {exists} {d}")

        print(f"  라벨 폴더: {len(label_dirs)}개")
        for d in label_dirs[:5]:
            print(f"    {d}")
        if len(label_dirs) > 5:
            print(f"    ... 외 {len(label_dirs)-5}개")

        if image_dirs and label_dirs:
            cnt = convert_dataset(
                "71769",
                image_dirs,
                label_dirs,
                os.path.join(OUTPUT_DIR, "71769")
            )
            total_converted += cnt or 0
        else:
            print("\n[71769] 필요한 폴더를 찾을 수 없습니다.")
            print("  이미지 이동 완료 후 다시 시도하세요.")

    # data.yaml 생성
    if total_converted > 0:
        ds_dir = os.path.join(OUTPUT_DIR, "71769")
        yaml_path = os.path.join(ds_dir, "data.yaml")
        ds_yaml = f"""path: {ds_dir}
train: images/train
val: images/val

names:
  0: crack
"""
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(ds_yaml)
        print(f"\ndata.yaml 저장: {yaml_path}")

    print("\n" + "="*60)
    print("변환 완료!")
    print("="*60)
    print(f"\n총 변환: {total_converted:,}개")
    print("\n다음 단계:")
    print("  python train_seg.py")

if __name__ == "__main__":
    main()
