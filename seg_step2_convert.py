"""
Seg Step 2: AI Hub segmentation 데이터셋 -> YOLO seg 포맷 변환

71769: SOC 시설물 균열패턴 (JSON with polygon)
567: 서울시 노후주택 균열 (JSON with polygon)

사용법:
  python seg_step2_convert.py           # 둘 다 변환
  python seg_step2_convert.py --71769   # 71769만
  python seg_step2_convert.py --567     # 567만
"""
import json
import glob
import os
import random
import shutil
import argparse
from pathlib import Path
from PIL import Image
from collections import Counter
from tqdm import tqdm

# ================== 설정 ==================
BASE_DIR = r"D:\AIHub_dataset"
OUTPUT_DIR = r"D:\crack_detection\dataset_seg"

VAL_RATIO = 0.15
CONVERT_TO_JPG = True

# 71769 클래스 매핑 (균열 타입)
CLASS_MAP_71769 = {
    "crack": 0,
    "Crack": 0,
    "균열": 0,
    "ConcreteCrack": 0,
}

# 567 클래스 매핑 (균열 타입)
CLASS_MAP_567 = {
    "crack": 0,
    "Crack": 0,
    "균열": 0,
    "wall_crack": 0,
    "floor_crack": 0,
}

CLASS_NAMES = ["crack"]
# =====================================================

def find_datasets():
    """다운로드된 데이터셋 폴더 탐색"""
    datasets = {"71769": [], "567": []}

    # 71769 패턴 탐색
    patterns_71769 = [
        "TL_*", "VL_*",  # 라벨
        "TS_*", "VS_*",  # 원천
    ]

    # 567 패턴 탐색
    patterns_567 = [
        "Tl_*", "Vl_*",  # 라벨 (add)
        "Ts_*", "Vs_*",  # 원천 (add)
        "TL_*", "VL_*",  # 원본 라벨
    ]

    for root, dirs, files in os.walk(BASE_DIR):
        for d in dirs:
            # 71769 데이터셋 폴더 탐색
            if any(d.startswith(p.replace("*", "")) for p in ["TL_지", "VL_지", "TS_지", "VS_지"]):
                if "시설물" in root or "71769" in root:
                    datasets["71769"].append(os.path.join(root, d))

            # 567 데이터셋 폴더 탐색
            if any(d.startswith(p.replace("*", "")) for p in ["Tl_", "Ts_", "VL_", "TL_"]):
                if "노후주택" in root or "567" in root:
                    datasets["567"].append(os.path.join(root, d))

    return datasets

def build_image_index(image_dirs):
    """이미지 경로 인덱싱"""
    image_index = {}
    extensions = ["*.jpg", "*.png", "*.tiff", "*.tif", "*.JPG", "*.PNG", "*.TIFF", "*.TIF"]

    for img_dir in image_dirs:
        if not os.path.isdir(img_dir):
            continue
        for ext in extensions:
            for p in glob.glob(os.path.join(img_dir, "**", ext), recursive=True):
                # 파일명과 stem 모두 인덱싱
                image_index[os.path.basename(p)] = p
                image_index[Path(p).stem] = p

    return image_index

def collect_json_files(label_dirs):
    """JSON 라벨 파일 수집"""
    json_files = []
    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        json_files.extend(glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True))

    random.shuffle(json_files)
    return json_files

def polygon_to_yolo(polygon, img_w, img_h):
    """
    Polygon 좌표를 YOLO segmentation 포맷으로 변환
    YOLO seg: class_id x1 y1 x2 y2 ... (normalized)
    """
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

        # Normalize
        nx = max(0, min(1, x / img_w))
        ny = max(0, min(1, y / img_h))
        normalized.extend([nx, ny])

    if len(normalized) < 6:  # 최소 3점 필요
        return None

    return normalized

def process_json_71769(jf, image_index, class_count):
    """71769 JSON 처리 (SOC 시설물)"""
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, "error"

    # 이미지 정보 추출
    img_info = data.get("images", [{}])[0] if data.get("images") else {}
    if not img_info:
        img_info = data.get("image", {})

    file_name = img_info.get("file_name", "")
    W = img_info.get("width", 0)
    H = img_info.get("height", 0)

    if not file_name or W <= 0 or H <= 0:
        return None, "error"

    # 이미지 찾기
    stem = Path(file_name).stem
    src_img_path = image_index.get(file_name) or image_index.get(stem)
    if not src_img_path:
        return None, "no_image"

    # 어노테이션 처리
    yolo_lines = []
    annotations = data.get("annotations", [])

    for ann in annotations:
        # 클래스 확인
        cls_str = ann.get("category_name", "") or ann.get("class", "") or \
                  ann.get("attributes", {}).get("class", "")

        cls_idx = CLASS_MAP_71769.get(cls_str, CLASS_MAP_71769.get("crack", 0))

        # Polygon 추출
        segmentation = ann.get("segmentation", [])
        if not segmentation:
            # points 형식 시도
            points = ann.get("points", [])
            if points:
                segmentation = [points]

        for seg in segmentation:
            if isinstance(seg, list) and len(seg) >= 6:
                # Flat list: [x1, y1, x2, y2, ...]
                if all(isinstance(x, (int, float)) for x in seg):
                    points = [(seg[i], seg[i+1]) for i in range(0, len(seg)-1, 2)]
                    normalized = polygon_to_yolo(points, W, H)
                else:
                    # List of points: [[x1,y1], [x2,y2], ...]
                    normalized = polygon_to_yolo(seg, W, H)

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

def process_json_567(jf, image_index, class_count):
    """567 JSON 처리 (서울시 노후주택)"""
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, "error"

    # 이미지 정보
    img_info = data.get("images", [{}])[0] if data.get("images") else data.get("image", {})

    file_name = img_info.get("file_name", "") or img_info.get("filename", "")
    W = img_info.get("width", 0)
    H = img_info.get("height", 0)

    if not file_name or W <= 0 or H <= 0:
        return None, "error"

    # 이미지 찾기
    stem = Path(file_name).stem
    src_img_path = image_index.get(file_name) or image_index.get(stem)
    if not src_img_path:
        return None, "no_image"

    # 어노테이션 처리
    yolo_lines = []
    annotations = data.get("annotations", []) or data.get("shapes", [])

    for ann in annotations:
        cls_idx = 0  # 모두 crack으로 처리

        # Polygon 추출 (다양한 형식 지원)
        segmentation = ann.get("segmentation", [])
        points = ann.get("points", [])
        polygon = ann.get("polygon", [])

        all_segs = []
        if segmentation:
            all_segs.extend(segmentation if isinstance(segmentation[0], list) else [segmentation])
        if points:
            all_segs.append(points)
        if polygon:
            all_segs.append(polygon)

        for seg in all_segs:
            if isinstance(seg, list) and len(seg) >= 3:
                # Flat list
                if all(isinstance(x, (int, float)) for x in seg) and len(seg) >= 6:
                    pts = [(seg[i], seg[i+1]) for i in range(0, len(seg)-1, 2)]
                    normalized = polygon_to_yolo(pts, W, H)
                else:
                    normalized = polygon_to_yolo(seg, W, H)

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

    # 이미지 저장
    if CONVERT_TO_JPG:
        out_img_path = os.path.join(output_dir, "images", split, stem + ".jpg")
        try:
            Image.open(src_img_path).convert("RGB").save(out_img_path, quality=95)
        except Exception as e:
            stats["error"] += 1
            return False
    else:
        ext = Path(src_img_path).suffix
        out_img_path = os.path.join(output_dir, "images", split, stem + ext)
        shutil.copy(src_img_path, out_img_path)

    # 라벨 저장
    with open(os.path.join(output_dir, "labels", split, stem + ".txt"), "w") as f:
        f.write("\n".join(result["yolo_lines"]))

    return True

def convert_dataset(dataset_key, process_func, image_dirs, label_dirs, output_dir):
    """데이터셋 변환 메인"""
    print(f"\n{'='*60}")
    print(f"[{dataset_key}] 변환 시작")
    print("="*60)

    # 출력 폴더 생성
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # 이미지 인덱싱
    print("\n이미지 인덱싱 중...")
    image_index = build_image_index(image_dirs)
    print(f"  총 이미지: {len(image_index):,}개")

    if not image_index:
        print("  [X] 이미지를 찾을 수 없습니다.")
        return

    # JSON 수집
    print("\nJSON 라벨 수집 중...")
    json_files = collect_json_files(label_dirs)
    print(f"  총 JSON: {len(json_files):,}개")

    if not json_files:
        print("  [X] JSON 파일을 찾을 수 없습니다.")
        return

    # 변환
    stats = {"converted": 0, "no_image": 0, "no_annotation": 0, "error": 0}
    class_count = Counter()
    val_count = int(len(json_files) * VAL_RATIO)

    print(f"\n변환 중... (Train: {len(json_files)-val_count:,}, Val: {val_count:,})")

    for i, jf in enumerate(tqdm(json_files, desc="변환", unit="파일")):
        result, status = process_func(jf, image_index, class_count)

        if status != "ok":
            stats[status] = stats.get(status, 0) + 1
            continue

        split = "val" if i < val_count else "train"
        if save_sample(result, split, output_dir, stats):
            stats["converted"] += 1

    # 결과 출력
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
    parser = argparse.ArgumentParser(description="Seg 데이터셋 YOLO 변환")
    parser.add_argument("--71769", dest="ds_71769", action="store_true")
    parser.add_argument("--567", dest="ds_567", action="store_true")
    args = parser.parse_args()

    # 둘 다 지정 안 하면 둘 다
    if not args.ds_71769 and not args.ds_567:
        args.ds_71769 = True
        args.ds_567 = True

    print("="*60)
    print("Seg Step 2: YOLO Segmentation 포맷 변환")
    print("="*60)
    import sys
    sys.stdout.flush()

    total_converted = 0

    # 71769 변환
    if args.ds_71769:
        print("\n[71769] 폴더 탐색 중... (HDD 사용량 높으면 오래 걸림)")
        sys.stdout.flush()

        # 71769 폴더 구조 탐색 - 특정 경로만
        base_71769 = os.path.join(BASE_DIR, "075.건물_균열_탐지_이미지_고도화_SOC_시설물_균열패턴_이미지_데이터")

        image_dirs = []
        label_dirs = []

        # 특정 경로만 탐색 (전체 BASE_DIR 대신)
        if os.path.isdir(base_71769):
            for root, dirs, files in os.walk(base_71769):
                for d in dirs:
                    full_path = os.path.join(root, d)
                    # 원천 데이터 (이미지)
                    if d.startswith(("TS_", "VS_")) and "지" in d:
                        image_dirs.append(full_path)
                    # 라벨 데이터
                    elif d.startswith(("TL_", "VL_")) and "지" in d:
                        label_dirs.append(full_path)

        if image_dirs or label_dirs:
            print(f"\n[71769] 발견된 폴더:")
            print(f"  이미지: {len(image_dirs)}개")
            print(f"  라벨: {len(label_dirs)}개")

            cnt = convert_dataset(
                "71769",
                process_json_71769,
                image_dirs,
                label_dirs,
                os.path.join(OUTPUT_DIR, "71769")
            )
            total_converted += cnt or 0
        else:
            print("\n[71769] 데이터를 찾을 수 없습니다.")
            print("  seg_step1_download.py를 먼저 실행하세요.")

    # 567 변환
    if args.ds_567:
        print("\n[567] 폴더 탐색 중...")
        sys.stdout.flush()

        # 567 폴더 - 특정 경로만
        base_567 = os.path.join(BASE_DIR, "189.서울시_노후_주택_균열_데이터")

        image_dirs = []
        label_dirs = []

        if os.path.isdir(base_567):
            for root, dirs, files in os.walk(base_567):
                for d in dirs:
                    full_path = os.path.join(root, d)
                    # 원천 데이터 (이미지) - 567 패턴
                    if d.startswith(("Ts_", "Vs_")):
                        image_dirs.append(full_path)
                    # 라벨 데이터
                    elif d.startswith(("Tl_", "Vl_", "VL_")):
                        label_dirs.append(full_path)

        if image_dirs or label_dirs:
            print(f"\n[567] 발견된 폴더:")
            print(f"  이미지: {len(image_dirs)}개")
            print(f"  라벨: {len(label_dirs)}개")

            cnt = convert_dataset(
                "567",
                process_json_567,
                image_dirs,
                label_dirs,
                os.path.join(OUTPUT_DIR, "567")
            )
            total_converted += cnt or 0
        else:
            print("\n[567] 데이터를 찾을 수 없습니다.")
            print("  seg_step1_download.py를 먼저 실행하세요.")

    # data.yaml 생성
    if total_converted > 0:
        yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

names:
  0: crack
"""
        # 각 데이터셋별 yaml
        for ds in ["71769", "567"]:
            ds_dir = os.path.join(OUTPUT_DIR, ds)
            if os.path.isdir(ds_dir):
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
    print("  python seg_step3_train.py")

if __name__ == "__main__":
    main()
