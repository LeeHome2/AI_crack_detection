"""
Seg Step 3: YOLOv8 Segmentation 모델 학습

사용법:
  python seg_step3_train.py              # 기본 (71769+567 병합)
  python seg_step3_train.py --71769      # 71769만
  python seg_step3_train.py --567        # 567만
"""
import os
import sys
import argparse
from datetime import datetime

# ================== 설정 ==================
BASE_OUTPUT_DIR = r"D:\crack_detection\dataset_seg"
RUNS_DIR = r"D:\crack_detection\runs\segment"

# 학습 하이퍼파라미터
MODEL = "yolov8s-seg.pt"  # segmentation 모델
IMGSZ = 640
BATCH = 8  # seg는 메모리 더 사용
EPOCHS = 100
PATIENCE = 25
WORKERS = 4

OPTIMIZER = "AdamW"
LR0 = 0.001
# =====================================================

def check_dataset(data_yaml):
    """데이터셋 확인"""
    if not os.path.exists(data_yaml):
        return False, 0, 0

    dataset_dir = os.path.dirname(data_yaml)
    train_dir = os.path.join(dataset_dir, "images", "train")
    val_dir = os.path.join(dataset_dir, "images", "val")

    train_count = 0
    val_count = 0

    if os.path.isdir(train_dir):
        train_count = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))])
    if os.path.isdir(val_dir):
        val_count = len([f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))])

    return True, train_count, val_count

def train_model(data_yaml, project_name):
    """모델 학습"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[X] Ultralytics가 설치되지 않았습니다.")
        print("    pip install ultralytics")
        return None

    # GPU 확인
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n[OK] GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            print("\n[!] GPU 없음. CPU로 학습합니다.")
    except:
        pass

    print(f"\n학습 설정:")
    print(f"  모델: {MODEL}")
    print(f"  이미지 크기: {IMGSZ}")
    print(f"  배치 크기: {BATCH}")
    print(f"  에포크: {EPOCHS}")
    print(f"  Patience: {PATIENCE}")

    print(f"\n출력: {RUNS_DIR}/{project_name}")
    print("\n" + "="*60)
    print("학습 시작...")
    print("="*60 + "\n")

    model = YOLO(MODEL)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        workers=WORKERS,
        project=RUNS_DIR,
        name=project_name,
        optimizer=OPTIMIZER,
        lr0=LR0,
        verbose=True,
        plots=True,
        save=True,
        exist_ok=True,
    )

    return results

def main():
    parser = argparse.ArgumentParser(description="Seg 모델 학습")
    parser.add_argument("--71769", dest="ds_71769", action="store_true")
    parser.add_argument("--567", dest="ds_567", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("Seg Step 3: YOLOv8 Segmentation 학습")
    print("="*60)

    # 데이터셋 확인
    datasets = []

    if args.ds_71769:
        yaml_path = os.path.join(BASE_OUTPUT_DIR, "71769", "data.yaml")
        exists, train, val = check_dataset(yaml_path)
        if exists and train > 0:
            datasets.append(("71769", yaml_path, train, val))
            print(f"\n[OK] 71769: Train {train:,}개, Val {val:,}개")
        else:
            print(f"\n[X] 71769 데이터 없음")

    if args.ds_567:
        yaml_path = os.path.join(BASE_OUTPUT_DIR, "567", "data.yaml")
        exists, train, val = check_dataset(yaml_path)
        if exists and train > 0:
            datasets.append(("567", yaml_path, train, val))
            print(f"\n[OK] 567: Train {train:,}개, Val {val:,}개")
        else:
            print(f"\n[X] 567 데이터 없음")

    # 둘 다 지정 안 하면 둘 다 확인
    if not args.ds_71769 and not args.ds_567:
        for ds_name in ["71769", "567"]:
            yaml_path = os.path.join(BASE_OUTPUT_DIR, ds_name, "data.yaml")
            exists, train, val = check_dataset(yaml_path)
            if exists and train > 0:
                datasets.append((ds_name, yaml_path, train, val))
                print(f"\n[OK] {ds_name}: Train {train:,}개, Val {val:,}개")

    if not datasets:
        print("\n[X] 학습 가능한 데이터셋이 없습니다.")
        print("    seg_step2_convert.py를 먼저 실행하세요.")
        sys.exit(1)

    # 학습 실행
    for ds_name, yaml_path, train_count, val_count in datasets:
        print(f"\n{'='*60}")
        print(f"[{ds_name}] 학습 시작")
        print(f"  Train: {train_count:,}개")
        print(f"  Val: {val_count:,}개")
        print("="*60)

        project_name = f"seg_{ds_name}_v1"
        results = train_model(yaml_path, project_name)

        if results:
            best_model = os.path.join(RUNS_DIR, project_name, "weights", "best.pt")
            if os.path.exists(best_model):
                print(f"\n[OK] Best 모델: {best_model}")

    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
