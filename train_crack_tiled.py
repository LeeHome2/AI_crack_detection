"""
타일 데이터로 YOLOv8 학습 (train_crack_tiled.py)
- 먼저 tile_split.py 실행해서 dataset_tiled/ 생성 후 이 스크립트 실행
- 타일은 이미 640x640라 학습 시 다운스케일 없음 -> 가는 균열선 보존
- RTX 3070 Ti (8GB) 기준 설정
- 1차 학습(train_v1, mAP50 0.118)과 before/after 비교용

실행:
  (venv) D:\crack_detection> python train_crack_tiled.py
"""
from ultralytics import YOLO

# tile_split.py가 학습량에 따라 폴더를 만듦 (전체 -> dataset_tiled_full).
DATA = r"D:\crack_detection\dataset_tiled_full\data_tiled.yaml"


def main():
    model = YOLO("yolov8s.pt")   # 1차와 동일 백본 -> 공정한 before/after 비교

    model.train(
        data=DATA,
        epochs=80,            # 전체 데이터 밤샘 학습 (patience로 조기종료 됨)
        imgsz=640,            # 타일이 이미 640 -> 축소 없음(핵심)
        batch=16,             # 타일은 작아 메모리 여유. OOM 나면 8로
        device=0,
        patience=20,          # 20에폭 개선 없으면 조기 종료
        project="runs/crack",
        name="train_tiled_full",
        degrees=10,
        fliplr=0.5,
        flipud=0.3,
        hsv_v=0.4,
    )

    metrics = model.val()
    print("\n===== 타일 학습 완료 =====")
    print(f"mAP50    : {metrics.box.map50:.4f}   (1차 0.118)")
    print(f"mAP50-95 : {metrics.box.map:.4f}   (1차 0.039)")
    print(f"Precision: {metrics.box.mp:.4f}   (1차 0.194)")
    print(f"Recall   : {metrics.box.mr:.4f}   (1차 0.227)")
    print("모델: runs/detect/runs/crack/train_tiled_full/weights/best.pt")


if __name__ == "__main__":
    main()
