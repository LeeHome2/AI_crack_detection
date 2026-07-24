"""
6종 타일 50% 학습 - 100% 데이터셋에서 50%만 사용
예상: ~20시간, mAP50 15~17%
"""
from ultralytics import YOLO

DATA = r"C:\dataset_6class_tiled_50pct\data.yaml"  # 50% 타일 데이터 (새로 생성)


def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data=DATA,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        # fraction 불필요 - 이미 50% 데이터
        patience=20,
        project=r"D:\crack_detection\runs\detect",
        name="defect6_tiled_50pct",
        degrees=10,
        fliplr=0.5,
        flipud=0.3,
        hsv_v=0.4,
        mosaic=0.7,
        close_mosaic=20,
        cos_lr=True,
    )

    metrics = model.val()
    print(f"\nmAP50: {float(metrics.box.map50):.4f}")
    print(f"mAP50-95: {float(metrics.box.map):.4f}")


if __name__ == "__main__":
    main()
