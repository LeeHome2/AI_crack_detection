"""
6종 타일 데이터 학습 (train_6class_tiled.py) — RTX 3070 Ti(8GB) 기준
================================================================================
tile_split_6class.py 로 dataset_6class_tiled 생성 후 실행.
통짜 640 학습(defect6_v2, mAP50 10.8%·결함 79~96% 미탐)의 근본 원인은
'풀해상도 → 640 다운스케일로 미세·저대비 결함이 배경에 묻힘'.
→ 타일은 이미 640이라 다운스케일 없음 = 미세 결함 신호 보존(크랙 MVP가 0.179 낸 방식).

기대: 철근노출·균열·박리 위주로 미탐(background 예측) 대폭 감소.
     백태·도장은 결함 자체가 저대비라 개선폭 제한적일 수 있음(정직).

실행:
  (venv) D:\crack_detection> python tile_split_6class.py     # 먼저 타일 생성
  (venv) D:\crack_detection> python train_6class_tiled.py
결과: runs/detect/defect6_tiled/weights/best.pt
      → models/yolov8s_defect6_final.pt 로 복사하면 앱이 자동 사용(config 후보 경로).
비교: defect6_v2(통짜) vs defect6_tiled 의 confusion_matrix.png·results.png
"""
from ultralytics import YOLO

DATA = r"D:\crack_detection\dataset_6class_tiled\data.yaml"


def main():
    model = YOLO("yolov8s.pt")   # 통짜본과 동일 백본 → 타일 효과만 공정 비교

    model.train(
        data=DATA,
        epochs=100,           # patience로 조기종료(통짜는 ep40 평탄 → 타일은 더 갈 여지)
        imgsz=640,            # 타일이 이미 640 → 축소 없음(핵심)
        batch=16,             # 타일은 작아 메모리 여유. OOM 시 8.
        device=0,
        patience=30,
        project="runs/detect",
        name="defect6_tiled",
        degrees=10, fliplr=0.5, flipud=0.3, hsv_v=0.4,
        mosaic=0.7, close_mosaic=20, cos_lr=True,
    )

    metrics = model.val()
    print("mAP50:", round(float(metrics.box.map50), 4),
          "| mAP50-95:", round(float(metrics.box.map), 4))
    try:
        names = model.names
        for j, ci in enumerate(list(metrics.box.ap_class_index)):
            print(f"  {names[int(ci)]}: AP50={round(float(metrics.box.ap50[j]),4)} "
                  f"R={round(float(metrics.box.r[j]),3)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
