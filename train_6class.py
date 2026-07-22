"""
162 6종 복합 결함 YOLOv8 다중클래스 학습 (train_6class.py) — RTX 3070 Ti(8GB) 기준
================================================================================
convert_aihub_6class_to_yolo.py 로 dataset_6class 생성 후 실행.
6종: crack·spalling·efflorescence·rebar_exposure·steel_defect·paint_damage

균열 전용(1차 MVP)과 달리 면적 결함(철근노출·박락 등)이 섞여 있어 타일 분할 없이
원본 리사이즈로 학습해도 blob 결함은 잘 잡힘. (가는 균열 recall이 중요하면 imgsz를
960~1280으로 올리거나 균열은 seg 모델에 맡기는 하이브리드로 보완)

실행:
  (venv) D:\crack_detection> python train_6class.py
결과: runs/detect/defect6/train/weights/best.pt
      → config.py 의 YOLO_WEIGHTS 를 이 경로로 바꾸면 복합 결함 진단 활성화(2차 MVP).
"""
from ultralytics import YOLO

DATA = r"D:\crack_detection\dataset_6class\data.yaml"


def main():
    model = YOLO("yolov8s.pt")   # 전이학습 (1차 MVP와 동일 백본)

    model.train(
        data=DATA,
        epochs=100,           # 6종·클래스 불균형 → 넉넉히. patience로 조기종료.
        imgsz=640,            # OOM 없이 안정. 가는 균열 recall이 아쉬우면 960/1280로.
        batch=16,             # 8GB 기준. OOM(CUDA out of memory) 나면 8→4로.
        device=0,
        patience=25,          # 25에폭 개선 없으면 조기 종료
        project="runs/detect",
        name="defect6",
        # 증강 — 면적 결함은 방향성이 적어 회전·반전 안전. 균열 국소화 위해 mosaic 살짝 낮춤.
        degrees=10,
        fliplr=0.5,
        flipud=0.3,
        hsv_v=0.4,            # 조도 대응
        mosaic=0.7,
        close_mosaic=20,      # 막판 20에폭 mosaic 끔 → 국소화 안정
        cos_lr=True,          # 코사인 LR 스케줄 → 수렴 안정
    )

    # 검증 + 클래스별 지표(어떤 결함이 잘/안 잡히는지 확인)
    metrics = model.val()
    print("mAP50:", round(float(metrics.box.map50), 3),
          "| mAP50-95:", round(float(metrics.box.map), 3))
    # 클래스별 AP (데이터 적은 클래스 파악 → 추가 수집·가중 판단)
    try:
        names = model.names
        for i, ap in enumerate(metrics.box.maps):
            print(f"  {names.get(i, i)}: AP50-95 {round(float(ap), 3)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
