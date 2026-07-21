"""
타일 데이터로 YOLOv8 재학습 v2 (train_crack_tiled_v2.py) — 3070 Ti(8GB) 기준
================================================================================
v1(train_tiled_full) 진단: mAP50이 40에폭부터 0.18에서 평평하게 수렴 → '덜 돌려서'가
아니라 데이터·태스크 천장. 그래서 v2는 epoch를 늘리는 게 아니라 **데이터 종류**와
**증강**을 바꿔서 (1) 오탐(precision 0.285) 감소, (2) 실사용 도메인 갭 축소를 노린다.

v1 대비 바뀐 점 (근거):
  1) HARD NEGATIVE 학습: 이음새/타일 줄눈/패널 경계/그림자 등 '균열 아님' 배경 이미지를
     라벨 없이(빈 .txt 또는 라벨파일 없음) train/images 에 섞으면 YOLO가 배경으로 학습 →
     precision↑, 타일 눈금 오탐↓ (postprocess 필터를 모델 단에서 보완).
     data_tiled.yaml 의 train 경로에 hard_neg 이미지를 함께 두거나 아래 HARD_NEG_DIR 병합.
  2) 도메인 사진 fine-tune: 폰 근접 촬영 실사진 100~300장을 라벨링해 train 에 추가하면
     드론→폰 도메인 갭을 가장 싸게 메움. (별도 폴더로 합쳐 data_tiled.yaml 갱신)
  3) 증강 재조정: 얇은 균열은 mosaic 과도가 국소화(localization)를 해쳐 → mosaic 1.0→0.5,
     close_mosaic 10→20(막판 20에폭은 mosaic 끔), scale 0.5 추가, cos_lr True 로 수렴 안정화.
  4) epochs 100/patience 30 (여유만; 수렴 자체는 데이터가 좌우).

백본 선택:
  - 기본 yolov8s: v1과 동일 → **공정한 before/after 비교**. (권장: 데이터 효과부터 확인)
  - yolov8m(25M): 몇 점 더 오를 수 있으나 8GB에선 batch 8 필요·학습 2배↑. MODEL 만 바꾸면 됨.
    (데이터가 더 큰 지렛대라 8s로 데이터 개선 효과부터 측정한 뒤 8m 시도 권장)

실행:
  (venv) D:\crack_detection> python train_crack_tiled_v2.py
비교: runs/crack/train_tiled_full(v1) vs runs/crack/train_tiled_v2(v2) 의 results.png
"""
import os
from ultralytics import YOLO

# ── 경로 (데스크탑 실제 경로로 유지) ─────────────────────────────────────────
DATA = r"D:\crack_detection\dataset_tiled_full\data_tiled.yaml"
MODEL = "yolov8s.pt"          # 공정 비교용. 8m 시도 시 "yolov8m.pt" + BATCH=8
BATCH = 16                    # 8s/640 기준. OOM(CUDA out of memory)이면 8
RUN_NAME = "train_tiled_v2"

# hard negative(균열 아님) 이미지 폴더 — 라벨 없이 넣을 배경 사진들.
#   여기 사진들을 dataset_tiled_full\images\train 로 복사만 하면 라벨 없는 배경으로 학습됨.
#   (YOLO는 라벨파일 없는 이미지를 자동으로 '객체 없음' 배경으로 취급)
#   None 이면 병합 단계 건너뜀 — 이미 train 폴더에 섞어뒀다는 뜻.
HARD_NEG_DIR = None   # 예: r"D:\crack_detection\hard_neg"  (타일 눈금·줄눈·그림자 사진)


def _merge_hard_negatives():
    """HARD_NEG_DIR 의 이미지를 학습 train/images 로 복사(라벨 없이). 배경 학습용."""
    if not HARD_NEG_DIR or not os.path.isdir(HARD_NEG_DIR):
        return 0
    import shutil, glob
    train_img = os.path.join(os.path.dirname(DATA), "images", "train")
    os.makedirs(train_img, exist_ok=True)
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in glob.glob(os.path.join(HARD_NEG_DIR, ext)):
            dst = os.path.join(train_img, "hn_" + os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy2(p, dst)      # 라벨(.txt) 안 만듦 → 배경 이미지로 학습
                n += 1
    print(f"[hard-neg] {n}장 배경 이미지 추가 (라벨 없음 → precision↑ 기대)")
    return n


def main():
    _merge_hard_negatives()
    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=100,
        imgsz=640,
        batch=BATCH,
        device=0,
        patience=30,
        project="runs/crack",
        name=RUN_NAME,
        # ── 증강: 얇은 균열 국소화 보존 위주 ──
        mosaic=0.5,          # v1 1.0 → 0.5 (과한 조각모자이크는 얇은 선 위치 흐림)
        close_mosaic=20,     # 막판 20에폭은 mosaic 꺼서 실분포로 마무리
        scale=0.5,           # 다양한 촬영 거리 대응(근접/원거리)
        degrees=10,
        fliplr=0.5,
        flipud=0.3,
        hsv_v=0.4,           # 조도 변화
        hsv_s=0.3,
        cos_lr=True,         # 코사인 LR → 후반 수렴 안정
        lr0=0.01,
        lrf=0.01,
    )

    metrics = model.val()
    print("\n===== v2 재학습 완료 =====")
    print(f"mAP50    : {metrics.box.map50:.4f}   (v1 0.179)")
    print(f"mAP50-95 : {metrics.box.map:.4f}   (v1 0.062)")
    print(f"Precision: {metrics.box.mp:.4f}   (v1 0.285)  ← hard-neg 효과 확인 포인트")
    print(f"Recall   : {metrics.box.mr:.4f}   (v1 0.241)")
    print(f"모델: runs/crack/{RUN_NAME}/weights/best.pt")
    print("※ precision이 오르면 이음새 오탐이 모델 단에서 준 것 — postprocess와 이중 방어.")


if __name__ == "__main__":
    main()
