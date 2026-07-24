# ===== [1] 파라미터 =====
EPOCHS   = 60
IMGSZ    = 640
BATCH    = 16
MODEL    = "yolov8s-seg.pt"
PROJECT  = "/kaggle/working/runs"
NAME     = "seg_crack_val"

# ===== [2] ultralytics 설치 =====
!pip -q install -U ultralytics
import ultralytics, torch
ultralytics.checks()
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

# ===== [3] 데이터 복사 (input은 읽기전용이라 working으로 복사) =====
import os, glob, zipfile, shutil
INP = "/kaggle/input"
WORK = "/kaggle/working/seg_data"

# 기존 폴더 삭제 후 재생성
if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK, exist_ok=True)

zips = glob.glob(INP + "/**/*.zip", recursive=True)
if zips:
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(WORK)
    print("zip 추출:", zips[0])
else:
    # 폴더인 경우 전체 복사
    src_yamls = glob.glob(INP + "/**/data.yaml", recursive=True)
    assert src_yamls, "seg_subset 못찾음"
    src_dir = os.path.dirname(src_yamls[0])
    shutil.copytree(src_dir, WORK, dirs_exist_ok=True)
    print("폴더 복사:", src_dir, "->", WORK)

yml = glob.glob(WORK + "/**/data.yaml", recursive=True)
BASE = os.path.dirname(yml[0]) if yml else WORK
print("BASE =", BASE)

# ===== [4] data.yaml 재작성 =====
import yaml
cfg = {"path": BASE, "train": "images/train", "val": "images/val", "names": {0: "crack"}}
with open(os.path.join(BASE, "data.yaml"), "w") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print(open(os.path.join(BASE, "data.yaml")).read())
nt = len(glob.glob(BASE + "/images/train/*.*"))
nv = len(glob.glob(BASE + "/images/val/*.*"))
print(f"train {nt}장 / val {nv}장")
assert nt > 0 and nv > 0, "이미지 없음"

# ===== [5] 학습 =====
from ultralytics import YOLO
model = YOLO(MODEL)
model.train(
    data=os.path.join(BASE, "data.yaml"),
    epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,
    patience=20, project=PROJECT, name=NAME,
    mosaic=0.7, fliplr=0.5, flipud=0.3, cos_lr=True, close_mosaic=15,
)

# ===== [6] 검증 =====
m = model.val()
print("=== SEG (mask) ===")
print("mask mAP50   :", round(float(m.seg.map50), 4))
print("mask mAP50-95:", round(float(m.seg.map), 4))
print("=== BOX (참고) ===")
print("box  mAP50   :", round(float(m.box.map50), 4))
print("box  mAP50-95:", round(float(m.box.map), 4))

# ===== [7] 시각화 + 모델 저장 =====
val_imgs = sorted(glob.glob(BASE + "/images/val/*.*"))[:6]
model.predict(val_imgs, save=True, project=PROJECT, name=NAME + "_pred", conf=0.25)

best = f"{PROJECT}/{NAME}/weights/best.pt"
dst = "/kaggle/working/yolov8s_seg_crack_best.pt"
shutil.copy(best, dst)
print("다운로드:", dst)

from IPython.display import Image as IdImage, display
for p in sorted(glob.glob(f"{PROJECT}/{NAME}_pred/*.jpg"))[:6]:
    display(IdImage(filename=p))
