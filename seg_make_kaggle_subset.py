"""
seg_make_kaggle_subset.py — 캐글 병렬 검증용 seg 서브셋 zip 생성 (데스크탑 실행)
================================================================================
목적: seg_step2_convert.py 산출물(dataset_seg/71769, 단일클래스 crack, YOLO-seg 폴리곤)에서
      소량(기본 3000/600)만 뽑아 zip → 캐글에 Dataset으로 업로드 → seg_validate_kaggle.ipynb 로
      마스크 mAP·비주얼 빠른 검증. (풀 71k 업로드는 비현실적이라 서브셋만.)

전제: seg_step2_convert.py 로 변환 완료돼 dataset_seg/71769 에
      images/{train,val} + labels/{train,val}(폴리곤 .txt) 가 있어야 함.
      (train/val 분할 폴더가 없고 flat 이면 자동으로 83:17 분할)

실행(데스크탑):
  (venv) D:\crack_detection> python seg_make_kaggle_subset.py
  → D:\crack_detection\seg_subset.zip 생성 → 캐글 "＋ New Dataset" 으로 업로드
"""
import os
import glob
import random
import shutil
import zipfile

# ================== 설정 ==================
SRC = r"D:\crack_detection\dataset_seg\71769"     # 변환 산출물 (또는 567)
OUT = r"D:\crack_detection\dataset_seg_kaggle_subset"
ZIP = r"D:\crack_detection\seg_subset.zip"
N_TRAIN, N_VAL = 3000, 600      # 검증용 소량 (P100 60ep ≈ 1~2h)
SEED = 0
CLASS_NAMES = ["crack"]
# =========================================

random.seed(SEED)
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")


def _label_of(img_path, lbl_dir):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(lbl_dir, stem + ".txt")


def _gather(img_dir, lbl_dir):
    """라벨이 실제로 있는 이미지만 수집(빈 라벨·미스매치 제외)."""
    imgs = []
    for ext in IMG_EXTS:
        imgs += glob.glob(os.path.join(img_dir, ext))
    out = []
    for p in imgs:
        lp = _label_of(p, lbl_dir)
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            out.append(p)
    return out


def _discover():
    """train/val 분할 폴더가 있으면 사용, 없으면 flat 수집 후 자동 분할."""
    ti = os.path.join(SRC, "images", "train")
    vi = os.path.join(SRC, "images", "val")
    if os.path.isdir(ti) and os.path.isdir(vi):
        train = _gather(ti, os.path.join(SRC, "labels", "train"))
        val = _gather(vi, os.path.join(SRC, "labels", "val"))
        return train, val
    # flat 구조 폴백: images/ + labels/
    fi = os.path.join(SRC, "images")
    fl = os.path.join(SRC, "labels")
    allimg = _gather(fi, fl)
    random.shuffle(allimg)
    cut = int(len(allimg) * 0.83)
    return allimg[:cut], allimg[cut:]


def _copy(imgs, split):
    oi = os.path.join(OUT, "images", split)
    ol = os.path.join(OUT, "labels", split)
    os.makedirs(oi, exist_ok=True)
    os.makedirs(ol, exist_ok=True)
    n = 0
    for p in imgs:
        # 라벨 경로: images/... → labels/... 치환
        lp = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        lp = os.path.splitext(lp)[0] + ".txt"
        if not os.path.exists(lp):
            continue
        shutil.copy(p, oi)
        shutil.copy(lp, ol)
        n += 1
    print(f"[{split}] {n}장 복사")
    return n


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(f"[중단] 변환 산출물 없음: {SRC}\n  seg_step2_convert.py 먼저 실행.")
    train, val = _discover()
    if not train or not val:
        raise SystemExit(f"[중단] 라벨 있는 이미지 부족 (train {len(train)} / val {len(val)}). "
                         "변환이 폴리곤 seg 라벨을 만들었는지 확인.")
    random.shuffle(train)
    random.shuffle(val)
    train, val = train[:N_TRAIN], val[:N_VAL]

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    _copy(train, "train")
    _copy(val, "val")

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: .\ntrain: images/train\nval: images/val\n\nnames:\n{names_block}\n")

    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, OUT))

    size_mb = os.path.getsize(ZIP) / 1e6
    print(f"\n✅ 완료 → {ZIP}  ({size_mb:.0f} MB)")
    print("   캐글 '＋ New Dataset' 으로 업로드 → seg_validate_kaggle.ipynb 실행")


if __name__ == "__main__":
    main()
