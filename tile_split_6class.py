"""
6종 타일 분할 (tile_split_6class.py)
- dataset_6class(원본 해상도 jpg + YOLO 6클래스 라벨)을 640 타일로 분할.
- 통짜 640 다운스케일 → 미세·저대비 결함(백태·도장·박리)이 배경에 묻히던 문제를 원본 해상도로 완화.
- tile_split.py(크랙 단일클래스)를 다중클래스로 확장: 클래스 id 보존 + 큰 면적결함 보호.

핵심 파라미터
  TILE=640, OVERLAP=0.2 → stride 512
  MIN_BOX_KEEP=0.25   : 타일에 박스가 25% 이상 남으면 채택(면적결함 고려해 크랙 0.30보다 완화)
  BIG_KEEP_FRAC=0.12  : 위 미달이어도 클립박스가 타일의 12% 이상이면 채택(타일보다 큰 면적결함 보존)
  MIN_BOX_PX=6        : 타일 안 박스 최소 변(px)
  EMPTY_TILE_RATIO=0.10 : 결함 없는 배경 타일을 10%만 포함(오탐 억제 hard-neg)

실행(데스크탑, dataset_6class 있는 곳):
  (venv) D:\crack_detection> python tile_split_6class.py
  → dataset_6class_tiled/ 생성 후  python train_6class_tiled.py
"""
import os
import glob
import random
import cv2

# ================== 설정 ==================
SRC = r"C:\dataset_6class"        # SSD에 복사된 원본
OUT = r"C:\dataset_6class_tiled_small"  # SSD에 타일 출력 (20% 빠른 테스트)

TILE = 640
OVERLAP = 0.2
MIN_BOX_KEEP = 0.25
BIG_KEEP_FRAC = 0.12
MIN_BOX_PX = 6
EMPTY_TILE_RATIO = 0.10
SUBSET_TRAIN = 4000    # 20% subset (원본 18293장 중 4000장)
SUBSET_VAL = 800       # 20% subset (원본 4155장 중 800장)
SEED = 0

CLASS_NAMES = ["crack", "spalling", "efflorescence", "rebar_exposure",
               "steel_defect", "paint_damage"]
# =========================================

STRIDE = int(TILE * (1 - OVERLAP))   # 512
random.seed(SEED)


def tile_positions(total, tile, stride):
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def read_yolo_label(path, W, H):
    """YOLO 정규화 라벨 → [cls, x1,y1,x2,y2](절대 픽셀) 목록. 클래스 id 보존."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            cls = int(float(p[0]))
            xc, yc, w, h = (float(v) for v in p[1:])
            xc, yc, w, h = xc * W, yc * H, w * W, h * H
            boxes.append([cls, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return boxes


def process_split(split, subset):
    img_dir = os.path.join(SRC, "images", split)
    lbl_dir = os.path.join(SRC, "labels", split)
    out_img = os.path.join(OUT, "images", split)
    out_lbl = os.path.join(OUT, "labels", split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if subset is not None:
        random.shuffle(images)
        images = images[:subset]

    n_tiles = n_with_box = 0
    per_cls = {n: 0 for n in CLASS_NAMES}

    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = read_yolo_label(os.path.join(lbl_dir, stem + ".txt"), W, H)

        for ty in tile_positions(H, TILE, STRIDE):
            for tx in tile_positions(W, TILE, STRIDE):
                tile_boxes = []
                for (cls, x1, y1, x2, y2) in boxes:
                    ix1, iy1 = max(x1, tx), max(y1, ty)
                    ix2, iy2 = min(x2, tx + TILE), min(y2, ty + TILE)
                    iw, ih = ix2 - ix1, iy2 - iy1
                    if iw <= 0 or ih <= 0:
                        continue
                    orig_area = (x2 - x1) * (y2 - y1)
                    if orig_area <= 0:
                        continue
                    clip_area = iw * ih
                    keep = (clip_area / orig_area >= MIN_BOX_KEEP) or \
                           (clip_area >= BIG_KEEP_FRAC * TILE * TILE)   # 큰 면적결함 보존
                    if not keep or iw < MIN_BOX_PX or ih < MIN_BOX_PX:
                        continue
                    lx1, ly1, lx2, ly2 = ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty
                    xc = (lx1 + lx2) / 2 / TILE
                    yc = (ly1 + ly2) / 2 / TILE
                    nw = (lx2 - lx1) / TILE
                    nh = (ly2 - ly1) / TILE
                    tile_boxes.append((cls, xc, yc, nw, nh))
                    per_cls[CLASS_NAMES[cls]] += 1

                if not tile_boxes and random.random() > EMPTY_TILE_RATIO:
                    continue

                tile_img = img[ty:ty + TILE, tx:tx + TILE]
                th, tw = tile_img.shape[:2]
                if th != TILE or tw != TILE:
                    tile_img = cv2.copyMakeBorder(
                        tile_img, 0, TILE - th, 0, TILE - tw,
                        cv2.BORDER_CONSTANT, value=(114, 114, 114))

                tname = f"{stem}_x{tx}_y{ty}"
                cv2.imwrite(os.path.join(out_img, tname + ".jpg"), tile_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(os.path.join(out_lbl, tname + ".txt"), "w") as f:
                    f.write("\n".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}"
                                      for c, a, b, w, h in tile_boxes))
                n_tiles += 1
                if tile_boxes:
                    n_with_box += 1

    print(f"[{split}] 원본 {len(images)}장 → 타일 {n_tiles}개 (결함포함 {n_with_box}, 배경 {n_tiles - n_with_box})")
    print("  클래스별 박스:", {k: v for k, v in per_cls.items()})
    return n_tiles


def main():
    print(f"타일 {TILE}px / overlap {OVERLAP} / stride {STRIDE}")
    process_split("train", SUBSET_TRAIN)
    process_split("val", SUBSET_VAL)
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n\nnames:\n{names_block}\n")
    print(f"data.yaml 생성: {os.path.join(OUT, 'data.yaml')}")


if __name__ == "__main__":
    main()
