"""
타일 분할 (tile_split.py)
- 기존 변환 dataset(2560x1440 jpg + YOLO 라벨)을 640x640 타일로 분할
- 타일에 걸치는 bbox를 타일 좌표계로 재계산 + 클리핑
- 목적: 2560->640 다운스케일로 소실되던 가는 균열선을 원본 해상도로 보존
- 출력: dataset_tiled/  (+ data_tiled.yaml)

핵심 파라미터
  TILE=640, OVERLAP=0.2 -> stride 512
  MIN_BOX_KEEP=0.30   : 타일에 박스가 30% 이상 남아야 유효 라벨로 채택
  MIN_BOX_PX=6        : 타일 안 박스의 최소 변 길이(px)
  EMPTY_TILE_RATIO=0.15 : 균열 없는 배경 타일을 15%만 포함(불균형 방지)
  SUBSET_TRAIN / SUBSET_VAL : None이면 전체, 숫자면 그만큼만 (빠른 시험용)
"""
import os
import glob
import random
import cv2

# ================== 설정 ==================
SRC = r"D:\crack_detection\dataset"          # 원본(변환 완료) 데이터셋
OUT_BASE = r"D:\crack_detection"              # 타일 출력 상위 폴더 (하위에 버전별 폴더 자동 생성)

TILE = 640
OVERLAP = 0.2
MIN_BOX_KEEP = 0.30
MIN_BOX_PX = 6
EMPTY_TILE_RATIO = 0.15

# 빠른 시험용: None = 전체 사용, 숫자 = 원본 이미지 그 개수만 타일링
SUBSET_TRAIN = None   # None = 전체 3,120장 사용 (밤샘 학습용)
SUBSET_VAL = None     # None = 전체 검증셋
SEED = 0
# =========================================

STRIDE = int(TILE * (1 - OVERLAP))   # 512
random.seed(SEED)

# 학습량에 따라 출력 폴더 이름을 자동 지정 -> 실행마다 버전 기록 (덮어쓰기 방지)
_tag = f"sub{SUBSET_TRAIN}" if SUBSET_TRAIN is not None else "full"
OUT = os.path.join(OUT_BASE, f"dataset_tiled_{_tag}")   # 예: dataset_tiled_sub1500


def tile_positions(total, tile, stride):
    """1차원 축에서 타일 시작 좌표 목록 (마지막 타일은 끝에 딱 맞춤)."""
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def read_yolo_label(path, W, H):
    """YOLO 정규화 라벨 -> 절대 픽셀 박스 [x1,y1,x2,y2] 목록."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            _, xc, yc, w, h = map(float, p)
            xc, yc, w, h = xc * W, yc * H, w * W, h * H
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
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

    n_tiles = n_tiles_with_box = n_boxes = 0

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
                for (x1, y1, x2, y2) in boxes:
                    # 타일과 교집합
                    ix1, iy1 = max(x1, tx), max(y1, ty)
                    ix2, iy2 = min(x2, tx + TILE), min(y2, ty + TILE)
                    iw, ih = ix2 - ix1, iy2 - iy1
                    if iw <= 0 or ih <= 0:
                        continue
                    orig_area = (x2 - x1) * (y2 - y1)
                    if orig_area <= 0:
                        continue
                    # 박스가 타일 안에 충분히 남아있고, 너무 작지 않을 때만 채택
                    if (iw * ih) / orig_area < MIN_BOX_KEEP:
                        continue
                    if iw < MIN_BOX_PX or ih < MIN_BOX_PX:
                        continue
                    # 타일 로컬 좌표 -> YOLO 정규화
                    lx1, ly1, lx2, ly2 = ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty
                    xc = (lx1 + lx2) / 2 / TILE
                    yc = (ly1 + ly2) / 2 / TILE
                    nw = (lx2 - lx1) / TILE
                    nh = (ly2 - ly1) / TILE
                    tile_boxes.append((xc, yc, nw, nh))

                # 배경 타일은 일부만 포함
                if not tile_boxes and random.random() > EMPTY_TILE_RATIO:
                    continue

                tile_img = img[ty:ty + TILE, tx:tx + TILE]
                # 가장자리 타일이 640 미만이면 우/하단 패딩
                th, tw = tile_img.shape[:2]
                if th != TILE or tw != TILE:
                    tile_img = cv2.copyMakeBorder(
                        tile_img, 0, TILE - th, 0, TILE - tw,
                        cv2.BORDER_CONSTANT, value=(114, 114, 114))

                tname = f"{stem}_x{tx}_y{ty}"
                cv2.imwrite(os.path.join(out_img, tname + ".jpg"), tile_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(os.path.join(out_lbl, tname + ".txt"), "w") as f:
                    f.write("\n".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}"
                                      for a, b, c, d in tile_boxes))
                n_tiles += 1
                if tile_boxes:
                    n_tiles_with_box += 1
                    n_boxes += len(tile_boxes)

    print(f"[{split}] 원본 {len(images)}장 -> 타일 {n_tiles}개 "
          f"(균열 포함 {n_tiles_with_box}, 배경 {n_tiles - n_tiles_with_box}, 총 박스 {n_boxes})")
    return n_tiles


def main():
    print(f"타일 {TILE}px / overlap {OVERLAP} / stride {STRIDE}")
    process_split("train", SUBSET_TRAIN)
    process_split("val", SUBSET_VAL)

    yaml = (f"path: {OUT}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"names:\n  0: crack\n")
    with open(os.path.join(OUT, "data_tiled.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)
    print(f"data_tiled.yaml 생성: {os.path.join(OUT, 'data_tiled.yaml')}")


if __name__ == "__main__":
    main()
