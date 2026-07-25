"""
Seg 타일 분할 (tile_split_seg.py)
- 1920x1080 원본 이미지를 640x640 타일로 분할
- YOLO Segmentation 폴리곤 라벨을 타일 경계에서 클리핑
- Shapely 사용하여 폴리곤 클리핑 처리

실행:
  python tile_split_seg.py
"""
import os
import glob
import random
import cv2
import numpy as np

try:
    from shapely.geometry import Polygon, box
    from shapely.validation import make_valid
    SHAPELY_OK = True
except ImportError:
    print("[!] Shapely 없음. pip install shapely")
    SHAPELY_OK = False

# ================== 설정 ==================
SRC = r"C:\dataset_seg\71769"
OUT = r"C:\dataset_seg_tiled"

TILE = 640
OVERLAP = 0.2
MIN_POLY_AREA_RATIO = 0.05  # 클리핑된 폴리곤이 타일의 0.5% 이상이면 보존
MIN_POINTS = 3              # 최소 폴리곤 점 개수
EMPTY_TILE_RATIO = 0.05     # 빈 타일 5%만 포함
SEED = 42
# ==========================================

STRIDE = int(TILE * (1 - OVERLAP))  # 512
random.seed(SEED)


def tile_positions(total, tile, stride):
    """타일 시작 위치 계산"""
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def read_seg_label(path, W, H):
    """YOLO seg 라벨 읽기 → [(cls, [(x,y), ...]), ...]"""
    polys = []
    if not os.path.exists(path):
        return polys
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # cls + 최소 3점 (6좌표)
                continue
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
            points = []
            for i in range(0, len(coords) - 1, 2):
                x = coords[i] * W
                y = coords[i + 1] * H
                points.append((x, y))
            if len(points) >= 3:
                polys.append((cls, points))
    return polys


def clip_polygon_to_tile(points, tx, ty, tile_size):
    """폴리곤을 타일 영역으로 클리핑"""
    if not SHAPELY_OK:
        return None

    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area < 1:
            return None

        tile_box = box(tx, ty, tx + tile_size, ty + tile_size)
        clipped = poly.intersection(tile_box)

        if clipped.is_empty or clipped.area < 1:
            return None

        # MultiPolygon인 경우 가장 큰 것 선택
        if clipped.geom_type == 'MultiPolygon':
            clipped = max(clipped.geoms, key=lambda g: g.area)
        elif clipped.geom_type == 'GeometryCollection':
            polys = [g for g in clipped.geoms if g.geom_type == 'Polygon']
            if not polys:
                return None
            clipped = max(polys, key=lambda g: g.area)

        if clipped.geom_type != 'Polygon':
            return None

        # 타일 로컬 좌표로 변환
        coords = list(clipped.exterior.coords)[:-1]  # 마지막 점 제거 (닫힌 폴리곤)
        if len(coords) < MIN_POINTS:
            return None

        local_coords = [(x - tx, y - ty) for x, y in coords]
        return local_coords, clipped.area

    except Exception:
        return None


def process_split(split):
    img_dir = os.path.join(SRC, "images", split)
    lbl_dir = os.path.join(SRC, "labels", split)
    out_img = os.path.join(OUT, "images", split)
    out_lbl = os.path.join(OUT, "labels", split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        images = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    n_tiles = n_with_poly = 0
    min_area = MIN_POLY_AREA_RATIO * TILE * TILE

    for idx, img_path in enumerate(images):
        if idx % 1000 == 0:
            print(f"  [{split}] {idx}/{len(images)}...")

        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        polys = read_seg_label(lbl_path, W, H)

        for ty in tile_positions(H, TILE, STRIDE):
            for tx in tile_positions(W, TILE, STRIDE):
                tile_polys = []

                for cls, points in polys:
                    result = clip_polygon_to_tile(points, tx, ty, TILE)
                    if result is None:
                        continue
                    local_coords, area = result
                    if area < min_area:
                        continue

                    # 정규화
                    norm_coords = [(x / TILE, y / TILE) for x, y in local_coords]
                    tile_polys.append((cls, norm_coords))

                # 빈 타일 필터링
                if not tile_polys and random.random() > EMPTY_TILE_RATIO:
                    continue

                # 타일 이미지 추출
                tile_img = img[ty:ty + TILE, tx:tx + TILE]
                th, tw = tile_img.shape[:2]
                if th != TILE or tw != TILE:
                    tile_img = cv2.copyMakeBorder(
                        tile_img, 0, TILE - th, 0, TILE - tw,
                        cv2.BORDER_CONSTANT, value=(114, 114, 114))

                tname = f"{stem}_x{tx}_y{ty}"
                cv2.imwrite(os.path.join(out_img, tname + ".jpg"), tile_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 라벨 저장
                with open(os.path.join(out_lbl, tname + ".txt"), "w") as f:
                    for cls, coords in tile_polys:
                        coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)
                        f.write(f"{cls} {coord_str}\n")

                n_tiles += 1
                if tile_polys:
                    n_with_poly += 1

    print(f"[{split}] 원본 {len(images)}장 → 타일 {n_tiles}개 (결함 {n_with_poly}, 배경 {n_tiles - n_with_poly})")
    return n_tiles


def main():
    if not SHAPELY_OK:
        print("Shapely 설치 필요: pip install shapely")
        return

    print(f"Seg 타일링: {TILE}px, overlap {OVERLAP}, stride {STRIDE}")
    print(f"입력: {SRC}")
    print(f"출력: {OUT}")
    print()

    process_split("train")
    process_split("val")

    # data.yaml 생성
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n  0: crack\n")

    print(f"\ndata.yaml 생성 완료: {os.path.join(OUT, 'data.yaml')}")


if __name__ == "__main__":
    main()
