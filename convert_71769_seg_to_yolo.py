"""
71769(SOC 시설물 균열패턴 이미지[고도화]) → YOLOv8-seg 변환 (convert_71769_seg_to_yolo.py)
================================================================================
목적: 71769의 세그멘테이션 라벨(폴리라인=얇은 균열 중심선 / 폴리곤=면적 결함)을
YOLOv8-seg 학습 포맷(정규화 폴리곤)으로 변환. hard negative(균열 미포함)도 포함.

왜 이 데이터: 162(bbox)의 얇은균열 한계·오탐을 근본 해결.
- 폴리라인 중심선 → 모델이 균열 라인을 직접 출력(OpenCV 스켈레톤 땜빵 제거)
- mm 폭 라벨 → 실제 폭 추정
- 균열 미포함 25k → precision(타일 오탐)

71769 라벨 JSON(가이드 '실제 예시' 기준):
  annotations: [
    { "label": "crack", "labelNum": 0,
      "points": [[x1,y1],[x2,y2], ...],
      "shape": "Polyline",              # 또는 "Polygon"
      "width": "0.1mm 초과, 0.3mm 이하",  # mm 폭 구간(클래스)
      "px": 1 }                          # 균열 픽셀 폭
  ]
  이미지 단위: name / width / height / object_included(Y/N)
  ※ 파일마다 최상위 키 경로가 조금 다를 수 있어(image/images 래핑 등) →
     _extract_meta() 가 여러 경로를 방어적으로 탐색. 실제 파일 1개로 최종 확인 권장.

YOLOv8-seg 라벨 포맷(정규화, 0~1):
  <cls> x1 y1 x2 y2 ... xn yn   (닫힌 폴리곤 외곽 좌표)
  hard negative: 라벨 파일 없음 or 빈 파일(이미지만 존재).

실행(데스크탑, 데이터가 있는 곳):
  (venv) D:\...> pip install shapely
  (venv) D:\...> python convert_71769_seg_to_yolo.py
"""
import os
import glob
import json
import random
import shutil

try:
    from shapely.geometry import LineString      # 폴리라인 → 띠 폴리곤 버퍼링
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

# ================== 설정 (데스크탑 실제 경로로) ==================
SRC = r"D:\AIHub_dataset\71769"           # 71769 데이터 루트(이미지+json 하위 포함)
OUT = r"D:\crack_detection\dataset_seg_71769"
VAL_RATIO = 0.2
SEED = 0

# 71769 개방데이터 실제 클래스(로컬 확인): linear/reticular/complex crack 3종.
#   (가이드의 10종은 전체 구축 스펙이고, 공개분 라벨은 균열 3종.) 소문자로 비교.
CLASS_MAP = {
    "linear crack": 0,      # 선형균열(대부분 폴리라인=얇은 중심선)
    "reticular crack": 1,   # 망상균열
    "complex crack": 2,     # 복합균열
}
INCLUDE_HARD_NEG = True     # object_included=N(균열 미포함) → 빈 라벨로 포함(precision↑)
MIN_POLYLINE_PX = 3         # 폴리라인 버퍼 최소 반폭(px). px 라벨이 너무 작을 때 하한.
MIN_POLY_POINTS = 3         # 폴리곤 최소 꼭짓점(이하이면 스킵)
# ==============================================================

random.seed(SEED)
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _extract_meta(obj):
    """JSON에서 (width, height, object_included, annotations, image_name) 방어적 추출.
    파일 스키마가 image/images 래핑 등으로 다를 수 있어 여러 경로를 시도한다.
    """
    # 이미지 정보가 들어갈 만한 후보 컨테이너
    img = obj
    for k in ("image", "images"):
        if k in obj:
            v = obj[k]
            img = v[0] if isinstance(v, list) and v else v
            break
    W = img.get("width") or obj.get("width") or 0
    H = img.get("height") or obj.get("height") or 0
    name = img.get("name") or obj.get("name") or ""
    inc = str(img.get("object_included", obj.get("object_included", "Y"))).strip().upper()
    anns = (img.get("annotations") or obj.get("annotations")
            or img.get("annotation") or obj.get("annotation") or [])
    return int(W or 0), int(H or 0), inc, anns, name


def _points_of(ann):
    """annotation에서 좌표 리스트 [[x,y],...] 추출(points 키 우선)."""
    pts = ann.get("points")
    if pts and isinstance(pts[0], (list, tuple)):
        return [(float(p[0]), float(p[1])) for p in pts]
    # [x1,y1,x2,y2,...] 평탄형 방어
    if pts and isinstance(pts[0], (int, float)):
        return [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts) - 1, 2)]
    return []


def _polyline_to_polygon(pts, half_px):
    """폴리라인(열린 선)을 half_px 만큼 양쪽 버퍼링해 닫힌 띠 폴리곤 외곽 좌표로."""
    if _HAS_SHAPELY and len(pts) >= 2:
        poly = LineString(pts).buffer(max(half_px, MIN_POLYLINE_PX), cap_style=2, join_style=2)
        if poly.is_empty:
            return []
        ext = poly.exterior if poly.geom_type == "Polygon" else poly.geoms[0].exterior
        return list(ext.coords)
    # shapely 없으면: 선 왕복(앞으로 갔다 뒤로) 으로 최소 닫힌 폴리곤 근사
    return pts + pts[::-1]


def _to_yolo_polygon(ann, W, H):
    """annotation → (cls, [정규화 좌표...]) 또는 None(스킵)."""
    label = str(ann.get("label", "")).strip().lower()
    if label not in CLASS_MAP:
        return None
    cls = CLASS_MAP[label]
    pts = _points_of(ann)
    if len(pts) < 2:
        return None
    shape = str(ann.get("shape", "")).strip().lower()
    if shape == "polyline" or (shape != "polygon" and len(pts) < MIN_POLY_POINTS):
        half = float(ann.get("px", 1) or 1) / 2.0
        poly = _polyline_to_polygon(pts, half)
    else:
        poly = pts
    if len(poly) < MIN_POLY_POINTS:
        return None
    # 정규화 + 클리핑
    norm = []
    for x, y in poly:
        norm.append(min(max(x / W, 0.0), 1.0))
        norm.append(min(max(y / H, 0.0), 1.0))
    return cls, norm


def _find_image(json_path, name, index):
    """json에 대응하는 이미지 경로 찾기 (name 필드 → 같은 basename 순)."""
    d = os.path.dirname(json_path)
    if name:
        for ext in IMG_EXTS:
            cand = os.path.join(d, name if name.lower().endswith(IMG_EXTS) else name + ext)
            if os.path.exists(cand):
                return cand
    base = os.path.splitext(os.path.basename(json_path))[0]
    for ext in IMG_EXTS:
        cand = os.path.join(d, base + ext)
        if os.path.exists(cand):
            return cand
    hit = index.get(base)
    return hit


def main():
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

    # 이미지 인덱스(basename → 경로) — name 매칭 실패 대비
    index = {}
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(SRC, "**", "*" + ext), recursive=True):
            index[os.path.splitext(os.path.basename(p))[0]] = p

    jsons = glob.glob(os.path.join(SRC, "**", "*.json"), recursive=True)
    print(f"발견 json {len(jsons)}개 / 이미지 {len(index)}개")

    n_ok = n_neg = n_skip = 0
    for jp in jsons:
        try:
            obj = json.load(open(jp, encoding="utf-8"))
        except Exception:
            n_skip += 1
            continue
        W, H, inc, anns, name = _extract_meta(obj)
        img_path = _find_image(jp, name, index)
        if not img_path or not W or not H:
            n_skip += 1
            continue

        lines = []
        for a in (anns or []):
            r = _to_yolo_polygon(a, W, H)
            if r:
                cls, norm = r
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in norm))

        is_neg = (inc == "N") or (not lines)
        if is_neg and not INCLUDE_HARD_NEG:
            n_skip += 1
            continue

        split = "val" if random.random() < VAL_RATIO else "train"
        dst_img = os.path.join(OUT, "images", split, os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)
        lbl = os.path.join(OUT, "labels", split,
                           os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        with open(lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))     # hard negative면 빈 파일
        if lines:
            n_ok += 1
        else:
            n_neg += 1

    # data yaml
    names = [k for k, _ in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]
    yaml = os.path.join(OUT, "data_seg.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(names)}\nnames: {names}\n")

    print(f"완료 — 균열 라벨 {n_ok} · hard-neg {n_neg} · 스킵 {n_skip}")
    print(f"shapely: {'ON' if _HAS_SHAPELY else 'OFF(폴리라인 근사 버퍼)'} · data: {yaml}")
    print("※ 실제 JSON 1개로 _extract_meta 경로·label 문자열·px 스케일 최종 확인 권장.")


if __name__ == "__main__":
    main()
