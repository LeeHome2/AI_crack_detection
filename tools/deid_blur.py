#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deid_blur.py — 현장 촬영 이미지 개인정보 자동 비식별화(전처리)
================================================================
목적
  프로젝트 9번 CP4("직접 촬영 현장 이미지 100~300장을 전처리하여 시연·테스트셋으로 활용")의
  '전처리' 조건을 충족하기 위한 스크립트. 현장 사진에 우발적으로 찍힌 개인정보
  (차량 번호판 · 사람 얼굴)를 자동 탐지해 가우시안 블러/모자이크로 가린다.

무엇을 하나
  1) EXIF 회전 보정   : 폰 사진의 세로/가로 회전 메타를 실제 픽셀에 적용(입력 규격 정규화)
  2) (선택)리사이즈    : --max-size 로 최대 변 길이 제한(테스트셋 입력 규격 통일)
  3) 얼굴·번호판 탐지  : OpenCV Haar(무설치·오프라인) + (선택)YOLO 가중치
  4) 블러/모자이크     : 탐지 영역을 가림. 원본은 절대 수정하지 않음
  5) 검수·이력 기록    : review/ 에 '무엇을 가렸는지' 박스 오버레이, manifest.csv 에 처리 이력

안전 설계
  - 원본 보존: 입력은 읽기만, 결과는 out_dir/ 에 새로 저장
  - 균열 데이터 훼손 방지: 번호판 후보를 가로세로비(가로로 긴 사각형)로 필터 →
    균열·텍스처를 번호판으로 오인해 뭉개는 사고를 줄임. 그래도 review/ 로 사람이 최종 확인
  - --dry-run: 실제 블러 없이 검수 오버레이 + manifest 만 먼저 뽑아 탐지 상태 점검

사용 예
  # 1) 먼저 무엇이 잡히는지 확인 (블러 안 함)
  python tools/deid_blur.py --in ./field_raw --out ./field_deid --dry-run
  # 2) 확인 후 실제 처리 (얼굴+번호판 블러, 최대 변 1600px 로 정규화)
  python tools/deid_blur.py --in ./field_raw --out ./field_deid --max-size 1600
  # 3) 모자이크로 가리고 싶으면
  python tools/deid_blur.py --in ./field_raw --out ./field_deid --mode mosaic
  # 4) (선택) 성능 좋은 YOLO 번호판/얼굴 모델이 있으면 추가
  python tools/deid_blur.py --in ./field_raw --out ./field_deid \
      --yolo-plate weights/plate.pt --yolo-face weights/face.pt

의존성
  pip install opencv-python pillow numpy        (필수)
  pip install ultralytics                        (선택: --yolo-* 사용 시)
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("[오류] opencv-python 이 필요합니다:  pip install opencv-python")

try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False  # EXIF 보정만 생략, 나머지는 동작

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# 이미지 로드 (EXIF 회전 보정 포함)
# ----------------------------------------------------------------------------
def load_bgr(path: Path):
    """EXIF 회전을 반영해 BGR ndarray 로 로드. 실패 시 None."""
    if _HAS_PIL:
        try:
            im = Image.open(path)
            im = ImageOps.exif_transpose(im)      # 폰 사진 회전 보정
            im = im.convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    # PIL 실패 시 OpenCV 폴백(EXIF 미반영)
    data = np.fromfile(str(path), dtype=np.uint8)   # 한글 경로 안전
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def save_bgr(img, path: Path):
    """한글 경로 안전 저장."""
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext if ext else ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


def resize_max(img, max_size: int):
    """최대 변을 max_size 로 축소(확대는 안 함). (img, scale) 반환."""
    if max_size <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return img, 1.0
    s = max_size / float(m)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA), s


# ----------------------------------------------------------------------------
# 탐지기
# ----------------------------------------------------------------------------
class HaarDetectors:
    """OpenCV 내장 Haar cascade — 별도 다운로드 없이 오프라인 동작."""

    def __init__(self):
        base = cv2.data.haarcascades
        self.face_front = self._load(base, "haarcascade_frontalface_alt2.xml")
        self.face_prof = self._load(base, "haarcascade_profileface.xml")
        self.plate = self._load(base, "haarcascade_russian_plate_number.xml")

    @staticmethod
    def _load(base, name):
        p = os.path.join(base, name)
        c = cv2.CascadeClassifier(p)
        return None if c.empty() else c

    def faces(self, gray):
        boxes = []
        for cas in (self.face_front, self.face_prof):
            if cas is None:
                continue
            for (x, y, w, h) in cas.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24)):
                boxes.append((x, y, w, h))
        # 측면 얼굴은 좌우 반전에서도 탐지(프로파일 cascade는 한쪽만 학습됨)
        if self.face_prof is not None:
            flip = cv2.flip(gray, 1)
            W = gray.shape[1]
            for (x, y, w, h) in self.face_prof.detectMultiScale(
                    flip, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24)):
                boxes.append((W - x - w, y, w, h))
        return boxes

    def plates(self, gray):
        if self.plate is None:
            return []
        out = []
        for (x, y, w, h) in self.plate.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=8, minSize=(28, 12)):
            ar = w / float(h) if h else 0
            # 번호판은 가로로 긴 사각형(대략 2~6:1). 균열/텍스처 오탐 억제.
            if 1.8 <= ar <= 6.5:
                out.append((x, y, w, h))
        return out


class YoloDetector:
    """선택: ultralytics YOLO 가중치로 얼굴/번호판 탐지(있으면 더 정확)."""

    def __init__(self, weights, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, img_bgr):
        r = self.model.predict(img_bgr, conf=self.conf, verbose=False)[0]
        boxes = []
        for b in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = b[:4]
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes


# ----------------------------------------------------------------------------
# 가림 처리
# ----------------------------------------------------------------------------
def _pad_clip(box, W, H, pad_frac):
    x, y, w, h = box
    px, py = int(w * pad_frac), int(h * pad_frac)
    x1 = max(0, x - px); y1 = max(0, y - py)
    x2 = min(W, x + w + px); y2 = min(H, y + h + py)
    return x1, y1, x2, y2


def redact(img, boxes, mode="blur", pad_frac=0.15):
    """boxes 영역을 가림. mode=blur|mosaic. 반영된 박스 목록 반환."""
    H, W = img.shape[:2]
    applied = []
    for box in boxes:
        x1, y1, x2, y2 = _pad_clip(box, W, H, pad_frac)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = img[y1:y2, x1:x2]
        if mode == "mosaic":
            bw = max(1, (x2 - x1) // 12)
            bh = max(1, (y2 - y1) // 12)
            small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
            img[y1:y2, x1:x2] = cv2.resize(
                small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        else:  # blur
            k = max(31, ((max(x2 - x1, y2 - y1) // 2) | 1))  # 홀수 커널
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        applied.append((x1, y1, x2, y2))
    return applied


def draw_overlay(img, faces, plates):
    """검수용: 얼굴=초록, 번호판=빨강 박스."""
    o = img.copy()
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(o, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(o, "FACE", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    for (x1, y1, x2, y2) in plates:
        cv2.rectangle(o, (x1, y1), (x2, y2), (0, 0, 230), 3)
        cv2.putText(o, "PLATE", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 230), 2)
    return o


# ----------------------------------------------------------------------------
# 메인 파이프라인
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="현장 이미지 개인정보(번호판·얼굴) 자동 블러 전처리")
    ap.add_argument("--in", dest="in_dir", required=True, help="원본 이미지 폴더")
    ap.add_argument("--out", dest="out_dir", required=True, help="결과 저장 폴더")
    ap.add_argument("--mode", choices=["blur", "mosaic"], default="blur")
    ap.add_argument("--max-size", type=int, default=0,
                    help="최대 변 길이(px)로 리사이즈. 0=원본 유지")
    ap.add_argument("--pad", type=float, default=0.15, help="박스 여유 비율")
    ap.add_argument("--faces-only", action="store_true")
    ap.add_argument("--plates-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="블러 없이 검수 오버레이+manifest 만 생성")
    ap.add_argument("--yolo-face", default="", help="(선택)얼굴 YOLO 가중치")
    ap.add_argument("--yolo-plate", default="", help="(선택)번호판 YOLO 가중치")
    ap.add_argument("--recursive", action="store_true", help="하위 폴더까지")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    review_dir = out_dir / "_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    files = []
    it = in_dir.rglob("*") if args.recursive else in_dir.glob("*")
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    if not files:
        sys.exit(f"[오류] 이미지가 없습니다: {in_dir}")

    haar = HaarDetectors()
    yolo_face = YoloDetector(args.yolo_face) if args.yolo_face else None
    yolo_plate = YoloDetector(args.yolo_plate) if args.yolo_plate else None

    do_faces = not args.plates_only
    do_plates = not args.faces_only

    manifest = out_dir / "manifest.csv"
    n_face_total = n_plate_total = n_hit_imgs = 0

    with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "width", "height", "faces", "plates",
                     "mode", "resized"])
        for i, p in enumerate(files, 1):
            img = load_bgr(p)
            if img is None:
                print(f"[건너뜀] 읽기 실패: {p.name}")
                continue
            img, scale = resize_max(img, args.max_size)
            H, W = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces, plates = [], []
            if do_faces:
                if yolo_face:
                    faces = [(x, y, x + w, y + h)
                             for (x, y, w, h) in yolo_face.detect(img)]
                else:
                    faces = [(x, y, x + w, y + h)
                             for (x, y, w, h) in haar.faces(gray)]
            if do_plates:
                if yolo_plate:
                    plates = [(x, y, x + w, y + h)
                              for (x, y, w, h) in yolo_plate.detect(img)]
                else:
                    plates = [(x, y, x + w, y + h)
                              for (x, y, w, h) in haar.plates(gray)]

            # 검수 오버레이는 항상 생성
            overlay = draw_overlay(img, faces, plates)
            save_bgr(overlay, review_dir / f"{p.stem}_review.jpg")

            if not args.dry_run:
                out_img = img.copy()
                redact(out_img, [(x1, y1, x2 - x1, y2 - y1)
                                 for (x1, y1, x2, y2) in faces],
                       mode=args.mode, pad_frac=args.pad)
                redact(out_img, [(x1, y1, x2 - x1, y2 - y1)
                                 for (x1, y1, x2, y2) in plates],
                       mode=args.mode, pad_frac=args.pad)
                save_bgr(out_img, out_dir / p.name)

            nf, npl = len(faces), len(plates)
            n_face_total += nf
            n_plate_total += npl
            if nf or npl:
                n_hit_imgs += 1
            wr.writerow([p.name, W, H, nf, npl, args.mode,
                         "1" if scale != 1.0 else "0"])
            print(f"[{i}/{len(files)}] {p.name}: 얼굴 {nf} · 번호판 {npl}"
                  + ("  (dry-run)" if args.dry_run else ""))

    print("\n===== 요약 =====")
    print(f"처리 이미지      : {len(files)}장")
    print(f"개인정보 검출 이미지: {n_hit_imgs}장")
    print(f"얼굴 총 {n_face_total}개 · 번호판 총 {n_plate_total}개")
    print(f"결과 폴더        : {out_dir}")
    print(f"검수 오버레이    : {review_dir}  ← 꼭 눈으로 확인하세요")
    print(f"처리 이력(증빙)  : {manifest}")
    if args.dry_run:
        print("※ dry-run 이었습니다. 오버레이 확인 후 --dry-run 빼고 다시 실행하세요.")


if __name__ == "__main__":
    main()
