"""
ROI 기반 탐지 테스트 - 원본 이미지만
=====================================
demo 폴더의 *_src.jpg 원본 이미지 + test1~3 원본만 테스트

실행: python test_roi_originals.py
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline import detector, roi_triage, features, postprocess

# 경로 설정
TEST_DIR = r"C:\Users\user\Desktop\crack_test"
DEMO_DIR = os.path.join(TEST_DIR, "demo")
OUTPUT_DIR = os.path.join(DEMO_DIR, "results_roi_v2")

# 원본 이미지 목록 (bbox, mask, result 제외)
ORIGINAL_IMAGES = [
    # demo 폴더 원본
    os.path.join(DEMO_DIR, "01_crack_clear_src.jpg"),
    os.path.join(DEMO_DIR, "02_crack_clear2_src.jpg"),
    os.path.join(DEMO_DIR, "03_floor_crack_src.jpg"),
    os.path.join(DEMO_DIR, "05_dataset_crack_src.jpg"),
    os.path.join(DEMO_DIR, "06_floor_crack2_src.jpg"),
    os.path.join(DEMO_DIR, "07_crack_pattern_src.jpg"),
    os.path.join(DEMO_DIR, "val_01_src.jpg"),
    os.path.join(DEMO_DIR, "val_02_src.jpg"),
    os.path.join(DEMO_DIR, "val_03_src.jpg"),
    os.path.join(DEMO_DIR, "val_04_src.jpg"),
    os.path.join(DEMO_DIR, "val_05_src.jpg"),
    # test 원본 (bbox/mask 없는 것)
    os.path.join(TEST_DIR, "test1_crack.jpg"),
    os.path.join(TEST_DIR, "test2_multi_crack.jpg"),
    os.path.join(TEST_DIR, "test3_spalling.jpg"),
]

# 결함 스타일
DEFECT_STYLE = {
    "crack": {"ko": "균열", "bgr": (0, 0, 255)},
    "spalling": {"ko": "박리/박락", "bgr": (0, 140, 255)},
    "efflorescence": {"ko": "백태/누수", "bgr": (230, 160, 0)},
    "rebar_exposure": {"ko": "철근노출", "bgr": (200, 0, 200)},
    "steel_defect": {"ko": "강재손상", "bgr": (170, 70, 70)},
    "paint_damage": {"ko": "도장손상", "bgr": (0, 170, 0)},
}


def get_style(label):
    return DEFECT_STYLE.get(label, {"ko": "결함", "bgr": (0, 0, 255)})


def annotate_image(img_bgr, det, rois=None, mode_label=""):
    """결과 이미지 생성"""
    vis = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    # ROI 영역 (파란 점선)
    if rois:
        for roi in rois:
            x1, y1 = int(roi["x1"] * w), int(roi["y1"] * h)
            x2, y2 = int(roi["x2"] * w), int(roi["y2"] * h)
            for j in range(x1, x2, 15):
                cv2.line(vis, (j, y1), (min(j+8, x2), y1), (255, 150, 0), 2)
                cv2.line(vis, (j, y2), (min(j+8, x2), y2), (255, 150, 0), 2)
            for j in range(y1, y2, 15):
                cv2.line(vis, (x1, j), (x1, min(j+8, y2)), (255, 150, 0), 2)
                cv2.line(vis, (x2, j), (x2, min(j+8, y2)), (255, 150, 0), 2)
            desc = roi.get("description", "")[:25]
            cv2.putText(vis, desc, (x1+5, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)

    # 균열 스켈레톤
    if det and det.detections:
        sk = features.skeleton_mask(img_bgr, det)
        if sk.any():
            sk = cv2.dilate(sk, np.ones((3,3), np.uint8), iterations=1)
            vis[sk > 0] = (0, 255, 255)

    # 탐지 박스
    for d in det.detections:
        x1, y1, x2, y2 = d.box
        label = getattr(d, "label", "crack")
        stl = get_style(label)
        cv2.rectangle(vis, (x1, y1), (x2, y2), stl["bgr"], 2)
        txt = f"{label[:5]} {d.conf:.2f}"
        cv2.putText(vis, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stl["bgr"], 2)

    # 모드 라벨
    if mode_label:
        cv2.rectangle(vis, (0, 0), (350, 35), (50, 50, 50), -1)
        cv2.putText(vis, mode_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


def detect_full(img_bgr):
    """기존 방식"""
    det = detector.detect_hybrid(img_bgr) if detector.is_hybrid_ready() else detector.detect(img_bgr)
    det = postprocess.filter_seams(img_bgr, det)
    if config.MERGE_OVERLAP_ENABLED:
        det = postprocess.merge_overlapping_boxes(det, label_filter="crack", margin=config.MERGE_OVERLAP_MARGIN)
    return det


def detect_roi(img_bgr):
    """ROI 방식"""
    det_fn = detector.detect_hybrid if detector.is_hybrid_ready() else detector.detect
    det, rois = roi_triage.detect_with_roi(img_bgr, det_fn)
    det = postprocess.filter_seams(img_bgr, det)
    if config.MERGE_OVERLAP_ENABLED:
        det = postprocess.merge_overlapping_boxes(det, label_filter="crack", margin=config.MERGE_OVERLAP_MARGIN)
    return det, rois


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 존재하는 파일만 필터
    images = [p for p in ORIGINAL_IMAGES if os.path.exists(p)]

    print("="*60)
    print("ROI 기반 탐지 테스트 (원본 이미지만)")
    print("="*60)
    print(f"이미지 수: {len(images)}")
    print(f"출력: {OUTPUT_DIR}")
    print("="*60)

    results = []

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        name = os.path.splitext(os.path.basename(img_path))[0].replace("_src", "")
        h, w = img.shape[:2]
        print(f"\n[{name}] {w}x{h}")

        # 기존 방식
        print("  기존...", end=" ", flush=True)
        det_full = detect_full(img)
        cnt_full = len(det_full.detections)
        print(f"{cnt_full}개")

        # ROI 방식
        print("  ROI...", end=" ", flush=True)
        det_roi, rois = detect_roi(img)
        cnt_roi = len(det_roi.detections)
        roi_cnt = len(rois) if rois else 0
        print(f"{cnt_roi}개 (ROI {roi_cnt}개)")

        # 이미지 저장
        vis_full = annotate_image(img, det_full, None, f"FULL: {cnt_full}")
        vis_roi = annotate_image(img, det_roi, rois, f"ROI: {cnt_roi} ({roi_cnt} ROIs)")

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_full.jpg"), vis_full)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_roi.jpg"), vis_roi)

        # 비교 이미지
        scale = min(600 / w, 800 / h, 1.0)
        vis_full_s = cv2.resize(vis_full, (int(w*scale), int(h*scale)))
        vis_roi_s = cv2.resize(vis_roi, (int(w*scale), int(h*scale)))
        compare = np.hstack([vis_full_s, vis_roi_s])
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_compare.jpg"), compare)

        results.append({
            "name": name, "full": cnt_full, "roi": cnt_roi,
            "diff": cnt_full - cnt_roi, "roi_cnt": roi_cnt
        })

    # 요약
    print("\n" + "="*60)
    print("결과 요약")
    print("="*60)
    print(f"{'이미지':<25} {'기존':>6} {'ROI':>6} {'차이':>8}")
    print("-"*60)

    total_full, total_roi = 0, 0
    for r in results:
        total_full += r["full"]
        total_roi += r["roi"]
        diff = f"{r['diff']:+d}" if r['diff'] != 0 else "0"
        print(f"{r['name']:<25} {r['full']:>6} {r['roi']:>6} {diff:>8}")

    print("-"*60)
    diff_total = total_full - total_roi
    pct = (diff_total / total_full * 100) if total_full > 0 else 0
    print(f"{'합계':<25} {total_full:>6} {total_roi:>6} {diff_total:>+8} ({pct:.1f}%)")
    print("="*60)
    print(f"\n결과 저장: {OUTPUT_DIR}")


if __name__ == "__main__":
    config.ROI_TRIAGE_ENABLED = True
    main()
