"""
ROI 기반 탐지 전체 테스트 + 기존 방식 비교
==========================================
모든 샘플 이미지에 대해:
1) 기존 방식 (전체 이미지 YOLO)
2) ROI 방식 (Vision ROI → YOLO)
결과를 비교하여 저장

실행: python test_roi_full.py
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline import detector, roi_triage, features, postprocess
from schemas import DetectResult

# 테스트 이미지 경로
TEST_DIR = r"C:\Users\user\Desktop\crack_test"
OUTPUT_DIR = os.path.join(TEST_DIR, "demo", "results_roi")

# 결함 스타일
DEFECT_STYLE = {
    "crack":          {"ko": "균열",      "bgr": (0, 0, 255),   "hex": "#dc2626"},
    "spalling":       {"ko": "박리/박락",  "bgr": (0, 140, 255), "hex": "#f97316"},
    "efflorescence":  {"ko": "백태/누수",  "bgr": (230, 160, 0), "hex": "#0ea5e9"},
    "rebar_exposure": {"ko": "철근노출",   "bgr": (200, 0, 200), "hex": "#c026d3"},
    "steel_defect":   {"ko": "강재손상",   "bgr": (170, 70, 70), "hex": "#4f46e5"},
    "paint_damage":   {"ko": "도장손상",   "bgr": (0, 170, 0),   "hex": "#16a34a"},
}
_DEFAULT = {"ko": "결함", "bgr": (0, 0, 255), "hex": "#dc2626"}


def get_style(label):
    return DEFECT_STYLE.get(label, _DEFAULT)


def annotate_image(img_bgr, det, rois=None, mode_label=""):
    """결과 이미지 생성 (박스 + ROI 영역 + 라벨)"""
    vis = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    # ROI 영역 표시 (파란 점선)
    if rois:
        for i, roi in enumerate(rois):
            x1 = int(roi["x1"] * w)
            y1 = int(roi["y1"] * h)
            x2 = int(roi["x2"] * w)
            y2 = int(roi["y2"] * h)
            # 점선 효과
            for j in range(x1, x2, 15):
                cv2.line(vis, (j, y1), (min(j+8, x2), y1), (255, 150, 0), 2)
                cv2.line(vis, (j, y2), (min(j+8, x2), y2), (255, 150, 0), 2)
            for j in range(y1, y2, 15):
                cv2.line(vis, (x1, j), (x1, min(j+8, y2)), (255, 150, 0), 2)
                cv2.line(vis, (x2, j), (x2, min(j+8, y2)), (255, 150, 0), 2)
            # ROI 라벨
            desc = roi.get("description", f"ROI {i+1}")[:25]
            cv2.putText(vis, desc, (x1+5, y1+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)

    # 균열 스켈레톤 (노란색)
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
        cv2.putText(vis, txt, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, stl["bgr"], 2)

    # 모드 라벨 (상단)
    if mode_label:
        cv2.rectangle(vis, (0, 0), (300, 35), (50, 50, 50), -1)
        cv2.putText(vis, mode_label, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


def detect_full_image(img_bgr):
    """기존 방식: 전체 이미지 YOLO"""
    if detector.is_hybrid_ready():
        det = detector.detect_hybrid(img_bgr)
    else:
        det = detector.detect(img_bgr)
    # 후처리
    det = postprocess.filter_seams(img_bgr, det)
    if config.MERGE_OVERLAP_ENABLED:
        det = postprocess.merge_overlapping_boxes(det, label_filter="crack",
                                                   margin=config.MERGE_OVERLAP_MARGIN)
    return det


def detect_with_roi(img_bgr):
    """ROI 방식: Vision ROI → YOLO"""
    if detector.is_hybrid_ready():
        det_fn = detector.detect_hybrid
    else:
        det_fn = detector.detect

    det, rois = roi_triage.detect_with_roi(img_bgr, det_fn)

    # 후처리
    det = postprocess.filter_seams(img_bgr, det)
    if config.MERGE_OVERLAP_ENABLED:
        det = postprocess.merge_overlapping_boxes(det, label_filter="crack",
                                                   margin=config.MERGE_OVERLAP_MARGIN)
    return det, rois


def count_by_label(det):
    """라벨별 개수"""
    from collections import Counter
    return Counter(getattr(d, "label", "crack") for d in det.detections)


def test_all_images():
    """모든 테스트 이미지에 대해 비교 테스트"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 이미지 파일 수집
    images = []
    for f in os.listdir(TEST_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(f)

    if not images:
        print(f"테스트 이미지 없음: {TEST_DIR}")
        return

    print("="*70)
    print("ROI 기반 탐지 전체 테스트")
    print("="*70)
    print(f"이미지 수: {len(images)}")
    print(f"출력 경로: {OUTPUT_DIR}")
    print("="*70)

    results = []

    for img_file in sorted(images):
        img_path = os.path.join(TEST_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] 로드 실패: {img_file}")
            continue

        name = os.path.splitext(img_file)[0]
        h, w = img.shape[:2]
        print(f"\n[{name}] {w}x{h}")

        # 1) 기존 방식
        print("  기존 방식...", end=" ", flush=True)
        det_full = detect_full_image(img)
        cnt_full = len(det_full.detections)
        print(f"{cnt_full}개 탐지")

        # 2) ROI 방식
        print("  ROI 방식...", end=" ", flush=True)
        det_roi, rois = detect_with_roi(img)
        cnt_roi = len(det_roi.detections)
        roi_count = len(rois) if rois else 0
        print(f"{cnt_roi}개 탐지 (ROI {roi_count}개)")

        # 결과 저장
        vis_full = annotate_image(img, det_full, None, f"FULL: {cnt_full} det")
        vis_roi = annotate_image(img, det_roi, rois, f"ROI: {cnt_roi} det ({roi_count} ROIs)")

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_1_full.jpg"), vis_full)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_2_roi.jpg"), vis_roi)

        # 나란히 비교 이미지
        scale = 800 / max(w, h)
        vis_full_s = cv2.resize(vis_full, (int(w*scale), int(h*scale)))
        vis_roi_s = cv2.resize(vis_roi, (int(w*scale), int(h*scale)))
        compare = np.hstack([vis_full_s, vis_roi_s])
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_compare.jpg"), compare)

        # 결과 기록
        diff = cnt_full - cnt_roi
        results.append({
            "name": name,
            "size": f"{w}x{h}",
            "full": cnt_full,
            "roi": cnt_roi,
            "diff": diff,
            "roi_count": roi_count,
            "full_labels": dict(count_by_label(det_full)),
            "roi_labels": dict(count_by_label(det_roi)),
        })

    # 요약 출력
    print("\n" + "="*70)
    print("결과 요약")
    print("="*70)
    print(f"{'이미지':<25} {'크기':<12} {'기존':>6} {'ROI':>6} {'차이':>6} {'ROI수':>6}")
    print("-"*70)

    total_full = 0
    total_roi = 0
    for r in results:
        total_full += r["full"]
        total_roi += r["roi"]
        diff_str = f"-{r['diff']}" if r['diff'] > 0 else (f"+{-r['diff']}" if r['diff'] < 0 else "0")
        print(f"{r['name']:<25} {r['size']:<12} {r['full']:>6} {r['roi']:>6} {diff_str:>6} {r['roi_count']:>6}")

    print("-"*70)
    total_diff = total_full - total_roi
    pct = (total_diff / total_full * 100) if total_full > 0 else 0
    print(f"{'합계':<25} {'':<12} {total_full:>6} {total_roi:>6} {-total_diff:>+6} ({pct:.1f}% 감소)")
    print("="*70)

    # 요약 파일 저장
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("ROI 기반 탐지 테스트 결과\n")
        f.write("="*60 + "\n\n")
        for r in results:
            diff_str = f"-{r['diff']}" if r['diff'] > 0 else (f"+{-r['diff']}" if r['diff'] < 0 else "0")
            f.write(f"{r['name']}: 기존 {r['full']}개 → ROI {r['roi']}개 ({diff_str})\n")
            f.write(f"  기존 라벨: {r['full_labels']}\n")
            f.write(f"  ROI 라벨: {r['roi_labels']}\n")
            if r['roi_count'] > 0:
                f.write(f"  ROI 영역: {r['roi_count']}개\n")
            f.write("\n")
        f.write("-"*60 + "\n")
        f.write(f"합계: {total_full}개 → {total_roi}개 ({pct:.1f}% 감소)\n")

    print(f"\n결과 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    # ROI 모드 강제 활성화 (테스트용)
    config.ROI_TRIAGE_ENABLED = True
    test_all_images()
