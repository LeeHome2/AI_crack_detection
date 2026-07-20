"""
폴더 일괄 테스트 (predict_folder.py)
- 지정 폴더의 모든 사진에 전체 파이프라인(탐지 -> Feature -> Rule) 실행
- 출력: 주석 이미지(박스+등급) + 요약 CSV
- RAG/보고서는 제외 (대량 테스트용)

실행:
  (venv) D:\crack_detection> python predict_folder.py
"""
import os
import glob
import csv
import cv2

import config
from pipeline import detector, features, rules

SRC_DIR = os.path.join(config.BASE_DIR, "Mobile Devices")     # 테스트할 사진 폴더
OUT_DIR = os.path.join(config.BASE_DIR, "runs", "crack", "mobile_test")
CSV_PATH = os.path.join(OUT_DIR, "_summary.csv")

GRADE_BGR = {"정상": (80, 160, 40), "주의": (0, 150, 220),
             "위험": (0, 0, 220), "긴급": (30, 20, 130)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not detector.is_ready():
        print("[중단] 탐지 모델(best.pt)이 없습니다. train_tiled_full 학습 후 실행하세요.")
        return

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
        imgs += glob.glob(os.path.join(SRC_DIR, "**", ext), recursive=True)
    imgs = sorted(set(imgs))
    print(f"대상 사진: {len(imgs)}장")

    rows = []
    grade_count = {}
    for i, path in enumerate(imgs, 1):
        img = cv2.imread(path)
        if img is None:
            continue
        det = detector.detect(img)
        feat = features.extract(img, det)
        risk = rules.evaluate(feat)   # 대량 테스트는 RAG 없이 Rule만

        # 주석 이미지
        vis = img.copy()
        color = GRADE_BGR.get(risk.grade, (80, 80, 80))
        for d in det.detections:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 4)
            cv2.putText(vis, f"{d.conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vis, f"{risk.grade} ({risk.score}) / crack {feat.crack_count}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(path)), vis)

        rows.append({
            "file": os.path.basename(path),
            "crack_count": feat.crack_count,
            "max_conf": feat.max_confidence,
            "score": risk.score,
            "grade": risk.grade,
            "max_length_ratio": feat.max_length_ratio,
            "avg_width_px": feat.avg_width_px,
        })
        grade_count[risk.grade] = grade_count.get(risk.grade, 0) + 1
        if i % 10 == 0:
            print(f"진행 {i}/{len(imgs)}...")

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    detected = sum(1 for r in rows if r["crack_count"] > 0)
    print("\n===== 일괄 테스트 완료 =====")
    print(f"총 {len(rows)}장 중 균열 탐지 {detected}장 ({detected/len(rows)*100:.0f}%)")
    print(f"등급 분포: {grade_count}")
    print(f"주석 이미지: {OUT_DIR}")
    print(f"요약 CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
