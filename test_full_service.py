"""
전체 서비스 통합 테스트
========================
트리아지 → 탐지 → 분석 → 시각화 → 보고서 생성

실행: python test_full_service.py
"""
import os
import sys
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline import orchestrator, features, triage, detector, report

# 설정
config.ROI_TRIAGE_ENABLED = True

SAMPLES_DIR = r"C:\Users\user\Desktop\crack_test\samples"
OUTPUT_DIR = r"C:\Users\user\Desktop\crack_test\service_test_result"

DEFECT_STYLE = {
    "crack": {"ko": "균열", "bgr": (0, 0, 255), "hex": "#dc2626"},
    "spalling": {"ko": "박리/박락", "bgr": (0, 140, 255), "hex": "#f97316"},
    "efflorescence": {"ko": "백태/누수", "bgr": (230, 160, 0), "hex": "#0ea5e9"},
    "rebar_exposure": {"ko": "철근노출", "bgr": (200, 0, 200), "hex": "#c026d3"},
    "steel_defect": {"ko": "강재손상", "bgr": (170, 70, 70), "hex": "#4f46e5"},
    "paint_damage": {"ko": "도장손상", "bgr": (0, 170, 0), "hex": "#16a34a"},
}


def get_style(label):
    return DEFECT_STYLE.get(label, {"ko": "결함", "bgr": (0, 0, 255)})


def annotate_image(img_bgr, state):
    """결과 시각화 이미지 생성"""
    vis = img_bgr.copy()
    det = state.detect

    if det and det.detections:
        # 스켈레톤
        sk = features.skeleton_mask(img_bgr, det)
        if sk.any():
            sk = cv2.dilate(sk, np.ones((3, 3), np.uint8), iterations=1)
            vis[sk > 0] = (0, 255, 255)

        # 박스
        for d in det.detections:
            x1, y1, x2, y2 = [int(v) for v in d.box]
            label = getattr(d, "label", "crack")
            stl = get_style(label)
            cv2.rectangle(vis, (x1, y1), (x2, y2), stl["bgr"], 2)
            txt = f"{label[:6]} {d.conf:.2f}"
            cv2.putText(vis, txt, (x1, max(y1-5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, stl["bgr"], 2)

    return vis


def create_header(img, fname, state):
    """이미지 상단에 정보 헤더 추가"""
    h, w = img.shape[:2]
    header_h = 50
    result = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    result[header_h:, :] = img

    # 헤더 배경
    triage_verdict = state.triage.verdict if state.triage else "unknown"
    if triage_verdict != "ok":
        cv2.rectangle(result, (0, 0), (w, header_h), (50, 50, 100), -1)  # 회색-파랑
    else:
        grade = state.risk.grade if state.risk else "N/A"
        grade_colors = {"정상": (0, 150, 0), "주의": (0, 180, 220),
                       "위험": (0, 100, 200), "긴급": (0, 0, 180)}
        cv2.rectangle(result, (0, 0), (w, header_h), grade_colors.get(grade, (50,50,50)), -1)

    # 텍스트
    det_cnt = len(state.detect.detections) if state.detect else 0
    score = state.risk.score if state.risk else 0
    grade = state.risk.grade if state.risk else "N/A"

    if triage_verdict != "ok":
        text = f"{fname} | Triage: {triage_verdict}"
    else:
        text = f"{fname} | Det: {det_cnt} | Score: {score} | Grade: {grade}"

    cv2.putText(result, text, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)

    images = sorted([f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith('.jpg')])

    print("=" * 70)
    print("전체 서비스 통합 테스트")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"이미지: {len(images)}장")
    print(f"출력: {OUTPUT_DIR}")
    print("=" * 70)

    results = []
    triage_stats = {}
    grade_stats = {}
    defect_stats = {}

    for i, fname in enumerate(images, 1):
        img_path = os.path.join(SAMPLES_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{i:02d}] {fname}: 읽기 실패")
            continue

        # 전체 파이프라인 실행
        state = orchestrator.analyze(img)

        # 트리아지 결과
        triage_verdict = state.triage.verdict if state.triage else "unknown"
        triage_stats[triage_verdict] = triage_stats.get(triage_verdict, 0) + 1

        # 탐지 결과
        det_cnt = len(state.detect.detections) if state.detect else 0

        # 위험도
        score = state.risk.score if state.risk else 0
        grade = state.risk.grade if state.risk else "N/A"

        if triage_verdict == "ok":
            grade_stats[grade] = grade_stats.get(grade, 0) + 1

        # 결함 유형별 집계
        labels = {}
        if state.detect:
            for d in state.detect.detections:
                lbl = getattr(d, "label", "crack")
                labels[lbl] = labels.get(lbl, 0) + 1
                defect_stats[lbl] = defect_stats.get(lbl, 0) + 1

        label_str = ", ".join([f"{k}:{v}" for k, v in labels.items()]) if labels else "-"

        # 결과 저장
        result_entry = {
            "filename": fname,
            "triage": triage_verdict,
            "detections": det_cnt,
            "score": score,
            "grade": grade,
            "defects": label_str
        }
        results.append(result_entry)

        # 상태 출력
        if triage_verdict != "ok":
            status = f"SKIP ({triage_verdict})"
        else:
            status = f"{det_cnt}det | {score}pt | {grade}"
        print(f"[{i:02d}] {fname[:25]:<25} | {status}")

        # 시각화 이미지 저장 (triage 통과한 것만)
        if triage_verdict == "ok":
            vis = annotate_image(img, state)
            vis_with_header = create_header(vis, fname, state)
            out_path = os.path.join(OUTPUT_DIR, "images", fname.replace(".jpg", "_result.jpg"))
            cv2.imwrite(out_path, vis_with_header)

            # 개별 보고서 생성 (상위 등급만)
            if grade in ["위험", "긴급"]:
                try:
                    report_path = os.path.join(OUTPUT_DIR, "reports", fname.replace(".jpg", "_report.md"))
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(f"# 점검 보고서: {fname}\n\n")
                        f.write(f"## 기본 정보\n")
                        f.write(f"- 파일: {fname}\n")
                        f.write(f"- 등급: **{grade}** ({score}점)\n")
                        f.write(f"- 탐지: {det_cnt}개\n")
                        f.write(f"- 결함: {label_str}\n\n")
                        if state.report:
                            f.write(f"## 상세 분석\n")
                            f.write(state.report.basic_info + "\n\n")
                            if state.report.recommendations:
                                f.write(f"## 권고사항\n")
                                f.write(state.report.recommendations + "\n")
                except Exception as e:
                    print(f"    보고서 생성 실패: {e}")
        else:
            # triage 실패 이미지도 표시 (원본 + 헤더)
            vis_with_header = create_header(img, fname, state)
            out_path = os.path.join(OUTPUT_DIR, "images", fname.replace(".jpg", "_skipped.jpg"))
            cv2.imwrite(out_path, vis_with_header)

    # 종합 보고서 생성
    print("\n" + "=" * 70)
    print("종합 보고서 생성 중...")

    summary_path = os.path.join(OUTPUT_DIR, "검증보고서_종합.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 서비스 검증 종합 보고서\n\n")
        f.write(f"## 1. 검증 개요\n\n")
        f.write(f"| 항목 | 내용 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 검증 일시 | {datetime.now().strftime('%Y-%m-%d %H:%M')} |\n")
        f.write(f"| 샘플 경로 | {SAMPLES_DIR} |\n")
        f.write(f"| 이미지 수 | {len(images)}장 |\n")
        f.write(f"| ROI 모드 | {'활성' if config.ROI_TRIAGE_ENABLED else '비활성'} |\n\n")

        f.write(f"## 2. 트리아지 결과\n\n")
        f.write(f"| 판정 | 수량 | 설명 |\n")
        f.write(f"|------|------|------|\n")
        triage_desc = {
            "ok": "정상 → 탐지 수행",
            "retake_far": "원거리 촬영 → 재촬영 요청",
            "retake_blur": "흐림/저화질 → 재촬영 요청",
            "not_crack": "균열 대상 아님 → 스킵"
        }
        for k, v in triage_stats.items():
            desc = triage_desc.get(k, k)
            f.write(f"| {k} | {v}장 | {desc} |\n")

        f.write(f"\n## 3. 등급 분포 (triage=ok)\n\n")
        f.write(f"| 등급 | 수량 | 비율 |\n")
        f.write(f"|------|------|------|\n")
        ok_count = triage_stats.get("ok", 0)
        for g in ["정상", "주의", "위험", "긴급"]:
            cnt = grade_stats.get(g, 0)
            pct = (cnt / ok_count * 100) if ok_count > 0 else 0
            f.write(f"| {g} | {cnt}장 | {pct:.1f}% |\n")

        f.write(f"\n## 4. 결함 유형 통계\n\n")
        f.write(f"| 결함 유형 | 탐지 수 |\n")
        f.write(f"|----------|--------|\n")
        for lbl, cnt in sorted(defect_stats.items(), key=lambda x: -x[1]):
            ko = DEFECT_STYLE.get(lbl, {}).get("ko", lbl)
            f.write(f"| {lbl} ({ko}) | {cnt}개 |\n")
        total_defects = sum(defect_stats.values())
        f.write(f"| **합계** | **{total_defects}개** |\n")

        f.write(f"\n## 5. 상세 결과\n\n")
        f.write(f"| 파일 | 트리아지 | 탐지 | 점수 | 등급 | 결함 |\n")
        f.write(f"|------|----------|------|------|------|------|\n")
        for r in results:
            f.write(f"| {r['filename']} | {r['triage']} | {r['detections']} | {r['score']} | {r['grade']} | {r['defects']} |\n")

        f.write(f"\n## 6. 결과 파일\n\n")
        f.write(f"```\n")
        f.write(f"{OUTPUT_DIR}/\n")
        f.write(f"├── images/          # 시각화 결과\n")
        f.write(f"│   ├── *_result.jpg # 탐지 결과 (triage=ok)\n")
        f.write(f"│   └── *_skipped.jpg # 스킵된 이미지 (triage!=ok)\n")
        f.write(f"├── reports/         # 개별 보고서 (위험/긴급)\n")
        f.write(f"└── 검증보고서_종합.md  # 본 문서\n")
        f.write(f"```\n\n")
        f.write(f"---\n*자동 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # 결과 출력
    print("=" * 70)
    print("트리아지 결과:")
    for k, v in triage_stats.items():
        print(f"  {k}: {v}장")

    print("\n등급 분포 (triage=ok):")
    for g in ["정상", "주의", "위험", "긴급"]:
        if g in grade_stats:
            print(f"  {g}: {grade_stats[g]}장")

    print("\n결함 유형:")
    for lbl, cnt in sorted(defect_stats.items(), key=lambda x: -x[1]):
        print(f"  {lbl}: {cnt}개")

    print("=" * 70)
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
