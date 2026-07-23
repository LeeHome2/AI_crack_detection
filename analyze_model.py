"""
6종 모델 분석 (analyze_model.py) — 데스크탑(모델·데이터 있는 곳)에서 실행.
전체 지표 + 클래스별 AP/P/R + (선택) 샘플 예측 요약을 콘솔·model_metrics.json 으로.

실행:
  (venv) D:\crack_detection> python analyze_model.py
  (경로 다르면)          python analyze_model.py models\yolov8s_defect6_final.pt D:\crack_detection\dataset_6class\data.yaml
  (샘플 예측도)          python analyze_model.py models\yolov8s_defect6_final.pt D:\crack_detection\dataset_6class\data.yaml  test_images\

결과 model_metrics.json 을 원격 세션(여기)에 업로드하면 상세 분석해줌.
"""
import sys
import json
import os
from collections import Counter

try:
    from ultralytics import YOLO
except Exception as e:
    print("ultralytics 필요:", e)
    sys.exit(1)

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join("models", "yolov8s_defect6_final.pt")
DATA = sys.argv[2] if len(sys.argv) > 2 else r"D:\crack_detection\dataset_6class\data.yaml"
SAMPLE_DIR = sys.argv[3] if len(sys.argv) > 3 else ""

m = YOLO(MODEL)
names = {int(k): v for k, v in m.names.items()}
print(f"[모델] {MODEL} · 클래스 {names}")

metrics = m.val(data=DATA, verbose=False)
out = {
    "model": os.path.basename(MODEL),
    "overall": {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    },
    "per_class": {},
}
# 클래스별 (val 에 등장한 클래스만 index 제공)
try:
    idxs = list(metrics.box.ap_class_index)
    for j, ci in enumerate(idxs):
        nm = names.get(int(ci), str(ci))
        out["per_class"][nm] = {
            "AP50": round(float(metrics.box.ap50[j]), 4),
            "AP50_95": round(float(metrics.box.ap[j]), 4),
            "precision": round(float(metrics.box.p[j]), 4),
            "recall": round(float(metrics.box.r[j]), 4),
        }
except Exception as e:
    out["per_class_error"] = str(e)

# (선택) 샘플 이미지 예측 — 클래스별 탐지 수·평균 신뢰도
if SAMPLE_DIR and os.path.isdir(SAMPLE_DIR):
    imgs = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))][:20]
    cnt, conf = Counter(), {}
    for r in m.predict(imgs, conf=0.15, verbose=False):
        for b in r.boxes:
            nm = names.get(int(b.cls[0]), "?")
            cnt[nm] += 1
            conf.setdefault(nm, []).append(float(b.conf[0]))
    out["sample_pred"] = {nm: {"n": cnt[nm], "avg_conf": round(sum(conf[nm]) / len(conf[nm]), 3)}
                          for nm in cnt}

print(json.dumps(out, ensure_ascii=False, indent=2))
with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n저장: model_metrics.json  (이 파일을 원격 세션에 업로드하면 분석해줌)")
print("혼동행렬: runs/detect/val*/confusion_matrix.png")
