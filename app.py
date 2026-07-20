"""
AI 시설물 균열 안전점검 시스템 — Streamlit 앱 (골격)
실행: (venv) streamlit run app.py

파이프라인: 업로드 -> 탐지 -> Feature -> (RAG + Rule) -> 보고서 -> 화면
모델/API가 없어도 실행되며, 준비 안 된 단계는 안내 메시지를 표시.
"""
import numpy as np
import cv2
import streamlit as st

import config
from pipeline import detector, features, rules, rag, report

GRADE_COLOR = {"정상": "#16a34a", "주의": "#d97706", "위험": "#dc2626", "긴급": "#7f1d1d"}

st.set_page_config(page_title="균열 안전점검", page_icon="🧱", layout="wide")
st.title("🧱 AI 시설물 균열 안전점검 시스템")
st.caption("사진을 올리면 균열을 탐지하고, 위험도와 안전기준 근거로 점검 보고서 초안을 만듭니다.")

# ---- 사이드바: 시스템 상태 ----
with st.sidebar:
    st.header("시스템 상태")
    st.write("🟢 탐지 모델" if detector.is_ready() else "🔴 탐지 모델 (train_tiled_full 없음)")
    st.write("🟢 RAG 지식베이스" if rag.is_ready() else "🟡 RAG (인덱스 미구축)")
    st.write("🟢 Claude API" if config.ANTHROPIC_API_KEY else "🟡 보고서 (API 키 없음 → 목업)")
    st.divider()
    st.subheader("모델 성능")
    m = config.MODEL_METRICS
    st.metric("mAP50", m["mAP50"])
    st.metric("Recall", m["recall"])

# ---- 업로드 ----
up = st.file_uploader("시설물 균열 사진 업로드", type=["jpg", "jpeg", "png"])
if up is None:
    st.info("사진을 업로드하면 분석이 시작됩니다.")
    st.stop()

file_bytes = np.frombuffer(up.read(), np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)   # BGR

# ---- 파이프라인 실행 ----
with st.spinner("분석 중..."):
    det = detector.detect(img)
    feat = features.extract(img, det)
    risk_pre = rules.evaluate(feat)                 # RAG 전 1차
    rag_res = rag.search(feat, risk_pre)
    risk = rules.evaluate(feat, rag_res)            # RAG 반영 최종
    rep = report.generate(feat, risk, rag_res)

# ---- 탐지 이미지 ----
vis = img.copy()
for d in det.detections:
    x1, y1, x2, y2 = d.box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(vis, f"{d.conf:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("분석 결과")
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
    if not detector.is_ready():
        st.warning("탐지 모델(best.pt)이 없어 박스가 표시되지 않습니다. train_tiled_full 학습 후 이용하세요.")

with col2:
    # 위험도 카드
    color = GRADE_COLOR.get(risk.grade, "#334155")
    st.markdown(
        f"<div style='padding:16px;border-radius:12px;background:{color};color:#fff'>"
        f"<div style='font-size:14px'>위험도 등급</div>"
        f"<div style='font-size:34px;font-weight:800'>{risk.grade}</div>"
        f"<div style='font-size:14px'>점수 {risk.score}</div></div>",
        unsafe_allow_html=True)
    st.subheader("Rule 기여 내역")
    if risk.contributions:
        st.table(risk.contributions)
    else:
        st.write("가점 항목 없음 (정상 범위)")

    st.subheader("Feature")
    st.json({"균열 개수": feat.crack_count,
             "최고 신뢰도": feat.max_confidence,
             "최장 길이 비율": feat.max_length_ratio,
             "평균 폭(px)": feat.avg_width_px})

# ---- RAG 근거 ----
st.subheader("📚 안전기준 근거 (RAG)")
if rag_res.evidences:
    for e in rag_res.evidences:
        st.markdown(f"> {e.text}  \n_출처: {e.source} (유사도 {e.score})_")
else:
    st.info("RAG 지식베이스가 아직 구축되지 않았습니다. knowledge/build_index.py 실행 후 표시됩니다.")

# ---- 보고서 초안 ----
st.subheader("📝 점검 보고서 초안")
if not config.ANTHROPIC_API_KEY:
    st.caption("※ API 키 미설정 → 템플릿 목업으로 표시됩니다.")
st.markdown(f"**요약**  \n{rep.summary}")
st.markdown(f"**위험도 설명**  \n{rep.risk_explain}")
st.markdown(f"**권고 조치**  \n{rep.actions}")
st.markdown(f"**전문점검 권고**  \n{rep.inspection_advice}")
