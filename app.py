"""
AI 시설물 균열 안전점검 시스템 — Streamlit 앱 (모바일 대응)
실행: (venv) streamlit run app.py

구조: UI(이 파일)는 얇게 — 입력·렌더·세션관리만. 파이프라인 흐름은 pipeline/orchestrator.
- 세션 캐시(st.session_state): 같은 사진이면 재분석 안 함 (Streamlit 재실행마다 YOLO 재계산 방지).
- 탐지 이미지에 OpenCV 스켈레톤(균열 중심선) 오버레이 — 재학습 없는 시각 품질 개선.
- 모바일 웹: st.camera_input(폰 카메라 직촬) + st.file_uploader(원본 업로드). 모델/API 없어도 동작.
"""
import numpy as np
import cv2
import streamlit as st

import config
from pipeline import orchestrator, features, detector, rag, report

GRADE_COLOR = {"정상": "#16a34a", "주의": "#d97706", "위험": "#dc2626", "긴급": "#7f1d1d"}


def annotate(img_bgr, det):
    """탐지 박스 + 균열 중심선(스켈레톤) 오버레이 → RGB 반환."""
    vis = img_bgr.copy()
    # 스켈레톤(중심선): 노란색으로, 폰에서 보이게 살짝 두껍게
    sk = features.skeleton_mask(img_bgr, det)
    if sk.any():
        sk = cv2.dilate(sk, np.ones((3, 3), np.uint8), iterations=1)
        vis[sk > 0] = (0, 255, 255)   # BGR 노란색
    # 탐지 박스
    for d in det.detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(vis, f"{d.conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="균열 안전점검", page_icon="🧱", layout="centered")
st.title("🧱 AI 균열 안전점검")
st.caption("균열 사진을 찍거나 올리면 위험도를 판정하고 점검 보고서 초안을 만듭니다.")

# ---- 사이드바: 시스템 상태 ----
with st.sidebar:
    st.header("시스템 상태")
    st.write("🟢 탐지 모델" if detector.is_ready() else "🔴 탐지 모델 (가중치 없음)")
    st.write("🟢 RAG 지식베이스" if rag.is_ready() else "🟡 RAG (인덱스 미구축)")
    _prov = report.provider_label()
    st.write(f"{'🟢' if _prov != '목업' else '🟡'} 보고서 LLM: {_prov}")
    st.divider()
    st.subheader("모델 성능")
    m = config.MODEL_METRICS
    st.metric("mAP50", m["mAP50"])
    st.metric("Recall", m["recall"])

# ---- 1) 입력: 촬영 또는 업로드 (모바일 대응) ----
st.subheader("1) 균열 사진 입력")
mode = st.radio("입력 방식", ["📷 촬영", "🖼️ 사진 선택"],
                horizontal=True, label_visibility="collapsed")
if mode == "📷 촬영":
    up = st.camera_input("균열 부위를 가까이서 촬영하세요")
    st.caption("※ 폰 카메라는 HTTPS 접속에서만 열립니다. 원거리·고해상도는 '사진 선택'을 권장합니다.")
else:
    up = st.file_uploader("균열 사진 업로드 (원본 고해상도 권장)",
                          type=["jpg", "jpeg", "png"])

if up is None:
    st.info("사진을 촬영하거나 업로드하면 분석이 시작됩니다.")
    st.stop()

data = up.getvalue()   # 촬영·업로드 공통 (read()와 달리 재호출 안전)
img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # BGR
if img is None:
    st.error("이미지를 읽을 수 없습니다. 다른 사진으로 시도해 주세요.")
    st.stop()

# ---- 세션 캐시: 같은 사진이면 재분석 안 함 (Streamlit 재실행 대비) ----
h = orchestrator.image_hash(data)
if st.session_state.get("hash") != h:
    with st.spinner("분석 중..."):
        state = orchestrator.analyze(img, h)
        vis_rgb = annotate(img, state.detect)
    st.session_state.update({"hash": h, "state": state, "vis": vis_rgb})

state = st.session_state["state"]
vis_rgb = st.session_state["vis"]
feat, risk, rag_res, rep = state.features, state.risk, state.rag, state.report

# ---- 2) 판정 결과 (세로 스택) ----
st.subheader("2) 판정 결과")

# 위험도 카드 (헤드라인)
color = GRADE_COLOR.get(risk.grade, "#334155")
st.markdown(
    f"<div style='padding:18px;border-radius:14px;background:{color};color:#fff;text-align:center'>"
    f"<div style='font-size:14px;opacity:.9'>위험도 등급</div>"
    f"<div style='font-size:40px;font-weight:800;line-height:1.15'>{risk.grade}</div>"
    f"<div style='font-size:15px'>점수 {risk.score} / 100</div></div>",
    unsafe_allow_html=True)
st.write("")

# 탐지 이미지 (박스 + 균열 중심선 오버레이)
st.image(vis_rgb, use_container_width=True, caption=f"탐지된 균열 {feat.crack_count}개")
st.caption("🟨 노란 선 = AI가 추출한 균열 중심선(OpenCV 스켈레톤) · 🟥 빨간 박스 = YOLO 탐지")
if not detector.is_ready():
    st.warning("탐지 모델(best.pt)이 없어 박스가 표시되지 않습니다.")

# 측정 특징·산정 근거 (접이식 — 모바일 화면 절약)
with st.expander("측정 특징 · 위험도 산정 근거"):
    st.json({"균열 개수": feat.crack_count,
             "최고 신뢰도": feat.max_confidence,
             "최장 길이 비율": feat.max_length_ratio,
             "평균 폭(px)": feat.avg_width_px})
    st.markdown("**Rule 기여 내역**")
    if risk.contributions:
        st.table(risk.contributions)
    else:
        st.write("가점 항목 없음 (정상 범위)")

# ---- 안전기준 근거 (RAG) ----
with st.expander("📚 안전기준 근거 (RAG)", expanded=bool(rag_res.evidences)):
    if rag_res.evidences:
        for e in rag_res.evidences:
            st.markdown(f"> {e.text}  \n_출처: {e.source} (유사도 {e.score})_")
    else:
        st.info("RAG 지식베이스가 아직 구축되지 않았습니다. build_index 실행 후 표시됩니다.")

# ---- 3) 점검 보고서 초안 (현업 6섹션 서식) ----
st.subheader("3) 점검 보고서 초안")
_prov = report.provider_label()
st.caption(f"※ 보고서 LLM: {_prov}" + ("  (LLM 키 없음 → 템플릿 목업)" if _prov == "목업" else ""))
report_md = rep.to_markdown()
for _title, _attr in rep.SECTIONS:
    st.markdown(f"#### {_title}")
    st.markdown(getattr(rep, _attr))
st.download_button(
    "📄 보고서 초안 내려받기 (.md)",
    data=report_md,
    file_name="균열_안전점검_결과보고서_초안.md",
    mime="text/markdown",
    use_container_width=True,
)
