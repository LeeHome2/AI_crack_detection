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
from pipeline import orchestrator, features, detector, rag, report, triage
from schemas import Stage

GRADE_COLOR = {"정상": "#16a34a", "주의": "#d97706", "위험": "#dc2626", "긴급": "#7f1d1d"}

# 결함별 시각 스타일 — 박스 색(BGR·cv2), 범례 색(hex), 한글명, 이미지 라벨(ASCII: cv2가 한글 못 그림)
DEFECT_STYLE = {
    "crack":          {"ko": "균열",      "code": "crack",  "bgr": (0, 0, 255),   "hex": "dc2626"},
    "spalling":       {"ko": "박리/박락",  "code": "spall",  "bgr": (0, 140, 255), "hex": "f97316"},
    "efflorescence":  {"ko": "백태/누수",  "code": "efflor", "bgr": (230, 160, 0), "hex": "0ea5e9"},
    "rebar_exposure": {"ko": "철근노출",   "code": "rebar",  "bgr": (200, 0, 200), "hex": "c026d3"},
    "steel_defect":   {"ko": "강재손상",   "code": "steel",  "bgr": (170, 70, 70), "hex": "4f46e5"},
    "paint_damage":   {"ko": "도장손상",   "code": "paint",  "bgr": (0, 170, 0),   "hex": "16a34a"},
}
_DEFAULT_STYLE = {"ko": "결함", "code": "obj", "bgr": (0, 0, 255), "hex": "dc2626"}


def _style(label):
    return DEFECT_STYLE.get(label, _DEFAULT_STYLE)


def annotate(img_bgr, det):
    """결함별 색상 박스 + 라벨 + 균열 중심선(스켈레톤) 오버레이 → RGB 반환."""
    vis = img_bgr.copy()
    # 스켈레톤(중심선): 균열에만(면적 결함 제외) — features.skeleton_mask가 crack만 처리. 노란색.
    sk = features.skeleton_mask(img_bgr, det)
    if sk.any():
        sk = cv2.dilate(sk, np.ones((3, 3), np.uint8), iterations=1)
        vis[sk > 0] = (0, 255, 255)   # BGR 노란색
    # 결함별 색상 박스 + 라벨(코드+신뢰도)
    for d in det.detections:
        x1, y1, x2, y2 = d.box
        stl = _style(getattr(d, "label", "crack"))
        col = stl["bgr"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
        label = f"{stl['code']} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ly = max(th + 6, y1)
        cv2.rectangle(vis, (x1, ly - th - 6), (x1 + tw + 6, ly), col, -1)
        cv2.putText(vis, label, (x1 + 3, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def defect_chips(det):
    """탐지 결함 종류별 색상 칩(HTML) — 한글명 + 개수. 없으면 빈 문자열."""
    if not det or not det.detections:
        return ""
    from collections import Counter
    cnt = Counter(getattr(d, "label", "crack") for d in det.detections)
    # crack 먼저, 그다음 개수 많은 순
    order = sorted(cnt.items(), key=lambda kv: (kv[0] != "crack", -kv[1]))
    chips = []
    for label, c in order:
        stl = _style(label)
        chips.append(
            f"<span style='display:inline-block;padding:3px 11px;margin:3px 4px 3px 0;"
            f"border-radius:12px;background:#{stl['hex']};color:#fff;font-size:13px;"
            f"font-weight:600'>{stl['ko']} {c}</span>")
    return "".join(chips)


st.set_page_config(page_title="시설물 안전점검", page_icon="🧱", layout="centered")
st.title("🧱 AI 시설물 안전점검")
st.caption("시설물 결함(균열·철근노출·박리·누수 등) 사진을 올리면 위험도를 판정하고 점검 보고서 초안을 만듭니다.")

# ---- 사이드바: 시스템 상태 ----
with st.sidebar:
    st.header("시스템 상태")
    _tri = triage.provider_label()
    st.write(f"{'🟢' if _tri == 'Claude 비전' else '🟡'} 1차 트리아지: {_tri}")
    st.write("🟢 탐지 모델" if detector.is_ready() else "🔴 탐지 모델 (가중치 없음)")
    st.write("🟢 RAG 지식베이스" if rag.is_ready() else "🟡 RAG (인덱스 미구축)")
    _prov = report.provider_label()
    st.write(f"{'🟢' if _prov != '목업' else '🟡'} 보고서 LLM: {_prov}")
    st.divider()
    st.subheader("모델 성능")
    m = config.MODEL_METRICS
    st.metric("mAP50", m["mAP50"])
    st.metric("Recall", m["recall"])

# ---- 1) 입력: 업로드(기본) 또는 촬영 (모바일 대응) ----
# 기본은 '사진 선택'. 촬영은 버튼을 눌러야 카메라가 켜짐(진입 즉시 카메라 안 열림).
st.subheader("1) 균열 사진 입력")
mode = st.radio("입력 방식", ["🖼️ 사진 선택", "📷 촬영"],
                horizontal=True, label_visibility="collapsed")   # 첫 항목=기본값
up = None
if mode == "🖼️ 사진 선택":
    up = st.file_uploader("균열 사진 업로드 (원본 고해상도 권장)",
                          type=["jpg", "jpeg", "png"])
else:
    # 카메라는 '켜기' 버튼을 눌러야 활성화 (원치 않는 자동 실행 방지)
    if not st.session_state.get("cam_on"):
        st.caption("※ 폰 카메라는 HTTPS 접속에서만 열립니다. 원거리·고해상도는 '사진 선택'을 권장합니다.")
        if st.button("📷 카메라 켜기", use_container_width=True):
            st.session_state["cam_on"] = True
            st.rerun()
    else:
        up = st.camera_input("균열 부위를 가까이서 촬영하세요")
        if st.button("카메라 끄기", use_container_width=True):
            st.session_state["cam_on"] = False
            st.rerun()

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
        # 트리아지 게이트로 조기 반환되면 detect 없음 → 어노테이트 생략
        vis_rgb = annotate(img, state.detect) if state.detect is not None else None
    st.session_state.update({"hash": h, "state": state, "vis": vis_rgb})

state = st.session_state["state"]
vis_rgb = st.session_state["vis"]

# ---- 1.5) 비전 트리아지 게이트: 재촬영/반려면 여기서 안내하고 멈춤 ----
if state.stage in (Stage.NEEDS_RETAKE, Stage.REJECTED):
    tri = state.triage
    _icon = "🚫" if state.stage == Stage.REJECTED else "📸"
    _bg = "#7f1d1d" if state.stage == Stage.REJECTED else "#d97706"
    st.markdown(
        f"<div style='padding:16px;border-radius:14px;background:{_bg};color:#fff'>"
        f"<div style='font-size:22px;font-weight:800'>{_icon} 다시 촬영이 필요해요</div>"
        f"<div style='font-size:15px;margin-top:6px'>{tri.message}</div></div>",
        unsafe_allow_html=True)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True,
             caption="업로드한 사진")
    st.caption(f"※ 1차 판정: {tri.provider} · 판정 {tri.verdict}"
               + (f" · 선명도 {tri.blur_score}" if tri.blur_score else ""))
    st.stop()

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

# 탐지 이미지 (결함별 색상 박스 + 균열 중심선 오버레이)
st.image(vis_rgb, use_container_width=True, caption="탐지 결과 (결함별 색상 구분)")
_chips = defect_chips(state.detect)
if _chips:
    st.markdown("**탐지된 결함:** " + _chips, unsafe_allow_html=True)
st.caption("🟨 노란 선 = 균열 중심선(OpenCV 스켈레톤) · 색상 박스 = 결함별 YOLO 탐지 (위 범례 색상)")
if not detector.is_ready():
    st.warning("탐지 모델(best.pt)이 없어 박스가 표시되지 않습니다.")

# 측정 특징·산정 근거 (접이식 — 모바일 화면 절약)
with st.expander("측정 특징 · 위험도 산정 근거"):
    _info = {"균열 개수": feat.crack_count,
             "최고 신뢰도(균열)": feat.max_confidence,
             "최장 길이 비율": feat.max_length_ratio,
             "평균 폭(px)": feat.avg_width_px}
    if getattr(feat, "defects", None):
        _info["복합 결함(균열 외)"] = {
            config.DEFECT_KO.get(k, k): {"개수": v.get("count"), "최고신뢰도": v.get("max_conf")}
            for k, v in feat.defects.items()}
    st.json(_info)
    st.markdown("**Rule 기여 내역**")
    if risk.contributions:
        st.table(risk.contributions)
    else:
        st.write("가점 항목 없음 (정상 범위)")

# ---- 안전기준 근거 (RAG) ----
with st.expander("📚 안전기준 근거 (RAG)", expanded=bool(rag_res.evidences)):
    if rag_res.evidences:
        for e in rag_res.evidences:
            src = f"[{e.source}]({e.url})" if e.url else e.source
            st.markdown(f"> {e.text}  \n_근거 출처: {src} (유사도 {e.score})_")
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
