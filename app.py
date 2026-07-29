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
from pipeline import orchestrator, features, detector, rag, report, triage, pdf_export, multiturn
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


def annotate(img_bgr, det, crack_mask=None):
    """결함별 색상 박스 + 균열 오버레이 → RGB 반환.
    - crack_mask(하이브리드 seg) 있으면: 정밀 마스크를 반투명 채움 + 외곽선으로 표시(균열 박스 생략).
    - 없으면: 기존 OpenCV 스켈레톤 중심선(노란색).
    """
    vis = img_bgr.copy()
    has_mask = crack_mask is not None and getattr(crack_mask, "any", lambda: False)()
    if has_mask:
        # seg 정밀 균열 영역: 반투명 노란 채움 + 선명한 외곽선(데모 임팩트)
        m = (crack_mask > 0)
        tint = vis.copy(); tint[m] = (0, 255, 255)
        vis = cv2.addWeighted(vis, 0.55, tint, 0.45, 0)
        cnts, _ = cv2.findContours((m.astype(np.uint8)) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 200, 255), 2)
    else:
        # 스켈레톤(중심선): 균열에만 — features.skeleton_mask가 crack만 처리. 노란색.
        sk = features.skeleton_mask(img_bgr, det)
        if sk.any():
            sk = cv2.dilate(sk, np.ones((3, 3), np.uint8), iterations=1)
            vis[sk > 0] = (0, 255, 255)   # BGR 노란색
    # 결함별 색상 박스 + 라벨(코드+신뢰도). 하이브리드면 균열은 마스크로 대체 → 박스 생략.
    for d in det.detections:
        if has_mask and getattr(d, "label", "crack") == "crack":
            continue
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


def _no_strike(s) -> str:
    """RAG/문서 원문의 등급구간 표기(0.1~0.3 등)가 마크다운 취소선(~..~)으로
    깨지는 것 방지. 단일 물결표만 이스케이프 → 표·굵게 등 다른 서식은 보존."""
    return str(s).replace("~", "\\~")


st.set_page_config(page_title=f"시설물 안전점검 · {config.APP_VARIANT}",
                   page_icon="🧱", layout="centered")
st.title("🧱 AI 시설물 안전점검")
st.caption(f"🏷️ **{config.APP_VARIANT}** — {config.APP_VARIANT_DESC}")
st.caption("시설물 결함(균열·철근노출·박리·누수 등) 사진을 올리면 위험도를 판정하고 점검 보고서 초안을 만듭니다.")

# ---- 사이드바: 시스템 상태 ----
with st.sidebar:
    st.header("시스템 상태")
    st.write(f"🏷️ 배포 라인: **{config.APP_VARIANT}**")
    st.caption(config.APP_VARIANT_DESC)

    st.divider()
    st.subheader("파이프라인")
    _tri = triage.provider_label()
    _roi_enabled = getattr(config, "ROI_TRIAGE_ENABLED", True)
    _roi = "활성" if _roi_enabled else "비활성"
    st.write(f"{'🟢' if _tri == 'Claude 비전' else '🟡'} 트리아지: {_tri}")
    st.write(f"{'🟢' if _roi_enabled else '⚪'} ROI 모드: {_roi}")
    _hybrid_enabled = getattr(config, "HYBRID_DETECT_ENABLED", True)
    _hybrid = "하이브리드" if _hybrid_enabled and detector.is_hybrid_ready() else "단일모델"
    st.write(f"🟢 탐지: {_hybrid}" if detector.is_ready() else "🔴 탐지 모델 없음")
    st.write("🟢 RAG 지식베이스" if rag.is_ready() else "🟡 RAG (미구축)")
    _prov = report.provider_label()
    st.write(f"{'🟢' if _prov != '목업' else '🟡'} 보고서: {_prov}")
    st.write(f"{'🟢' if pdf_export.is_available() else '🟡'} PDF: {'활성' if pdf_export.is_available() else 'fpdf2 필요'}")

    st.divider()
    st.subheader("모델 성능")
    m = config.MODEL_METRICS
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("mAP50", f"{m['mAP50']:.1%}")
        st.metric("Precision", f"{m['precision']:.1%}")
    with col_m2:
        st.metric("mAP50-95", f"{m['mAP50_95']:.1%}")
        st.metric("Recall", f"{m['recall']:.1%}")

    st.divider()
    st.subheader("탐지 대상 결함")
    defect_list = "균열 · 박리 · 백태 · 철근노출 · 강재손상 · 도장손상"
    st.caption(defect_list)

    st.divider()
    st.subheader("테스트 옵션")
    skip_report = st.toggle("⚡ 빠른 테스트 (보고서 생략)", value=False,
                            help="LLM 보고서 생성을 건너뛰어 검출만 빠르게 확인")

# ---- 1) 시설물 정보 입력 (사진 업로드 전에 미리 입력 가능) ----
st.subheader("1) 시설물 정보")
st.caption("보고서에 들어갈 정보를 미리 입력하세요. 사진 분석 시 자동 반영됩니다.")
col1, col2 = st.columns(2)
with col1:
    facility_name = st.text_input("시설물명", placeholder="예: ○○아파트 101동",
                                  value=st.session_state.get("user_info", {}).get("facility_name", ""))
    location = st.text_input("위치", placeholder="예: 서울시 강남구 ○○로 123",
                             value=st.session_state.get("user_info", {}).get("location", ""))
with col2:
    inspector = st.text_input("점검자", placeholder="예: 홍길동",
                              value=st.session_state.get("user_info", {}).get("inspector", ""))
    part_detail = st.text_input("점검 부위 상세", placeholder="예: 지하주차장 B2층 기둥",
                                value=st.session_state.get("user_info", {}).get("part_detail", ""))
remarks = st.text_area("비고", placeholder="특이사항이나 추가 메모", height=68,
                       value=st.session_state.get("user_info", {}).get("remarks", ""))

# 입력값 즉시 세션에 저장 (분석 시 자동 반영)
user_info = {
    "facility_name": facility_name.strip() if facility_name else "",
    "location": location.strip() if location else "",
    "inspector": inspector.strip() if inspector else "",
    "part_detail": part_detail.strip() if part_detail else "",
    "remarks": remarks.strip() if remarks else "",
}
st.session_state["user_info"] = user_info

# ---- 2) 사진 입력: 업로드(기본) 또는 촬영 (모바일 대응) ----
st.subheader("2) 균열 사진 입력")
mode = st.radio("입력 방식", ["🖼️ 사진 선택", "📷 촬영"],
                horizontal=True, label_visibility="collapsed")
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
cache_key = f"{h}_{skip_report}"
if st.session_state.get("cache_key") != cache_key:
    with st.status("분석 준비 중…", expanded=True) as _status:
        def _prog(label):
            _status.update(label=label + " …")
            st.write("✔ " + label)
        state = orchestrator.analyze(img, h, progress=_prog, skip_report=skip_report,
                                     user_info=user_info)
        # 트리아지 게이트로 조기 반환되면 detect 없음 → 어노테이트 생략
        vis_rgb = annotate(img, state.detect, getattr(state, "crack_mask", None)) \
            if state.detect is not None else None
        _done = "재촬영 안내" if state.detect is None else "분석 완료"
        _status.update(label=_done, state="complete", expanded=False)
    st.session_state.update({"hash": h, "cache_key": cache_key, "state": state,
                             "vis": vis_rgb, "img_bgr": img})

state = st.session_state["state"]
vis_rgb = st.session_state["vis"]

# ---- 기본현황 표만 업데이트 (탐지/보고서 재생성 없이) ----
if st.session_state.pop("regenerate_report", False) and state.report is not None:
    user_info = st.session_state.get("user_info", {})
    meta = getattr(state.triage, "meta", None) if state.triage else None
    report.update_basic_info(state.report, user_info, meta)
    st.session_state["state"] = state

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
st.subheader("3) 판정 결과")

# 탐지 0건 백스톱 — 트리아지는 통과했으나 모델이 아무 결함도 못 잡은 경우.
#  원거리·전경이면 근접 재촬영을 권고(확신 있는 '정상' 오해 방지). 실제 무결함일 수도 있어 중립적 안내.
_ndet = len(getattr(state.detect, "detections", None) or []) if state.detect is not None else 0
if _ndet == 0:
    st.warning(
        "🔍 **결함이 탐지되지 않았습니다.** 실제로 결함이 없는 상태이거나, "
        "원거리·전경으로 촬영돼 결함이 너무 작게 찍혔을 수 있어요. "
        "결함이 의심되면 해당 부위에 **가까이 다가가 화면을 채우도록 다시 촬영**해 주세요.")

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
if getattr(state, "crack_mask", None) is not None:
    st.caption("🟨 노란 영역 = 균열 세그멘테이션 마스크(픽셀 정밀) · 색상 박스 = 면적 결함 YOLO 탐지 (위 범례 색상)")
else:
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
            st.markdown(f"> {_no_strike(e.text)}  \n_근거 출처: {src} (유사도 {e.score})_")
    else:
        st.info("RAG 지식베이스가 아직 구축되지 않았습니다. build_index 실행 후 표시됩니다.")

# ---- 4) 점검 보고서 초안 (현업 7섹션 서식) ----
st.subheader("4) 점검 보고서 초안")
if rep is None:
    st.info("⚡ 빠른 테스트 모드: 보고서 생성이 생략되었습니다. 사이드바에서 토글을 끄면 보고서가 생성됩니다.")
else:
    _prov = report.provider_label()
    st.caption(f"※ 보고서 LLM: {_prov}" + ("  (LLM 키 없음 → 템플릿 목업)" if _prov == "목업" else ""))
    report_md = rep.to_markdown()
    for _title, _attr in rep.SECTIONS:
        st.markdown(f"#### {_title}")
        st.markdown(_no_strike(getattr(rep, _attr)))
    # 다운로드 버튼들
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📄 마크다운 (.md)",
            data=report_md,
            file_name="균열_안전점검_결과보고서_초안.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_dl2:
        if pdf_export.is_available():
            try:
                pdf_bytes = pdf_export.generate_pdf(
                    report=rep,
                    risk=risk,
                    img_original_bgr=st.session_state.get("img_bgr", img),
                    img_annotated_rgb=vis_rgb,
                    user_info=st.session_state.get("user_info"),
                )
                st.download_button(
                    "📑 PDF 내려받기",
                    data=pdf_bytes,
                    file_name="균열_안전점검_결과보고서.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"PDF 생성 실패: {e}")
        else:
            st.info("PDF 내보내기: `pip install fpdf2` 필요")

    # 기본현황 정보 수정 버튼
    st.divider()
    with st.expander("✏️ 기본현황 정보 수정", expanded=False):
        st.caption("분석 후 시설물 정보를 수정하면 보고서 기본현황 표만 업데이트됩니다.")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            edit_facility = st.text_input("시설물명 ", value=st.session_state.get("user_info", {}).get("facility_name", ""), key="edit_facility")
            edit_location = st.text_input("위치 ", value=st.session_state.get("user_info", {}).get("location", ""), key="edit_location")
        with col_e2:
            edit_inspector = st.text_input("점검자 ", value=st.session_state.get("user_info", {}).get("inspector", ""), key="edit_inspector")
            edit_part = st.text_input("점검 부위 ", value=st.session_state.get("user_info", {}).get("part_detail", ""), key="edit_part")
        if st.button("📋 기본현황 표에 반영", type="primary", use_container_width=True):
            st.session_state["user_info"] = {
                "facility_name": edit_facility.strip() if edit_facility else "",
                "location": edit_location.strip() if edit_location else "",
                "inspector": edit_inspector.strip() if edit_inspector else "",
                "part_detail": edit_part.strip() if edit_part else "",
                "remarks": st.session_state.get("user_info", {}).get("remarks", ""),
            }
            st.session_state["regenerate_report"] = True
            st.success("기본현황 표가 업데이트됩니다.")
            st.rerun()

# ---- 5) 대화형 정보 보충 (Claude 챗봇) ----
from pipeline.chat_agent import ReportChatAgent, is_available as chat_agent_available

if chat_agent_available() and rep is not None:
    st.divider()
    st.subheader("5) 대화형 정보 보충")
    st.caption("AI 어시스턴트와 대화하며 보고서에 필요한 추가 정보를 수집합니다.")

    # 에이전트 초기화 (보고서 텍스트와 현재 user_info 전달)
    def get_chat_agent():
        report_md = rep.to_markdown() if rep else ""
        current_info = st.session_state.get("user_info", {})
        agent_key = f"chat_agent_{hash(report_md[:100])}"

        if agent_key not in st.session_state:
            st.session_state[agent_key] = ReportChatAgent(report_md, current_info)
            st.session_state["chat_messages"] = []
            st.session_state["chat_started"] = False
            st.session_state["current_agent_key"] = agent_key
        return st.session_state[agent_key]

    def reset_chat():
        report_md = rep.to_markdown() if rep else ""
        current_info = st.session_state.get("user_info", {})
        agent_key = st.session_state.get("current_agent_key", "chat_agent")
        st.session_state[agent_key] = ReportChatAgent(report_md, current_info)
        st.session_state["chat_messages"] = []
        st.session_state["chat_started"] = False

    agent = get_chat_agent()

    # 첫 대화 시작 (보고서 분석 후 첫 질문)
    if not st.session_state.get("chat_started"):
        with st.spinner("보고서 분석 중..."):
            greeting = agent.start()
        st.session_state["chat_messages"] = [{"role": "assistant", "content": greeting}]
        st.session_state["chat_started"] = True

    # 보고서 재생성 준비 완료
    if agent.is_ready_to_regenerate():
        st.success("추가 정보 수집 완료! 보고서를 재생성할 준비가 되었습니다.")

        collected = agent.get_collected_info()
        if collected:
            # 한글 라벨 매핑
            label_map = {
                "facility_name": "시설물명",
                "location": "위치",
                "inspector": "점검자",
                "part_detail": "점검 부위",
                "discovery_time": "발견 시점",
                "repair_history": "보수 이력",
                "concerns": "우려사항",
            }
            with st.expander("수집된 추가 정보", expanded=True):
                for k, v in collected.items():
                    if v:
                        label = label_map.get(k, k)
                        st.write(f"**{label}**: {v}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("보고서 재생성", type="primary", use_container_width=True):
                # 수집된 정보를 user_info에 병합
                merged = st.session_state.get("user_info", {}).copy()

                # 기본현황 필드 매핑
                field_mapping = {
                    "facility_name": "facility_name",
                    "location": "location",
                    "inspector": "inspector",
                    "part_detail": "part_detail",
                }
                for collected_key, user_info_key in field_mapping.items():
                    if collected.get(collected_key) and not merged.get(user_info_key):
                        merged[user_info_key] = collected[collected_key]

                # 나머지 정보(발견시점, 보수이력, 우려사항)는 remarks에 추가
                extra_info = []
                extra_keys = ["discovery_time", "repair_history", "concerns"]
                for k in extra_keys:
                    if collected.get(k):
                        label = {"discovery_time": "발견시점", "repair_history": "보수이력", "concerns": "우려사항"}.get(k, k)
                        extra_info.append(f"{label}: {collected[k]}")
                if extra_info:
                    existing_remarks = merged.get("remarks", "").strip()
                    merged["remarks"] = (existing_remarks + "\n" + "\n".join(extra_info)).strip()

                st.session_state["user_info"] = merged
                # 전체 보고서 재생성 플래그
                st.session_state["cache_key"] = None
                st.success("보고서를 재생성합니다...")
                st.rerun()
        with col2:
            if st.button("새 대화 시작", use_container_width=True):
                reset_chat()
                st.rerun()
    else:
        # 대화 기록 표시
        for msg in st.session_state.get("chat_messages", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # 사용자 입력
        if chat_input := st.chat_input("메시지를 입력하세요...", key="claude_chat"):
            st.session_state["chat_messages"].append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.write(chat_input)

            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    response = agent.respond(chat_input)
                st.write(response)
            st.session_state["chat_messages"].append({"role": "assistant", "content": response})

            # 수집된 정보를 user_info에 즉시 반영 (기본현황 수정 폼과 연동)
            collected = agent.get_collected_info()
            if collected:
                current_user_info = st.session_state.get("user_info", {}).copy()
                sync_fields = ["facility_name", "location", "inspector", "part_detail"]
                for field in sync_fields:
                    if collected.get(field) and not current_user_info.get(field):
                        current_user_info[field] = collected[field]
                st.session_state["user_info"] = current_user_info

            st.rerun()

        # 새 대화 버튼
        if st.button("🔄 대화 초기화", use_container_width=True):
            reset_chat()
            st.rerun()
elif rep is not None:
    st.divider()
    st.subheader("5) 대화형 정보 보충")
    st.info("Claude API 키가 설정되지 않아 대화 기능을 사용할 수 없습니다. config의 ANTHROPIC_API_KEY를 확인해주세요.")
