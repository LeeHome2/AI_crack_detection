"""
[오케스트레이터] 파이프라인 흐름·상태 관리 (orchestrator.py)
- UI(app.py)에서 '흐름/상태'를 분리한다. 각 노드는 AgentState를 받아 갱신해 반환 →
  추후 LangGraph(노드=triage/detect/collect/report) 이식이 그대로 되도록 설계.
- 지금 단계(PR-A′): detect → features → rule → rag → report 를 한 번에 수행하고
  결과를 AgentState에 담아 반환. app.py는 이 state를 session_state에 캐시해
  Streamlit 재실행 시 무거운 YOLO 추론을 재계산하지 않는다.
- 다음 단계: triage(비전 게이트·재촬영), collect(멀티턴 정보수집) 노드를 앞뒤로 삽입 예정.
"""
import hashlib

import config
from pipeline import triage, detector, features, rules, rag, report, postprocess, segmenter
from schemas import AgentState, Stage, DetectResult


def image_hash(data: bytes) -> str:
    """업로드 바이트의 해시 (세션 캐시 키). 같은 사진이면 재분석 안 함."""
    return hashlib.md5(data).hexdigest()


def analyze(img_bgr, image_hash_str: str = "", progress=None, skip_report=False,
            user_info=None) -> AgentState:
    """이미지 1장 → 전체 파이프라인 → 채워진 AgentState.
    triage(1차 게이트) → detect·features·rule·rag·report.
    각 단계는 모델/API 없어도 안전하게 동작(트리아지 실패 시 통과 폴백).
    progress(label): 단계 진입 시 호출되는 선택적 콜백(UI 진행 표시용). None이면 무시.
    """
    _p = progress if callable(progress) else (lambda *_a, **_k: None)
    state = AgentState(stage=Stage.AWAIT_IMAGE, image_hash=image_hash_str)

    # [1] 비전 트리아지: 재촬영/반려면 무거운 YOLO를 돌리지 않고 조기 반환
    _p("① 사진 적합성 확인 (비전 트리아지)")
    state.triage = triage.triage(img_bgr)
    if not state.triage.ok:
        state.stage = Stage.NEEDS_RETAKE if state.triage.verdict != "not_crack" \
            else Stage.REJECTED
        return state

    # [하이브리드] crack 모델 + defect6 모델 조합 (기본 ON)
    if detector.is_hybrid_ready():
        _p("② 결함 탐지 (하이브리드: crack + defect6)")
        det = detector.detect_hybrid(img_bgr)
    else:
        _p("② 결함 탐지 (YOLO 타일 추론)")
        det = detector.detect(img_bgr)
    # 후처리: 이음새(타일 줄눈 등) 오탐 박스 제거 → 이후 특징·위험도·근거·보고서에 반영
    det = postprocess.filter_seams(img_bgr, det)

    # 후처리: 겹치는 균열 박스 병합 (타일 중복 제거)
    if config.MERGE_OVERLAP_ENABLED:
        det = postprocess.merge_overlapping_boxes(det, label_filter="crack",
                                                   margin=config.MERGE_OVERLAP_MARGIN)

    # [고도화·기본 OFF] seg 하이브리드: 균열은 seg 마스크에서, 면적결함은 bbox에서.
    #   활성 + seg 준비됐을 때만. 예외 시 조용히 bbox 단독으로 폴백(데모 안전).
    crack_mask = None
    if config.SEG_HYBRID_ENABLED and segmenter.is_ready():
        try:
            _p("② 균열 세그멘테이션 (하이브리드)")
            seg_res, crack_mask = segmenter.segment(img_bgr)
            merged = DetectResult(image_size=det.image_size)
            merged.detections = [d for d in det.detections if d.label != "crack"]
            merged.detections += seg_res.detections   # 균열=seg 마스크 성분으로 대체
            det = merged
        except Exception as e:
            print(f"[orchestrator] seg 하이브리드 실패 → bbox 폴백: {e}")
            crack_mask = None
    state.detect = det
    state.crack_mask = crack_mask

    _p("③ 형태 특징 추출 (OpenCV)")
    state.features = features.extract(img_bgr, state.detect)
    if crack_mask is not None and crack_mask.any():
        # [하이브리드] 균열 특징을 seg 마스크에서 직접 계산(박스 OpenCV보다 정밀).
        cnt, length, width = segmenter.mask_features(crack_mask)
        H, W = img_bgr.shape[:2]
        diag = (W ** 2 + H ** 2) ** 0.5
        state.features.crack_count = cnt
        state.features.max_length_ratio = round(length / diag, 4) if diag else 0.0
        state.features.avg_width_px = round(width, 2)
        _cc = [d.conf for d in state.detect.detections if d.label == "crack"]
        if _cc:
            state.features.max_confidence = round(max(_cc), 3)
    else:
        # 물리적 균열 개수 보정(bbox 경로): 이어진 균열 덩어리 수 (한 줄이 여러 박스여도 1개)
        state.features.crack_count = postprocess.physical_crack_count(img_bgr, state.detect)

    _p("④ 안전기준 근거 검색 (RAG)")
    risk_pre = rules.evaluate(state.features)            # RAG 전 1차 위험도
    state.rag = rag.search(state.features, risk_pre)     # 근거 검색
    state.risk = rules.evaluate(state.features, state.rag)   # RAG 반영 최종 위험도

    if not skip_report:
        _p("⑤ 점검 보고서 생성 (LLM)")
        # 트리아지가 읽어낸 메타(구조부위·재질·양상)를 보고서 기본현황·점검결과에 반영
        state.report = report.generate(state.features, state.risk, state.rag,
                                       meta=state.triage.meta, user_info=user_info)
    else:
        _p("⑤ 보고서 생성 생략 (빠른 테스트)")
        state.report = None

    state.stage = Stage.ANALYZED
    return state
