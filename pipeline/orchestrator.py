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

from pipeline import triage, detector, features, rules, rag, report
from schemas import AgentState, Stage


def image_hash(data: bytes) -> str:
    """업로드 바이트의 해시 (세션 캐시 키). 같은 사진이면 재분석 안 함."""
    return hashlib.md5(data).hexdigest()


def analyze(img_bgr, image_hash_str: str = "") -> AgentState:
    """이미지 1장 → 전체 파이프라인 → 채워진 AgentState.
    triage(1차 게이트) → detect·features·rule·rag·report.
    각 단계는 모델/API 없어도 안전하게 동작(트리아지 실패 시 통과 폴백).
    """
    state = AgentState(stage=Stage.AWAIT_IMAGE, image_hash=image_hash_str)

    # [1] 비전 트리아지: 재촬영/반려면 무거운 YOLO를 돌리지 않고 조기 반환
    state.triage = triage.triage(img_bgr)
    if not state.triage.ok:
        state.stage = Stage.NEEDS_RETAKE if state.triage.verdict != "not_crack" \
            else Stage.REJECTED
        return state

    state.detect = detector.detect(img_bgr)
    state.features = features.extract(img_bgr, state.detect)

    risk_pre = rules.evaluate(state.features)            # RAG 전 1차 위험도
    state.rag = rag.search(state.features, risk_pre)     # 근거 검색
    state.risk = rules.evaluate(state.features, state.rag)   # RAG 반영 최종 위험도
    # 트리아지가 읽어낸 메타(구조부위·재질·양상)를 보고서 기본현황·점검결과에 반영
    state.report = report.generate(state.features, state.risk, state.rag,
                                   meta=state.triage.meta)

    state.stage = Stage.ANALYZED
    return state
