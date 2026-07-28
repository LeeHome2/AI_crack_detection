"""
상태 스키마 정의 (LangGraph TypedDict)
"""
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


# ---- 필드 정의 ----
REQUIRED_FIELDS = ["facility_name", "location"]
OPTIONAL_FIELDS = ["floor", "inspector", "discovery_time", "detail", "remarks"]
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

FIELD_INFO = {
    "facility_name": {
        "ko": "시설물명",
        "prompt": "점검하실 시설물 이름을 알려주세요.",
        "example": "예: OO아파트 101동, XX빌딩",
        "required": True,
    },
    "location": {
        "ko": "위치",
        "prompt": "시설물의 주소 또는 위치를 알려주세요.",
        "example": "예: 서울시 강남구 역삼동",
        "required": True,
    },
    "floor": {
        "ko": "층수",
        "prompt": "점검 부위의 층수를 알려주세요.",
        "example": "예: 지하1층, 3층, 옥상",
        "required": False,
    },
    "inspector": {
        "ko": "점검자",
        "prompt": "점검자 성함을 알려주세요.",
        "example": "",
        "required": False,
    },
    "discovery_time": {
        "ko": "발견시점",
        "prompt": "결함을 발견한 시점이 언제인가요?",
        "example": "예: 오늘, 일주일 전",
        "required": False,
    },
    "detail": {
        "ko": "점검부위",
        "prompt": "점검 부위를 상세히 알려주세요.",
        "example": "예: 외벽, 천장, 기둥",
        "required": False,
    },
    "remarks": {
        "ko": "비고",
        "prompt": "추가로 기록할 사항이 있으신가요?",
        "example": "없으면 '없음'",
        "required": False,
    },
}


def add_messages(left: list, right: list) -> list:
    """메시지 리스트 병합 (reducer)."""
    return left + right


class InspectionState(TypedDict):
    """점검 정보 수집 상태.

    LangGraph의 상태는 노드 간 전달되며,
    각 노드는 필요한 필드를 읽고/갱신한다.
    """
    # ---- 수집된 정보 ----
    facility_name: str | None
    location: str | None
    floor: str | None
    inspector: str | None
    discovery_time: str | None
    detail: str | None
    remarks: str | None

    # ---- 대화 상태 ----
    # messages: 대화 기록 (HumanMessage, AIMessage)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # current_field: 현재 질문 중인 필드 (None이면 완료)
    current_field: str | None
    # pending_fields: 아직 수집 안 된 필드 목록
    pending_fields: list[str]
    # last_user_input: 가장 최근 사용자 입력 (파싱용)
    last_user_input: str | None

    # ---- 메타 ----
    turn_count: int
    complete: bool
    error: str | None


def initial_state() -> InspectionState:
    """초기 상태 생성."""
    return InspectionState(
        # 수집 정보
        facility_name=None,
        location=None,
        floor=None,
        inspector=None,
        discovery_time=None,
        detail=None,
        remarks=None,
        # 대화 상태
        messages=[],
        current_field=None,
        pending_fields=list(ALL_FIELDS),
        last_user_input=None,
        # 메타
        turn_count=0,
        complete=False,
        error=None,
    )


def get_user_info(state: InspectionState) -> dict:
    """상태에서 user_info 딕셔너리 추출 (orchestrator/report용)."""
    return {
        "facility_name": state.get("facility_name") or "",
        "location": state.get("location") or "",
        "floor": state.get("floor") or "",
        "inspector": state.get("inspector") or "",
        "discovery_time": state.get("discovery_time") or "",
        "detail": state.get("detail") or "",
        "remarks": state.get("remarks") or "",
    }


def missing_required(state: InspectionState) -> list[str]:
    """아직 수집 안 된 필수 필드 목록."""
    return [f for f in REQUIRED_FIELDS if not state.get(f)]


def next_field(state: InspectionState) -> str | None:
    """다음 수집할 필드. 필수 우선, 없으면 선택, 다 채워졌으면 None."""
    pending = state.get("pending_fields", [])
    # 필수 먼저
    for f in REQUIRED_FIELDS:
        if f in pending:
            return f
    # 선택 필드
    for f in OPTIONAL_FIELDS:
        if f in pending:
            return f
    return None
