"""
LangGraph 노드 함수
각 노드는 InspectionState를 받아 갱신된 상태를 반환
"""
import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage

from .state import (
    InspectionState, FIELD_INFO, ALL_FIELDS,
    next_field, missing_required
)
from .prompts import (
    GREETING, EXTRACT_PROMPT, format_fields_desc, format_summary
)


# ---- LLM 설정 (lazy import) ----
_llm = None


def get_llm():
    """LLM 인스턴스 (lazy load)."""
    global _llm
    if _llm is None:
        try:
            # Anthropic 우선
            import os
            if os.environ.get("ANTHROPIC_API_KEY"):
                from langchain_anthropic import ChatAnthropic
                _llm = ChatAnthropic(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                )
            else:
                # 폴백: 규칙 기반 (LLM 없음)
                _llm = None
        except ImportError:
            _llm = None
    return _llm


# ============================================================
# 노드 함수
# ============================================================

def greet_node(state: InspectionState) -> dict:
    """첫 인사 노드. 인사말 + 첫 질문."""
    first_field = next_field(state)
    field_info = FIELD_INFO.get(first_field, {})

    greeting = GREETING
    question = f"\n\n{field_info.get('prompt', '')} {field_info.get('example', '')}"

    return {
        "messages": [AIMessage(content=greeting + question)],
        "current_field": first_field,
        "turn_count": 0,
    }


def ask_node(state: InspectionState) -> dict:
    """질문 노드. 다음 필드에 대한 질문 생성."""
    current = state.get("current_field")
    if not current:
        # 다음 필드 찾기
        current = next_field(state)
        if not current:
            # 모든 필드 수집 완료
            return {"complete": True}

    field_info = FIELD_INFO.get(current, {})
    prompt = field_info.get("prompt", f"{current}을(를) 알려주세요.")
    example = field_info.get("example", "")

    question = f"{prompt} {example}".strip()

    return {
        "messages": [AIMessage(content=question)],
        "current_field": current,
    }


def receive_input_node(state: InspectionState) -> dict:
    """사용자 입력 수신 노드.

    실제로는 외부에서 input을 받아 상태에 주입.
    이 노드는 turn_count만 증가.
    """
    return {
        "turn_count": state.get("turn_count", 0) + 1,
    }


def parse_node(state: InspectionState) -> dict:
    """파싱 노드. 사용자 입력에서 정보 추출."""
    user_input = state.get("last_user_input", "")
    if not user_input:
        return {}

    pending = state.get("pending_fields", [])
    current = state.get("current_field")

    # 스킵 키워드 체크
    skip_keywords = ["없음", "없어요", "모름", "skip", "패스", "-", "해당없음"]
    if user_input.strip().lower() in skip_keywords:
        # 현재 필드 스킵
        new_pending = [f for f in pending if f != current]
        nxt = next_field({**state, "pending_fields": new_pending})
        return {
            "pending_fields": new_pending,
            "current_field": nxt,
        }

    # LLM 파싱 시도
    llm = get_llm()
    extracted = {}

    if llm:
        try:
            fields_desc = format_fields_desc(pending, FIELD_INFO)
            chain = EXTRACT_PROMPT | llm
            result = chain.invoke({
                "fields_desc": fields_desc,
                "user_input": user_input,
            })
            # JSON 추출
            content = result.content
            match = re.search(r'\{[^}]*\}', content)
            if match:
                extracted = json.loads(match.group())
        except Exception as e:
            print(f"[parse_node] LLM 파싱 실패: {e}")

    # LLM 실패 시 규칙 기반 폴백
    if not extracted and current:
        extracted = {current: user_input.strip()}

    # 상태 업데이트
    updates = {}
    new_pending = list(pending)

    for field, value in extracted.items():
        if field in ALL_FIELDS and value:
            updates[field] = value
            if field in new_pending:
                new_pending.remove(field)

    updates["pending_fields"] = new_pending

    # 다음 필드 결정
    nxt = next_field({**state, **updates})
    updates["current_field"] = nxt

    return updates


def validate_node(state: InspectionState) -> dict:
    """검증 노드. 필수 필드 체크."""
    missing = missing_required(state)

    if not missing and not state.get("pending_fields"):
        # 모든 필드 완료
        return {"complete": True}

    if not missing:
        # 필수는 다 채움, 선택 필드 남음
        # 선택 필드는 최대 2개만 더 물어봄
        pending = state.get("pending_fields", [])
        if len(pending) <= 2:
            nxt = next_field(state)
            return {"current_field": nxt}
        else:
            # 선택 필드 스킵하고 완료
            return {"complete": True, "pending_fields": []}

    return {}


def summarize_node(state: InspectionState) -> dict:
    """요약 노드. 수집된 정보 요약."""
    summary = format_summary(state, FIELD_INFO)
    return {
        "messages": [AIMessage(content=summary)],
        "complete": True,
    }


# ============================================================
# 라우터 함수 (조건부 엣지)
# ============================================================

def route_after_parse(state: InspectionState) -> Literal["validate", "ask"]:
    """파싱 후 라우팅."""
    return "validate"


def route_after_validate(state: InspectionState) -> Literal["summarize", "ask"]:
    """검증 후 라우팅."""
    if state.get("complete"):
        return "summarize"
    return "ask"


def should_continue(state: InspectionState) -> Literal["continue", "end"]:
    """대화 계속 여부."""
    if state.get("complete"):
        return "end"
    if state.get("turn_count", 0) >= 15:
        return "end"
    return "continue"
