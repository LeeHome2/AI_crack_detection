"""
LangGraph 그래프 구성
"""
from langgraph.graph import StateGraph, END

from .state import InspectionState, initial_state
from .nodes import (
    greet_node,
    ask_node,
    parse_node,
    validate_node,
    summarize_node,
    route_after_validate,
)


def build_graph() -> StateGraph:
    """점검 정보 수집 그래프 구성.

    그래프 구조:
        START → greet → ask ←──┐
                  ↓            │
                parse          │
                  ↓            │
               validate ───────┘
                  ↓
               summarize → END
    """
    # 그래프 생성
    graph = StateGraph(InspectionState)

    # 노드 추가
    graph.add_node("greet", greet_node)
    graph.add_node("ask", ask_node)
    graph.add_node("parse", parse_node)
    graph.add_node("validate", validate_node)
    graph.add_node("summarize", summarize_node)

    # 엣지 연결
    graph.set_entry_point("greet")
    graph.add_edge("greet", "ask")

    # ask → parse (사용자 입력 후)
    # 실제로는 외부에서 입력을 받아 parse로 진입
    # 여기서는 ask 후 대기 상태로 전환 (interrupt)

    # parse → validate
    graph.add_edge("parse", "validate")

    # validate → 조건부 (완료면 summarize, 아니면 ask)
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "summarize": "summarize",
            "ask": "ask",
        }
    )

    # summarize → END
    graph.add_edge("summarize", END)

    return graph


def compile_graph():
    """그래프 컴파일. 실행 가능한 Runnable 반환."""
    graph = build_graph()
    return graph.compile()


# ---- 대화형 실행 헬퍼 ----

class ConversationRunner:
    """대화형 그래프 실행기.

    LangGraph의 interrupt 기능 대신
    수동으로 상태를 관리하며 대화를 진행.
    """

    def __init__(self):
        self.state = initial_state()
        self.started = False

    def start(self) -> str:
        """대화 시작. 첫 메시지 반환."""
        if self.started:
            return self.get_last_message()

        # greet 노드 실행
        updates = greet_node(self.state)
        self._apply_updates(updates)

        # ask 노드 실행
        updates = ask_node(self.state)
        self._apply_updates(updates)

        self.started = True
        return self.get_last_message()

    def respond(self, user_input: str) -> str:
        """사용자 입력 처리. 에이전트 응답 반환."""
        from langchain_core.messages import HumanMessage

        # 사용자 메시지 추가
        self.state["messages"] = list(self.state.get("messages", [])) + [
            HumanMessage(content=user_input)
        ]
        self.state["last_user_input"] = user_input
        self.state["turn_count"] = self.state.get("turn_count", 0) + 1

        # parse 노드 실행
        updates = parse_node(self.state)
        self._apply_updates(updates)

        # validate 노드 실행
        updates = validate_node(self.state)
        self._apply_updates(updates)

        # 완료 여부에 따라 분기
        if self.state.get("complete"):
            updates = summarize_node(self.state)
            self._apply_updates(updates)
        else:
            updates = ask_node(self.state)
            self._apply_updates(updates)

        return self.get_last_message()

    def _apply_updates(self, updates: dict):
        """상태 업데이트 적용."""
        for key, value in updates.items():
            if key == "messages":
                # 메시지는 추가
                self.state["messages"] = list(self.state.get("messages", [])) + list(value)
            else:
                self.state[key] = value

    def get_last_message(self) -> str:
        """마지막 AI 메시지 반환."""
        from langchain_core.messages import AIMessage
        messages = self.state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return ""

    def is_complete(self) -> bool:
        """수집 완료 여부."""
        return self.state.get("complete", False)

    def get_state(self) -> InspectionState:
        """현재 상태 반환."""
        return self.state

    def get_user_info(self) -> dict:
        """수집된 user_info 딕셔너리."""
        from .state import get_user_info
        return get_user_info(self.state)
