"""
멀티턴 에이전트 인터페이스
외부에서 사용하는 진입점
"""
from .graph import ConversationRunner
from .state import get_user_info, InspectionState


class InspectionAgent:
    """시설물 점검 정보 수집 에이전트.

    사용법:
        agent = InspectionAgent()

        # 대화 시작
        greeting = agent.start()
        print(greeting)

        # 대화 루프
        while not agent.is_complete():
            user_input = input("You: ")
            response = agent.respond(user_input)
            print(f"Agent: {response}")

        # 결과 획득
        user_info = agent.get_user_info()
    """

    def __init__(self):
        self._runner = ConversationRunner()

    def start(self) -> str:
        """대화 시작. 첫 인사말 반환."""
        return self._runner.start()

    def respond(self, user_input: str) -> str:
        """사용자 입력 처리. 에이전트 응답 반환."""
        return self._runner.respond(user_input)

    def is_complete(self) -> bool:
        """정보 수집 완료 여부."""
        return self._runner.is_complete()

    def get_user_info(self) -> dict:
        """수집된 정보를 딕셔너리로 반환.

        Returns:
            dict: orchestrator/report에서 사용하는 형식
                {
                    "facility_name": "...",
                    "location": "...",
                    "floor": "...",
                    ...
                }
        """
        return self._runner.get_user_info()

    def get_state(self) -> InspectionState:
        """현재 전체 상태 반환 (디버깅용)."""
        return self._runner.get_state()

    def get_messages(self) -> list:
        """대화 기록 반환."""
        return list(self._runner.get_state().get("messages", []))

    @property
    def turn_count(self) -> int:
        """현재 턴 수."""
        return self._runner.get_state().get("turn_count", 0)


# ---- 편의 함수 ----

def create_agent() -> InspectionAgent:
    """에이전트 인스턴스 생성."""
    return InspectionAgent()


def is_enabled() -> bool:
    """LangGraph 에이전트 활성화 여부.

    환경변수 LANGGRAPH_AGENT_ENABLED로 제어.
    기본: False (검증 후 활성화)
    """
    import os
    val = os.environ.get("LANGGRAPH_AGENT_ENABLED", "0")
    return val not in ("0", "false", "no", "off", "")
