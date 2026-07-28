"""
[고도화] 멀티턴 대화 에이전트 (multiturn.py)
- 시설물 정보를 자연스러운 대화로 수집
- 수집 완료 후 user_info 딕셔너리 반환 → orchestrator/report에서 사용
- 독립 모듈: 단독 테스트 후 통합 가능

사용법:
    agent = MultiturnAgent()
    response, state, complete = agent.process("OO아파트입니다", {})
    # complete=True이면 state["user_info"]에 수집된 정보
"""
import json
import re
from dataclasses import dataclass, field
from typing import Optional
import os

import config

# ---- 수집 필드 정의 ----
REQUIRED_FIELDS = ["facility_name", "location"]
OPTIONAL_FIELDS = ["floor", "discovery_time", "inspector", "detail", "remarks"]
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

FIELD_PROMPTS = {
    "facility_name": "점검하실 시설물 이름을 알려주세요. (예: OO아파트 101동, XX빌딩)",
    "location": "시설물의 주소 또는 위치를 알려주세요. (예: 서울시 강남구 역삼동)",
    "floor": "점검 부위의 층수를 알려주세요. (예: 지하1층, 3층, 옥상) 없으면 '없음'",
    "discovery_time": "결함을 발견한 시점이 언제인가요? (예: 오늘, 일주일 전)",
    "inspector": "점검자 성함을 알려주세요.",
    "detail": "점검 부위를 상세히 알려주세요. (예: 외벽, 천장, 기둥)",
    "remarks": "추가로 기록할 사항이 있으신가요? 없으면 '없음'",
}

FIELD_KO = {
    "facility_name": "시설물명",
    "location": "위치",
    "floor": "층수",
    "discovery_time": "발견시점",
    "inspector": "점검자",
    "detail": "점검부위",
    "remarks": "비고",
}


@dataclass
class CollectState:
    """멀티턴 수집 상태."""
    user_info: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # [(role, msg), ...]
    current_field: Optional[str] = None
    complete: bool = False
    turn_count: int = 0


class MultiturnAgent:
    """멀티턴 대화 에이전트.

    process()를 반복 호출하며 정보 수집.
    LLM 사용 가능 시 자연어 파싱, 불가 시 규칙 기반 폴백.
    """

    def __init__(self, use_llm: bool = True, max_turns: int = 15):
        self.use_llm = use_llm and self._llm_available()
        self.max_turns = max_turns

    def _llm_available(self) -> bool:
        """LLM API 사용 가능 여부."""
        return bool(config.ANTHROPIC_API_KEY or config.UPSTAGE_API_KEY)

    def start(self) -> tuple[str, CollectState]:
        """대화 시작. 첫 인사말 + 초기 상태 반환."""
        state = CollectState()
        state.current_field = "facility_name"
        greeting = (
            "안녕하세요! 시설물 안전점검 보고서 작성을 도와드리겠습니다.\n\n"
            f"{FIELD_PROMPTS['facility_name']}"
        )
        state.history.append(("assistant", greeting))
        return greeting, state

    def process(self, user_msg: str, state: CollectState) -> tuple[str, CollectState, bool]:
        """사용자 메시지 처리.

        Returns:
            response: 에이전트 응답
            state: 갱신된 상태
            complete: 수집 완료 여부
        """
        state.turn_count += 1
        state.history.append(("user", user_msg))

        # 최대 턴 초과 시 강제 완료
        if state.turn_count >= self.max_turns:
            state.complete = True
            response = self._summary_response(state)
            state.history.append(("assistant", response))
            return response, state, True

        # 사용자 응답에서 정보 추출
        if self.use_llm:
            extracted = self._extract_with_llm(user_msg, state)
        else:
            extracted = self._extract_rule_based(user_msg, state)

        # 추출된 정보 저장
        for k, v in extracted.items():
            if v and v.lower() not in ("없음", "없어요", "모름", "skip", "-"):
                state.user_info[k] = v

        # 다음 필드 결정
        next_field = self._next_missing_field(state)

        if next_field is None:
            # 모든 필수 필드 수집 완료
            state.complete = True
            response = self._summary_response(state)
        else:
            state.current_field = next_field
            response = self._build_response(state, extracted)

        state.history.append(("assistant", response))
        return response, state, state.complete

    def _next_missing_field(self, state: CollectState) -> Optional[str]:
        """다음 수집해야 할 필드. 필수 우선, 그 다음 선택."""
        for f in REQUIRED_FIELDS:
            if f not in state.user_info:
                return f
        # 필수 완료 후 선택 필드는 한 번씩만 물어봄 (이미 물어본 건 스킵)
        asked = {h[1] for h in state.history if h[0] == "assistant"}
        for f in OPTIONAL_FIELDS:
            if f not in state.user_info:
                prompt = FIELD_PROMPTS[f]
                # 이미 물어봤으면 스킵
                if not any(prompt in a for a in asked):
                    return f
        return None

    def _extract_rule_based(self, user_msg: str, state: CollectState) -> dict:
        """규칙 기반 추출 (LLM 없을 때 폴백)."""
        extracted = {}
        current = state.current_field
        if current:
            # 현재 질문에 대한 답변으로 간주
            extracted[current] = user_msg.strip()
        return extracted

    def _extract_with_llm(self, user_msg: str, state: CollectState) -> dict:
        """LLM으로 자연어에서 정보 추출."""
        # 수집해야 할 필드
        missing = [f for f in ALL_FIELDS if f not in state.user_info]

        prompt = f"""사용자가 시설물 점검 정보를 제공하고 있습니다.
다음 메시지에서 아래 필드에 해당하는 정보를 추출하세요.

수집 대상 필드: {', '.join(missing)}
필드 설명:
{chr(10).join(f'- {f}: {FIELD_KO[f]}' for f in missing)}

사용자 메시지: "{user_msg}"

JSON 형식으로 추출된 정보만 반환하세요. 없는 정보는 포함하지 마세요.
예: {{"facility_name": "OO아파트", "location": "서울시 강남구"}}
"""
        try:
            result = self._call_llm(prompt)
            # JSON 파싱
            match = re.search(r'\{[^}]+\}', result)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"[multiturn] LLM 추출 실패: {e}")

        # 실패 시 규칙 기반 폴백
        return self._extract_rule_based(user_msg, state)

    def _call_llm(self, prompt: str) -> str:
        """LLM 호출 (Claude → Solar → 실패)."""
        # Claude 시도
        if config.ANTHROPIC_API_KEY:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                resp = client.messages.create(
                    model=config.ANTHROPIC_MODEL,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.content[0].text
            except Exception as e:
                print(f"[multiturn] Claude 실패: {e}")

        # Solar 시도
        if config.UPSTAGE_API_KEY:
            try:
                import requests
                resp = requests.post(
                    config.SOLAR_CHAT_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {config.UPSTAGE_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config.SOLAR_CHAT_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 256
                    },
                    timeout=30
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[multiturn] Solar 실패: {e}")

        raise RuntimeError("LLM 호출 실패")

    def _build_response(self, state: CollectState, extracted: dict) -> str:
        """다음 질문 응답 생성."""
        parts = []

        # 방금 추출된 정보 확인
        if extracted:
            confirmed = [f"{FIELD_KO[k]}: {v}" for k, v in extracted.items()
                        if v and k in state.user_info]
            if confirmed:
                parts.append(f"네, {', '.join(confirmed)}(으)로 기록했습니다.")

        # 다음 질문
        next_field = state.current_field
        if next_field:
            parts.append(FIELD_PROMPTS[next_field])

        return "\n".join(parts) if parts else "계속 진행하겠습니다."

    def _summary_response(self, state: CollectState) -> str:
        """수집 완료 요약."""
        info = state.user_info
        lines = ["입력하신 정보를 확인해 주세요:\n"]
        for f in ALL_FIELDS:
            if f in info:
                lines.append(f"- {FIELD_KO[f]}: {info[f]}")
        lines.append("\n이제 사진을 업로드하시면 점검을 시작합니다.")
        return "\n".join(lines)

    def get_user_info(self, state: CollectState) -> dict:
        """orchestrator/report에 전달할 user_info 딕셔너리."""
        return state.user_info.copy()


# ---- 편의 함수 ----
def is_enabled() -> bool:
    """멀티턴 수집 활성화 여부 (기본 OFF)."""
    return config.MULTITURN_ENABLED
    # config에서 이미 파싱됨


def create_agent() -> MultiturnAgent:
    """에이전트 인스턴스 생성."""
    return MultiturnAgent()
