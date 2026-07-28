"""
LLM 프롬프트 템플릿
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---- 정보 추출 프롬프트 ----
EXTRACT_SYSTEM = """당신은 시설물 점검 정보를 수집하는 AI 어시스턴트입니다.
사용자의 응답에서 다음 필드에 해당하는 정보를 추출하세요.

수집 대상 필드:
{fields_desc}

규칙:
1. 사용자 응답에서 해당 필드의 정보가 있으면 추출
2. 여러 필드 정보가 한 문장에 있으면 모두 추출
3. "없음", "모름", "skip" 등은 해당 필드 건너뛰기로 처리
4. 확실하지 않으면 추출하지 말 것

JSON 형식으로 응답:
{{"field_name": "추출된 값", ...}}
정보가 없으면 빈 객체: {{}}
"""

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EXTRACT_SYSTEM),
    ("human", "사용자 응답: {user_input}"),
])


# ---- 질문 생성 프롬프트 ----
ASK_SYSTEM = """당신은 친절한 시설물 점검 어시스턴트입니다.
사용자에게 점검 정보를 자연스럽게 물어보세요.

수집할 정보: {field_ko}
힌트: {field_prompt}
예시: {field_example}

규칙:
1. 한국어로 자연스럽게 질문
2. 1-2문장으로 간결하게
3. 이전 대화 맥락 고려
4. 이미 수집된 정보는 언급하지 말 것
"""

ASK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ASK_SYSTEM),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "다음 질문을 생성하세요."),
])


# ---- 인사말 ----
GREETING = """안녕하세요! 시설물 안전점검 보고서 작성을 도와드리겠습니다.

몇 가지 정보를 여쭤볼게요. 모르시거나 해당 없는 항목은 '없음'이라고 답해주시면 됩니다."""


# ---- 요약 ----
SUMMARY_TEMPLATE = """입력하신 정보를 확인해 주세요:

{info_lines}

정보 수집이 완료되었습니다.
이제 사진을 업로드하시면 점검을 시작합니다."""


def format_fields_desc(fields: list[str], field_info: dict) -> str:
    """필드 목록을 설명 문자열로."""
    lines = []
    for f in fields:
        info = field_info.get(f, {})
        ko = info.get("ko", f)
        lines.append(f"- {f}: {ko}")
    return "\n".join(lines)


def format_summary(state: dict, field_info: dict) -> str:
    """수집된 정보 요약."""
    lines = []
    for f in ["facility_name", "location", "floor", "inspector",
              "discovery_time", "detail", "remarks"]:
        val = state.get(f)
        if val:
            ko = field_info.get(f, {}).get("ko", f)
            lines.append(f"- {ko}: {val}")

    if not lines:
        lines.append("- (수집된 정보 없음)")

    return SUMMARY_TEMPLATE.format(info_lines="\n".join(lines))
