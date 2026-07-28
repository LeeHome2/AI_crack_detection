"""
v3 LangGraph 에이전트 Streamlit 앱
"""
import streamlit as st

st.set_page_config(
    page_title="시설물 점검 정보 수집",
    page_icon="🏗️",
    layout="centered",
)

st.title("시설물 점검 정보 수집")
st.caption("LangGraph 기반 멀티턴 대화 에이전트")


def get_agent():
    """에이전트 인스턴스 가져오기 (캐시)."""
    from v3_langgraph import InspectionAgent
    if "agent_instance" not in st.session_state:
        st.session_state.agent_instance = InspectionAgent()
    return st.session_state.agent_instance


def reset_agent():
    """에이전트 리셋."""
    from v3_langgraph import InspectionAgent
    st.session_state.agent_instance = InspectionAgent()
    st.session_state.chat_messages = []
    st.session_state.agent_started = False


# 세션 상태 초기화
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "agent_started" not in st.session_state:
    st.session_state.agent_started = False

agent = get_agent()

# 대화 시작
if not st.session_state.agent_started:
    greeting = agent.start()
    st.session_state.chat_messages.append({"role": "assistant", "content": greeting})
    st.session_state.agent_started = True

# 대화 기록 표시
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 완료 시 정보 표시
if agent.is_complete():
    st.success("정보 수집 완료!")

    with st.expander("수집된 정보", expanded=True):
        user_info = agent.get_user_info()
        for key, value in user_info.items():
            if value:
                st.write(f"**{key}**: {value}")

    # 리셋 버튼
    if st.button("새로 시작", type="primary"):
        reset_agent()
        st.rerun()

# 사용자 입력
else:
    if user_input := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # 에이전트 응답
        response = agent.respond(user_input)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

        st.rerun()

# 사이드바 - 상태 정보
with st.sidebar:
    st.header("상태")
    st.write(f"**턴 수**: {agent.turn_count}")
    st.write(f"**완료**: {'예' if agent.is_complete() else '아니오'}")

    st.divider()

    st.header("수집 현황")
    user_info = agent.get_user_info()

    field_names = {
        "facility_name": "시설물명",
        "location": "위치",
        "floor": "층수",
        "inspector": "점검자",
        "discovery_time": "발견시점",
        "detail": "점검부위",
        "remarks": "비고",
    }

    for key, value in user_info.items():
        status = "✅" if value else "⬜"
        label = field_names.get(key, key)
        st.write(f"{status} {label}")

    st.divider()

    if st.button("리셋"):
        reset_agent()
        st.rerun()
