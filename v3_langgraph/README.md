# v3 LangGraph 기반 멀티턴 에이전트

> 시설물 점검 정보를 대화로 수집하는 LangGraph 기반 에이전트

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        InspectionGraph                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    │
│   │  START  │───▶│  greet  │───▶│   ask    │◀──┐│  END    │    │
│   └─────────┘    └─────────┘    └────┬─────┘   ││         │    │
│                                      │         ││         │    │
│                                      ▼         ││         │    │
│                                 ┌─────────┐    ││         │    │
│                                 │  parse  │    ││         │    │
│                                 └────┬────┘    ││         │    │
│                                      │         ││         │    │
│                                      ▼         ││         │    │
│                                 ┌──────────┐   ││         │    │
│                                 │ validate │───┘│         │    │
│                                 └────┬─────┘    │         │    │
│                                      │          │         │    │
│                                      ▼          │         │    │
│                                 ┌──────────┐    │         │    │
│                                 │ summarize│────┴────────▶│    │
│                                 └──────────┘              │    │
│                                                           │    │
└─────────────────────────────────────────────────────────────────┘
```

## 상태 스키마 (InspectionState)

```python
class InspectionState(TypedDict):
    # 수집된 정보
    facility_name: str | None
    location: str | None
    floor: str | None
    inspector: str | None
    discovery_time: str | None
    detail: str | None
    remarks: str | None

    # 대화 상태
    messages: list[BaseMessage]
    current_field: str | None
    pending_fields: list[str]

    # 메타
    turn_count: int
    complete: bool
```

## 노드

| 노드 | 역할 |
|------|------|
| `greet` | 첫 인사 + 첫 질문 |
| `ask` | 다음 필드 질문 생성 |
| `parse` | 사용자 응답에서 정보 추출 (LLM) |
| `validate` | 추출된 정보 검증 |
| `summarize` | 수집 완료 요약 |

## 조건부 엣지

```python
def route_after_validate(state):
    if state["complete"]:
        return "summarize"
    elif state["current_field"]:  # 재질문 필요
        return "ask"
    else:
        return "ask"  # 다음 필드
```

## 사용법

```python
from v3_langgraph.agent import InspectionAgent

agent = InspectionAgent()

# 대화 시작
response = agent.start()
print(response)

# 사용자 응답 처리
response = agent.respond("OO아파트 101동입니다")
print(response)

# 완료 확인
if agent.is_complete():
    user_info = agent.get_user_info()
```

## 파일 구조

```
v3_langgraph/
├── README.md
├── state.py      # 상태 스키마
├── nodes.py      # 노드 함수
├── graph.py      # 그래프 구성
├── agent.py      # 에이전트 인터페이스
├── prompts.py    # LLM 프롬프트
└── test_agent.py # 테스트
```

## 의존성

```bash
pip install langgraph langchain-anthropic langchain-core
```

## orchestrator 통합 (예정)

```python
# pipeline/orchestrator.py
from v3_langgraph.agent import InspectionAgent

def analyze_with_collect(img_bgr, ...):
    # 1. 정보 수집 (멀티턴)
    agent = InspectionAgent()
    # ... 대화 루프 ...
    user_info = agent.get_user_info()

    # 2. 기존 파이프라인
    return analyze(img_bgr, user_info=user_info, ...)
```
