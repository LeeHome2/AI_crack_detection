#!/usr/bin/env python
"""
LangGraph 에이전트 테스트

사용법:
    python -m v3_langgraph.test_agent           # 대화형
    python -m v3_langgraph.test_agent --auto    # 자동 시나리오
"""
import sys
import os

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_interactive():
    """대화형 테스트."""
    from v3_langgraph import InspectionAgent

    print("=" * 60)
    print("LangGraph 멀티턴 에이전트 테스트")
    print("=" * 60)
    print("'quit' 또는 'q'를 입력하면 종료합니다.\n")

    agent = InspectionAgent()
    greeting = agent.start()
    print(f"[Agent] {greeting}\n")

    while not agent.is_complete():
        try:
            user_input = input("[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n테스트 종료.")
            break

        if user_input.lower() in ("quit", "q", "exit"):
            print("테스트 종료.")
            break

        response = agent.respond(user_input)
        print(f"\n[Agent] {response}\n")

    if agent.is_complete():
        print("-" * 40)
        print("수집 완료! user_info:")
        for k, v in agent.get_user_info().items():
            if v:
                print(f"  {k}: {v}")
        print(f"총 턴 수: {agent.turn_count}")


def test_auto():
    """자동 시나리오 테스트."""
    from v3_langgraph import InspectionAgent

    print("=" * 60)
    print("자동 시나리오 테스트")
    print("=" * 60)

    scenarios = [
        {
            "name": "기본 시나리오 (순차 입력)",
            "inputs": [
                "강남 OO아파트 101동",
                "서울시 강남구 역삼동 123-45",
                "3층",
                "홍길동",
                "오늘 아침",
                "외벽 균열",
                "없음",
            ]
        },
        {
            "name": "최소 입력 (필수만)",
            "inputs": [
                "XX빌딩",
                "부산시 해운대구",
                "없음",
                "없음",
                "없음",
                "없음",
                "없음",
            ]
        },
        {
            "name": "복합 입력 (한 문장에 여러 정보)",
            "inputs": [
                "서울 강남구 OO아파트 관리인 김철수입니다",
                "3층 외벽",
                "오늘",
                "없음",
            ]
        },
    ]

    for scenario in scenarios:
        print(f"\n{'=' * 40}")
        print(f"시나리오: {scenario['name']}")
        print("=" * 40)

        agent = InspectionAgent()
        greeting = agent.start()
        print(f"[Agent] {greeting[:80]}...")

        for user_input in scenario["inputs"]:
            if agent.is_complete():
                break
            print(f"[You] {user_input}")
            response = agent.respond(user_input)
            print(f"[Agent] {response[:80]}...")

        print(f"\n결과: complete={agent.is_complete()}")
        print(f"수집된 정보: {agent.get_user_info()}")
        print(f"턴 수: {agent.turn_count}")


def test_graph_structure():
    """그래프 구조 테스트."""
    from v3_langgraph.graph import build_graph

    print("=" * 60)
    print("그래프 구조 테스트")
    print("=" * 60)

    graph = build_graph()
    print(f"노드: {list(graph.nodes.keys())}")
    print(f"엣지: {graph.edges}")

    # Mermaid 다이어그램 출력 (가능하면)
    try:
        compiled = graph.compile()
        print("\n그래프 컴파일 성공!")
    except Exception as e:
        print(f"\n그래프 컴파일 실패: {e}")


if __name__ == "__main__":
    if "--auto" in sys.argv:
        test_auto()
    elif "--graph" in sys.argv:
        test_graph_structure()
    else:
        test_interactive()
