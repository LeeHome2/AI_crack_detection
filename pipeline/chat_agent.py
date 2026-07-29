"""
Claude 기반 대화형 보고서 보조 에이전트
- 생성된 보고서를 분석하여 빠진/불명확한 정보 파악
- 자연스러운 대화로 추가 정보 수집
- 충분히 수집되면 보고서 재생성 제안
"""
import config

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


def is_available() -> bool:
    """Claude 대화 에이전트 사용 가능 여부."""
    key = (config.ANTHROPIC_API_KEY or "").strip()
    return _HAS_ANTHROPIC and bool(key) and key.isascii()


class ReportChatAgent:
    """보고서 보완을 위한 Claude 대화 에이전트."""

    SYSTEM_PROMPT = """당신은 시설물 안전점검 보고서 작성을 돕는 전문 어시스턴트입니다.

역할:
1. 생성된 보고서를 분석하여 빠지거나 불명확한 정보를 파악합니다.
2. 사용자와 자연스럽게 대화하며 추가 정보를 수집합니다.
3. 충분한 정보가 모이면 보고서 재생성을 제안합니다.

대화 스타일:
- 친절하고 전문적인 톤
- 한 번에 1-2개의 질문만 (부담 주지 않기)
- 사용자가 "모르겠다"라고 하면 넘어가고 다른 정보 질문
- 이미 입력된 정보는 다시 묻지 않기

수집 대상 정보:
- 시설물명, 위치, 점검자, 점검 부위
- 균열/결함 발견 시점 또는 이력
- 최근 보수 이력
- 특이사항이나 우려 사항

응답 형식:
- 짧고 명확하게 (1-3문장)
- 이모지 사용 가능
- 마크다운 불필요"""

    def __init__(self, report_text: str, user_info: dict = None):
        """
        Args:
            report_text: 생성된 보고서 마크다운 텍스트
            user_info: 이미 입력된 사용자 정보
        """
        self.report_text = report_text
        self.user_info = user_info or {}
        self.messages = []
        self.collected_info = {}
        self.ready_to_regenerate = False

        if is_available():
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        else:
            self.client = None

    def _build_context(self) -> str:
        """대화 컨텍스트 구성."""
        ctx = f"[현재 보고서]\n{self.report_text[:3000]}\n\n"

        if self.user_info:
            ctx += "[이미 입력된 정보]\n"
            for k, v in self.user_info.items():
                if v:
                    ctx += f"- {k}: {v}\n"
            ctx += "\n"

        if self.collected_info:
            ctx += "[대화에서 수집한 정보]\n"
            for k, v in self.collected_info.items():
                if v:
                    ctx += f"- {k}: {v}\n"

        return ctx

    def start(self) -> str:
        """첫 인사 및 초기 질문 생성."""
        if not self.client:
            return "Claude API가 설정되지 않았습니다. config의 ANTHROPIC_API_KEY를 확인해주세요."

        # 보고서 분석하여 빠진 정보 파악하고 첫 질문 생성
        context = self._build_context()
        prompt = f"""{context}

위 보고서를 분석하고, 아직 입력되지 않았거나 불명확한 정보 중 가장 중요한 것 1-2개를 자연스럽게 질문해주세요.
이미 (미입력)으로 표시된 항목이나 명확하지 않은 부분을 우선적으로 물어보세요.
첫 인사와 함께 질문을 시작하세요."""

        try:
            response = self.client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=300,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            reply = response.content[0].text
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"오류가 발생했습니다: {e}"

    def respond(self, user_message: str) -> str:
        """사용자 메시지에 응답."""
        if not self.client:
            return "Claude API가 설정되지 않았습니다."

        self.messages.append({"role": "user", "content": user_message})

        # 컨텍스트 + 대화 이력 기반 응답
        context = self._build_context()
        system = self.SYSTEM_PROMPT + f"\n\n[컨텍스트]\n{context}"

        # 대화 이력을 API 형식으로 변환
        api_messages = []
        for msg in self.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        # 마지막에 지시 추가
        instruction = """
사용자의 응답을 분석하여:
1. 새로운 정보가 있으면 기억해두세요
2. 아직 물어볼 게 있으면 자연스럽게 다음 질문을 하세요
3. 사용자가 모르겠다고 하면 "알겠습니다" 하고 다른 것을 물어보세요
4. 충분한 정보가 수집되었다고 판단되면 "수집된 정보로 보고서를 다시 작성할까요?" 라고 물어보세요
5. 사용자가 재작성을 원하면 응답 맨 앞에 [REGENERATE]를 붙여주세요

★★★ 정보 추출 태그 (매우 중요!) ★★★
사용자가 정보를 제공하면 응답 맨 끝에 INFO 태그를 붙이세요.
핵심: 값에는 해당 필드의 정보만! 다른 내용은 절대 포함하지 마세요!

올바른 예시:
- 사용자: "점검자 이호민이고 발견시기는 모름"
  → [INFO:점검자=이호민]  (발견시점은 모르므로 태그 없음)

- 사용자: "한양아파트 101동 지하주차장입니다"
  → [INFO:시설물명=한양아파트 101동]
  → [INFO:점검부위=지하주차장]

잘못된 예시 (하지 마세요!):
- [INFO:점검자=점검자 이호민 발견시기는 모름]  ← 전체 문장 포함 금지!
- [INFO:발견시점=모름]  ← "모름"은 정보가 아님, 태그 생략!

태그 형식:
[INFO:시설물명=한양아파트 101동]
[INFO:위치=서울시 강남구]
[INFO:점검자=홍길동]
[INFO:점검부위=지하주차장 B2층 기둥]
[INFO:발견시점=지난주]
[INFO:보수이력=작년 외벽 보수]
[INFO:우려사항=균열이 커지고 있음]"""

        api_messages[-1]["content"] = api_messages[-1]["content"] + f"\n\n[시스템 지시]\n{instruction}"

        try:
            response = self.client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=500,
                system=system,
                messages=api_messages
            )
            reply = response.content[0].text

            # 재생성 플래그 확인
            if "[REGENERATE]" in reply:
                self.ready_to_regenerate = True
                reply = reply.replace("[REGENERATE]", "").strip()

            # Claude 응답에서 INFO 태그 파싱 및 제거
            reply = self._extract_info_tags(reply)

            # 대화에서 정보 추출 (백업용 패턴 매칭)
            self._extract_info(user_message)

            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"오류가 발생했습니다: {e}"

    def _extract_info_tags(self, reply: str) -> str:
        """Claude 응답에서 [INFO:key=value] 태그를 파싱하고 제거."""
        import re

        # 태그 매핑 (한글 키 -> 영문 필드명)
        tag_mapping = {
            "시설물명": "facility_name",
            "위치": "location",
            "점검자": "inspector",
            "점검부위": "part_detail",
            "발견시점": "discovery_time",
            "보수이력": "repair_history",
            "우려사항": "concerns",
        }

        # [INFO:key=value] 패턴 찾기
        pattern = r'\[INFO:([^=]+)=([^\]]+)\]'
        matches = re.findall(pattern, reply)

        for key, value in matches:
            key = key.strip()
            value = value.strip()
            if key in tag_mapping:
                field = tag_mapping[key]
                self.collected_info[field] = value

        # 태그 제거하여 깨끗한 응답 반환
        clean_reply = re.sub(pattern, '', reply).strip()
        # 연속된 줄바꿈 정리
        clean_reply = re.sub(r'\n{3,}', '\n\n', clean_reply)
        return clean_reply

    def _extract_info(self, text: str):
        """사용자 응답에서 정보 추출 (백업용 패턴 매칭).

        주의: INFO 태그로 이미 추출된 값은 덮어쓰지 않음.
        이 메서드는 Claude가 INFO 태그를 생성하지 못한 경우에만 동작.
        """
        import re

        # 이미 INFO 태그로 추출된 필드는 건드리지 않음
        # (Claude의 INFO 태그 추출이 더 정확함)

        # 점검자 이름 추출 - "이호민입니다", "점검자 홍길동" 패턴
        if "inspector" not in self.collected_info:
            # "OOO입니다" 패턴
            name_match = re.search(r'([가-힣]{2,4})입니다', text)
            if name_match:
                self.collected_info["inspector"] = name_match.group(1)
            else:
                # "점검자 OOO", "점검자는 OOO" 패턴
                name_match = re.search(r'점검자[는은]?\s*([가-힣]{2,4})', text)
                if name_match:
                    self.collected_info["inspector"] = name_match.group(1)

        # 시설물명 - "OO아파트", "OO빌딩 101동" 패턴
        if "facility_name" not in self.collected_info:
            facility_match = re.search(r'([가-힣A-Za-z0-9]+(?:아파트|빌딩|건물|교량|터널|학교|병원|센터)(?:\s*\d+동)?)', text)
            if facility_match:
                self.collected_info["facility_name"] = facility_match.group(1)

        # 위치 - "서울시 강남구", "B2층" 패턴
        if "location" not in self.collected_info:
            # 주소 패턴
            addr_match = re.search(r'([가-힣]+(?:시|도)\s*[가-힣]+(?:구|군|시)(?:\s*[가-힣]+(?:동|읍|면))?)', text)
            if addr_match:
                self.collected_info["location"] = addr_match.group(1)
            else:
                # 층수 패턴
                floor_match = re.search(r'((?:지하\s*)?[B]?\d+층)', text)
                if floor_match:
                    self.collected_info["location"] = floor_match.group(1)

        # 점검 부위 - "기둥", "외벽" 등 구조 부재명
        if "part_detail" not in self.collected_info:
            part_keywords = ["기둥", "보", "슬래브", "외벽", "내벽", "바닥", "천장", "계단", "난간", "옹벽", "교각", "교대"]
            for kw in part_keywords:
                if kw in text:
                    # 키워드 주변 문맥 추출 (앞뒤 10자)
                    idx = text.find(kw)
                    start = max(0, idx - 10)
                    end = min(len(text), idx + len(kw) + 10)
                    context = text[start:end].strip()
                    self.collected_info["part_detail"] = context
                    break

        # 발견 시점 - "지난주", "3일 전" 패턴만 (모르겠다는 응답 제외)
        if "discovery_time" not in self.collected_info:
            if "모르" not in text and "모름" not in text:
                time_match = re.search(r'(어제|오늘|지난주|지난달|최근|\d+(?:일|주|달|개월|년)\s*전)', text)
                if time_match:
                    self.collected_info["discovery_time"] = time_match.group(1)

    def is_ready_to_regenerate(self) -> bool:
        """보고서 재생성 준비 완료 여부."""
        return self.ready_to_regenerate

    def get_collected_info(self) -> dict:
        """수집된 추가 정보 반환."""
        return self.collected_info

    def get_all_info(self) -> dict:
        """기존 정보 + 수집된 정보 병합."""
        merged = self.user_info.copy()
        merged.update(self.collected_info)
        return merged
