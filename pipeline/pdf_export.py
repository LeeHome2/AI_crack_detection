"""
PDF 보고서 내보내기 (pdf_export.py)
- 원본 사진, 탐지 결과 이미지, 6섹션 보고서를 PDF로 생성
- fpdf2 사용 (pip install fpdf2)
- 한글 폰트 자동 설정 (맑은 고딕 or NanumGothic)
"""
import io
import os
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# PDF 라이브러리 (없으면 안내)
try:
    from fpdf import FPDF
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False
    FPDF = object   # 임포트 안전: fpdf 미설치 시에도 모듈·클래스 정의는 되게(실사용은 is_available/generate_pdf가 차단)
                    # app.py가 최상단에서 이 모듈을 import하므로 여기서 죽으면 앱 전체가 안 뜸.

from schemas import Report, RiskResult


# 한글 폰트 경로 후보
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",       # Windows 맑은 고딕
    "C:/Windows/Fonts/NanumGothic.ttf",  # Windows 나눔고딕
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
    "/System/Library/Fonts/AppleGothic.ttf",  # macOS
]


def _find_font():
    """사용 가능한 한글 폰트 경로 반환."""
    for fp in _FONT_CANDIDATES:
        if os.path.exists(fp):
            return fp
    return None


def is_available() -> bool:
    """PDF 내보내기 사용 가능 여부."""
    return _HAS_FPDF and _find_font() is not None


class KoreanPDF(FPDF):
    """한글 지원 PDF 클래스."""

    def __init__(self):
        super().__init__()
        font_path = _find_font()
        if font_path:
            self.add_font("Korean", "", font_path, uni=True)
            self.add_font("Korean", "B", font_path, uni=True)  # Bold도 같은 폰트 사용
            self._korean_font = True
        else:
            self._korean_font = False

    def set_korean_font(self, size=10, bold=False):
        if self._korean_font:
            style = "B" if bold else ""
            self.set_font("Korean", style, size)
        else:
            self.set_font("Helvetica", "", size)


def _cv2_to_temp_file(img_bgr) -> str:
    """OpenCV BGR 이미지를 임시 PNG 파일로 저장하고 경로 반환."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, img_bgr)
    return path


def _strip_emoji(text: str) -> str:
    """이모지 제거 (PDF 폰트 호환성)."""
    import re
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


def _md_to_plain(md_text: str) -> str:
    """마크다운을 단순 텍스트로 변환 (볼드/이탤릭 제거, 표는 유지)."""
    import re
    text = md_text
    # 볼드/이탤릭 제거
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # 블록쿼트 > 제거
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    # 이모지 제거
    text = _strip_emoji(text)
    return text


def generate_pdf(
    report: Report,
    risk: RiskResult,
    img_original_bgr,
    img_annotated_rgb,
    user_info: dict = None
) -> bytes:
    """PDF 바이트 생성.

    Args:
        report: Report 객체
        risk: RiskResult 객체 (등급/점수)
        img_original_bgr: 원본 이미지 (BGR)
        img_annotated_rgb: 탐지 결과 이미지 (RGB)
        user_info: 사용자 입력 정보

    Returns:
        PDF 파일 바이트
    """
    if not _HAS_FPDF:
        raise RuntimeError("fpdf2 라이브러리가 설치되지 않았습니다. pip install fpdf2")

    pdf = KoreanPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 헤더
    pdf.set_korean_font(18, bold=True)
    pdf.cell(0, 12, "시설물 안전점검 결과보고서", ln=True, align="C")
    pdf.ln(3)

    # 기본 정보 (점검일, 등급)
    pdf.set_korean_font(10)
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    grade_text = f"점검일시: {today}  |  위험도 등급: {risk.grade} ({risk.score}점)"
    pdf.cell(0, 8, grade_text, ln=True, align="C")
    pdf.ln(5)

    # 이미지 섹션
    temp_files = []
    try:
        # 원본 이미지
        pdf.set_korean_font(12, bold=True)
        pdf.cell(0, 8, "[ 원본 사진 ]", ln=True)
        orig_path = _cv2_to_temp_file(img_original_bgr)
        temp_files.append(orig_path)
        # 이미지 크기 조정 (최대 폭 180mm)
        pdf.image(orig_path, x=15, w=180)
        pdf.ln(5)

        # 탐지 결과 이미지
        pdf.set_korean_font(12, bold=True)
        pdf.cell(0, 8, "[ 탐지 결과 ]", ln=True)
        # RGB to BGR for cv2.imwrite
        annotated_bgr = cv2.cvtColor(img_annotated_rgb, cv2.COLOR_RGB2BGR)
        annot_path = _cv2_to_temp_file(annotated_bgr)
        temp_files.append(annot_path)
        pdf.image(annot_path, x=15, w=180)
        pdf.ln(8)

        # 보고서 섹션들
        sections = [
            ("1. 기본현황", report.basic_info),
            ("2. 점검결과", report.inspection_result),
            ("3. 안전등급", report.safety_grade),
            ("4. 종합의견", report.overall_opinion),
            ("5. 판단근거", report.evidence_basis),
            ("6. 유의사항", report.caveats),
        ]

        for title, content in sections:
            pdf.add_page()
            pdf.set_korean_font(14, bold=True)
            pdf.cell(0, 10, title, ln=True)
            pdf.ln(2)

            # 마크다운 표 처리
            pdf.set_korean_font(10)
            plain = _md_to_plain(content)

            # 표인지 확인
            lines = plain.strip().split("\n")
            in_table = False
            table_rows = []

            for line in lines:
                line = line.strip()
                if line.startswith("|") and line.endswith("|"):
                    # 구분선(|---|---|) 무시
                    if set(line.replace("|", "").replace("-", "").strip()) == set():
                        continue
                    in_table = True
                    cells = [c.strip() for c in line.split("|")[1:-1]]
                    table_rows.append(cells)
                else:
                    # 표 끝나면 출력
                    if in_table and table_rows:
                        _draw_table(pdf, table_rows)
                        table_rows = []
                        in_table = False
                    # 일반 텍스트
                    if line:
                        pdf.multi_cell(0, 6, line)
                        pdf.ln(1)

            # 남은 표 출력
            if table_rows:
                _draw_table(pdf, table_rows)

    finally:
        # 임시 파일 정리
        for fp in temp_files:
            try:
                os.unlink(fp)
            except Exception:
                pass

    # PDF 바이트 반환
    return pdf.output()


def _draw_table(pdf: KoreanPDF, rows: list):
    """간단한 표 그리기."""
    if not rows:
        return

    # 열 개수 & 폭 계산
    n_cols = len(rows[0])
    page_width = pdf.w - 30  # 여백 제외
    col_width = page_width / n_cols

    pdf.set_korean_font(9)

    for i, row in enumerate(rows):
        # 첫 행은 헤더 (볼드)
        if i == 0:
            pdf.set_korean_font(9, bold=True)
            pdf.set_fill_color(230, 230, 230)
            fill = True
        else:
            pdf.set_korean_font(9)
            fill = False

        for cell in row:
            # 셀 텍스트 (길면 줄임)
            text = str(cell)[:40]
            pdf.cell(col_width, 7, text, border=1, fill=fill)
        pdf.ln()

    pdf.ln(3)
