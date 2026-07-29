"""
PDF 보고서 내보내기 (pdf_export.py)
- 원본 사진, 탐지 결과 이미지, 7섹션 보고서를 PDF로 생성
- fpdf2 사용 (pip install fpdf2)
- 한글 폰트 자동 설정 (맑은 고딕 or NanumGothic)
"""
import io
import os
import re
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# PDF 라이브러리 (없으면 안내)
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False
    FPDF = object
    XPos = YPos = None

from schemas import Report, RiskResult


# 한글 폰트 경로 후보 (Regular 폰트만 사용)
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


class KoreanPDF(FPDF):
    """한글 지원 PDF 클래스."""

    def __init__(self, font_path: str):
        super().__init__()
        self.font_path = font_path
        self._korean_font = "Korean"
        # 폰트 등록
        self.add_font(self._korean_font, style="", fname=font_path)

    def set_korean_font(self, size: int = 10):
        """한글 폰트 설정."""
        self.set_font(self._korean_font, size=size)


def is_available() -> bool:
    """PDF 내보내기 사용 가능 여부."""
    return _HAS_FPDF and _find_font() is not None


def _cv2_to_temp_file(img_bgr) -> str:
    """OpenCV BGR 이미지를 임시 PNG 파일로 저장하고 경로 반환."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, img_bgr)
    return path


def _strip_emoji(text: str) -> str:
    """이모지 제거 (PDF 폰트 호환성). 한글은 보존."""
    # 이모지만 정확히 제거 (한글 범위 U+AC00-U+D7AF 보존)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended
        "\U00002600-\U000026FF"  # misc symbols
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


def _md_to_plain(md_text: str) -> str:
    """마크다운을 단순 텍스트로 변환 (볼드/이탤릭 제거, 표는 유지)."""
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
    """PDF 바이트 생성."""
    if not _HAS_FPDF:
        raise RuntimeError("fpdf2 라이브러리가 설치되지 않았습니다. pip install fpdf2")

    # 한글 폰트 확인
    font_path = _find_font()
    if not font_path:
        raise RuntimeError("한글 폰트를 찾을 수 없습니다.")

    # PDF 생성 (한글 폰트 지원)
    pdf = KoreanPDF(font_path)
    pdf.set_auto_page_break(auto=True, margin=15)

    # 첫 페이지
    pdf.add_page()

    # 헤더
    pdf.set_korean_font(size=18)
    pdf.cell(0, 12, text="시설물 안전점검 결과보고서", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(3)

    # 기본 정보 (점검일, 등급)
    pdf.set_korean_font(size=10)
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    grade_text = f"점검일시: {today}  |  위험도 등급: {risk.grade} ({risk.score}점)"
    pdf.cell(0, 8, text=grade_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)

    # 이미지 섹션
    temp_files = []
    try:
        # 원본 이미지
        orig_path = _cv2_to_temp_file(img_original_bgr)
        temp_files.append(orig_path)

        # 탐지 결과 이미지
        annotated_bgr = cv2.cvtColor(img_annotated_rgb, cv2.COLOR_RGB2BGR)
        annot_path = _cv2_to_temp_file(annotated_bgr)
        temp_files.append(annot_path)

        # 이미지 라벨
        pdf.set_korean_font(size=10)
        pdf.cell(90, 6, text="[ 원본 사진 ]", align="C")
        pdf.cell(90, 6, text="[ 탐지 결과 ]", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        y_pos = pdf.get_y()
        pdf.image(orig_path, x=10, y=y_pos, w=85)
        pdf.image(annot_path, x=105, y=y_pos, w=85)
        pdf.ln(70)

    finally:
        for fp in temp_files:
            try:
                os.unlink(fp)
            except Exception:
                pass

    # 보고서 섹션들
    sections = [
        ("1. 기본현황", report.basic_info),
        ("2. 점검결과", report.inspection_result),
        ("3. 안전등급", report.safety_grade),
        ("4. 종합의견", report.overall_opinion),
        ("5. 권고사항", report.recommendations),
        ("6. 판단근거", report.evidence_basis),
        ("7. 유의사항", report.caveats),
    ]

    # DEBUG: 섹션 내용 로그
    print("[PDF DEBUG] === Report sections ===")
    for title, content in sections:
        has_content = bool(content and content.strip())
        preview = (content[:100] if content else "(없음)")
        print(f"  {title}: {'있음' if has_content else '없음'} - {preview}")

    for title, content in sections:
        if not content or not content.strip():
            print(f"[PDF DEBUG] Skipping empty section: {title}")
            continue

        pdf.add_page()

        # 섹션 제목
        pdf.set_korean_font(size=14)
        pdf.cell(0, 10, text=_strip_emoji(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        # 내용 처리
        pdf.set_korean_font(size=10)
        plain = _md_to_plain(content)
        lines = plain.strip().split("\n")

        in_table = False
        table_rows = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 표 라인 체크
            if line.startswith("|") and line.endswith("|"):
                inner = line.replace("|", "").replace("-", "").replace(":", "").strip()
                if not inner:
                    continue
                in_table = True
                cells = [c.strip() for c in line.split("|")[1:-1]]
                table_rows.append(cells)
            else:
                # 표 끝나면 그리기
                if in_table and table_rows:
                    _draw_table(pdf, table_rows)
                    table_rows = []
                    in_table = False

                # 일반 텍스트
                pdf.set_korean_font(size=10)
                pdf.multi_cell(0, 6, text=line)
                pdf.ln(1)

        # 남은 표 출력
        if table_rows:
            _draw_table(pdf, table_rows)

    return bytes(pdf.output())


def _draw_table(pdf: KoreanPDF, rows: list):
    """표 그리기 (한글 폰트 사용)."""
    if not rows:
        return

    n_cols = len(rows[0]) if rows else 0
    if n_cols == 0:
        return

    page_width = pdf.w - 30
    col_width = page_width / n_cols

    for i, row in enumerate(rows):
        # 헤더 행
        if i == 0:
            pdf.set_korean_font(size=9)
            pdf.set_fill_color(230, 230, 230)
            fill = True
        else:
            pdf.set_korean_font(size=9)
            fill = False

        # 열 개수 맞추기
        while len(row) < n_cols:
            row.append("")

        for cell_text in row[:n_cols]:
            text = _strip_emoji(str(cell_text))[:50]
            pdf.cell(col_width, 7, text=text, border=1, fill=fill)
        pdf.ln()

    pdf.ln(3)
