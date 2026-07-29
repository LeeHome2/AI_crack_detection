"""
PDF 보고서 내보내기 (pdf_export.py)
- 상단에 원본/탐지 사진 2컷 → 이어서 7섹션 보고서를 현업 서식(파란 헤더 표)으로 렌더.
- fpdf2 네이티브 table() 사용 → 셀 안에서 자동 줄바꿈(텍스트가 표 밖으로 넘치지 않음).
- 마크다운 잔재(###·---·**bold**·> 인용·이모지)는 렌더 전에 정리.
- 한글 폰트 자동 설정 (맑은 고딕 or NanumGothic). 폰트 없으면 is_available()=False.
"""
import os
import re
import tempfile
from datetime import datetime

import cv2
import numpy as np

# PDF 라이브러리 (없으면 안내)
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    from fpdf.fonts import FontFace
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False
    FPDF = object
    XPos = YPos = FontFace = None

from schemas import Report, RiskResult


# ---- 서식 색상 (현업 보고서 톤: 스틸블루 헤더 + 연청색 교차음영) ----
_C_TITLE = (37, 99, 175)      # 섹션 제목·소제목 파란색
_C_TH_BG = (46, 108, 142)     # 표 헤더 배경(스틸블루)
_C_TH_FG = (255, 255, 255)    # 표 헤더 글자(흰색)
_C_STRIPE = (238, 243, 247)   # 짝수행 연청색 음영
_C_GRID = (200, 200, 200)     # 표 격자 연회색
_C_RULE = (46, 108, 142)      # 섹션 제목 밑줄


# 한글 폰트 경로 후보 (Regular 폰트만 사용)
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",       # Windows 맑은 고딕
    "C:/Windows/Fonts/NanumGothic.ttf",  # Windows 나눔고딕
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux (fonts-nanum)
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
        self.add_font(self._korean_font, style="", fname=font_path)

    def set_korean_font(self, size: int = 10):
        self.set_font(self._korean_font, size=size)


def is_available() -> bool:
    """PDF 내보내기 사용 가능 여부 (fpdf2 설치 + 한글 폰트 존재)."""
    return _HAS_FPDF and _find_font() is not None


def _cv2_to_temp_file(img_bgr, max_side: int = 1400, quality: int = 82) -> str:
    """OpenCV BGR 이미지를 임시 JPEG로 저장하고 경로 반환.
    긴 변을 max_side로 다운스케일 + JPEG 압축 → PDF 용량 급감(원본 4000px면 수십MB).
    """
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


# \uC774\uBAA8\uC9C0\u00B7\uBCC0\uC774\uC120\uD0DD\uC790(U+FE0F)\u00B7ZWJ(U+200D) \uB4F1 \uD3F0\uD2B8\uC5D0 \uC5C6\uB294 \uAE30\uD638 \uC81C\uAC70 (\uD55C\uAE00 U+AC00~D7AF\uB294 \uBCF4\uC874)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E0-\U0001F1FF"
    "\u2190-\u21FF\u2B00-\u2BFF\uFE00-\uFE0F\u200D\u20E3]+",
    flags=re.UNICODE)


def _clean(text: str) -> str:
    """마크다운 잔재·이모지 정리 (한글 보존). 표 셀·본문 공용."""
    if text is None:
        return ""
    t = str(text)
    t = re.sub(r'\*\*([^*]+)\*\*', r'\1', t)   # **bold**
    t = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', t)  # *italic*
    t = t.replace("`", "")
    t = _EMOJI.sub("", t)
    return t.strip()


def _img_dims(img, max_w, max_h):
    """종횡비 유지하며 (max_w, max_h) 상자에 맞는 (w, h) mm 반환."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return max_w, max_h
    ratio = w / h
    dw, dh = max_w, max_w / ratio
    if dh > max_h:
        dh, dw = max_h, max_h * ratio
    return dw, dh


def _section_title(pdf: KoreanPDF, title: str):
    """파란 섹션 제목 + 밑줄. 페이지 하단에 걸리면 새 페이지로."""
    if pdf.get_y() > pdf.h - 45:
        pdf.add_page()
    pdf.set_text_color(*_C_TITLE)
    pdf.set_korean_font(size=14)
    pdf.cell(0, 9, text=_clean(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    y = pdf.get_y() + 0.5
    pdf.set_draw_color(*_C_RULE)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(3.5)
    pdf.set_text_color(0, 0, 0)


def _sub_heading(pdf: KoreanPDF, text: str):
    """### 소제목 → 파란 소제목."""
    pdf.ln(1)
    pdf.set_text_color(*_C_TITLE)
    pdf.set_korean_font(size=11)
    pdf.multi_cell(0, 6, text=_clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(0.5)


def _bullet(pdf: KoreanPDF, text: str, ordered_prefix: str = None):
    """- 불릿 / 1. 번호 항목 → 들여쓰기 + 말머리."""
    pdf.set_korean_font(size=10)
    mark = ordered_prefix if ordered_prefix else "•"
    indent = 6
    pdf.set_x(pdf.l_margin + indent)
    pdf.cell(5, 6, text=mark)
    pdf.multi_cell(pdf.w - pdf.r_margin - pdf.l_margin - indent - 5, 6,
                   text=_clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _paragraph(pdf: KoreanPDF, text: str):
    pdf.set_korean_font(size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, text=_clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _col_widths(n_cols: int, rows: list):
    """열 너비 배분: 2열은 라벨 좁게/값 넓게, 그 외는 텍스트 많은 열에 가중."""
    if n_cols == 2:
        return (30, 70)
    # 열별 평균 글자수로 가중 (최소 하한 둬서 좁은 열도 읽히게)
    maxlens = [1] * n_cols
    for r in rows:
        for i in range(min(n_cols, len(r))):
            maxlens[i] = max(maxlens[i], len(_clean(r[i])))
    weights = [max(8, m) for m in maxlens]
    total = sum(weights)
    return tuple(100 * w / total for w in weights)


def _draw_table(pdf: KoreanPDF, rows: list):
    """fpdf2 네이티브 표 — 셀 자동 줄바꿈(넘침 방지) + 파란 헤더 + 교차 음영."""
    rows = [[_clean(c) for c in r] for r in rows if any(str(c).strip() for c in r)]
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]  # 열 수 정규화

    if pdf.get_y() > pdf.h - 30:
        pdf.add_page()

    pdf.set_korean_font(size=9)
    pdf.set_draw_color(*_C_GRID)
    pdf.set_line_width(0.2)
    headings = FontFace(color=_C_TH_FG, fill_color=_C_TH_BG)

    with pdf.table(
        col_widths=_col_widths(n_cols, rows),
        headings_style=headings,
        cell_fill_color=_C_STRIPE,
        cell_fill_mode="ROWS",
        borders_layout="ALL",
        line_height=6,
        padding=1.6,
        text_align="LEFT",
        v_align="TOP",
        first_row_as_headings=True,
        width=pdf.w - pdf.l_margin - pdf.r_margin,
    ) as table:
        for r in rows:
            row = table.row()
            for cell in r:
                row.cell(cell)
    pdf.ln(3)


def _render_content(pdf: KoreanPDF, content: str):
    """섹션 본문 파싱: 표(|...|)·소제목(###)·불릿(-,•,1.)·문단을 각각 렌더."""
    table_rows = []

    def flush():
        nonlocal table_rows
        if table_rows:
            _draw_table(pdf, table_rows)
            table_rows = []

    for raw in content.strip().split("\n"):
        line = raw.strip()
        if not line:
            continue

        # 표 행 (인용부호 정리 전에 판정 — 표 라인엔 > 안 붙음)
        if line.startswith("|") and line.endswith("|"):
            inner = line.replace("|", "").replace("-", "").replace(":", "").strip()
            if not inner:          # |---|---| 구분선
                continue
            table_rows.append([c.strip() for c in line.split("|")[1:-1]])
            continue
        flush()

        line = re.sub(r'^>\s?', '', line).strip()   # 블록쿼트 > 제거
        if not line or line in ("---", "***", "___"):
            continue               # 빈 줄·수평선 잔재 생략

        if line.startswith("###") or line.startswith("##"):
            _sub_heading(pdf, line.lstrip("#").strip())
        elif re.match(r'^\d+\.\s', line):
            m = re.match(r'^(\d+\.)\s*(.*)', line)
            if m.group(2).strip():                  # 빈 번호 항목(예: 끊긴 "2.") 생략
                _bullet(pdf, m.group(2), ordered_prefix=m.group(1))
        elif re.match(r'^\d+\.$', line):
            continue               # 텍스트 없는 "2." 같은 잘린 항목 생략
        elif line.startswith(("- ", "• ", "* ")):
            if line[2:].strip():
                _bullet(pdf, line[2:])
        else:
            _paragraph(pdf, line)

    flush()


def generate_pdf(
    report: Report,
    risk: RiskResult,
    img_original_bgr,
    img_annotated_rgb,
    user_info: dict = None,
) -> bytes:
    """PDF 바이트 생성 (상단 사진 2컷 + 7섹션)."""
    if not _HAS_FPDF:
        raise RuntimeError("fpdf2 라이브러리가 설치되지 않았습니다. pip install fpdf2")
    font_path = _find_font()
    if not font_path:
        raise RuntimeError("한글 폰트를 찾을 수 없습니다.")

    pdf = KoreanPDF(font_path)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- 표지 헤더 ----
    pdf.set_text_color(*_C_TITLE)
    pdf.set_korean_font(size=18)
    pdf.cell(0, 12, text="시설물 안전점검 결과보고서",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_text_color(90, 90, 90)
    pdf.set_korean_font(size=10)
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 7, text=f"점검일시 {today}   |   위험도 등급 {risk.grade} ({risk.score}점)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ---- 상단 사진 2컷 (원본 / 탐지) ----
    temp_files = []
    try:
        orig_path = _cv2_to_temp_file(img_original_bgr)
        temp_files.append(orig_path)
        annotated_bgr = cv2.cvtColor(img_annotated_rgb, cv2.COLOR_RGB2BGR)
        annot_path = _cv2_to_temp_file(annotated_bgr)
        temp_files.append(annot_path)

        avail = pdf.w - pdf.l_margin - pdf.r_margin
        gap = 8
        half = (avail - gap) / 2
        max_h = 95
        ow, oh = _img_dims(img_original_bgr, half, max_h)
        aw, ah = _img_dims(annotated_bgr, half, max_h)
        row_h = max(oh, ah)

        pdf.set_korean_font(size=9)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(half, 6, text="[ 원본 사진 ]", align="C")
        pdf.cell(gap, 6, text="")
        pdf.cell(half, 6, text="[ 탐지 결과 ]", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

        y0 = pdf.get_y()
        # 각 반쪽 안에서 가로 가운데 정렬
        pdf.image(orig_path, x=pdf.l_margin + (half - ow) / 2, y=y0, w=ow, h=oh)
        pdf.image(annot_path, x=pdf.l_margin + half + gap + (half - aw) / 2, y=y0, w=aw, h=ah)
        pdf.set_y(y0 + row_h + 6)
    finally:
        for fp in temp_files:
            try:
                os.unlink(fp)
            except Exception:
                pass

    # ---- 7섹션 (연속 흐름, 섹션당 페이지 강제 안 함) ----
    sections = [
        ("1. 기본현황", report.basic_info),
        ("2. 점검결과", report.inspection_result),
        ("3. 안전등급", report.safety_grade),
        ("4. 종합의견", report.overall_opinion),
        ("5. 권고사항", report.recommendations),
        ("6. 판단근거", report.evidence_basis),
        ("7. 유의사항", report.caveats),
    ]
    for title, content in sections:
        if not content or not content.strip():
            continue
        _section_title(pdf, title)
        _render_content(pdf, content)
        pdf.ln(2)

    return bytes(pdf.output())
