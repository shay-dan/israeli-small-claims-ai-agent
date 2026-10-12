from __future__ import annotations

import io
import os
import re
from pathlib import Path

from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, KeepTogether, PageBreak, SimpleDocTemplate, Spacer


def find_hebrew_font() -> Path:
    candidates = [
        Path(__file__).parent.parent / "assets" / "fonts" / "Rubik-Regular.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No Hebrew-capable TTF found. Add assets/fonts/Rubik-Regular.ttf")


def _register_font() -> str:
    name = "HebrewUI"
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, str(find_hebrew_font())))
    return name


class RTLTextBlock(Flowable):
    """Wrap logical Hebrew text first, then apply bidi ordering per rendered line."""

    def __init__(
        self,
        text: str,
        font_name: str,
        font_size: float = 8.8,
        leading: float = 11.5,
        color=colors.HexColor("#202124"),
        align: str = "right",
        background=None,
        border=None,
        padding: float = 0,
    ):
        super().__init__()
        self.text = text
        self.font_name = font_name
        self.font_size = font_size
        self.leading = leading
        self.color = color
        self.align = align
        self.background = background
        self.border = border
        self.padding = padding
        self.lines: list[str] = []

    def keep_with_next(self) -> "RTLTextBlock":
        self.keepWithNext = 1
        return self

    def wrap(self, avail_width, avail_height):
        usable = max(20, avail_width - self.padding * 2)
        self.lines = _wrap_logical_text(self.text, usable, self.font_name, self.font_size)
        self.width = avail_width
        self.height = len(self.lines) * self.leading + self.padding * 2
        return self.width, self.height

    def draw(self):
        canvas = self.canv
        if self.background:
            canvas.setFillColor(self.background)
            canvas.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        if self.border:
            canvas.setStrokeColor(self.border)
            canvas.roundRect(0, 0, self.width, self.height, 4, fill=0, stroke=1)
        canvas.setFont(self.font_name, self.font_size)
        canvas.setFillColor(self.color)
        y = self.height - self.padding - self.leading + 3
        for logical_line in self.lines:
            visual = get_display(logical_line, base_dir="R")
            if self.align == "center":
                canvas.drawCentredString(self.width / 2, y, visual)
            else:
                canvas.drawRightString(self.width - self.padding, y, visual)
            y -= self.leading


def _wrap_logical_text(text: str, width: float, font_name: str, font_size: float) -> list[str]:
    words: list[str] = []
    for token in text.split():
        if pdfmetrics.stringWidth(token, font_name, font_size) <= width:
            words.append(token)
            continue
        chunk = ""
        for character in token:
            candidate = chunk + character
            if chunk and pdfmetrics.stringWidth(candidate, font_name, font_size) > width:
                words.append(chunk)
                chunk = character
            else:
                chunk = candidate
        if chunk:
            words.append(chunk)
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def markdown_to_pdf_bytes(markdown: str, title: str = "טיוטת כתב תביעה") -> bytes:
    buffer = io.BytesIO()
    font_name = _register_font()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="Small Claims AI Agent",
    )
    story = []
    declaration_story = None
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "## הצהרה":
            declaration_story = []
        target_story = declaration_story if declaration_story is not None else story
        if line == "<!-- PAGE_BREAK -->":
            target_story.append(PageBreak())
            continue
        if line.startswith("# "):
            target_story.extend([Spacer(1, 3), RTLTextBlock(_clean_inline(line[2:]), font_name, 16, 21, colors.HexColor("#D04A02"), "center"), Spacer(1, 3)])
        elif line.startswith("## "):
            heading = RTLTextBlock(_clean_inline(line[3:]), font_name, 12.5, 17, colors.HexColor("#D04A02")).keep_with_next()
            gap = Spacer(1, 2)
            gap.keepWithNext = 1
            target_story.extend([Spacer(1, 3), heading, gap])
        elif line.startswith("### "):
            target_story.append(RTLTextBlock(_clean_inline(line[4:]), font_name, 11.5, 16, colors.HexColor("#D04A02")).keep_with_next())
        elif line.startswith("> "):
            target_story.append(RTLTextBlock(_clean_inline(line[2:]), font_name, 9, 13.5, colors.HexColor("#555555"), background=colors.HexColor("#F4F4F4"), border=colors.HexColor("#D9D9D9"), padding=6))
        elif re.match(r"^\d+\.\s", line):
            number, text = line.split(". ", 1)
            target_story.append(RTLTextBlock(f"{number}. {_clean_inline(text)}", font_name))
        elif line.startswith("- "):
            target_story.append(RTLTextBlock(f"• {_clean_inline(line[2:])}", font_name))
        else:
            target_story.append(RTLTextBlock(_clean_inline(line), font_name))
    if declaration_story is not None:
        story.append(KeepTogether(declaration_story))
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return buffer.getvalue()


def _clean_inline(value: str) -> str:
    value = re.sub(r"\[AUTH:[^\]]+\]", "", value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", value)
    value = value.replace("**", "").replace("`", "")
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _page_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D04A02"))
    canvas.setLineWidth(0.6)
    canvas.line(14 * mm, 13 * mm, A4[0] - 14 * mm, 13 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawCentredString(A4[0] / 2, 8 * mm, f"{doc.page}")
    canvas.restoreState()


def save_pdf(markdown: str, path: str | os.PathLike[str], title: str = "טיוטת כתב תביעה") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(markdown_to_pdf_bytes(markdown, title=title))
    return target
