"""Typography with HarfBuzz shaping (kerning) drawn through Skia.  Montserrat only."""
from functools import lru_cache

import numpy as np
import skia
import uharfbuzz as hb

from .config import FONTS


def rgba(h, a=1.0):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return skia.Color(r, g, b, int(np.clip(a, 0, 1) * 255))


@lru_cache(None)
def _hb_font(weight):
    blob = hb.Blob.from_file_path(str(FONTS[weight]))
    face = hb.Face(blob)
    font = hb.Font(face)
    return font, face.upem


@lru_cache(None)
def _sk_typeface(weight):
    return skia.Typeface.MakeFromFile(str(FONTS[weight]))


@lru_cache(4096)
def shape(text, weight, size, tracking=0.0):
    """-> (glyph ids, x positions, total advance) at `size` px; tracking in em."""
    font, upem = _hb_font(weight)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})
    k = size / upem
    gids, xs = [], []
    x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gids.append(info.codepoint)
        xs.append(x + pos.x_offset * k)
        x += pos.x_advance * k + tracking * size
    if gids:
        x -= tracking * size
    return tuple(gids), tuple(xs), x


def text_width(text, weight, size, tracking=0.0):
    return shape(text, weight, size, tracking)[2]


def draw_text(canvas, text, x, y, weight="semibold", size=40, color="#3E5D70", alpha=1.0, tracking=0.0,
              align="left", blur=0.0):
    """Draw text with its baseline at y.  align: left/center/right."""
    gids, xs, w = shape(text, weight, size, tracking)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    font = skia.Font(_sk_typeface(weight), size)
    font.setSubpixel(True)
    font.setEdging(skia.Font.Edging.kAntiAlias)
    builder = skia.TextBlobBuilder()
    builder.allocRunPosH(font, list(gids), list(xs), 0.0)
    blob = builder.make()
    paint = skia.Paint(AntiAlias=True, Color=rgba(color, alpha))
    if blur > 0.3:
        paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    canvas.drawTextBlob(blob, x, y, paint)
    return w


def cap_height(size, weight="semibold"):
    return size * 0.70


def wrap(text, weight, size, max_w, tracking=0.0):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if text_width(cand, weight, size, tracking) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def rrect(canvas, x, y, w, h, r, color, alpha=1.0, blur=0.0, stroke=0.0):
    p = skia.Paint(AntiAlias=True, Color=rgba(color, alpha))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
    if blur > 0.3:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r), p)


def circle(canvas, x, y, r, color, alpha=1.0, stroke=0.0, blur=0.0):
    p = skia.Paint(AntiAlias=True, Color=rgba(color, alpha))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
    if blur > 0.3:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    canvas.drawCircle(x, y, r, p)
