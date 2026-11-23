"""2D graphic elements (brand style of the clinic's posts)."""
import math

import numpy as np
import skia

from . import typo as T
from .config import C, CLINIC
from .easing import clamp, lin, settle, out_cubic, in_out_cubic, smooth, out_quint, in_cubic


def capsule(canvas, text, cx, cy, size, t_in, t, t_out=None, align="center", W=1080, color=None, alpha=1.0):
    """Rounded capsule, brand deep blue, white uppercase SemiBold; slides in from the right."""
    fs = size
    tw = T.text_width(text, "semibold", fs, 0.08)
    h = fs * 2.05
    w = tw + fs * 2.2
    x_rest = cx - w / 2 if align == "center" else cx
    k = settle(lin(t, t_in, t_in + 1.0), overshoot=0.035)
    x = x_rest + (W - x_rest + 40) * (1 - k)
    a = alpha * smooth(lin(t, t_in, t_in + 0.25))
    if t_out is not None:
        ko = in_cubic(lin(t, t_out, t_out + 0.55))
        x = x - (x_rest + w + 60) * ko
        a *= 1 - smooth(lin(t, t_out + 0.25, t_out + 0.55))
    if a <= 0.003:
        return None
    y = cy - h / 2
    T.rrect(canvas, x + 2, y + fs * 0.25, w, h, h / 2, "#3E5D70", 0.18 * a, blur=fs * 0.45)
    T.rrect(canvas, x, y, w, h, h / 2, color or C["brand_deep"], a)
    T.draw_text(canvas, text, x + w / 2, cy + T.cap_height(fs) / 2, "semibold", fs, "#FFFFFF", a, 0.08, "center")
    return (x, y, w, h)


def step_label(canvas, n, text, cx, cy, size, t_in, t_out, t, align="center", max_w=900):
    a = smooth(lin(t, t_in, t_in + 0.35)) * (1 - smooth(lin(t, t_out - 0.25, t_out + 0.05)))
    if a <= 0.003:
        return
    dy = (1 - out_cubic(lin(t, t_in, t_in + 0.5))) * size * 0.8
    fs = size
    tw = T.text_width(text, "semibold", fs)
    h = fs * 2.3
    r = h * 0.36
    w = tw + h + fs * 1.3
    x = cx - w / 2 if align == "center" else cx
    y = cy - h / 2 + dy
    T.rrect(canvas, x, y + fs * 0.25, w, h, h / 2, "#3E5D70", 0.12 * a, blur=fs * 0.6)
    T.rrect(canvas, x, y, w, h, h / 2, "#FFFFFF", 0.94 * a)
    ccx, ccy = x + h / 2, y + h / 2
    k = out_quint(lin(t, t_in + 0.05, t_in + 0.45))
    T.circle(canvas, ccx, ccy, r * (0.6 + 0.4 * k), C["brand_light"], a)
    T.draw_text(canvas, str(n), ccx, ccy + T.cap_height(fs * 0.95) / 2, "bold", fs * 0.95, "#FFFFFF", a, 0, "center")
    T.draw_text(canvas, text, x + h + fs * 0.25, ccy + T.cap_height(fs) / 2, "semibold", fs, C["brand_dark"], a)


def result_tag(canvas, text, cx, cy, size, t_in, t_out, t, align="center"):
    a = smooth(lin(t, t_in, t_in + 0.35)) * (1 - smooth(lin(t, t_out - 0.25, t_out + 0.05)))
    if a <= 0.003:
        return
    fs = size
    tw = T.text_width(text, "semibold", fs)
    h = fs * 2.3
    w = tw + h + fs * 1.3
    x = cx - w / 2 if align == "center" else cx
    y = cy - h / 2 + (1 - out_cubic(lin(t, t_in, t_in + 0.5))) * size * 0.8
    T.rrect(canvas, x, y + fs * 0.25, w, h, h / 2, "#3E5D70", 0.12 * a, blur=fs * 0.6)
    T.rrect(canvas, x, y, w, h, h / 2, C["brand_deep"], 0.96 * a)
    ccx, ccy = x + h / 2, y + h / 2
    r = h * 0.34
    T.circle(canvas, ccx, ccy, r, "#FFFFFF", a)
    # check mark drawn progressively
    k = out_cubic(lin(t, t_in + 0.15, t_in + 0.55))
    p = skia.Path()
    pts = [(ccx - r * 0.45, ccy + r * 0.02), (ccx - r * 0.1, ccy + r * 0.36), (ccx + r * 0.5, ccy - r * 0.34)]
    L1 = math.dist(pts[0], pts[1])
    L2 = math.dist(pts[1], pts[2])
    d = k * (L1 + L2)
    p.moveTo(*pts[0])
    if d <= L1:
        f = d / L1
        p.lineTo(pts[0][0] + (pts[1][0] - pts[0][0]) * f, pts[0][1] + (pts[1][1] - pts[0][1]) * f)
    else:
        p.lineTo(*pts[1])
        f = (d - L1) / L2
        p.lineTo(pts[1][0] + (pts[2][0] - pts[1][0]) * f, pts[1][1] + (pts[2][1] - pts[1][1]) * f)
    paint = skia.Paint(AntiAlias=True, Color=T.rgba(C["brand_deep"], a), Style=skia.Paint.kStroke_Style,
                       StrokeWidth=r * 0.24, StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)
    canvas.drawPath(p, paint)
    T.draw_text(canvas, text, x + h + fs * 0.25, ccy + T.cap_height(fs) / 2, "semibold", fs, "#FFFFFF", a)


def bands(canvas, t, t0, W, H, dur=0.95):
    """Diagonal brand-blue bands sweeping right -> left (chapter transitions)."""
    x = lin(t, t0, t0 + dur)
    if x <= 0 or x >= 1:
        return
    D = math.hypot(W, H)
    ang = math.radians(-24)
    specs = [(0.00, 0.20, 0.92), (0.08, 0.11, 0.65), (0.15, 0.05, 0.45)]   # delay, width (of D), alpha
    for delay, wf, al in specs:
        k = in_out_cubic(clamp((x - delay) / (1 - 0.15)))
        cx = W * 1.35 - (W * 1.7 + D * wf) * k
        bw = D * wf * (0.55 + 0.45 * math.sin(math.pi * k))
        canvas.save()
        canvas.translate(cx, H / 2)
        canvas.rotate(math.degrees(ang) + 90)
        p = skia.Paint(AntiAlias=True, Color=T.rgba(C["brand_light"], al))
        canvas.drawRect(skia.Rect.MakeXYWH(-D, -bw / 2, 2 * D, bw), p)
        canvas.restore()


def wordmark(canvas, cx, y_top, size, alpha=1.0, color=None, align="center", reveal=1.0):
    """«Clínica dental» bold geometric over «OdontoMaipu» light, widely tracked."""
    color = color or C["brand_deep"]
    a1 = alpha * smooth(clamp(reveal * 2))
    a2 = alpha * smooth(clamp(reveal * 2 - 0.6))
    s1 = size
    s2 = size * 0.78
    T.draw_text(canvas, CLINIC["wordmark_top"], cx, y_top + s1 * 0.75, "bold", s1, color, a1, 0.02, align)
    tr = 0.42 + 0.25 * (1 - smooth(clamp(reveal * 1.4 - 0.4)))
    T.draw_text(canvas, CLINIC["wordmark_bottom"], cx, y_top + s1 * 0.75 + s2 * 1.55, "light", s2, color, a2, tr, align)
    return s1 * 0.75 + s2 * 1.55


def subtitle(canvas, lines, cx, cy, size, alpha, max_w):
    if alpha <= 0.003 or not lines:
        return
    fs = size
    lh = fs * 1.32
    h = lh * len(lines) + fs * 0.7
    w = max(T.text_width(l, "semibold", fs) for l in lines) + fs * 1.4
    x = cx - w / 2
    y = cy - h / 2
    T.rrect(canvas, x, y, w, h, fs * 0.6, C["brand_dark"], 0.86 * alpha)
    for i, l in enumerate(lines):
        T.draw_text(canvas, l, cx, y + fs * 0.35 + lh * i + lh * 0.5 + T.cap_height(fs) / 2 - 1, "semibold", fs,
                    "#FFFFFF", alpha, 0.0, "center")


def chat_icon(canvas, x, y, s, color, alpha=1.0):
    """Generic chat bubble (not any company's logo)."""
    p = skia.Paint(AntiAlias=True, Color=T.rgba(color, alpha))
    rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, s, s * 0.78), s * 0.3, s * 0.3)
    canvas.drawRRect(rr, p)
    tail = skia.Path()
    tail.moveTo(x + s * 0.2, y + s * 0.66)
    tail.lineTo(x + s * 0.12, y + s * 0.98)
    tail.lineTo(x + s * 0.46, y + s * 0.74)
    tail.close()
    canvas.drawPath(tail, p)
    for i in range(3):
        T.circle(canvas, x + s * (0.3 + 0.2 * i), y + s * 0.39, s * 0.065, "#FFFFFF", alpha)


def photo_panel(canvas, x, y, w, h, alpha, photo=None):
    """Doctor photo cropped into a rounded panel; a neutral placeholder if none is supplied."""
    if alpha <= 0.003:
        return
    rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), w * 0.08, w * 0.08)
    canvas.save()
    canvas.clipRRect(rr, True)
    if photo is not None:
        img = photo
        iw, ih = img.width(), img.height()
        s = max(w / iw, h / ih)
        dw, dh = iw * s, ih * s
        paint = skia.Paint(AntiAlias=True)
        paint.setAlphaf(alpha)
        canvas.drawImageRect(img, skia.Rect.MakeXYWH(x + (w - dw) / 2, y + (h - dh) * 0.2, dw, dh),
                             skia.SamplingOptions(skia.CubicResampler.Mitchell()), paint)
    else:
        sh = skia.GradientShader.MakeLinear([skia.Point(x, y), skia.Point(x, y + h)],
                                            [T.rgba(C["brand_light"], alpha), T.rgba(C["brand_deep"], alpha)])
        canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, h), skia.Paint(Shader=sh, AntiAlias=True))
        # generic silhouette
        T.circle(canvas, x + w / 2, y + h * 0.36, w * 0.17, "#FFFFFF", 0.30 * alpha)
        T.rrect(canvas, x + w * 0.2, y + h * 0.56, w * 0.6, h * 0.6, w * 0.3, "#FFFFFF", 0.30 * alpha)
        T.draw_text(canvas, "[foto autorizada del doctor/a]", x + w / 2, y + h - w * 0.07, "medium", w * 0.052,
                    "#FFFFFF", 0.9 * alpha, 0, "center")
    canvas.restore()


def glass_card(canvas, x, y, w, h, alpha, image=None, marker=None, mk=1.0):
    """Floating glass card with a digital radiograph and a planning marker."""
    if alpha <= 0.003:
        return None
    r = w * 0.06
    rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r)
    T.rrect(canvas, x, y + w * 0.03, w, h, r, "#3E5D70", 0.18 * alpha, blur=w * 0.05)
    canvas.save()
    canvas.clipRRect(rr, True)
    T.rrect(canvas, x, y, w, h, r, "#13232E", 0.86 * alpha)
    if image is not None:
        paint = skia.Paint(AntiAlias=True)
        paint.setAlphaf(alpha * 0.95)
        canvas.drawImageRect(image, skia.Rect.MakeXYWH(x, y, w, h), skia.SamplingOptions(skia.CubicResampler.Mitchell()), paint)
    # glass sheen
    sh = skia.GradientShader.MakeLinear([skia.Point(x, y), skia.Point(x + w, y + h)],
                                        [skia.Color(255, 255, 255, int(60 * alpha)), skia.Color(255, 255, 255, 0)])
    canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, h), skia.Paint(Shader=sh))
    canvas.restore()
    T.rrect(canvas, x, y, w, h, r, "#FFFFFF", 0.55 * alpha, stroke=max(1.5, w * 0.006))
    if marker is not None and mk > 0:
        mx, my = x + marker[0] * w, y + marker[1] * h
        s = w * 0.07
        a = alpha * mk
        p = skia.Paint(AntiAlias=True, Color=T.rgba(C["line_core"], a), Style=skia.Paint.kStroke_Style, StrokeWidth=w * 0.008)
        canvas.drawCircle(mx, my, s * (1.6 - 0.6 * mk), p)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            canvas.drawLine(mx + dx * s * 0.5, my + dy * s * 0.5, mx + dx * s * 1.35, my + dy * s * 1.35, p)
        return (mx, my)
    return None
