"""Frame compositor: stage + 3D plate + light line + graphics + subtitles.

    python3 -m om.compose --fmt 9x16 [--start 0 --end 1799] [--out file.mp4] [--plates build/plates] [--preview]
"""
import argparse
import math
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import skia

from . import graphics as G
from . import logo as LG
from . import typo as T
from .config import C, CLINIC, CHAPTERS, FORMATS, PLATES, FPS, NFRAMES, ROOT, DOCTOR_PHOTO, PLACEHOLDERS
from .director import line_items
from .easing import lin, smooth, out_cubic, in_out_cubic, settle, clamp, in_out_sine, out_quint
from .layout import layout
from .lightline import draw_lightline, draw_dots
from .timeline import B, CHAPTER_STEPS, CHAPTER_T0, CHAPTER_END, RESULT_TAG, TRANSITIONS, phrases

SAMP = skia.SamplingOptions(skia.CubicResampler.Mitchell())


def _img(arr):
    arr = np.ascontiguousarray(arr)
    return skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)


@lru_cache(None)
def grain_tiles(w, h, n=6):
    rng = np.random.default_rng(5)
    tiles = []
    for i in range(n):
        g = rng.normal(128, 26, (h, w)).clip(0, 255).astype(np.uint8)
        a = np.dstack([g, g, g, np.full_like(g, 255)])
        tiles.append(_img(a))
    return tiles


@lru_cache(None)
def logo_img(h, white=False):
    return _img(LG.raster_cached(int(h), white))


@lru_cache(None)
def doctor_photo():
    if DOCTOR_PHOTO.exists():
        from PIL import Image
        return _img(np.array(Image.open(DOCTOR_PHOTO).convert("RGBA")))
    return None


@lru_cache(None)
def xray_image():
    p = ROOT / "build" / "xray_card.png"
    if not p.exists():
        make_xray_card(p)
    from PIL import Image
    return _img(np.array(Image.open(p).convert("RGBA")))


def make_xray_card(path):
    """Digital radiograph of the implant site, computed from the same 3D world."""
    from . import world as W
    Mc = W.site_frame(W.SITE_IMPLANT)
    nx, nz, ny = 320, 240, 48
    xs = np.linspace(-1.9, 1.9, nx)
    zs = np.linspace(0.6, -2.3, nz)
    ys = np.linspace(-0.8, 0.8, ny)
    X, Z, Y = np.meshgrid(xs, zs, ys, indexing="xy")
    P = Mc[:3, 3] + X[..., None] * Mc[:3, 0] + Y[..., None] * Mc[:3, 1] + Z[..., None] * Mc[:3, 2]
    P = P.reshape(-1, 3)
    teeth = W.teeth_union(P, lambda d: (not d["upper"]) and d["present"])
    bone = W.bone_sdf(P, sockets=False)
    dens = (teeth < 0) * 1.0 + (bone < 0) * 0.33
    img = dens.reshape(nz, nx, ny).sum(-1) / ny
    img = np.clip(img / img.max(), 0, 1) ** 0.8
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, 1.0)
    rgb = np.stack([0.07 + 0.88 * img, 0.12 + 0.85 * img, 0.16 + 0.82 * img], -1)
    rgb += np.random.default_rng(2).normal(0, 0.02, rgb.shape)
    a = np.ones(img.shape)
    out = (np.dstack([rgb, a]).clip(0, 1) * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(out).save(path)


# ------------------------------------------------------------------ subtitles
def subtitle_chunks():
    out = []
    ph = phrases()
    for i, p in enumerate(ph):
        t0 = p["words"][0]["t0"] - 0.08
        t1 = p["words"][-1]["t1"] + 0.35
        if i + 1 < len(ph):
            t1 = min(t1, ph[i + 1]["words"][0]["t0"] - 0.06)
        out.append(dict(text=p["text"], t0=t0, t1=t1))
    return out


# ------------------------------------------------------------------ frame
class Comp:
    def __init__(self, fmt, plates_dir, plate_scale=0.75):
        self.fmt = fmt
        F = FORMATS[fmt]
        self.W, self.H = F["w"], F["h"]
        self.plate = F["plate"]
        self.PW, self.PH = PLATES[self.plate]["w"], PLATES[self.plate]["h"]
        self.L = layout(fmt)
        self.plates = Path(plates_dir) / self.plate
        self.subs = subtitle_chunks()
        self.surface = skia.Surface(self.W, self.H)
        self.oy = (self.PH - self.H) / 2 if fmt == "4x5" else 0.0

    # -------------------------------------------------- stage
    def stage(self, c, t, f):
        W, H = self.W, self.H
        sh = skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)],
                                            [T.rgba(C["bg_top"]), T.rgba(C["bg_bottom"])])
        c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Shader=sh))
        gx, gy, gr = self.L["glow"]
        breath = 0.5 + 0.5 * math.sin(2 * math.pi * t / 4.8 - 1.2)      # 2 bars at 100 BPM
        gx += 30 * math.sin(t * 0.21)
        gy += 40 * math.sin(t * 0.17 + 1)
        r = gr * (0.92 + 0.12 * breath)
        rg = skia.GradientShader.MakeRadial(skia.Point(gx, gy), r,
                                            [T.rgba(C["brand_light"], 0.20 + 0.08 * breath), T.rgba(C["brand_light"], 0.0)])
        c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Shader=rg))

    def grain(self, c, f, amount=0.035):
        tiles = grain_tiles(self.W // 2, self.H // 2)
        p = skia.Paint()
        p.setBlendMode(skia.BlendMode.kOverlay)
        p.setAlphaf(amount)
        c.drawImageRect(tiles[f % len(tiles)], skia.Rect.MakeWH(self.W, self.H), skia.SamplingOptions(), p)

    # -------------------------------------------------- 3D plate
    def plate_alpha(self, t):
        return 1.0 - smooth(lin(t, B["s6.snap"] - 0.1, B["s6.snap"] + 0.9))

    def draw_plate(self, c, t, f):
        a = self.plate_alpha(t)
        if a <= 0.003:
            return
        p = self.plates / f"{f:04d}.png"
        if not p.exists():
            return
        img = skia.Image.open(str(p))
        paint = skia.Paint(AntiAlias=True)
        paint.setAlphaf(a)
        c.drawImageRect(img, skia.Rect.MakeXYWH(0, -self.oy, self.PW, self.PH), SAMP, paint)

    # -------------------------------------------------- logo + wordmark (opening)
    def opening(self, c, t):
        L = self.L
        lo = L["logo_open"]
        if t > B["tooth.in1"] + 0.2:
            return
        head = in_out_sine(lin(t, B["logo.trace0"], B["logo.trace1"]))
        out_k = in_out_cubic(lin(t, B["tooth.in0"] - 0.05, B["tooth.in0"] + 0.65))
        a = 1 - out_k
        if a <= 0.003 or head <= 0:
            return
        h = int(lo["h"])
        if head >= 1.0 and out_k <= 0:
            # at rest: the official raster, unscaled, on whole pixels
            c.drawImage(logo_img(h), round(lo["x"]), round(lo["y"]))
        else:
            arr = LG.revealed(h, head)
            s = 1 + 0.18 * out_k
            img = _img(arr)
            paint = skia.Paint(AntiAlias=True)
            paint.setAlphaf(a)
            w = arr.shape[1]
            cx, cy = lo["x"] + w / 2, lo["y"] + h / 2
            c.drawImageRect(img, skia.Rect.MakeXYWH(cx - w * s / 2, cy - h * s / 2, w * s, h * s), SAMP, paint)
        wm = L["wordmark_open"]
        rev = lin(t, B["logo.trace0"] + 0.6, B["logo.trace1"] + 0.35)
        if rev > 0:
            G.wordmark(c, wm["cx"], wm["y"] - 30 * out_k, wm["size"], alpha=a, align=wm["align"], reveal=rev)

    # -------------------------------------------------- chapter graphics
    def chapter(self, c, t):
        L = self.L
        for ch in CHAPTERS:
            k = ch["key"]
            t0 = CHAPTER_T0[k]
            t_end = CHAPTER_END[k]
            if not (t0 - 0.1 < t < t_end + 0.8):
                continue
            cp = L["capsule"]
            G.capsule(c, ch["title"], cp["cx"], cp["cy"], cp["size"], t0, t, t_out=t_end - 0.5,
                      align=cp["align"], W=self.W)
            steps = CHAPTER_STEPS[k]
            st = L["step"]
            if st.get("list"):
                self.step_list(c, t, ch, steps, st)
            else:
                for i, name in enumerate(ch["steps"]):
                    G.step_label(c, i + 1, name, st["cx"], st["cy"], st["size"], steps[i], steps[i + 1], t,
                                 align=st["align"])
            G.result_tag(c, RESULT_TAG[k], st["cx"], st["cy"] if not st.get("list") else st["cy"] + 3 * st["size"] * 2.6,
                         st["size"], steps[3] + 0.15, t_end - 0.55, t, align=st["align"])
            if k == "impl":
                self.osseo_label(c, t, steps)

    def step_list(self, c, t, ch, steps, st):
        """16:9: the three steps as a list on the right; the current one is highlighted."""
        fs = st["size"]
        t0 = CHAPTER_T0[ch["key"]]
        t_end = CHAPTER_END[ch["key"]]
        a = smooth(lin(t, steps[0] - 0.6, steps[0] - 0.1)) * (1 - smooth(lin(t, t_end - 0.6, t_end - 0.2)))
        if a <= 0.003:
            return
        for i, name in enumerate(ch["steps"]):
            y = st["cy"] + i * fs * 2.6
            active = smooth(lin(t, steps[i], steps[i] + 0.3)) * (1 - smooth(lin(t, steps[i + 1] - 0.1, steps[i + 1] + 0.2)))
            done = smooth(lin(t, steps[i + 1], steps[i + 1] + 0.3))
            h = fs * 2.2
            w = T.text_width(name, "semibold", fs) + h + fs * 1.4
            x = st["cx"]
            T.rrect(c, x, y - h / 2, w, h, h / 2, "#FFFFFF", a * (0.55 + 0.4 * active))
            col = C["brand_light"] if (active > 0.5 or done > 0.5) else "#B9CCD9"
            T.circle(c, x + h / 2, y, h * 0.34, col, a)
            T.draw_text(c, str(i + 1), x + h / 2, y + T.cap_height(fs * 0.9) / 2, "bold", fs * 0.9, "#FFFFFF", a, 0, "center")
            T.draw_text(c, name, x + h + fs * 0.2, y + T.cap_height(fs) / 2, "semibold", fs,
                        C["brand_dark"], a * (0.55 + 0.45 * max(active, done)))

    def osseo_label(self, c, t, steps):
        """Clinic-supplied osseointegration time, shown during step 3."""
        st = self.L["step"]
        t0, t1 = steps[2] + 0.4, steps[3]
        a = smooth(lin(t, t0, t0 + 0.3)) * (1 - smooth(lin(t, t1 - 0.2, t1 + 0.1)))
        if a <= 0.003:
            return
        fs = st["size"] * 0.62
        text = PLACEHOLDERS["osseo_time"]
        if st.get("list"):
            x, y, al = st["cx"], st["cy"] + 3 * st["size"] * 2.6 + fs * 2.0, "left"
        else:
            x, y, al = st["cx"], st["cy"] + st["size"] * 1.9, st["align"]
        w = T.text_width(text, "medium", fs) + fs * 1.6
        h = fs * 2.0
        bx = x - w / 2 if al == "center" else x
        T.rrect(c, bx, y - h / 2, w, h, h / 2, C["brand_light"], 0.9 * a)
        T.draw_text(c, text, bx + w / 2, y + T.cap_height(fs) / 2, "medium", fs, "#FFFFFF", a, 0, "center")

    def xray_card(self, c, t):
        L = self.L["card"]
        t0 = B["c3.s1"]
        a = smooth(lin(t, t0, t0 + 0.45)) * (1 - smooth(lin(t, B["c3.s2"] + 0.1, B["c3.s2"] + 0.5)))
        self.L["_xray_marker"] = None
        if a <= 0.003:
            return
        y = L["y"] + 30 * (1 - out_cubic(lin(t, t0, t0 + 0.6))) + 6 * math.sin(t * 1.7)
        mk = out_cubic(lin(t, t0 + 0.35, t0 + 0.7))
        m = G.glass_card(c, L["x"], y, L["w"], L["h"], a, image=xray_image(), marker=(0.5, 0.42), mk=mk)
        self.L["_xray_marker"] = m

    # -------------------------------------------------- closing
    def closing(self, c, t):
        L = self.L
        if t < B["s6"] + 0.1:
            return
        end = smooth(lin(t, B["s7"], B["s7"] + 0.4))
        a_all = 1 - end
        lo = L["logo_close"]
        h = int(lo["h"])
        head = in_out_sine(lin(t, B["s6.snap"] - 0.3, B["s6.snap"] + 0.05))
        if head > 0 and a_all > 0:
            if head >= 1 and a_all >= 1:
                c.drawImage(logo_img(h), round(lo["x"]), round(lo["y"]))
            else:
                p = skia.Paint(AntiAlias=True)
                p.setAlphaf(a_all)
                c.drawImage(_img(LG.revealed(h, head)), round(lo["x"]), round(lo["y"]), SAMP, p)
            wm = L["wordmark_close"]
            G.wordmark(c, wm["cx"], wm["y"], wm["size"], alpha=a_all, align=wm["align"],
                       reveal=lin(t, B["s6.snap"], B["s6.snap"] + 0.8))
        # treatments
        k0 = B["s6.stack"]
        ti = L["title"]
        ta = smooth(lin(t, k0, k0 + 0.4)) * a_all
        if ta > 0:
            T.draw_text(c, "Nuestros tratamientos", ti["x"], ti["y"], "extrabold", ti["size"], C["brand_dark"], ta, 0,
                        ti["align"])
        cp = L["caps"]
        for i, ch in enumerate(CHAPTERS):
            col = i // 2 if cp.get("cols") == 2 else 0
            row = i % 2 if cp.get("cols") == 2 else i
            x = cp["x"] + col * cp.get("colgap", 0)
            cy = cp["y"] + row * cp["gap"] * (1.0 if cp.get("cols") != 2 else 1.0) + cp["size"]
            self.stack_capsule(c, ch["title"], x, cy, cp["w"], cp["size"], k0 + 0.15 + 0.14 * i, t, a_all)
        ph = L["photo"]
        pk = out_quint(lin(t, k0 + 0.1, k0 + 0.9))
        G.photo_panel(c, ph["x"] - (ph["x"] + ph["w"] + 40) * (1 - pk), ph["y"], ph["w"], ph["h"], a_all * smooth(pk * 3),
                      doctor_photo())
        self.cta(c, t, a_all)
        # S7 end plate
        if end > 0:
            el = L["end_logo"]
            hh = int(el["h"])
            p = skia.Paint(AntiAlias=True)
            p.setAlphaf(end)
            if end >= 1:
                c.drawImage(logo_img(hh), round(el["x"]), round(el["y"]))
            else:
                c.drawImage(logo_img(hh), round(el["x"]), round(el["y"]), SAMP, p)
            G.wordmark(c, self.W / 2, el["y"] + hh + 40, 34, alpha=end, align="center")
            d = L["disclaimer"]
            lines = T.wrap(CLINIC["disclaimer"], "medium", d["size"], d["max_w"])
            for i, ln in enumerate(lines):
                T.draw_text(c, ln, d["cx"], d["y"] + i * d["size"] * 1.45, "medium", d["size"], C["brand_dark"],
                            end * 0.9, 0, "center")

    def stack_capsule(self, c, text, x, cy, w, size, t_in, t, a_all):
        k = settle(lin(t, t_in, t_in + 0.8), overshoot=0.03)
        if k <= 0:
            return
        h = size * 2.3
        xx = x + (self.W - x + 30) * (1 - k)
        a = smooth(lin(t, t_in, t_in + 0.2)) * a_all
        T.rrect(c, xx + 2, cy - h / 2 + size * 0.25, w, h, h / 2, "#3E5D70", 0.16 * a, blur=size * 0.5)
        T.rrect(c, xx, cy - h / 2, w, h, h / 2, C["brand_deep"], a)
        fs = size if T.text_width(text, "semibold", size, 0.06) < w - size * 1.6 else size * (w - size * 1.6) / T.text_width(text, "semibold", size, 0.06)
        T.draw_text(c, text, xx + w / 2, cy + T.cap_height(fs) / 2, "semibold", fs, "#FFFFFF", a, 0.06, "center")

    def cta(self, c, t, a_all):
        L = self.L["cta"]
        t0 = B["s6.cta"]
        k = out_cubic(lin(t, t0, t0 + 0.6))
        a = smooth(lin(t, t0, t0 + 0.35)) * a_all
        if a <= 0.003:
            return
        fs = L["size"]
        y = L["y"] + 40 * (1 - k)
        al = L["align"]
        cx = L["cx"]
        line1 = CLINIC["cta"]
        w1 = T.text_width(line1, "semibold", fs) + fs * 1.5
        x1 = cx - w1 / 2 if al == "center" else cx
        G.chat_icon(c, x1, y - fs * 0.95, fs * 1.15, C["brand_deep"], a)
        T.draw_text(c, line1, x1 + fs * 1.5, y, "semibold", fs, C["brand_dark"], a)
        T.draw_text(c, CLINIC["whatsapp"], cx, y + fs * 1.95, "extrabold", fs * 1.6, C["brand_deep"], a, 0.01, al)
        T.draw_text(c, CLINIC["address"], cx, y + fs * 3.15, "medium", fs * 0.78, C["brand_dark"], a, 0, al)
        T.draw_text(c, CLINIC["instagram"], cx, y + fs * 4.25, "semibold", fs * 0.86, C["brand_deep"], a, 0.02, al)

    # -------------------------------------------------- subtitles
    def subtitles(self, c, t):
        S = self.L["sub"]
        for ch in self.subs:
            if ch["t0"] - 0.12 < t < ch["t1"] + 0.12:
                a = smooth(lin(t, ch["t0"] - 0.12, ch["t0"] + 0.05)) * (1 - smooth(lin(t, ch["t1"] - 0.05, ch["t1"] + 0.12)))
                lines = T.wrap(ch["text"], "semibold", S["size"], S["max_w"])
                cy = S["cy"]
                if self.fmt != "16x9" and t >= B["s6"] + 0.3:
                    cy = self.H - S["size"] * 1.9 * (1 if len(lines) == 1 else 1.4)
                G.subtitle(c, lines, S["cx"], cy, S["size"], a, S["max_w"])

    # -------------------------------------------------- main
    def overlay(self, c, t):
        self.xray_card(c, t)
        self.opening(c, t)
        for it in line_items(t, self.fmt, self.L):
            if it["kind"] == "line":
                draw_lightline(c, it["pts"], head=it["head"], tail=it["tail"], width=it["width"],
                               intensity=it["intensity"], core=it.get("core"), halo=it.get("halo"),
                               spark=it.get("spark", True), fade_tail=it.get("fade_tail", 0.35), vis=it.get("vis"),
                               dash=it.get("dash"), closed=it.get("closed", False))
            else:
                draw_dots(c, it["pts"], size=it.get("size", 2.2), intensity=it["intensity"], core=it.get("core"),
                          halo=it.get("halo"), vis=it.get("vis"))
        self.chapter(c, t)
        for tt in TRANSITIONS:
            G.bands(c, t, tt, self.W, self.H)
        self.closing(c, t)
        self.subtitles(c, t)

    SUB = (-1 / 6, 0.0, 1 / 6)   # 180-degree shutter, 3 sub-frames (real motion blur for the 2D layer)

    @staticmethod
    def subframes(t):
        """More sub-frames while 2D elements move fast (capsules, bands, closing stack) to avoid strobing."""
        fast = []
        for k, t0 in CHAPTER_T0.items():
            fast += [(t0 - 0.05, t0 + 0.9), (CHAPTER_END[k] - 0.55, CHAPTER_END[k] + 0.1)]
        fast += [(tt - 0.05, tt + 1.0) for tt in TRANSITIONS]
        fast += [(B["s6.stack"], B["s6.stack"] + 1.6), (B["s6.cta"], B["s6.cta"] + 0.7), (B["s7"], B["s7"] + 0.45),
                 (B["logo.trace0"], B["logo.trace1"]), (B["s6"] - 0.3, B["s6.snap"] + 0.1)]
        if any(a <= t <= b for a, b in fast):
            return tuple(np.linspace(-0.25, 0.25, 9))
        return (-1 / 6, 0.0, 1 / 6)

    def frame(self, f):
        t = f / FPS
        c = self.surface.getCanvas()
        c.clear(skia.Color(255, 255, 255))
        self.stage(c, t, f)
        self.draw_plate(c, t, f)
        self.grain(c, f)
        if not hasattr(self, "ov"):
            self.ov = skia.Surface(skia.ImageInfo.Make(self.W, self.H, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType))
        acc = None
        subs = self.subframes(t)
        for d in subs:
            oc = self.ov.getCanvas()
            oc.clear(skia.Color(0, 0, 0, 0))
            self.overlay(oc, t + d / FPS)
            a = self.ov.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
            acc = a.astype(np.uint16) if acc is None else acc + a
        n = len(subs)
        avg = ((acc + n // 2) // n).astype(np.uint8)
        img = skia.Image.fromarray(avg, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
        c.drawImage(img, 0, 0)
        arr = self.surface.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)
        return arr[..., :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="9x16")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=NFRAMES - 1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--plates", default=str(ROOT / "build" / "plates"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None, help="write PNG frames to this folder instead of video")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--crf", type=int, default=17)
    a = ap.parse_args()
    comp = Comp(a.fmt, a.plates)
    frames = [int(x) for x in a.frames.split(",")] if a.frames else list(range(a.start, a.end + 1, a.step))
    t0 = time.time()
    if a.png:
        from PIL import Image
        Path(a.png).mkdir(parents=True, exist_ok=True)
        for f in frames:
            Image.fromarray(comp.frame(f)).save(Path(a.png) / f"{f:04d}.png")
        print("png frames", len(frames), round(time.time() - t0, 1), "s")
        return
    out = a.out or str(ROOT / "build" / f"comp_{a.fmt}.mp4")
    fps = FPS / a.step
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{comp.W}x{comp.H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        pr.stdin.write(comp.frame(f).tobytes())
        if i % 150 == 0:
            print(f"  {a.fmt} frame {f} ({time.time() - t0:.0f}s)", flush=True)
    pr.stdin.close()
    pr.wait()
    print("wrote", out, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()


# ====================================================================== capsules
class CapsuleCards(Comp):
    """3 s logo intro and 3 s contact outro for the stand-alone chapter clips."""

    def intro(self, f, chapter):
        t = f / FPS
        c = self.surface.getCanvas()
        c.clear(skia.Color(255, 255, 255))
        self.stage(c, t + 1.0, f)
        self.grain(c, f)
        L = self.L
        lo = dict(L["logo_open"])
        h = int(lo["h"] * 0.82)
        w = h * LG.ASPECT
        x = lo["x"] + (lo["w"] - w) / 2
        y = lo["y"] + (lo["h"] - h) / 2 - (40 if self.fmt != "16x9" else 0)
        head = in_out_sine(lin(t, 0.15, 1.15))
        out_k = smooth(lin(t, 2.55, 3.0))
        if head >= 1 and out_k <= 0:
            c.drawImage(logo_img(h), round(x), round(y))
        else:
            p = skia.Paint(AntiAlias=True)
            p.setAlphaf(1 - out_k)
            c.drawImage(_img(LG.revealed(h, head)), round(x), round(y), SAMP, p)
        if head < 1:
            pts, u, on, _ = LG.trace(h)
            draw_lightline(c, pts + [x, y], head=head, tail=max(0, head - 0.16), width=5, vis=0.35 + 0.65 * on, fade_tail=0.6)
        wm = L["wordmark_open"]
        G.wordmark(c, wm["cx"] if self.fmt != "16x9" else wm["cx"], wm["y"] - (40 if self.fmt != "16x9" else 0),
                   wm["size"], alpha=1 - out_k, align=wm["align"], reveal=lin(t, 0.5, 1.6))
        cp = L["capsule"]
        cy = (wm["y"] + wm["size"] * 3.6) if self.fmt != "16x9" else wm["y"] + wm["size"] * 3.4
        cx = cp["cx"] if self.fmt == "9x16" else (self.W / 2 if self.fmt == "4x5" else wm["cx"])
        al = "center" if self.fmt != "16x9" else "left"
        G.capsule(c, chapter["title"], cx, cy, cp["size"], 1.2, t, t_out=2.5, align=al, W=self.W)
        return self.surface.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)[..., :3]

    def outro(self, f, sub_chunk=None):
        t = f / FPS
        c = self.surface.getCanvas()
        c.clear(skia.Color(255, 255, 255))
        self.stage(c, t + 3.0, f)
        self.grain(c, f)
        L = self.L
        k = out_cubic(lin(t, 0.0, 0.6))
        a = smooth(lin(t, 0.0, 0.35))
        lo = L["logo_close"]
        hh = int(lo["h"])
        p = skia.Paint(AntiAlias=True)
        p.setAlphaf(a)
        if a >= 1:
            c.drawImage(logo_img(hh), round(lo["x"]), round(lo["y"]))
        else:
            c.drawImage(logo_img(hh), round(lo["x"]), round(lo["y"]), SAMP, p)
        wm = L["wordmark_close"]
        G.wordmark(c, wm["cx"], wm["y"], wm["size"], alpha=a, align=wm["align"])
        # contact block, centred lower half
        fs = L["cta"]["size"] * (1.15 if self.fmt != "16x9" else 1.25)
        cx = self.W / 2
        y = (self.H * 0.55 if self.fmt != "16x9" else self.H * 0.52) + 40 * (1 - k)
        line1 = CLINIC["cta"]
        w1 = T.text_width(line1, "semibold", fs) + fs * 1.5
        x1 = cx - w1 / 2
        G.chat_icon(c, x1, y - fs * 0.95, fs * 1.15, C["brand_deep"], a)
        T.draw_text(c, line1, x1 + fs * 1.5, y, "semibold", fs, C["brand_dark"], a)
        T.draw_text(c, CLINIC["whatsapp"], cx, y + fs * 2.0, "extrabold", fs * 1.65, C["brand_deep"], a, 0.01, "center")
        T.draw_text(c, CLINIC["address"], cx, y + fs * 3.25, "medium", fs * 0.8, C["brand_dark"], a, 0, "center")
        T.draw_text(c, CLINIC["instagram"], cx, y + fs * 4.4, "semibold", fs * 0.9, C["brand_deep"], a, 0.02, "center")
        d = L["disclaimer"]
        for i, ln in enumerate(T.wrap(CLINIC["disclaimer"], "medium", d["size"] * 0.85, d["max_w"])):
            T.draw_text(c, ln, cx, self.H - d["size"] * (2.6 - 1.3 * i) - (30 if self.fmt == "16x9" else 60), "medium",
                        d["size"] * 0.85, C["brand_dark"], a * 0.8, 0, "center")
        if sub_chunk:
            S = L["sub"]
            sa = smooth(lin(t, sub_chunk[0] - 0.1, sub_chunk[0] + 0.05)) * (1 - smooth(lin(t, sub_chunk[1] - 0.05, sub_chunk[1] + 0.1)))
            lines = T.wrap(sub_chunk[2], "semibold", S["size"], S["max_w"])
            G.subtitle(c, lines, S["cx"], self.H * (0.86 if self.fmt == "9x16" else 0.80) if self.fmt != "16x9" else S["cy"] - 10,
                       S["size"], sa, S["max_w"])
        return self.surface.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)[..., :3]
