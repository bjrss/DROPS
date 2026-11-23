"""Per-format layouts: genuine re-designs (positions, sizes, what goes where), not crops."""
from . import logo as LG


def layout(fmt):
    if fmt == "9x16":
        W, H = 1080, 1920
        lh = 560
        L = dict(W=W, H=H, s=1.0,
                 logo_open=dict(h=lh, w=lh * LG.ASPECT, x=(W - lh * LG.ASPECT) / 2, y=680),
                 wordmark_open=dict(cx=W / 2, y=1290, size=50, align="center"),
                 capsule=dict(cx=W / 2, cy=250, size=50, align="center"),
                 step=dict(cx=W / 2, cy=1345, size=42, align="center"),
                 sub=dict(cx=W / 2, cy=1580, size=44, max_w=940),
                 card=dict(x=610, y=430, w=400, h=300),
                 glow=(W / 2, H * 0.46, W * 0.75),
                 # closing
                 logo_close=dict(h=300, w=300 * LG.ASPECT, x=(W - 300 * LG.ASPECT) / 2, y=120),
                 wordmark_close=dict(cx=W / 2, y=448, size=38, align="center"),
                 title=dict(x=W / 2, y=640, size=50, align="center"),
                 photo=dict(x=60, y=700, w=440, h=600),
                 caps=dict(x=540, y=730, w=480, gap=140, size=27),
                 cta=dict(cx=W / 2, y=1395, size=36, align="center"),
                 end_logo=dict(h=230, w=230 * LG.ASPECT, x=(W - 230 * LG.ASPECT) / 2, y=740),
                 disclaimer=dict(cx=W / 2, y=1600, size=26, max_w=860))
    elif fmt == "4x5":
        W, H = 1080, 1350
        lh = 470
        L = dict(W=W, H=H, s=1.0,
                 logo_open=dict(h=lh, w=lh * LG.ASPECT, x=(W - lh * LG.ASPECT) / 2, y=440),
                 wordmark_open=dict(cx=W / 2, y=950, size=44, align="center"),
                 capsule=dict(cx=60, cy=110, size=40, align="left"),
                 step=dict(cx=60, cy=1000, size=38, align="left"),
                 sub=dict(cx=W / 2, cy=1190, size=40, max_w=960),
                 card=dict(x=650, y=170, w=370, h=278),
                 glow=(W / 2, H * 0.48, W * 0.7),
                 logo_close=dict(h=190, w=190 * LG.ASPECT, x=70, y=70),
                 wordmark_close=dict(cx=70 + 190 * LG.ASPECT + 34, y=118, size=36, align="left"),
                 title=dict(x=560, y=370, size=40, align="left"),
                 photo=dict(x=60, y=320, w=450, h=560),
                 caps=dict(x=560, y=410, w=460, gap=118, size=25),
                 cta=dict(cx=W / 2, y=910, size=32, align="center"),
                 end_logo=dict(h=200, w=200 * LG.ASPECT, x=(W - 200 * LG.ASPECT) / 2, y=460),
                 disclaimer=dict(cx=W / 2, y=1100, size=24, max_w=900))
    else:  # 16x9
        W, H = 1920, 1080
        lh = 470
        L = dict(W=W, H=H, s=1.0,
                 logo_open=dict(h=lh, w=lh * LG.ASPECT, x=W * 0.5 - lh * LG.ASPECT - 70, y=(H - lh) / 2 - 30),
                 wordmark_open=dict(cx=W * 0.5 + 10, y=430, size=58, align="left"),
                 capsule=dict(cx=1225, cy=290, size=40, align="left"),
                 step=dict(cx=1225, cy=420, size=34, align="left", list=True),
                 sub=dict(cx=W / 2, cy=985, size=38, max_w=1500),
                 card=dict(x=1225, y=560, w=400, h=300),
                 glow=(W * 0.42, H * 0.5, W * 0.45),
                 logo_close=dict(h=230, w=230 * LG.ASPECT, x=760, y=110),
                 wordmark_close=dict(cx=760 + 230 * LG.ASPECT + 34, y=168, size=42, align="left"),
                 title=dict(x=760, y=450, size=42, align="left"),
                 photo=dict(x=90, y=110, w=560, h=760),
                 caps=dict(x=760, y=500, w=470, gap=112, size=25, cols=2, colgap=500),
                 cta=dict(cx=760, y=760, size=32, align="left"),
                 end_logo=dict(h=230, w=230 * LG.ASPECT, x=(W - 230 * LG.ASPECT) / 2, y=330),
                 disclaimer=dict(cx=W / 2, y=760, size=26, max_w=1400))
    return L
