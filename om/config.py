"""Odonto Maipú — «De la raíz a la sonrisa».

Single source of truth for brand, formats and copy. Everything (Blender scenes,
2D compositor, audio) imports from here, so a change in one place propagates.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "brand"
BUILD = ROOT / "build"          # heavy intermediates (git-ignored)
OUT = ROOT / "entregables"      # deliverables

FPS = 30
DURATION = 60.0                 # 58 s + 2 s end plate
NFRAMES = int(DURATION * FPS)   # 1800  (frames 0..1799)

# ---------------------------------------------------------------- colours
def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


C = {
    # stage
    "bg_top": "#F4F8FB",
    "bg_bottom": "#DCE8F1",
    # brand (exact, from official artwork)
    "brand_light": "#729FBB",
    "brand_deep": "#557A91",
    "brand_dark": "#3E5D70",   # long text on light stage when contrast is short
    "white": "#FFFFFF",
    # light line
    "line_core": "#CFEAFF",
    "line_halo": "#729FBB",
    # materials
    "enamel": "#F3EEE2",
    "dentin": "#E8D9BC",
    "pulp": "#E59C95",
    "pulp_inflamed": "#D9534F",
    "gum": "#E89A9A",
    "bone": "#EDE4D3",
    "titanium": "#9EA9B3",
    "ceramic": "#F6F3EC",
    "gutta": "#E8916B",
    "caries": "#6B4F3A",
}
RGB = {k: hex2rgb(v) for k, v in C.items()}

# ---------------------------------------------------------------- fonts
FONT_DIR = Path("/usr/share/fonts/opentype/montserrat")
FONTS = {
    "extrabold": FONT_DIR / "Montserrat-ExtraBold.otf",
    "bold": FONT_DIR / "Montserrat-Bold.otf",
    "semibold": FONT_DIR / "Montserrat-SemiBold.otf",
    "medium": FONT_DIR / "Montserrat-Medium.otf",
    "regular": FONT_DIR / "Montserrat-Regular.otf",
    "light": FONT_DIR / "Montserrat-Light.otf",
}

LOGO_AZUL = BRAND / "logo" / "odontomaipu_logo_azul.svg"
LOGO_BLANCO = BRAND / "logo" / "odontomaipu_logo_blanco.svg"
DOCTOR_PHOTO = BRAND / "photos" / "doctor.png"     # authorised photo (drop in; placeholder used if absent)

# ---------------------------------------------------------------- formats
# plate_scale: internal 3D render scale (upscaled with Lanczos in the compositor;
# all type, logo and the light line are drawn at full output resolution).
FORMATS = {
    "9x16": dict(w=1080, h=1920, plate="9x16"),
    "4x5": dict(w=1080, h=1350, plate="9x16"),   # own layout, 3D plate shared with 9:16 (same horizontal FOV)
    "16x9": dict(w=1920, h=1080, plate="16x9"),
}
PLATES = {
    "9x16": dict(w=1080, h=1920),
    "16x9": dict(w=1920, h=1080),
}

# ---------------------------------------------------------------- copy
CLINIC = dict(
    name="Odonto Maipú",
    wordmark_top="Clínica dental",
    wordmark_bottom="OdontoMaipu",
    whatsapp="+56 9 4164 9307",
    address="Av. Los Pajaritos 3195, of. 1306, Maipú",
    instagram="@odontomaipu_",
    cta="Escríbenos por WhatsApp",
    tagline="De la raíz a tu sonrisa",
    disclaimer="Imágenes referenciales. Todo tratamiento requiere evaluación profesional previa.",
)

# Values the clinic must confirm. They are rendered verbatim, so edit here and re-run.
PLACEHOLDERS = dict(
    osseo_time="[tiempo de oseointegración que indique la clínica]",
    impression_kind="Impresión",          # or "Escaneo digital" — [según lo que use la clínica]
    dentist_name="[nombre del odontólogo/a]",
)

CHAPTERS = [
    dict(key="endo", title="ENDODONCIA", t0=4.0,
         steps=["Anestesia local", "Limpieza de los conductos", "Sellado"]),
    dict(key="ciru", title="CIRUGÍA ORAL", t0=16.0,
         steps=["Anestesia local", "Extracción", "Sutura y cicatrización"]),
    dict(key="impl", title="IMPLANTOLOGÍA", t0=28.0,
         steps=["Planificación", "Implante", "Oseointegración"]),
    dict(key="rehab", title="REHABILITACIÓN ORAL", t0=40.0,
         steps=[PLACEHOLDERS["impression_kind"], "Pilar y corona", "Mordida"]),
]

# Voice-over script (target start time in seconds, text). One TTS call; the
# real word timings from the recording re-time every animation beat.
VO_SCRIPT = [
    (0.40, "Así se ve tu tratamiento por dentro."),
    (4.60, "Endodoncia."),
    (6.00, "Cuando una caries llega al nervio, limpiamos el interior del diente…"),
    (11.00, "…lo desinfectamos y lo sellamos, para conservar tu diente."),
    (16.60, "Cirugía oral."),
    (18.00, "Extracciones complejas, como las muelas del juicio, con anestesia local…"),
    (23.00, "…y una cicatrización controlada paso a paso."),
    (28.60, "Implantología."),
    (30.00, "Un implante de titanio reemplaza la raíz perdida y se une al hueso…"),
    (35.60, "…hasta quedar firme e integrado."),
    (40.60, "Rehabilitación oral."),
    (42.00, "Sobre ese implante colocamos una corona hecha a tu medida…"),
    (47.20, "…para que vuelvas a masticar y sonreír con confianza."),
    (53.00, "Odonto Maipú. De la raíz a tu sonrisa."),
    (55.40, "Escríbenos por WhatsApp y agenda tu evaluación."),
]
