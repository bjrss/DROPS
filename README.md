# Odonto Maipú — «De la raíz a la sonrisa»

Video explicativo 3D (60 s) de los cuatro tratamientos de Odonto Maipú — Endodoncia, Cirugía oral,
Implantología y Rehabilitación oral — construido íntegramente en código:

* **3D**: Blender 4.5 LTS controlado por Python (`bpy`), EEVEE headless, motion blur real y profundidad de campo.
  Toda la anatomía (dientes, pulpa y conductos, encía, hueso trabecular, muela del juicio, implante, pilar,
  corona) se modela proceduralmente con SDF + *marching cubes* (`om/teeth.py`, `om/world.py`,
  `tools/make_geometry.py`): no hay modelos de terceros, así que no hay licencias que gestionar.
* **2D**: compositor en Python + Skia + HarfBuzz (`om/compose.py`): escenario, tipografía Montserrat,
  cápsulas, rótulos, subtítulos quemados, bandas diagonales y **la línea de luz**, dibujada por una sola
  función compartida (`om/lightline.py → draw_lightline`).
* **Audio**: voz ElevenLabs (una sola llamada por toma, 2 tomas, `audio/voice.py`), música original y ~100
  efectos sintetizados en numpy (`audio/music.py`, `audio/sfx.py`), mezcla con verificación por palabra
  (`audio/mix.py`).
* **Una única fuente de tiempo**: `om/timeline.py` lee los tiempos reales de cada palabra grabada; Blender,
  el compositor y el sonido usan las mismas funciones (`om/anim.py`, `om/director.py`).

## Entregables (`entregables/`)

| archivo | qué es |
|---|---|
| `OdontoMaipu_DeLaRaizATuSonrisa_9x16.mp4` | principal, 1080×1920, 30 fps |
| `OdontoMaipu_DeLaRaizATuSonrisa_4x5.mp4` | 1080×1350 (posts), maqueta propia |
| `OdontoMaipu_DeLaRaizATuSonrisa_16x9.mp4` | 1920×1080 (web / sala de espera), maqueta propia |
| `capsulas/…_capsula{1-4}_{endo,ciru,impl,rehab}_{formato}.mp4` | 3 s logo + 12 s capítulo + 3,3 s contacto |
| `…_subtitulos_es.srt` + `capsulas/*.srt` | subtítulos (los mismos que van quemados) |
| `audio/` | mezcla master (-14 LUFS / -1 dBTP) y stems de voz, música y SFX (WAV 48 kHz / 24 bit) |
| `…_ANIMATIC_baja_9x16.mp4` | animatic en baja para la validación clínica |
| `docs/BRIEF.md` | brief con tiempos exactos por fotograma, voz, cues de sonido y pendientes |

## Reconstruir

```bash
python3 tools/make_geometry.py                      # mallas procedurales -> build/geo
/opt/om/venv311/bin/python blender/build_scene.py   # escena + animación horneada -> build/scene.blend
tools/render_plates.sh                              # placas 3D 9:16 y 16:9 (EEVEE, 0,75x, 4 muestras)
python3 audio/voice.py record                       # 2 tomas ElevenLabs + elección + tiempos por palabra
python3 audio/music.py && python3 audio/sfx.py && python3 audio/mix.py
python3 tools/deliver.py compose 9x16 4x5 16x9 && python3 tools/deliver.py main
python3 tools/deliver.py srt && python3 tools/deliver.py capsules && python3 tools/deliver.py stems
python3 tools/make_brief.py
```

Requisitos: Python 3.11 + `bpy==4.5.16` (Blender como módulo), Python 3.13 con `numpy scipy scikit-image
skia-python uharfbuzz cairosvg pyloudnorm soundfile`, `ffmpeg`, fuentes Montserrat (OFL).
La clave de ElevenLabs la inyecta el proxy del entorno; el código nunca la lee ni la guarda.

## Marca y salud

* Logo oficial: el trazado se extrajo sin modificar de los PDF entregados (`brand/logo/*.svg`, PNG con
  transparencia). Solo se mueve, escala, funde o enmascara; en reposo es píxel exacto (#557A91 verificado).
  El *wordmark* («Clínica dental / OdontoMaipu») no venía en los archivos: está compuesto en Montserrat
  imitando su estilo, como pide el brief para los rótulos de marca.
* Sin sangre, agujas, taladros ni sonidos de instrumental; sin promesas, precios ni antes/después; sin
  marcas comerciales ni nombres registrados de técnicas; ícono de chat genérico (no el logo de WhatsApp).

## Pendientes de la clínica

Valores marcados entre corchetes en `om/config.py → PLACEHOLDERS` (se muestran tal cual en pantalla):
tiempo de oseointegración, «Impresión» vs «Escaneo digital», nombre del odontólogo/a para la validación.
La foto autorizada del doctor/a va en `brand/photos/doctor.png`; mientras no esté, el cierre muestra un
marcador neutro rotulado «[foto autorizada del doctor/a]».
