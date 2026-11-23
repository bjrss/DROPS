"""Generate docs/BRIEF.md (frame-exact timings) from the live timeline, VO timing and SFX cue sheet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audio"))
from om.config import CHAPTERS, CLINIC, PLACEHOLDERS, FPS  # noqa: E402
from om.timeline import B, CHAPTER_STEPS, CHAPTER_T0, RESULT_TAG, TRANSITIONS, phrases  # noqa: E402

LABELS = {
    "open.line_in": "La línea ya se mueve en el fotograma 0 y entra hacia el trazo del logo",
    "logo.trace0": "La línea recorre el trazo: corona izq., corona der., raíz natural y las 6 roscas, una por una",
    "logo.trace1": "Logo oficial completo (Clínica dental / OdontoMaipu); la línea se apaga",
    "logo.rest0": "Logo en reposo, píxel exacto, nada encima",
    "logo.rest1": "Fin del reposo del logo",
    "logo.detach": "La línea se desprende de la punta de la raíz",
    "tooth.in0": "El diente 3D (36) se materializa exactamente donde estaba el logo (barrido de luz)",
    "jaw.reveal0": "Encía, hueso y dientes vecinos crecen alrededor del diente",
    "tooth.in1": "Diente 3D completo",
    "xray0": "Modo rayos X: barrido de abajo hacia arriba; se abre el corte del hueso",
    "jaw.reveal1": "Arcada completa revelada",
    "xray1": "Diente en vidrio con borde fresnel azul de marca",
    "c1": "ENDODONCIA: cápsula entra desde la derecha",
    "c1.capsule": "Cápsula asentada a 1,2 s",
    "c1.problem": "Problema: caries (marrón) y pulpa inflamada (#D9534F) latiendo",
    "c1.caries": "«caries»: la línea rodea la caries",
    "c1.nervio": "«nervio»: la línea baja a la pulpa; pulso máximo",
    "c1.s1": "Paso 1 «Anestesia local»: ondas azules suaves (sin aguja)",
    "c1.s2": "Paso 2 «Limpieza de los conductos»: la línea entra por la corona y baja por cada conducto; la pulpa se disuelve en partículas",
    "c1.irrig": "«desinfectamos»: brillo de irrigación por los 3 conductos",
    "c1.s3": "Paso 3 «Sellado»: la línea se enfría a gutapercha y rellena los conductos; restauración cierra la corona",
    "c1.result": "Resultado «Diente conservado»: la pulpa ya no late",
    "c1.out": "Se cierra el corte; la línea viaja a la muela del juicio",
    "c2": "CIRUGÍA ORAL: cápsula",
    "c2.cut": "Corte del hueso en la zona del 38",
    "c2.problem": "Problema: muela del juicio retenida e inclinada contra la vecina (contorno punteado)",
    "c2.juicio": "«muelas del juicio»: pulso suave en el punto de contacto",
    "c2.s1": "Paso 1 «Anestesia local» (sincronizado con «anestesia local»)",
    "c2.s2": "Paso 2 «Extracción»: incisión de luz, colgajo translúcido, ventana ósea, la muela sale girando",
    "c2.s3": "Paso 3 «Sutura y cicatrización»: 3 puntadas de luz que se apagan; timelapse de relleno óseo",
    "c2.result": "Resultado «Encía sana»: la muela vecina queda libre",
    "c2.out": "La línea cruza la arcada hacia el 46",
    "c3": "IMPLANTOLOGÍA: cápsula",
    "c3.cut": "Corte del hueso en el espacio del 46",
    "c3.problem": "Problema: espacio sin diente; contorno fantasma punteado",
    "c3.raiz": "«raíz»: el contorno de la raíz perdida se ilumina",
    "c3.s1": "Paso 1 «Planificación»: radiografía digital en vidrio con marcador; guía punteada al hueso",
    "c3.s2": "Paso 2 «Implante»: la línea se enrolla como rosca, el implante entra y se enfría a titanio cepillado",
    "c3.s3": "Paso 3 «Oseointegración»: timelapse del hueso abrazando las roscas; rótulo de tiempo; tornillo de cicatrización",
    "c3.result": "Resultado «Implante integrado»; la cámara se queda en el mismo implante",
    "c3.out": "Bandas diagonales; sin cambio de escena",
    "c4": "REHABILITACIÓN ORAL: cápsula (mismo implante)",
    "c4.problem": "Problema: implante integrado pero espacio vacío (corona fantasma punteada)",
    "c4.s1": "Paso 1 «%s»: se retira el tornillo de cicatrización; barrido y malla de puntos azul" % PLACEHOLDERS["impression_kind"],
    "c4.s2": "Paso 2 «Pilar y corona»: el pilar baja",
    "c4.click": "Click limpio del pilar",
    "c4.seat": "La corona hecha sobre la malla baja y se asienta",
    "c4.s3": "Paso 3 «Mordida»: se cierra el corte, la arcada superior cierra, puntos de contacto parejos",
    "c4.result": "Resultado «Sonrisa completa»: la cámara se aleja a la arcada completa",
    "c4.smile0": "La línea dibuja la curva de la sonrisa",
    "c4.smile1": "Curva completa",
    "s6": "Cierre: la cámara se aleja; la línea sube hacia el logo",
    "s6.snap": "La línea encaja en el contorno del logo (acorde pleno)",
    "s6.stack": "«Nuestros tratamientos»: 4 cápsulas apiladas + foto del doctor/a a la izquierda",
    "s6.cta": "Escríbenos por WhatsApp + ícono genérico + teléfono + dirección + Instagram",
    "s7": "Placa final: logo pequeño + aviso legal",
    "end": "Fin",
}


def main():
    L = []
    L.append("# Odonto Maipú — «De la raíz a la sonrisa»\n")
    L.append("Brief técnico generado desde el código (`om/timeline.py`, `build/vo/timing.json`, `audio/sfx.py`). "
             "30 fps, 60 s = 1800 fotogramas (0–1799). Formatos: 9:16 1080×1920 (principal), 4:5 1080×1350, 16:9 1920×1080.\n")
    L.append("## Línea de tiempo (bloqueada a la voz grabada)\n")
    L.append("| s | fotograma | evento |\n|---:|---:|---|")
    for k, v in sorted(B.items(), key=lambda kv: kv[1]):
        L.append(f"| {v:6.2f} | {round(v * FPS):5d} | {LABELS.get(k, k)} |")
    L.append("\nTransiciones con bandas diagonales #729FBB: " + ", ".join(f"{t:.2f} s (f{round(t * FPS)})" for t in TRANSITIONS) + "\n")
    L.append("## Voz en off (ElevenLabs, toma elegida, colocada por palabra)\n")
    L.append("| inicio | fin | guion | frase |\n|---:|---:|---:|---|")
    for p in phrases():
        L.append(f"| {p['words'][0]['t0']:.2f} | {p['words'][-1]['t1']:.2f} | {p['target']:.2f} | {p['text']} |")
    L.append("\nAjustes por la grabación real: «Odonto Maipú. De la raíz a tu sonrisa.» empieza en 52,40 s (no cabe en la "
             "ventana de 2,4 s del guion a velocidad natural) y la línea encaja en el logo en ese mismo instante; "
             "«Escríbenos…» empieza en 55,70 s y termina a 58,79 s, por lo que la placa final empieza a 58,54 s.\n")
    L.append("## Pasos por capítulo\n")
    for ch in CHAPTERS:
        st = CHAPTER_STEPS[ch["key"]]
        L.append(f"- **{ch['title']}** (t0 {CHAPTER_T0[ch['key']]:.1f} s): " +
                 "; ".join(f"{i + 1} «{n}» {st[i]:.2f} s" for i, n in enumerate(ch["steps"])) +
                 f"; resultado «{RESULT_TAG[ch['key']]}» {st[3]:.2f} s")
    L.append("\n## Cues de sonido\n")
    sys.argv = ["x"]
    from sfx import cues
    cs = sorted(cues(), key=lambda c: c[0])
    L.append(f"{len(cs)} cues sintetizados en código (sin muestras de librería; ningún sonido de taladro, aspiración ni instrumental).\n")
    L.append("| s | fotograma | cue |\n|---:|---:|---|")
    for t, name, _, _ in cs:
        L.append(f"| {t:6.2f} | {round(t * FPS):5d} | {name} |")
    L.append("\n## Pendientes de la clínica (se muestran tal cual en pantalla)\n")
    for k, v in PLACEHOLDERS.items():
        L.append(f"- `{k}`: {v}")
    from om.config import DOCTOR_PHOTO
    if DOCTOR_PHOTO.exists():
        L.append("- Foto autorizada del doctor: incluida (`brand/photos/doctor.png`), recortada al rostro y torso en cada formato.")
    else:
        L.append("- Foto autorizada del doctor/a: colocar en `brand/photos/doctor.png` (hoy se ve un marcador neutro).")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "BRIEF.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("docs/BRIEF.md", len(L), "lines")


if __name__ == "__main__":
    main()
