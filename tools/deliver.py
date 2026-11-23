"""Build every deliverable once plates + audio exist.

    python3 tools/deliver.py compose 9x16 4x5 16x9    # composite videos (no audio) -> build/comp_*.mp4
    python3 tools/deliver.py main                     # mux with the master mix -> entregables/
    python3 tools/deliver.py srt                      # subtitles .srt (same chunks as the burned-in ones)
    python3 tools/deliver.py capsules 9x16 4x5 16x9   # 4 stand-alone chapter clips per format
    python3 tools/deliver.py stems                    # mix + stems (WAV 48 kHz / 24-bit)
    python3 tools/deliver.py animatic                 # low-res review cut for clinical validation
"""
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audio"))
from om.config import CHAPTERS, FPS, FORMATS, OUT  # noqa: E402

BUILD = ROOT / "build"
NAME = "OdontoMaipu_DeLaRaizATuSonrisa"
OUT.mkdir(parents=True, exist_ok=True)


def run(cmd):
    print(" ".join(map(str, cmd))[:200], flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def ffmpeg(*args):
    run(["ffmpeg", "-y", "-v", "error", *args])


# ------------------------------------------------------------------ compose
def _compose_part(args):
    fmt, a, b, out, plates = args
    run([sys.executable, "-m", "om.compose", "--fmt", fmt, "--start", a, "--end", b, "--out", out, "--plates", plates,
         "--crf", "16"])
    return out


def compose(fmts, workers=3, plates=None, parts=3):
    plates = plates or str(BUILD / "plates")
    jobs = []
    for fmt in fmts:
        edges = np.linspace(0, 1800, parts + 1).astype(int)
        for i in range(parts):
            jobs.append((fmt, edges[i], edges[i + 1] - 1, str(BUILD / f"comp_{fmt}_p{i}.mp4"), plates))
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_compose_part, jobs))
    for fmt in fmts:
        lst = BUILD / f"comp_{fmt}.txt"
        lst.write_text("".join(f"file '{BUILD / f'comp_{fmt}_p{i}.mp4'}'\n" for i in range(parts)))
        ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", BUILD / f"comp_{fmt}.mp4")


def main_mux():
    for fmt in FORMATS:
        v = BUILD / f"comp_{fmt}.mp4"
        if not v.exists():
            continue
        ffmpeg("-i", v, "-i", BUILD / "audio" / "mix.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy",
               "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest", "-movflags", "+faststart",
               OUT / f"{NAME}_{fmt}.mp4")


# ------------------------------------------------------------------ subtitles
def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt():
    from om.compose import subtitle_chunks
    ch = subtitle_chunks()
    txt = "".join(f"{i + 1}\n{srt_time(c['t0'])} --> {srt_time(c['t1'])}\n{c['text']}\n\n" for i, c in enumerate(ch))
    (OUT / f"{NAME}_subtitulos_es.srt").write_text(txt, encoding="utf-8")
    # capsules
    for k, chap in enumerate(CHAPTERS):
        t0 = chap["t0"]
        lines = []
        for c in ch:
            if c["t1"] > t0 and c["t0"] < t0 + 12:
                lines.append((max(c["t0"], t0) - t0 + 3.0, min(c["t1"], t0 + 12) - t0 + 3.0, c["text"]))
        cta = ch[-1]
        lines.append((15.0 + 0.15, 15.0 + 0.15 + (cta["t1"] - cta["t0"]), cta["text"]))
        txt = "".join(f"{i + 1}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(lines))
        (OUT / "capsulas").mkdir(exist_ok=True)
        (OUT / "capsulas" / f"{NAME}_capsula{k + 1}_{chap['key']}_subtitulos_es.srt").write_text(txt, encoding="utf-8")
    print("srt written")


# ------------------------------------------------------------------ capsules
INTRO, MAIN, OUTRO = 3.0, 12.0, 3.3


def capsule_audio(chap):
    import soundfile as sf
    from synth import SR
    from mix import limiter, lufs, true_peak_db
    A = BUILD / "audio"
    vo, _ = sf.read(A / "stem_vo.wav")
    mu, _ = sf.read(A / "stem_music.wav")
    fx, _ = sf.read(A / "stem_sfx.wav")
    t0 = chap["t0"]
    cta0 = 55.55

    def seg(x, a, d):
        return x[int(a * SR): int(a * SR) + int(d * SR)]

    def xfade(parts, fade=0.12):
        n = int(fade * SR)
        out = parts[0].copy()
        for p in parts[1:]:
            r = np.linspace(1, 0, n)[:, None]
            out[-n:] = out[-n:] * r + p[:n] * (1 - r)
            out = np.vstack([out, p[n:]])
        return out

    intro = seg(mu, 0.0, INTRO + 0.12) + seg(fx, 0.05, INTRO + 0.12)
    mid = seg(vo, t0, MAIN + 0.12) + seg(mu, t0, MAIN + 0.12) + seg(fx, t0, MAIN + 0.12)
    outro = seg(vo, cta0, OUTRO) + seg(mu, cta0, OUTRO) + seg(fx, cta0, OUTRO)
    x = xfade([intro, mid, outro])
    tt = np.arange(len(x)) / SR
    x *= np.clip((tt[-1] - tt) / 0.5, 0, 1)[:, None]
    g = 0.0
    for _ in range(4):
        g += -14.0 - lufs(limiter(x * 10 ** (g / 20)))
    y = limiter(x * 10 ** (g / 20))
    return y, lufs(y), true_peak_db(y)


def _capsule_cards(args):
    fmt, k = args
    from om.compose import CapsuleCards
    from om.timeline import phrases
    chap = CHAPTERS[k]
    cc = CapsuleCards(fmt, str(BUILD / "plates"))
    cta = phrases()[-1]
    sub = (0.15, 0.15 + (cta["words"][-1]["t1"] - cta["words"][0]["t0"]) + 0.3, cta["text"])
    paths = []
    for part, n, fn in (("intro", int(INTRO * FPS), lambda f: cc.intro(f, chap)),
                        ("outro", int(OUTRO * FPS), lambda f: cc.outro(f, sub))):
        out = BUILD / f"cap_{fmt}_{k}_{part}.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cc.W}x{cc.H}", "-r", str(FPS),
               "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", str(out)]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in range(n):
            pr.stdin.write(fn(f).tobytes())
        pr.stdin.close()
        pr.wait()
        paths.append(out)
    return paths


def capsules(fmts):
    import soundfile as sf
    from synth import SR
    (OUT / "capsulas").mkdir(exist_ok=True)
    jobs = [(fmt, k) for fmt in fmts for k in range(4)]
    with ProcessPoolExecutor(3) as ex:
        list(ex.map(_capsule_cards, jobs))
    report = []
    for k, chap in enumerate(CHAPTERS):
        y, L, tp = capsule_audio(chap)
        wav = BUILD / f"cap_audio_{k}.wav"
        sf.write(wav, y.astype(np.float32), SR, subtype="PCM_24")
        report.append(f"capsule {k + 1} {chap['key']}: {L:.2f} LUFS, {tp:.2f} dBTP")
        for fmt in fmts:
            main = BUILD / f"comp_{fmt}.mp4"
            out = OUT / "capsulas" / f"{NAME}_capsula{k + 1}_{chap['key']}_{fmt}.mp4"
            ffmpeg("-i", BUILD / f"cap_{fmt}_{k}_intro.mp4", "-ss", f"{chap['t0']:.3f}", "-t", f"{MAIN:.3f}", "-i", main,
                   "-i", BUILD / f"cap_{fmt}_{k}_outro.mp4", "-i", wav,
                   "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-map", "3:a",
                   "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS),
                   "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", out)
    (BUILD / "audio" / "capsules_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


def stems():
    d = OUT / "audio"
    d.mkdir(exist_ok=True)
    A = BUILD / "audio"
    for src, dst in (("mix.wav", "mezcla_master_-14LUFS.wav"), ("stem_vo.wav", "stem_voz.wav"),
                     ("stem_music.wav", "stem_musica.wav"), ("stem_sfx.wav", "stem_sfx.wav")):
        ffmpeg("-i", A / src, "-c:a", "pcm_s24le", "-ar", "48000", d / dst)
    (d / "informe_mezcla.txt").write_text((A / "mix_report.txt").read_text() +
                                          ((A / "capsules_report.txt").read_text() if (A / "capsules_report.txt").exists() else ""))
    (d / "cues_sfx.txt").write_text((A / "sfx_cues.txt").read_text())


def animatic():
    plates = BUILD / "animatic_plates"
    out = BUILD / "animatic_9x16_full.mp4"
    run([sys.executable, "-m", "om.compose", "--fmt", "9x16", "--start", "1", "--end", "1799", "--step", "2",
         "--plates", plates, "--out", out, "--crf", "20"])
    # half resolution, 15 fps, with the final mix
    ffmpeg("-i", out, "-i", BUILD / "audio" / "mix.wav", "-vf", "scale=540:960:flags=lanczos", "-c:v", "libx264",
           "-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
           OUT / f"{NAME}_ANIMATIC_baja_9x16.mp4")


if __name__ == "__main__":
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == "compose":
        compose(args or list(FORMATS))
    elif cmd == "main":
        main_mux()
    elif cmd == "srt":
        srt()
    elif cmd == "capsules":
        capsules(args or list(FORMATS))
    elif cmd == "stems":
        stems()
    elif cmd == "animatic":
        animatic()
