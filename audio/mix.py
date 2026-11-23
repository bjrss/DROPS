"""Mix: VO ≈ -16 LUFS, music always ≥ 12 dB under every word (verified per word),
SFX tucked under the voice, master -14 LUFS integrated / -1 dBTP.

Writes build/audio/{mix.wav, stem_vo.wav, stem_music.wav, stem_sfx.wav, mix_report.txt}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import resample_poly, lfilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "audio"))
from synth import SR, hp  # noqa: E402

A = ROOT / "build" / "audio"
BED_MAX = -19.0   # music level between phrases (pre-master); higher = closer to -16 LUFS voice but more pumping
VO = ROOT / "build" / "vo"
meter = pyln.Meter(SR)


def lufs(x):
    return meter.integrated_loudness(x)


def kweight(x):
    y = x.copy()
    for f in meter._filters.values():
        y = lfilter(f.b, f.a, y, axis=0)
    return y


def true_peak_db(x):
    up = resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


def limiter(x, ceiling_db=-1.2, look=0.0015, release=0.08, return_env=False):
    c = 10 ** (ceiling_db / 20)
    up = np.abs(resample_poly(x, 4, 1, axis=0)).max(1)
    pk = up.reshape(-1, 4).max(1)[: len(x)]
    pk = maximum_filter1d(pk, size=int(look * SR) * 2 + 1)
    g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    # smooth: instant attack (already looked ahead), exponential release
    out = np.empty_like(g)
    a = np.exp(-1 / (release * SR))
    cur = 1.0
    for i in range(len(g)):
        cur = g[i] if g[i] < cur else a * cur + (1 - a) * g[i]
        out[i] = cur
    out = np.minimum(out, uniform_filter1d(g, int(look * SR) + 1))
    if return_env:
        return x * out[:, None], out
    return x * out[:, None]


def compress(x, thresh_db=-24.0, ratio=3.0, attack=0.005, release=0.08):
    """Gentle feed-forward RMS compressor for the voice (tames TTS peaks before mastering)."""
    lvl = np.sqrt(uniform_filter1d(x[:, 0] ** 2, int(0.01 * SR)) + 1e-12)
    db = 20 * np.log10(lvl)
    over = np.maximum(db - thresh_db, 0)
    gr = -over * (1 - 1 / ratio)
    g = np.empty_like(gr)
    aa, ar = np.exp(-1 / (attack * SR)), np.exp(-1 / (release * SR))
    cur = 0.0
    for i in range(len(gr)):
        c = aa if gr[i] < cur else ar
        cur = c * cur + (1 - c) * gr[i]
        g[i] = cur
    return x * (10 ** (g / 20))[:, None]


def speech_mask(vo, timing):
    m = np.zeros(len(vo))
    for p in timing["phrases"]:
        for w in p["words"]:
            a = int((w["t0"] - 0.10) * SR)
            b = int((w["t1"] + 0.12) * SR)
            m[max(a, 0):b] = 1.0
    # smooth: 60 ms attack-ish / 350 ms release-ish (symmetric smoothing of a dilated mask)
    m = maximum_filter1d(m, int(0.25 * SR))
    m = uniform_filter1d(m, int(0.18 * SR))
    return np.clip(m, 0, 1)


def word_margins(vo, mus, timing):
    vk, mk = kweight(vo), kweight(mus)
    res = []
    for p in timing["phrases"]:
        for w in p["words"]:
            a, b = int(w["t0"] * SR), int(max(w["t1"], w["t0"] + 0.08) * SR)
            lv = 10 * np.log10(np.mean(vk[a:b] ** 2) + 1e-12)
            lm = 10 * np.log10(np.mean(mk[a:b] ** 2) + 1e-12)
            res.append((w["w"], w["t0"], lv - lm))
    return res


def main():
    timing = json.loads((VO / "timing.json").read_text())
    vo, sr = sf.read(VO / "vo_raw.wav")
    vo = resample_poly(vo, 160, 147)  # 44.1k -> 48k
    N = int(60.0 * SR)
    vo = np.pad(vo, (0, max(0, N - len(vo))))[:N]
    vo = hp(vo, 75)
    vo = np.stack([vo, vo], 1)
    vo *= 10 ** ((-16.0 - lufs(vo)) / 20)
    vo = compress(vo)
    vo *= 10 ** ((-16.0 - lufs(vo)) / 20)
    music, _ = sf.read(A / "music.wav")
    sfx, _ = sf.read(A / "sfx.wav")
    music = music[:N]
    sfx = sfx[:N]
    mask = speech_mask(vo[:, 0], timing)[:, None]
    music0, sfx0 = music.copy(), sfx.copy()
    bed = -21.0
    for outer in range(8):
        music = music0 * 10 ** ((bed - lufs(music0)) / 20)
        duck_db = -9.0
        for _ in range(14):
            mus = music * 10 ** ((duck_db * mask) / 20)
            worst = min(m for _, _, m in word_margins(vo, mus, timing))
            if worst >= 12.5:
                break
            duck_db -= (12.5 - worst) + 0.5
        sfx_g = sfx0 * 10 ** ((bed - 2.5 - lufs(sfx0)) / 20) * 10 ** ((-8.0 * mask) / 20)
        worst_sfx = min(m for _, _, m in word_margins(vo, sfx_g, timing))
        if worst_sfx < 12.5:
            sfx_g *= 10 ** ((worst_sfx - 12.5) / 20)
        mix = vo + mus + sfx_g
        g = 0.0
        for _ in range(4):
            g += -14.0 - lufs(limiter(mix * 10 ** (g / 20)))
        if abs(g) < 0.35 or bed >= BED_MAX - 1e-6:
            break
        bed = min(BED_MAX, bed + g)  # beds carry the extra loudness, capped to avoid audible pumping
    master, env = limiter(mix * 10 ** (g / 20), return_env=True)
    L = lufs(master)
    tp = true_peak_db(master)
    E = 10 ** (g / 20) * env[:, None]          # stems carry the master gain + limiter envelope: they sum to the mix
    for name, x in (("mix", master), ("stem_vo", vo * E), ("stem_music", mus * E), ("stem_sfx", sfx_g * E)):
        sf.write(A / f"{name}.wav", x.astype(np.float32), SR, subtype="PCM_24")
    mfinal = word_margins(vo, mus, timing)
    rep = [f"master integrated: {L:.2f} LUFS", f"master true peak: {tp:.2f} dBTP",
           f"voice in master: {lufs(vo * 10 ** (g / 20) * env[:, None]):.2f} LUFS",
           f"music bed between phrases: {bed:.1f} LUFS (pre-master)",
           f"music duck under voice: {duck_db:.1f} dB",
           f"min music margin under any word: {min(m for _, _, m in mfinal):.1f} dB",
           f"min sfx margin under any word: {min(m for _, _, m in word_margins(vo, sfx_g, timing)):.1f} dB"]
    worst5 = sorted(mfinal, key=lambda r: r[2])[:5]
    rep.append("closest words (music): " + ", ".join(f"{w}@{t:.2f}s {m:.1f}dB" for w, t, m in worst5))
    (A / "mix_report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
