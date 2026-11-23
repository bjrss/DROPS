"""Voice-over with ElevenLabs: whole script in ONE call per take, 2 takes, with
character timestamps.  Each phrase is then cut from the take and placed at its
target time; the real word timings drive the animation (build/vo/timing.json).

The API key is injected by the session proxy (header xi-api-key); this script
never reads or stores it.

    python3 audio/voice.py record      # 2 takes -> build/vo/take_{a,b}.(mp3|json)
    python3 audio/voice.py pick        # analyse, choose, slice, write timing.json + vo.wav
"""
import base64
import json
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from om.config import VO_SCRIPT  # noqa: E402

VO = ROOT / "build" / "vo"
VO.mkdir(parents=True, exist_ok=True)

VOICE_ID = "EXAVITQu4vr4xnSDxMaL"     # premade «Sarah»: calm, reassuring female voice
MODEL = "eleven_multilingual_v2"
SETTINGS = dict(stability=0.58, similarity_boost=0.8, style=0.12, use_speaker_boost=True, speed=1.0)
SR = 44100
# phrase index -> [(first word index, start time), ...]
PLACE = {13: [(0, 52.40), (2, 53.65)], 14: [(0, 55.70)]}


def tts_text():
    parts = []
    for i, (_, txt) in enumerate(VO_SCRIPT):
        t = txt.replace("…", "...").strip()
        if t.startswith("..."):
            t = t[3:].strip()
        parts.append(t)
    # short pauses between phrases (each phrase is re-timed afterwards anyway)
    return ' <break time="0.7s" /> '.join(parts)


def record(take, seed):
    body = dict(text=tts_text(), model_id=MODEL, language_code="es", voice_settings=SETTINGS, seed=seed)
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps?output_format=mp3_44100_128",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "xi-api-key": "proxy-injected"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    (VO / f"take_{take}.mp3").write_bytes(base64.b64decode(d["audio_base64"]))
    d.pop("audio_base64")
    (VO / f"take_{take}.json").write_text(json.dumps(d, ensure_ascii=False))
    print("recorded", take, round(d["alignment"]["character_end_times_seconds"][-1], 2), "s")


def load_wav(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def norm(s):
    s = unicodedata.normalize("NFC", s)
    return s


def phrases_in(al):
    chars = al["characters"]
    st = al["character_start_times_seconds"]
    en = al["character_end_times_seconds"]
    text = "".join(chars)
    out = []
    pos = 0
    for t0, txt in VO_SCRIPT:
        t = txt.replace("…", "...").strip()
        if t.startswith("..."):
            t = t[3:].strip()
        i = text.find(t, pos)
        if i < 0:
            raise RuntimeError(f"phrase not found in alignment: {t}")
        j = i + len(t)
        # words with times
        words = []
        k = i
        while k < j:
            while k < j and not chars[k].isalnum():
                k += 1
            if k >= j:
                break
            a = k
            while k < j and (chars[k].isalnum() or chars[k] in "áéíóúñüÁÉÍÓÚÑ"):
                k += 1
            words.append(dict(w=text[a:k], t0=st[a], t1=en[k - 1]))
        out.append(dict(target=t0, text=txt, t0=st[i], t1=en[j - 1], words=words))
        pos = j
    return out


def analyse(take):
    al = json.loads((VO / f"take_{take}.json").read_text())["alignment"]
    ph = phrases_in(al)
    y = load_wav(VO / f"take_{take}.mp3")
    score = 0.0
    notes = []
    for p in ph:
        # every word should have a plausible duration (no stretched / clipped words)
        for w in p["words"]:
            dur = w["t1"] - w["t0"]
            per = dur / max(len(w["w"]), 1)
            if per > 0.16 or per < 0.03:
                score += 1.0
                notes.append(f"odd word '{w['w']}' {dur:.2f}s")
        # energy gaps inside the phrase (dropouts)
        a, b = int(p["t0"] * SR), int(p["t1"] * SR)
        seg = y[a:b]
        if len(seg) > 0:
            fr = seg[: len(seg) // 441 * 441].reshape(-1, 441)
            rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
            gap = 0
            run = 0
            for r in rms:
                run = run + 1 if r < 0.004 else 0
                gap = max(gap, run)
            if gap > 45:  # >450 ms silence inside a phrase
                score += 0.5
                notes.append(f"long pause in '{p['text'][:30]}' {gap * 10} ms")
        # fit: phrase must end before the next phrase target
    for i, p in enumerate(ph):
        nxt = ph[i + 1]["target"] if i + 1 < len(ph) else 58.0
        dur = p["t1"] - p["t0"]
        slack = nxt - p["target"] - dur
        if slack < 0.15:
            score += 0.3 + max(0, -slack)
            notes.append(f"tight: '{p['text'][:28]}' dur {dur:.2f}s window {nxt - p['target']:.2f}s")
    peak = float(np.abs(y).max())
    if peak > 0.99:
        score += 1
        notes.append("clipping")
    return score, notes, ph


def pick():
    res = {}
    for take in ("a", "b"):
        if (VO / f"take_{take}.json").exists():
            res[take] = analyse(take)
            print(f"take {take}: score {res[take][0]:.2f}")
            for n in res[take][1]:
                print("   ", n)
    best = min(res, key=lambda k: res[k][0])
    print("chosen take:", best)
    ph = res[best][2]
    y = load_wav(VO / f"take_{best}.mp3")
    # place each phrase at its target; if a phrase would overlap the next, nudge
    # it earlier (never later than the script), keeping >= 0.12 s air.
    # PLACE overrides re-time the closing lines to the real recording (the
    # script windows there are shorter than the natural read).
    total = np.zeros(int(60.0 * SR), np.float32)
    timing = []
    prev_end = 0.0
    fade = int(0.015 * SR)
    for i, p in enumerate(ph):
        groups = PLACE.get(i)
        if groups is None:
            dur = p["t1"] - p["t0"]
            start = p["target"]
            nxt = ph[i + 1]["target"] if i + 1 < len(ph) else 58.6
            if start + dur > nxt - 0.12:
                start = max(prev_end + 0.15, nxt - 0.12 - dur)
            groups = [(0, start)]
        segs = []
        for gi, (w0, start) in enumerate(groups):
            w1 = groups[gi + 1][0] if gi + 1 < len(groups) else len(p["words"])
            ws = p["words"][w0:w1]
            ta = ws[0]["t0"] if gi > 0 else p["t0"]
            tb = ws[-1]["t1"] if gi + 1 < len(groups) else p["t1"]
            pad_a, pad_b = 0.04, 0.12
            a = max(int((ta - pad_a) * SR), 0)
            b = min(int((tb + pad_b) * SR), len(y))
            seg = y[a:b].copy()
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
            s0 = int((start - pad_a) * SR)
            total[s0:s0 + len(seg)] += seg
            off = start - ta
            segs += [dict(w=w["w"], t0=round(w["t0"] + off, 3), t1=round(w["t1"] + off, 3)) for w in ws]
            prev_end = start + (tb - ta)
        timing.append(dict(text=p["text"], target=p["target"], start=segs[0]["t0"], end=round(prev_end, 3), words=segs))
    (VO / "timing.json").write_text(json.dumps(dict(take=best, voice=VOICE_ID, model=MODEL, phrases=timing),
                                               ensure_ascii=False, indent=1))
    import soundfile as sf
    sf.write(VO / "vo_raw.wav", total, SR, subtype="FLOAT")
    print("wrote timing.json and vo_raw.wav")
    for t in timing:
        print(f"  {t['start']:6.2f}-{t['end']:6.2f}  (script {t['target']:5.2f})  {t['text']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pick"
    if cmd == "record":
        record("a", 1101)
        record("b", 2903)
    pick()
