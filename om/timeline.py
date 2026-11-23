"""Master beat sheet (seconds).  Locked to the recorded voice-over: beats that
belong to a spoken word read its real start time from build/vo/timing.json
(copied to brand/vo/timing.json), so re-recording the voice re-times the film.
"""
import json
from pathlib import Path

from .config import ROOT

_TIMING = None
for p in (ROOT / "build" / "vo" / "timing.json", ROOT / "brand" / "vo" / "timing.json"):
    if p.exists():
        _TIMING = json.loads(p.read_text())
        break


def phrases():
    return _TIMING["phrases"] if _TIMING else []


def word(w, after=0.0, default=None):
    """Start time of the first occurrence of word `w` at or after `after`."""
    for p in phrases():
        for x in p["words"]:
            if x["w"].lower() == w.lower() and x["t0"] >= after - 1e-6:
                return x["t0"]
    if default is None:
        raise KeyError(w)
    return default


def word_end(w, after=0.0, default=None):
    for p in phrases():
        for x in p["words"]:
            if x["w"].lower() == w.lower() and x["t0"] >= after - 1e-6:
                return x["t1"]
    if default is None:
        raise KeyError(w)
    return default


W = word
B = {}

# ---------------------------------------------------------------- S1 opening 0–4
B["open.line_in"] = 0.0
B["logo.trace0"] = 0.20
B["logo.trace1"] = 1.40
B["logo.rest0"] = 1.40
B["logo.rest1"] = 2.40
B["logo.detach"] = 2.40
B["tooth.in0"] = 2.55
B["tooth.in1"] = 3.25
B["jaw.reveal0"] = 3.12
B["jaw.reveal1"] = 3.9
B["xray0"] = 3.35
B["xray1"] = 4.05

# ---------------------------------------------------------------- C1 endodoncia 4–16
C1 = 4.0
B["c1"] = C1
B["c1.capsule"] = C1
B["c1.problem"] = C1 + 1.2
B["c1.caries"] = W("caries", 5, 6.6)
B["c1.nervio"] = W("nervio", 6, 7.65)
B["c1.s1"] = 7.45                                  # anestesia local
B["c1.s2"] = W("limpiamos", 7, 8.24) + 0.15          # limpieza de los conductos
B["c1.irrig"] = W("desinfectamos", 10, 11.2) - 0.2
B["c1.s3"] = W("sellamos", 11, 12.37) - 0.45         # sellado
B["c1.result"] = 13.5
B["c1.out"] = 15.0

# ---------------------------------------------------------------- C2 cirugía 16–28
C2 = 16.0
B["c2"] = C2
B["c2.cut"] = C2 + 0.15
B["c2.problem"] = C2 + 1.2
B["c2.juicio"] = W("muelas", 18, 19.99)
B["c2.s1"] = W("anestesia", 20, 21.48) - 0.45
B["c2.s2"] = W("local", 21, 22.04) + 0.25
B["c2.s3"] = 24.35
B["c2.result"] = 26.2
B["c2.out"] = 27.0

# ---------------------------------------------------------------- C3 implantología 28–40
C3 = 28.0
B["c3"] = C3
B["c3.cut"] = C3 + 0.3
B["c3.problem"] = C3 + 1.2
B["c3.raiz"] = W("raíz", 31, 32.32)
B["c3.s1"] = 32.0
B["c3.s2"] = W("une", 33, 33.65) - 0.15
B["c3.s3"] = W("hasta", 35, 35.6)
B["c3.result"] = 37.7
B["c3.out"] = 39.5

# ---------------------------------------------------------------- C4 rehabilitación 40–52
C4 = 40.0
B["c4"] = C4
B["c4.problem"] = C4 + 1.2
B["c4.s1"] = 42.9
B["c4.s2"] = W("corona", 44, 44.17) + 0.25
B["c4.click"] = B["c4.s2"] + 0.55
B["c4.seat"] = B["c4.s2"] + 1.75
B["c4.s3"] = W("para", 47, 47.2) - 0.35
B["c4.result"] = W("sonreír", 48, 48.8) - 0.2
B["c4.smile0"] = B["c4.result"] + 1.2
B["c4.smile1"] = B["c4.smile0"] + 1.1

# ---------------------------------------------------------------- S6 closing 52–58, S7 58–60
B["s6"] = 52.0
B["s6.snap"] = W("odonto", 52, 52.4)                 # line re-enters the logo: full chord
B["s6.stack"] = B["s6.snap"] + 0.55
B["s6.cta"] = W("escríbenos", 55, 55.7) - 0.2
B["s7"] = max(58.0, word_end("evaluación", 57, 58.79) - 0.25)
B["end"] = 60.0

CHAPTER_STEPS = {
    "endo": [B["c1.s1"], B["c1.s2"], B["c1.s3"], B["c1.result"]],
    "ciru": [B["c2.s1"], B["c2.s2"], B["c2.s3"], B["c2.result"]],
    "impl": [B["c3.s1"], B["c3.s2"], B["c3.s3"], B["c3.result"]],
    "rehab": [B["c4.s1"], B["c4.s2"], B["c4.s3"], B["c4.result"]],
}
CHAPTER_T0 = {"endo": C1, "ciru": C2, "impl": C3, "rehab": C4}
CHAPTER_PROBLEM = {"endo": B["c1.problem"], "ciru": B["c2.problem"], "impl": B["c3.problem"], "rehab": B["c4.problem"]}
CHAPTER_END = {"endo": 16.0, "ciru": 28.0, "impl": 40.0, "rehab": 52.0}
RESULT_TAG = {"endo": "Diente conservado", "ciru": "Encía sana", "impl": "Implante integrado", "rehab": "Sonrisa completa"}
TRANSITIONS = [15.55, 27.55, 39.55, 51.75]   # diagonal band sweeps


def table():
    return dict(sorted(B.items(), key=lambda kv: kv[1]))


if __name__ == "__main__":
    for k, v in table().items():
        print(f"{v:7.3f}s  f{round(v * 30):5d}  {k}")
