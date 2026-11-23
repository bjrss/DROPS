"""Original score, generated in code.  D major, 100 BPM (1 bar = 2.4 s).

Piano, plucks, pad and sub from frame 0; light percussion (shaker, soft kick)
from chapter 1; a small lift into every chapter change; a full chord when the
line snaps back into the logo, ringing under the close.  The bar grid is
anchored so that the snap falls on a downbeat.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audio"))
from synth import SR, add, bell, lp, midi, n, pad, pan, piano, pluck, reverb, reverb_ir, shaker, sub_kick, sweep_noise, sine, t_axis  # noqa
from om.timeline import B  # noqa: E402

BPM = 100
BEAT = 60 / BPM
BAR = 4 * BEAT
DUR = 60.0
SNAP = B["s6.snap"]
GRID0 = SNAP - round((SNAP - 0.0) / BAR) * BAR       # first downbeat (may be < 0)
while GRID0 > 0:
    GRID0 -= BAR

# chord per bar (cycle), voicings as note names
PROG = [
    dict(root="D2", pad=["D3", "A3", "E4", "F#4"], arp=["D4", "A4", "E5", "F#5", "A5"]),        # Dadd9
    dict(root="A1", pad=["C#3", "A3", "E4", "B4"], arp=["C#4", "A4", "E5", "B4", "E5"]),        # A/C#
    dict(root="B1", pad=["D3", "A3", "D4", "F#4"], arp=["B3", "F#4", "D5", "A5", "F#5"]),       # Bm7
    dict(root="G1", pad=["D3", "B3", "F#4", "A4"], arp=["G3", "D4", "B4", "F#5", "A5"]),        # Gmaj9
]
FINAL = dict(root="D2", pad=["D3", "A3", "E4", "F#4", "A4", "D5"], arp=["D4", "A4", "E5", "F#5", "A5", "D6"])


def bars():
    t = GRID0
    i = 0
    while t < DUR:
        yield i, t
        t += BAR
        i += 1


def render():
    N = int(DUR * SR)
    stems = {k: np.zeros((N, 2)) for k in ("piano", "pluck", "pad", "sub", "perc", "fx")}
    perc_on = 4.0 + 0.4
    for i, t0 in bars():
        if t0 + BAR <= 0:
            continue
        final = t0 >= SNAP - 1e-6
        ch = FINAL if final else PROG[i % 4]
        if final and t0 > SNAP + 0.01:
            continue  # the final chord rings by itself
        # ---- pad (sustained, overlapping bars)
        dur = BAR + 1.4 if not final else DUR - t0
        p = pad([midi(n(x)) for x in ch["pad"]], dur, vel=0.30 if not final else 0.42, cutoff=1500 if not final else 2600,
                attack=0.9 if not final else 0.05, release=1.4 if not final else 3.5, seed=i)
        add(stems["pad"], p, max(t0, 0))
        # ---- sub
        s = sine(midi(n(ch["root"])), dur if not final else DUR - t0, vel=0.32, a=0.08, r=0.6)
        add(stems["sub"], pan(s, 0), max(t0, 0))
        # ---- piano: gentle arpeggio in eighths, chord on the downbeat
        if final:
            for k, nm in enumerate(ch["arp"]):
                add(stems["piano"], pan(piano(midi(n(nm)), DUR - t0, vel=0.55 - 0.04 * k), -0.3 + 0.12 * k), t0 + 0.012 * k)
            continue
        pattern = [0, 2, 1, 3, 4, 3, 1, 2]
        for e in range(8):
            te = t0 + e * BEAT / 2
            if te < 0:
                continue
            nm = ch["arp"][pattern[e] % len(ch["arp"])]
            vel = 0.42 if e == 0 else 0.26 + 0.06 * ((e + 1) % 2)
            add(stems["piano"], pan(piano(midi(n(nm)), 2.2, vel=vel), -0.25 + 0.06 * e), te)
        # ---- plucks: sparkle answer on beats 3 and 4& every other bar
        if i % 2 == 1:
            for k, (b, nm) in enumerate(((2.0, ch["arp"][-1]), (3.5, ch["arp"][-2]))):
                tp = t0 + b * BEAT
                if tp >= 0 and tp < SNAP - 0.3:
                    add(stems["pluck"], pan(pluck(midi(n(nm)) * 2, 1.4, vel=0.22, seed=i * 7 + k), 0.45), tp)
        # ---- percussion from chapter 1 (soft kick on 1 & 3, shaker 8ths)
        for b in range(4):
            tb = t0 + b * BEAT
            if tb < perc_on or tb >= SNAP - 0.05:
                continue
            if b in (0, 2):
                add(stems["perc"], pan(sub_kick(0.42 if b == 0 else 0.3), 0), tb)
            for h in range(2):
                th = tb + h * BEAT / 2
                add(stems["perc"], pan(shaker(0.10 if h else 0.07, seed=int(th * 100)), 0.35), th)
    # ---- lifts into each chapter change: soft riser + reverse swell
    for tc in (4.0, 16.0, 28.0, 40.0):
        r = sweep_noise(1.5, 400, 7000, vel=0.12, seed=int(tc), shape="rise")
        add(stems["fx"], pan(r, 0.0), tc - 1.5)
        tone = np.concatenate([np.zeros(1)])
        sw = pad([midi(n("A4")), midi(n("D5"))], 1.6, vel=0.10, cutoff=4000, attack=1.4, release=0.15, seed=int(tc))
        add(stems["fx"], sw, tc - 1.55)
    # big moment: shimmer on the snap
    for k, nm in enumerate(["D6", "F#6", "A6", "D7"]):
        add(stems["fx"], pan(bell(midi(n(nm)), 3.5, vel=0.18), -0.4 + 0.27 * k), SNAP + 0.03 * k)
    ir = reverb_ir(2.8, 2.6, seed=3)
    for k in stems:
        mix = {"piano": 0.32, "pluck": 0.45, "pad": 0.25, "sub": 0.0, "perc": 0.12, "fx": 0.4}[k]
        if mix > 0:
            stems[k] = reverb(stems[k], mix, ir)
    stems["sub"] = lp(stems["sub"], 160)
    gains = {"piano": 1.0, "pluck": 0.9, "pad": 0.85, "sub": 0.8, "perc": 0.75, "fx": 0.9}
    music = sum(stems[k] * gains[k] for k in stems)
    # fade at the very end
    fade = np.clip((DUR - t_axis(DUR)) / 1.2, 0, 1)
    music *= fade[:, None]
    return music, stems


if __name__ == "__main__":
    import soundfile as sf
    out = ROOT / "build" / "audio"
    out.mkdir(parents=True, exist_ok=True)
    m, stems = render()
    sf.write(out / "music.wav", m.astype(np.float32), SR, subtype="FLOAT")
    print("music", m.shape, "peak", float(np.abs(m).max()), "grid0", round(GRID0, 3), "snap", SNAP)
