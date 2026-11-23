"""Sound design: every visible movement has a sound.  All sounds are synthesised
here (no library samples, nothing that sounds like a drill, suction or a dental
instrument).  Cue times come from the same beat sheet as the picture.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audio"))
from synth import SR, add, bell, hp, lp, bp, midi, n, noise, pan, sine, sweep_noise, t_axis, reverb, reverb_ir, piano  # noqa
from om.timeline import B, CHAPTER_STEPS, CHAPTER_T0, CHAPTER_END, TRANSITIONS  # noqa: E402

DUR = 60.0


# ------------------------------------------------------------------ sound palette
def shimmer(dur=1.2, base="D6", vel=0.2, seed=0, notes=8):
    rng = np.random.default_rng(seed)
    out = np.zeros((int((dur + 1.5) * SR), 2))
    scale = ["D", "E", "F#", "A", "B"]
    for k in range(notes):
        t0 = k * dur / notes + rng.uniform(0, 0.03)
        nm = scale[rng.integers(0, 5)] + str(6 + (k % 2))
        add(out, pan(bell(midi(n(nm)), 1.2, vel=vel * (0.6 + 0.4 * rng.random())), rng.uniform(-0.6, 0.6)), t0)
    return out


def whoosh(dur=0.9, vel=0.22, up=True, seed=0):
    f0, f1 = (300, 2600) if up else (2400, 280)
    return pan(sweep_noise(dur, f0, f1, vel=vel, seed=seed, q=1.8), 0.0)


def swipe(vel=0.2, seed=0):
    s = sweep_noise(0.42, 5000, 1400, vel=vel, seed=seed, q=2.2)
    tick = sine(midi(n("A6")), 0.08, vel=0.05, a=0.001, r=0.06)
    out = np.zeros((int(0.6 * SR), 2))
    add(out, pan(s, 0.6), 0)
    add(out, pan(tick, 0.2), 0.36)
    return out


def tone(nm, vel=0.22, dur=1.4):
    return pan(bell(midi(n(nm)), dur, vel=vel, bright=0.4), 0.0)


def click(vel=0.5):
    """Clean, precise click (abutment seating) — short, bright, not metallic-dental."""
    N = int(0.08 * SR)
    tt = np.arange(N) / SR
    x = hp(noise(0.08, 9), 2500) * np.exp(-tt * 260) * 0.6
    x += np.sin(2 * np.pi * 1850 * tt) * np.exp(-tt * 90) * 0.5
    x += np.sin(2 * np.pi * 620 * tt) * np.exp(-tt * 60) * 0.35
    return pan(x * vel, 0.0)


def tock(vel=0.35, f=330):
    tt = t_axis(0.35)
    x = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 22) + 0.4 * np.sin(2 * np.pi * f * 2.7 * tt) * np.exp(-tt * 40)
    return pan(x * vel, 0.0)


def chime(vel=0.24):
    out = np.zeros((int(2.6 * SR), 2))
    add(out, pan(bell(midi(n("D6")), 2.2, vel=vel), -0.25), 0)
    add(out, pan(bell(midi(n("A6")), 2.2, vel=vel * 0.85), 0.25), 0.12)
    add(out, pan(bell(midi(n("F#7")), 1.6, vel=vel * 0.35), 0.0), 0.24)
    return out


def wave(vel=0.25):
    """Soft blue wave (anaesthesia): a warm low swell, no needle."""
    tt = t_axis(1.3)
    x = np.sin(2 * np.pi * 196 * tt + 0.8 * np.sin(2 * np.pi * 2.5 * tt)) * np.sin(np.pi * tt / 1.3) ** 2
    air = lp(noise(1.3, 4), 900) * np.sin(np.pi * tt / 1.3) ** 2 * 0.15
    return pan((x * 0.6 + air) * vel, 0.0)


def sparkles(dur=1.5, vel=0.12, seed=0, density=14, lo="A6"):
    rng = np.random.default_rng(seed)
    out = np.zeros((int((dur + 0.6) * SR), 2))
    for k in range(int(density * dur)):
        t0 = rng.uniform(0, dur)
        f = midi(n(lo)) * 2 ** (rng.integers(0, 12) / 12)
        add(out, pan(bell(f, 0.4, vel=vel * rng.uniform(0.4, 1.0)), rng.uniform(-0.7, 0.7)), t0)
    return out


def rise_tone(dur=0.8, f0=330, f1=660, vel=0.15):
    tt = t_axis(dur)
    f = f0 * (f1 / f0) ** (tt / dur)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.3 * np.sin(2 * ph)) * np.sin(np.pi * tt / dur) ** 1.5
    return pan(x * vel, 0.0)


def zip_line(dur=0.45, vel=0.14, seed=0):
    return pan(sweep_noise(dur, 1800, 6500, vel=vel, seed=seed, q=3.0), 0.2)


# ------------------------------------------------------------------ cue sheet
def cues():
    C = []

    def cue(t, name, sig, gain=1.0):
        C.append((t, name, sig, gain))

    # S1
    cue(0.0, "line glides in", whoosh(0.6, 0.12, seed=1))
    cue(B["logo.trace0"], "logo drawn (crystal shimmer)", shimmer(1.2, vel=0.16, seed=2, notes=9))
    for k in range(6):  # each implant thread lights up
        cue(B["logo.trace0"] + 0.86 + 0.055 * k, f"thread {k + 1}", pan(bell(midi(n("A6")) * 2 ** (k / 12), 0.5, vel=0.07), 0.3))
    cue(B["logo.trace0"] + 0.7, "wordmark", swipe(0.08, seed=3))
    cue(B["logo.detach"], "line detaches", rise_tone(0.5, 440, 880, 0.08))
    cue(B["tooth.in0"], "3D tooth materialises (scan)", pan(sweep_noise(0.8, 2500, 600, vel=0.1, seed=5), 0))
    cue(B["jaw.reveal0"], "jaw grows in", whoosh(1.0, 0.14, seed=6))
    cue(B["xray0"], "x-ray scan", pan(sweep_noise(0.75, 700, 4200, vel=0.1, seed=7), 0))
    cue(B["xray0"] + 0.15, "cut-away opens", whoosh(0.7, 0.1, up=False, seed=8))
    # chapters
    names = ["endo", "ciru", "impl", "rehab"]
    step_notes = ["F#5", "A5", "D6"]
    for k in names:
        t0 = CHAPTER_T0[k]
        cue(t0, f"{k}: capsule swipe", swipe(0.2, seed=int(t0)))
        st = CHAPTER_STEPS[k]
        for i in range(3):
            cue(st[i], f"{k}: step {i + 1} tone", tone(step_notes[i], 0.16))
        cue(st[3], f"{k}: result chime", chime(0.2))
        cue(CHAPTER_END[k] - 0.5, f"{k}: capsule out", swipe(0.1, seed=int(t0) + 1))
    for tt in TRANSITIONS:
        cue(tt, "diagonal bands whoosh", whoosh(0.95, 0.2, up=False, seed=int(tt * 10)))
    # endo
    cue(B["c1.problem"] + 0.3, "caries highlighted", pan(bell(midi(n("B4")), 1.0, vel=0.08), 0))
    cue(B["c1.nervio"] - 0.35, "pulp throbs", pan(lp(sine(70, 1.6, 0.25, a=0.1, r=0.6), 300), 0))
    cue(B["c1.s1"] + 0.1, "anaesthesia wave", wave(0.22))
    cue(B["c1.s2"], "line enters the canals", zip_line(0.5, 0.1, seed=11))
    for i in range(3):
        cue(B["c1.s2"] + 0.6 + i * 0.75, f"canal {i + 1}", rise_tone(0.5, 600, 300, 0.05))
    cue(B["c1.s2"] + 0.3, "pulp dissolves (particles)", sparkles(1.8, 0.07, seed=12, lo="E6"))
    cue(B["c1.irrig"], "irrigation glint", pan(sweep_noise(0.8, 3000, 9000, vel=0.07, seed=13), 0))
    for i in range(3):
        cue(B["c1.s3"] + 0.12 + 0.28 * i, f"gutta-percha fills canal {i + 1}", rise_tone(0.55, 260, 390, 0.07))
    cue(B["c1.s3"] + 1.05, "restoration closes", tock(0.18, 520))
    cue(B["c1.out"], "line travels to the wisdom tooth", whoosh(1.4, 0.16, seed=14))
    # surgery
    cue(B["c2.cut"], "cut-away opens", whoosh(0.7, 0.1, up=False, seed=15))
    cue(B["c2.problem"] + 0.2, "outline of the retained tooth", zip_line(1.0, 0.06, seed=16))
    cue(B["c2.juicio"] - 0.2, "contact pulse", pan(bell(midi(n("F#4")), 1.0, vel=0.06), 0))
    cue(B["c2.s1"] + 0.1, "anaesthesia wave", wave(0.22))
    cue(B["c2.s2"], "incision line", zip_line(0.45, 0.13, seed=17))
    cue(B["c2.s2"] + 0.35, "flap opens", whoosh(0.5, 0.09, seed=18))
    cue(B["c2.s2"] + 0.6, "bone window clears", sparkles(0.5, 0.06, seed=19, lo="D6"))
    cue(B["c2.s2"] + 0.95, "tooth rises out gently", rise_tone(1.2, 220, 440, 0.1))
    cue(B["c2.s3"], "flap returns", whoosh(0.5, 0.08, up=False, seed=20))
    for j in range(3):
        cue(B["c2.s3"] + 0.45 + 0.22 * j, f"stitch {j + 1}", pan(bell(midi(n("E6")) * 2 ** (j * 2 / 12), 0.6, vel=0.09), 0.2))
    cue(B["c2.s3"] + 0.75, "healing time-lapse", sparkles(1.4, 0.05, seed=21, lo="A5"))
    cue(B["c2.out"], "line crosses the arch", whoosh(1.8, 0.2, seed=22))
    # implant
    cue(B["c3.cut"], "cut-away opens", whoosh(0.7, 0.1, up=False, seed=23))
    cue(B["c3.problem"] + 0.3, "ghost tooth outline", zip_line(1.1, 0.05, seed=24))
    cue(B["c3.s1"], "glass radiograph appears", pan(bell(midi(n("B6")), 1.0, vel=0.12, bright=1.4), -0.3))
    cue(B["c3.s1"] + 0.4, "planning marker", pan(bell(midi(n("F#6")), 0.6, vel=0.08), -0.3))
    cue(B["c3.s2"], "line winds into the thread", rise_tone(1.0, 300, 900, 0.1))
    cue(B["c3.s2"] + 0.35, "implant settles", whoosh(1.2, 0.08, up=False, seed=25))
    cue(B["c3.s2"] + 1.2, "titanium cools", pan(sweep_noise(0.8, 5000, 2500, vel=0.04, seed=26), 0))
    cue(B["c3.s3"] + 0.1, "osseointegration sparkles", sparkles(1.6, 0.06, seed=27, lo="B5"))
    cue(B["c3.s3"] + 1.05, "healing screw seated", tock(0.12, 700))
    # rehab
    cue(B["c4.problem"] + 0.2, "ghost crown outline", zip_line(0.9, 0.05, seed=28))
    cue(B["c4.s1"], "healing screw removed", rise_tone(0.6, 500, 700, 0.06))
    cue(B["c4.s1"] + 0.6, "scan sweep", pan(sweep_noise(0.95, 900, 5200, vel=0.09, seed=29), 0))
    cue(B["c4.s1"] + 0.7, "scan points", sparkles(0.9, 0.04, seed=30, lo="D7"))
    cue(B["c4.s2"], "abutment descends", whoosh(0.5, 0.06, up=False, seed=31))
    cue(B["c4.click"], "abutment click", click(0.42))
    cue(B["c4.s2"] + 0.2, "crown made to measure (points)", sparkles(0.6, 0.05, seed=32, lo="A6"))
    cue(B["c4.click"] + 0.55, "crown descends", whoosh(1.0, 0.07, up=False, seed=33))
    cue(B["c4.seat"], "crown seats", tock(0.22, 380))
    cue(B["c4.s3"] + 0.05, "opposing arch closes", whoosh(1.0, 0.1, up=False, seed=34))
    cue(B["c4.s3"] + 1.1, "bite closes", tock(0.2, 240))
    for k in range(5):
        cue(B["c4.s3"] + 1.0 + 0.1 * k, f"contact point {k + 1}", pan(bell(midi(n("A6")), 0.35, vel=0.05), -0.4 + 0.2 * k))
    cue(B["c4.result"] - 0.1, "camera pulls back", whoosh(1.8, 0.12, seed=35))
    cue(B["c4.smile0"], "smile curve drawn", shimmer(1.0, base="A6", vel=0.08, seed=36, notes=6))
    # close
    cue(B["s6"] - 0.2, "line lifts to the logo", rise_tone(0.6, 440, 880, 0.08))
    cue(B["s6.snap"], "logo snap sparkle", shimmer(0.4, vel=0.12, seed=37, notes=5))
    for i in range(4):
        cue(B["s6.stack"] + 0.15 + 0.14 * i, f"treatment capsule {i + 1}", swipe(0.12, seed=40 + i))
    cue(B["s6.stack"] + 0.1, "doctor photo slides in", whoosh(0.8, 0.08, seed=45))
    cue(B["s6.cta"], "contact block appears", tock(0.12, 660))
    cue(B["s7"], "end plate", pan(bell(midi(n("D6")), 1.8, vel=0.1), 0))
    return C


def render():
    N = int(DUR * SR)
    buf = np.zeros((N, 2))
    sheet = cues()
    for t, name, sig, g in sheet:
        add(buf, sig * g, t)
    buf = reverb(buf, 0.22, reverb_ir(1.8, 1.6, seed=9))
    return buf, sheet


if __name__ == "__main__":
    import soundfile as sf
    out = ROOT / "build" / "audio"
    out.mkdir(parents=True, exist_ok=True)
    x, sheet = render()
    sf.write(out / "sfx.wav", x.astype(np.float32), SR, subtype="FLOAT")
    with open(out / "sfx_cues.txt", "w") as fh:
        for t, name, _, _ in sorted(sheet, key=lambda c: c[0]):
            fh.write(f"{t:7.3f}  {name}\n")
    print("sfx cues:", len(sheet), "peak", float(np.abs(x).max()))
