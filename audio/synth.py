"""Small numpy synthesis toolkit shared by music and SFX (48 kHz stereo)."""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 48000


def t_axis(dur):
    return np.arange(int(dur * SR)) / SR


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
        "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def n(name):
    """'F#4' -> midi number."""
    nm, octv = name[:-1], int(name[-1])
    return 12 * (octv + 1) + NOTE[nm]


def env_adsr(N, a, d, s, r, sr=SR):
    A, D, R = int(a * sr), int(d * sr), int(r * sr)
    S = max(N - A - D - R, 0)
    e = np.concatenate([np.linspace(0, 1, max(A, 1)), np.linspace(1, s, max(D, 1)), np.full(S, s),
                        np.linspace(s, 0, max(R, 1))])
    return e[:N] if len(e) >= N else np.pad(e, (0, N - len(e)))


def lp(x, fc, order=2):
    sos = butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def hp(x, fc, order=2):
    sos = butter(order, fc, "high", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def bp(x, lo, hi, order=2):
    sos = butter(order, [lo, min(hi, SR * 0.45)], "band", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def pan(x, p):
    """mono -> stereo, p in [-1, 1] (equal power)."""
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1)


def add(buf, sig, t0):
    i = int(round(t0 * SR))
    if i >= len(buf):
        return
    if i < 0:
        sig = sig[-i:]
        i = 0
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def reverb_ir(dur=2.6, decay=2.2, seed=0, bright=6000):
    rng = np.random.default_rng(seed)
    N = int(dur * SR)
    tt = np.arange(N) / SR
    ir = rng.normal(0, 1, (N, 2)) * np.exp(-tt * (6.9 / decay))[:, None]
    ir = lp(ir, bright)
    ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))[:, None]
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))
    return ir


def reverb(x, mix=0.25, ir=None):
    ir = reverb_ir() if ir is None else ir
    wet = np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)
    return x * (1 - mix) + wet * mix


def piano(f, dur, vel=0.6, bright=1.0):
    """Soft felt-piano-like tone: inharmonic partials, two-stage decay, hammer softened."""
    tt = t_axis(dur)
    B = 0.0004
    out = np.zeros_like(tt)
    for k in range(1, 9):
        fk = f * k * np.sqrt(1 + B * k * k)
        if fk > SR * 0.45:
            break
        amp = (1.0 / k ** (1.6 - 0.4 * bright)) * (0.9 if k % 2 else 1.0)
        dec = 0.9 + 3.2 / k
        out += amp * np.sin(2 * np.pi * fk * tt + k) * (0.7 * np.exp(-tt * dec * 1.6) + 0.3 * np.exp(-tt * dec * 0.35))
    att = np.clip(tt / 0.006, 0, 1)
    rel = np.clip((dur - tt) / 0.25, 0, 1)
    return out * att * rel * vel * 0.32


def pluck(f, dur, vel=0.5, damp=0.996, seed=0):
    """Karplus-Strong pluck."""
    N = int(dur * SR)
    L = max(2, int(SR / f))
    rng = np.random.default_rng(seed)
    buf = lp(rng.uniform(-1, 1, L), 5000, 1)
    out = np.empty(N)
    for i in range(N):
        v = buf[i % L]
        out[i] = v
        buf[i % L] = damp * 0.5 * (v + buf[(i + 1) % L])
    return out * vel * np.clip(np.arange(N) / (0.004 * SR), 0, 1)


def pad(freqs, dur, vel=0.3, cutoff=1800, attack=1.2, release=1.6, seed=0):
    tt = t_axis(dur)
    rng = np.random.default_rng(seed)
    L = np.zeros_like(tt)
    R = np.zeros_like(tt)
    for f in freqs:
        for det, side in ((-6, 0), (0, 1), (6, 0), (3, 1)):
            ff = f * 2 ** (det / 1200)
            ph = rng.uniform(0, 2 * np.pi)
            # soft saw (few harmonics)
            w = sum(np.sin(2 * np.pi * ff * k * tt + ph * k) / k ** 1.3 for k in range(1, 7))
            (L if side == 0 else R)[:] += w
    x = np.stack([L, R], 1) / (len(freqs) * 2)
    x = lp(x, cutoff, 2)
    e = env_adsr(len(tt), attack, 0.5, 0.9, release)
    return x * e[:, None] * vel


def sine(f, dur, vel=0.5, a=0.005, r=0.1):
    tt = t_axis(dur)
    return np.sin(2 * np.pi * f * tt) * env_adsr(len(tt), a, 0.01, 1.0, r) * vel


def bell(f, dur=1.6, vel=0.4, bright=1.0):
    tt = t_axis(dur)
    parts = [(1.0, 1.0, 1.4), (2.0, 0.45, 2.2), (2.76, 0.28, 3.0), (5.4, 0.12 * bright, 5.0), (8.9, 0.05 * bright, 7.0)]
    out = sum(a * np.sin(2 * np.pi * f * r * tt) * np.exp(-tt * d) for r, a, d in parts)
    return out * np.clip(tt / 0.002, 0, 1) * vel * 0.4


def noise(dur, seed=0):
    return np.random.default_rng(seed).normal(0, 1, int(dur * SR))


def sweep_noise(dur, f0, f1, vel=0.5, seed=0, q=1.6, shape="hann"):
    """Band-passed noise with a moving centre frequency (whooshes)."""
    x = noise(dur, seed)
    N = len(x)
    out = np.zeros(N)
    blocks = 64
    edges = np.linspace(0, N, blocks + 1).astype(int)
    fc = np.geomspace(f0, f1, blocks)
    zi = None
    for b in range(blocks):
        lo, hi = fc[b] / q, fc[b] * q
        sos = butter(2, [max(lo, 30), min(hi, SR * 0.45)], "band", fs=SR, output="sos")
        seg = x[edges[b]: edges[b + 1]]
        out[edges[b]: edges[b + 1]] = sosfilt(sos, seg)
    e = np.hanning(N) if shape == "hann" else np.minimum(np.linspace(0, 1, N) ** 2 * 1.0, 1) * np.linspace(1, 0, N) ** 0.3
    return out * e * vel


def sub_kick(vel=0.6):
    tt = t_axis(0.45)
    f = 45 + 55 * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-tt * 9) * vel


def shaker(vel=0.12, seed=0):
    x = hp(noise(0.09, seed), 6000)
    tt = t_axis(0.09)
    return x * np.exp(-tt * 55) * np.clip(tt / 0.004, 0, 1) * vel
