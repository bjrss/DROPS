"""Easing / interpolation helpers shared by Blender, compositor and audio."""
import math

import numpy as np


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lin(t, t0, t1):
    """0..1 progress of t across [t0, t1] (clamped)."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return clamp((t - t0) / (t1 - t0))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def in_out_sine(x):
    x = clamp(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def in_out_cubic(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def in_out_quint(x):
    x = clamp(x)
    return 16 * x ** 5 if x < 0.5 else 1 - (-2 * x + 2) ** 5 / 2


def in_out_expo(x):
    x = clamp(x)
    if x in (0.0, 1.0):
        return x
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


def out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def out_quint(x):
    x = clamp(x)
    return 1 - (1 - x) ** 5


def out_expo(x):
    x = clamp(x)
    return 1.0 if x == 1 else 1 - 2 ** (-10 * x)


def in_cubic(x):
    x = clamp(x)
    return x ** 3


def out_back(x, s=1.2):
    x = clamp(x)
    c3 = s + 1
    return 1 + c3 * (x - 1) ** 3 + s * (x - 1) ** 2


def settle(x, overshoot=0.06, freq=1.6):
    """Ease-out with one gentle overshoot (for capsules settling)."""
    x = clamp(x)
    base = out_cubic(x)
    return base + overshoot * math.sin(math.pi * freq * x) * (1 - x) ** 2


def window(t, t0, t1, fin=0.3, fout=0.3, ease=smooth):
    """1 inside [t0, t1] with eased ramps of length fin/fout (outside)."""
    a = ease(lin(t, t0 - fin, t0)) if fin > 0 else (1.0 if t >= t0 else 0.0)
    b = 1 - ease(lin(t, t1, t1 + fout)) if fout > 0 else (1.0 if t <= t1 else 0.0)
    return min(a, b)


def mix(a, b, x):
    return a + (b - a) * x


def mixv(a, b, x):
    return tuple(ai + (bi - ai) * x for ai, bi in zip(a, b))


def keyed(t, keys, ease=in_out_cubic):
    """Piecewise interpolation through [(t, value), ...] with an easing per span.
    value can be float or tuple/np.array."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (ta, va), (tb, vb) in zip(keys[:-1], keys[1:]):
        if t <= tb:
            x = ease(lin(t, ta, tb))
            if isinstance(va, (tuple, list, np.ndarray)):
                return tuple(np.asarray(va, float) + (np.asarray(vb, float) - np.asarray(va, float)) * x)
            return va + (vb - va) * x
    return keys[-1][1]


def noise1(t, seed=0.0, octaves=3):
    """Smooth deterministic 1D noise in [-1, 1] (sum of sines) — for drift."""
    v = 0.0
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        f = (0.37 + 0.21 * o) * (1.9 ** o)
        v += amp * math.sin(t * f * 2 * math.pi * 0.25 + seed * (1.7 + o) + o * 1.3)
        tot += amp
        amp *= 0.5
    return v / tot
