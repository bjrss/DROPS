"""SDF -> triangle mesh (marching cubes) with light smoothing."""
import numpy as np


def grid(lo, hi, step):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    n = np.ceil((hi - lo) / step).astype(int) + 1
    xs = [lo[i] + np.arange(n[i]) * step for i in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    return np.stack([X, Y, Z], -1).astype(np.float32), lo


def mesh_sdf(fn, lo, hi, step, chunk=None):
    """Evaluate fn on a grid and polygonise the zero level set."""
    from skimage.measure import marching_cubes

    P, lo = grid(lo, hi, step)
    D = np.empty(P.shape[:3], np.float32)
    nz = P.shape[2]
    if chunk is None:
        chunk = max(1, int(4e6 // (P.shape[0] * P.shape[1])))
    for k in range(0, nz, chunk):
        D[:, :, k:k + chunk] = fn(P[:, :, k:k + chunk].astype(np.float64))
    # make sure the border is outside so the surface is closed
    D[0, :, :] = D[-1, :, :] = step
    D[:, 0, :] = D[:, -1, :] = step
    D[:, :, 0] = D[:, :, -1] = step
    v, f, n, _ = marching_cubes(D, 0.0, spacing=(step, step, step), allow_degenerate=False)
    v = v + lo
    return v.astype(np.float64), f.astype(np.int32)


def laplacian(v, f, iters=2, lam=0.5):
    """Taubin-ish smoothing to remove marching-cubes stair steps."""
    n = len(v)
    import scipy.sparse as sp

    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    for _ in range(iters):
        avg = A @ v / deg[:, None]
        v = v + lam * (avg - v)
        avg = A @ v / deg[:, None]
        v = v - (lam + 0.03) * (avg - v) * 0.98
    return v
