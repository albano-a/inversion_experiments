"""Interpola um horizonte (IL, XL, Y, X, Z) de 6 formas e compara em vista de mapa.

Metodos: Linear, Cubic, Nearest, RBF (thin-plate, smoothing=0, epsilon=None),
IDW (potencia 2) e Kriging ordinario (variograma esferico com auto-fit).

Uso:
    python horizon_interpolation.py [horizonte.txt] [--out saida] [--max-pixels 1000]
                                    [--mask-factor 5] [--workers N]

Sem arquivo de entrada usa a amostra embutida (SAMPLE). A amostra tem pontos
colineares, entao linear/cubic (triangulacao de Delaunay) retornam NaN; use o
horizonte completo para o resultado real.

Eficiencia:
  * os 6 metodos rodam em paralelo (um processo cada);
  * so se interpola nos pixels dentro da mascara de distancia (nao no bbox todo);
  * vizinhanca via cKDTree(workers=4); kriging resolvido em lotes (np.linalg.solve
    vetorizado) distribuidos em threads.

Saidas (em --out): horizon_<metodo>.npy (grade 2D, NaN fora da mascara),
grid_x.npy, grid_y.npy e horizon_comparison.png.
"""

import argparse
import io
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# BLAS single-thread: o paralelismo ja vem de processos + threads proprias.
# Sem isso, 6 processos x N threads BLAS estouram o limite do OpenBLAS.
# Precisa vir antes do import do numpy (os processos filhos herdam o ambiente).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

SAMPLE = """IL, XL, Y, X, Z
1436 1650 7561348.00000 412523.62500 2804.16602
1436 1651 7561348.00000 412536.12500 2803.66870
1436 1652 7561348.00000 412548.62500 2803.13281
1436 1653 7561348.00000 412561.12500 2802.58765
1436 1654 7561348.00000 412573.62500 2802.00757
1436 1655 7561348.00000 412586.12500 2801.39575
1436 1656 7561348.00000 412598.62500 2800.75073
1436 1657 7561348.00000 412611.12500 2800.12695
1436 1658 7561348.00000 412623.62500 2799.47656
1436 1659 7561348.00000 412636.12500 2798.79175
1436 1660 7561348.00000 412648.62500 2798.07690
"""

METHODS = [
    "Linear",
    "Cubic",
    "Nearest",
    "RBF (Thin-plate)",
    "IDW (p=2)",
    "Kriging (spherical)",
]


def slug(name):
    return name.split(" (")[0].lower().replace("-", "_")


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def read_horizon(path=None):
    """Retorna arrays (il, xl, y, x, z)."""
    if path:
        with open(path) as f:
            text = f.read()
    else:
        text = SAMPLE
    lines = [ln.replace(",", " ") for ln in text.splitlines() if ln.strip()]
    if not lines[0].split()[0].lstrip("-").replace(".", "", 1).isdigit():
        lines = lines[1:]  # cabecalho
    data = np.loadtxt(io.StringIO("\n".join(lines)), ndmin=2)
    il, xl, y, x, z = data[:, :5].T
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return il[ok], xl[ok], y[ok], x[ok], z[ok]


# ----------------------------------------------------------------------------
# Metodos (todos recebem xy (n,2) centrado, z (n,), q (m,2) e devolvem (m,))
# ----------------------------------------------------------------------------
def _chunks(m, size):
    return [slice(i, min(i + size, m)) for i in range(0, m, size)]


def _tri_interp(cls, xy, z, q):
    try:
        f = cls(xy, z)
    except Exception as e:  # QhullError em dados colineares / poucos pontos
        warnings.warn(f"{cls.__name__} falhou ({type(e).__name__}); retornando NaN")
        return np.full(len(q), np.nan)
    return f(q)


def interp_linear(xy, z, q):
    return _tri_interp(LinearNDInterpolator, xy, z, q)


def interp_cubic(xy, z, q):
    return _tri_interp(CloughTocher2DInterpolator, xy, z, q)


def interp_nearest(xy, z, q):
    return NearestNDInterpolator(xy, z)(q, workers=4)


def interp_rbf(xy, z, q):
    # thin-plate, smoothing=0, epsilon vazio (None). Com muitos pontos usa
    # vizinhanca local para evitar o sistema denso n x n.
    n = len(z)
    kw = {} if n <= 3000 else {"neighbors": 100}
    f = RBFInterpolator(
        xy, z, kernel="thin_plate_spline", smoothing=0.0, epsilon=None, **kw
    )
    out = np.empty(len(q))
    with ThreadPoolExecutor(max_workers=4) as ex:
        for sl, r in zip(
            _chunks(len(q), 20000), ex.map(lambda s: f(q[s]), _chunks(len(q), 20000))
        ):
            out[sl] = r
    return out


def interp_idw(xy, z, q, power=2.0, k=32):
    k = min(k, len(z))
    d, i = cKDTree(xy).query(q, k=k, workers=4)
    d = d.reshape(len(q), -1)
    i = i.reshape(len(q), -1)
    exact = d[:, 0] < 1e-12
    w = 1.0 / np.maximum(d, 1e-12) ** power
    out = (w * z[i]).sum(1) / w.sum(1)
    out[exact] = z[i[exact, 0]]
    return out


def _spherical(h, nugget, psill, rng):
    h = np.asarray(h, float)
    g = nugget + psill * np.where(h < rng, 1.5 * h / rng - 0.5 * (h / rng) ** 3, 1.0)
    return np.where(h == 0, 0.0, g)


def fit_spherical_variogram(xy, z, n_lags=20, max_pts=2000, seed=0):
    rs = np.random.default_rng(seed)
    idx = rs.choice(len(z), min(len(z), max_pts), replace=False)
    p, v = xy[idx], z[idx]
    h = pdist(p)
    g = 0.5 * pdist(v[:, None], "sqeuclidean")
    hmax = h.max() / 2.0
    edges = np.linspace(0, hmax, n_lags + 1)
    b = np.digitize(h, edges) - 1
    ok = (b >= 0) & (b < n_lags)
    cnt = np.bincount(b[ok], minlength=n_lags)
    lag = np.bincount(b[ok], h[ok], n_lags)
    sv = np.bincount(b[ok], g[ok], n_lags)
    m = cnt > 0
    lag, sv = lag[m] / cnt[m], sv[m] / cnt[m]
    var = max(np.var(z), 1e-12)
    p0 = [0.0, var, hmax / 2]
    try:
        popt, _ = curve_fit(
            _spherical,
            lag,
            sv,
            p0=p0,
            bounds=([0, 1e-12, hmax * 1e-3], [var * 2, var * 10, hmax * 4]),
        )
    except Exception:
        popt = p0
    return tuple(popt)


def interp_kriging(xy, z, q, k=24, chunk=10000):
    nug, ps, rng = fit_spherical_variogram(xy, z)
    k = min(k, len(z))
    tree = cKDTree(xy)
    reg = 1e-10 * (nug + ps)

    def solve(sl):
        qq = q[sl]
        d0, idx = tree.query(qq, k=k)
        d0, idx = d0.reshape(len(qq), -1), idx.reshape(len(qq), -1)
        p = xy[idx]  # (m,k,2)
        dij = np.linalg.norm(p[:, :, None] - p[:, None], axis=-1)  # (m,k,k)
        A = np.ones((len(qq), k + 1, k + 1))
        A[:, :k, :k] = _spherical(dij, nug, ps, rng) + reg * np.eye(k)
        A[:, k, k] = 0.0
        b = np.ones((len(qq), k + 1))
        b[:, :k] = _spherical(d0, nug, ps, rng)
        w = np.linalg.solve(A, b[..., None])[:, :k, 0]
        return (w * z[idx]).sum(1)

    out = np.empty(len(q))
    sls = _chunks(len(q), chunk)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for sl, r in zip(sls, ex.map(solve, sls)):
            out[sl] = r
    return out


FUNCS = {
    "Linear": interp_linear,
    "Cubic": interp_cubic,
    "Nearest": interp_nearest,
    "RBF (Thin-plate)": interp_rbf,
    "IDW (p=2)": interp_idw,
    "Kriging (spherical)": interp_kriging,
}


def run_method(name, xy, z, q):
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        res = FUNCS[name](xy, z, q)
    return name, res, time.perf_counter() - t


# ----------------------------------------------------------------------------
# Grade / mascara
# ----------------------------------------------------------------------------
def build_grid(x, y, max_pixels, mask_factor):
    xy = np.column_stack([x, y])
    d, _ = cKDTree(xy).query(xy, k=2)
    spacing = float(np.median(d[:, 1])) or 1.0
    pad = mask_factor * spacing
    x0, x1 = x.min() - pad, x.max() + pad
    y0, y1 = y.min() - pad, y.max() + pad
    step = max(spacing / 2, max(x1 - x0, y1 - y0) / max_pixels)
    gx = np.arange(x0, x1 + step, step)
    gy = np.arange(y0, y1 + step, step)
    return gx, gy, pad


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot(results, gx, gy, x, y, z, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin, vmax = np.percentile(z, [1, 99]) if len(z) > 50 else (z.min(), z.max())
    ext = [gx[0], gx[-1], gy[0], gy[-1]]
    fig, axs = plt.subplots(
        2, 3, figsize=(18, 11), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, name in zip(axs.ravel(), METHODS):
        im = ax.imshow(
            results[name],
            origin="lower",
            extent=ext,
            cmap="viridis_r",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        ax.scatter(x, y, s=2, c="k", alpha=0.4, linewidths=0)
        ax.set_title(name)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(alpha=0.3, lw=0.4)
    for ax in axs[1]:
        ax.set_xlabel("X (m)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Y (m)")
    fig.colorbar(im, ax=axs, shrink=0.7, label="Z")
    fig.suptitle("Horizonte - comparacao de interpolacoes (vista de mapa)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input", nargs="?", help="arquivo do horizonte (IL XL Y X Z); omitido = amostra"
    )
    ap.add_argument("--out", default="horizon_interp_out")
    ap.add_argument(
        "--max-pixels", type=int, default=1000, help="maximo de pixels no maior lado"
    )
    ap.add_argument(
        "--mask-factor",
        type=float,
        default=5.0,
        help="pixels a mais que isso x espacamento mediano da amostra viram NaN",
    )
    ap.add_argument("--workers", type=int, default=len(METHODS))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    il, xl, y, x, z = read_horizon(args.input)
    print(f"{len(z)} pontos lidos")

    gx, gy, max_dist = build_grid(x, y, args.max_pixels, args.mask_factor)
    GX, GY = np.meshgrid(gx, gy)
    ox, oy = x.mean(), y.mean()  # centraliza p/ estabilidade numerica
    xy = np.column_stack([x - ox, y - oy])
    qall = np.column_stack([GX.ravel() - ox, GY.ravel() - oy])

    dist, _ = cKDTree(xy).query(qall, workers=4)
    valid = dist <= max_dist
    q = qall[valid]
    print(f"grade {GX.shape[1]}x{GX.shape[0]}, {valid.sum()} pixels validos")

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_method, m, xy, z, q) for m in METHODS]
        for f in futs:
            name, res, dt = f.result()
            full = np.full(qall.shape[0], np.nan, dtype=np.float32)
            full[valid] = res
            results[name] = full.reshape(GX.shape)
            np.save(os.path.join(args.out, f"horizon_{slug(name)}.npy"), results[name])
            print(f"  {name:22s} {dt:7.2f}s")

    np.save(os.path.join(args.out, "grid_x.npy"), gx)
    np.save(os.path.join(args.out, "grid_y.npy"), gy)
    plot(results, gx, gy, x, y, z, os.path.join(args.out, "horizon_comparison.png"))
    print(f"Resultados salvos em {args.out}/")


if __name__ == "__main__":
    main()
