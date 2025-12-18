# ===========================
# NS Turbulence: PAPER-GRADE PIPELINE v2
# + IAAFT surrogate
# + spatial depth alternatives (rho / local_edge_mean / local_tet_vol_mean)
# ===========================
import io, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from scipy.spatial import Delaunay

# ---------------------------
# Constants
# ---------------------------
PHI = (1 + 5**0.5) / 2
PHI_INV = 1 / PHI
SQRT_2_3 = (2/3)**0.5

# ---------------------------
# Config (locked)
# ---------------------------
DEPTH_K = 7
TIME_DEPTH_METHOD = "log"
TIME_DEPTH_REVERSE = True

SPACE_DEPTH_METHOD = "quantile"
SPACE_DEPTH_REVERSE = False

FAMILIES = ["phi", "exp", "pow2", "linear", "none"]
ALPHAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

EDGE_WEIGHT_MODE = "min"
SCALE_MODE = "baseline_sqrt_2_3"

TRAIN_FRAC = 0.70
RNG_SEED_SPLIT = 42

N_PERM = 3000
N_SURR = 50

# IAAFT parameters
IAAFT_MAX_ITERS = 200
IAAFT_TOL = 1e-6

# ---------------------------
# Preprocess
# ---------------------------
def robust_center_scale(points, mode="global"):
    pts = points.astype(float).copy()
    pts -= np.mean(pts, axis=0, keepdims=True)
    if mode == "none":
        return pts
    if mode == "global":
        r = np.linalg.norm(pts, axis=1)
        s = np.sqrt(np.mean(r**2)) if r.size else 1.0
        if (not np.isfinite(s)) or (s <= 0):
            s = 1.0
        return pts / s
    raise ValueError("Unknown mode")

# ---------------------------
# Delaunay helpers
# ---------------------------
def delaunay_edges(points):
    tri = Delaunay(points)
    simplices = tri.simplices
    m = simplices.shape[1]  # 4 in 3D, 3 in 2D
    edges_list = []
    for a in range(m):
        for b in range(a+1, m):
            edges_list.append(np.sort(simplices[:, [a, b]], axis=1))
    edges = np.unique(np.vstack(edges_list), axis=0)
    return tri, edges

def dt_radii(points, edges, weights=None, edge_weight_mode="min"):
    n = points.shape[0]
    L = np.linalg.norm(points[edges[:,0]] - points[edges[:,1]], axis=1)

    if weights is None:
        inc = [[] for _ in range(n)]
        for (u,v), l in zip(edges, L):
            inc[u].append(l); inc[v].append(l)
        return np.array([np.mean(lst) if lst else np.nan for lst in inc], dtype=float)

    w = np.asarray(weights, dtype=float)
    if w.shape[0] != n:
        raise ValueError(f"weights length {w.shape[0]} != n {n}")
    w = np.clip(w, 0.0, np.inf)

    num = np.zeros(n, dtype=float)
    den = np.zeros(n, dtype=float)

    for (u,v), l in zip(edges, L):
        wu, wv = w[u], w[v]
        if edge_weight_mode == "min":
            we = wu if wu < wv else wv
        elif edge_weight_mode == "geom":
            we = math.sqrt(wu*wv)
        elif edge_weight_mode == "mean":
            we = 0.5*(wu+wv)
        else:
            raise ValueError("Unknown edge_weight_mode")

        if we > 0 and np.isfinite(we):
            num[u] += we*l; den[u] += we
            num[v] += we*l; den[v] += we

    return np.where(den > 0, num/den, np.nan)

def tetra_volumes(points, simplices):
    """
    Volume of tetrahedron (a,b,c,d):
      V = |det(b-a, c-a, d-a)| / 6
    """
    A = points[simplices[:,0]]
    B = points[simplices[:,1]]
    C = points[simplices[:,2]]
    D = points[simplices[:,3]]
    M = np.stack([B-A, C-A, D-A], axis=1)  # (T,3,3)
    det = np.linalg.det(M)
    return np.abs(det) / 6.0

def per_vertex_mean_incident_tet_volume(points, tri):
    n = points.shape[0]
    simplices = tri.simplices
    if simplices.shape[1] != 4:
        # if data is effectively 2D, fallback to NaNs
        return np.full(n, np.nan, dtype=float)

    vols = tetra_volumes(points, simplices)
    acc = np.zeros(n, dtype=float)
    cnt = np.zeros(n, dtype=int)
    for t, tet in enumerate(simplices):
        v = vols[t]
        for u in tet:
            acc[u] += v
            cnt[u] += 1
    out = np.where(cnt > 0, acc/cnt, np.nan)
    return out

# ---------------------------
# Depth (locked)
# ---------------------------
def depth_from_variable(v, K=7, method="quantile", reverse=False):
    v = np.asarray(v, dtype=float)
    mask = np.isfinite(v)
    vc = v[mask]
    if vc.size == 0:
        return np.zeros_like(v)

    if method == "quantile":
        edges = np.quantile(vc, np.linspace(0,1,K+1))
    elif method == "log":
        vv = vc - np.min(vc)
        vv = np.log(vv + 1.0)
        edges_log = np.linspace(np.min(vv), np.max(vv), K+1)
        edges = np.exp(edges_log) + np.min(vc) - 1.0
    elif method == "linear":
        edges = np.linspace(np.min(vc), np.max(vc), K+1)
    else:
        raise ValueError("Unknown method")

    edges = np.unique(edges)
    if edges.size <= 1:
        bins = np.zeros_like(v, dtype=int)
    else:
        bins = np.digitize(v, edges[1:-1]) if edges.size > 2 else np.zeros_like(v, dtype=int)
        bins = np.clip(bins, 0, K-1)

    depth = bins.astype(float)
    denom = max(1, int(np.max(depth[np.isfinite(depth)])))
    depth = depth / denom
    if reverse:
        depth = 1.0 - depth
    return depth

# ---------------------------
# Weight families
# ---------------------------
def weights_from_depth(depth, family="phi", alpha=1.0):
    d = np.asarray(depth, dtype=float)
    if family == "none":
        return np.ones_like(d)
    if family == "phi":
        return PHI ** (-alpha * d)
    if family == "exp":
        return np.exp(-alpha * d)
    if family == "pow2":
        return 2.0 ** (-alpha * d)
    if family == "linear":
        return np.clip(1.0 - alpha*d, 0.0, np.inf)
    raise ValueError("Unknown family")

# ---------------------------
# Eval
# ---------------------------
def summarize(valid):
    return dict(
        n=int(valid.size),
        median=float(np.median(valid)),
        q25=float(np.quantile(valid, 0.25)),
        q75=float(np.quantile(valid, 0.75)),
        mean=float(np.mean(valid)),
        std=float(np.std(valid, ddof=1)) if valid.size > 1 else float("nan")
    )

def eval_on_subset(points_sub, depth_sub, family, alpha):
    tri, edges = delaunay_edges(points_sub)
    r_base = dt_radii(points_sub, edges, weights=None)

    w = None if family == "none" else weights_from_depth(depth_sub, family=family, alpha=alpha)
    r = dt_radii(points_sub, edges, weights=w, edge_weight_mode=EDGE_WEIGHT_MODE)

    vb = r_base[np.isfinite(r_base)]
    v = r[np.isfinite(r)]
    if v.size < 10 or vb.size < 10:
        return None

    if SCALE_MODE == "baseline_sqrt_2_3":
        scale = SQRT_2_3 / np.median(vb)
        r = r * scale
        v = r[np.isfinite(r)]
        if v.size < 10:
            return None

    st = summarize(v)
    st["err_phi"] = abs(st["median"] - PHI_INV)
    st["err_sqrt_2_3"] = abs(st["median"] - SQRT_2_3)
    st["family"] = family
    st["alpha"] = float(alpha)
    st["edges"] = int(edges.shape[0])
    return st

def grid_eval(points_sub, depth_sub, label):
    rows = []
    for fam in FAMILIES:
        for a in ALPHAS:
            st = eval_on_subset(points_sub, depth_sub, fam, a)
            if st is None:
                continue
            st = dict(st)
            st["label"] = label
            rows.append(st)
    return pd.DataFrame(rows)

def select_best(df, target="phi"):
    if df.empty:
        return None
    key = "err_phi" if target == "phi" else "err_sqrt_2_3"
    return df.sort_values(key).iloc[0].to_dict()

def permutation_test(points_test, depth_test, family, alpha, target_value=PHI_INV, n_perm=N_PERM):
    base = eval_on_subset(points_test, depth_test, family, alpha)
    if base is None:
        return None
    base_err = abs(base["median"] - target_value)

    rng = np.random.default_rng(0)
    errs = []
    count = 0
    for _ in range(n_perm):
        d = depth_test.copy()
        rng.shuffle(d)
        st = eval_on_subset(points_test, d, family, alpha)
        if st is None:
            continue
        e = abs(st["median"] - target_value)
        errs.append(e)
        if e <= base_err:
            count += 1
    errs = np.asarray(errs, dtype=float)
    p = (count + 1) / (errs.size + 1)
    return dict(base=base, base_err=base_err, p_value=p, perm_errs=errs)

# ---------------------------
# Surrogates: Gaussian / Phase / IAAFT
# ---------------------------
def gaussian_surrogate(x, rng):
    x = np.asarray(x, dtype=float)
    mu, sig = np.mean(x), np.std(x, ddof=0)
    if sig <= 0 or not np.isfinite(sig):
        return np.full_like(x, mu)
    return rng.normal(mu, sig, size=x.shape[0])

def phase_randomized_surrogate(x, rng):
    x = np.asarray(x, dtype=float)
    x0 = x - np.mean(x)
    n = x0.size
    if n < 8:
        return x.copy()

    X = np.fft.rfft(x0)
    mag = np.abs(X)
    phase = rng.uniform(0, 2*np.pi, size=X.shape)
    phase[0] = 0.0
    if (n % 2) == 0:
        phase[-1] = 0.0

    Y = mag * np.exp(1j*phase)
    y = np.fft.irfft(Y, n=n)

    # restore mean/var
    y = y + np.mean(x)
    if np.std(y) > 1e-12:
        y = (y - np.mean(y)) / np.std(y) * np.std(x) + np.mean(x)
    return y

def iaaft_surrogate(x, rng, max_iters=IAAFT_MAX_ITERS, tol=IAAFT_TOL):
    """
    IAAFT: preserve amplitude distribution + power spectrum approximately.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 16:
        return x.copy()

    x_mean = np.mean(x)
    x0 = x - x_mean

    # target distribution (sorted values)
    x_sorted = np.sort(x0)

    # target spectrum magnitudes
    X = np.fft.rfft(x0)
    mag = np.abs(X)

    # init with phase randomized
    y = phase_randomized_surrogate(x0, rng)
    y = y - np.mean(y)

    prev = None
    for _ in range(max_iters):
        # enforce spectrum
        Y = np.fft.rfft(y)
        phase = np.angle(Y)
        Y2 = mag * np.exp(1j*phase)
        y_spec = np.fft.irfft(Y2, n=n)

        # enforce distribution via rank ordering
        ranks = np.argsort(np.argsort(y_spec))
        y_new = x_sorted[ranks]

        if prev is not None:
            rel = np.linalg.norm(y_new - prev) / (np.linalg.norm(prev) + 1e-12)
            if rel < tol:
                y = y_new
                break
        prev = y_new
        y = y_new

    y = y + x_mean
    # match mean/var exactly
    if np.std(y) > 1e-12 and np.std(x) > 1e-12:
        y = (y - np.mean(y)) / np.std(y) * np.std(x) + np.mean(x)
    else:
        y = np.full_like(y, np.mean(x))
    return y

def run_surrogates_time(kind, pts_raw, pts_proc, train_idx, test_idx, rng_seed=123):
    rng = np.random.default_rng(rng_seed)
    time = pts_raw[:, 0].astype(float)

    rows = []
    for s in range(N_SURR):
        if kind == "gaussian":
            t_s = gaussian_surrogate(time, rng)
        elif kind == "phase":
            t_s = phase_randomized_surrogate(time, rng)
        elif kind == "iaaft":
            t_s = iaaft_surrogate(time, rng)
        else:
            raise ValueError("unknown surrogate kind")

        depth_time_s = depth_from_variable(t_s, K=DEPTH_K, method=TIME_DEPTH_METHOD, reverse=TIME_DEPTH_REVERSE)

        tr_df = grid_eval(pts_proc[train_idx], depth_time_s[train_idx], label=f"time_train_{kind}_{s}")
        best = select_best(tr_df, target="phi")
        if best is None:
            continue

        te = eval_on_subset(pts_proc[test_idx], depth_time_s[test_idx], best["family"], best["alpha"])
        if te is None:
            continue

        rows.append(dict(
            surrogate=kind,
            surr_id=s,
            best_family=best["family"],
            best_alpha=best["alpha"],
            test_median=te["median"],
            test_err_phi=te["err_phi"],
            test_n=te["n"],
        ))
    return pd.DataFrame(rows)

# ---------------------------
# Spatial depth alternatives
# ---------------------------
def compute_spatial_depths(points_proc):
    """
    Returns dict[name] = depth array (0..1)
    Depth variables:
      - rho: distance to median center
      - local_edge_mean: unweighted DT per-vertex incident edge mean
      - local_tet_vol_mean: per-vertex mean incident tetra volume
    """
    center = np.median(points_proc, axis=0)
    rho = np.linalg.norm(points_proc - center, axis=1)

    tri, edges = delaunay_edges(points_proc)
    local_edge_mean = dt_radii(points_proc, edges, weights=None)

    local_tet_vol_mean = per_vertex_mean_incident_tet_volume(points_proc, tri)

    depths = {}
    depths["rho"] = depth_from_variable(rho, K=DEPTH_K, method=SPACE_DEPTH_METHOD, reverse=SPACE_DEPTH_REVERSE)
    depths["local_edge_mean"] = depth_from_variable(local_edge_mean, K=DEPTH_K, method=SPACE_DEPTH_METHOD, reverse=SPACE_DEPTH_REVERSE)
    depths["local_tet_vol_mean"] = depth_from_variable(local_tet_vol_mean, K=DEPTH_K, method=SPACE_DEPTH_METHOD, reverse=SPACE_DEPTH_REVERSE)
    return depths

# ---------------------------
# Plot helpers
# ---------------------------
def empirical_p(real_err, arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.nan
    return (np.sum(arr <= real_err) + 1) / (arr.size + 1)

def plot_hist_and_cdf(real_err, surr_errs_dict, title):
    plt.figure()
    for name, arr in surr_errs_dict.items():
        arr = np.asarray(arr, dtype=float)
        plt.hist(arr, bins=20, alpha=0.5, label=f"{name} (n={len(arr)})", density=True)
    plt.axvline(real_err, linewidth=2, label=f"real err={real_err:.4f}")
    plt.title(title + " — Err_phi histogram")
    plt.xlabel("err_phi")
    plt.ylabel("density")
    plt.legend()
    plt.show()

    plt.figure()
    for name, arr in surr_errs_dict.items():
        arr = np.sort(np.asarray(arr, dtype=float))
        if arr.size == 0:
            continue
        y = np.arange(1, arr.size+1) / arr.size
        plt.plot(arr, y, label=name)
    plt.axvline(real_err, linewidth=2, label="real")
    plt.title(title + " — Err_phi CDF")
    plt.xlabel("err_phi")
    plt.ylabel("CDF")
    plt.legend()
    plt.show()

# ---------------------------
# Main
# ---------------------------
def main():
    print("="*72)
    print(" NS Turbulence: PIPELINE v2 (IAAFT + spatial depth alternatives)")
    print("="*72)

    uploaded = files.upload()
    fname = next(iter(uploaded.keys()))
    df = pd.read_csv(io.StringIO(uploaded[fname].decode("utf-8")),
                     delim_whitespace=True, comment="#", header=None)

    pts_raw = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
    print(f"\n📊 Raw data shape: {pts_raw.shape}")

    pts = robust_center_scale(pts_raw, mode="global")

    # Train/test split
    n = pts.shape[0]
    rng = np.random.default_rng(RNG_SEED_SPLIT)
    idx = np.arange(n); rng.shuffle(idx)
    split = int(TRAIN_FRAC*n)
    train_idx, test_idx = idx[:split], idx[split:]
    print(f"\n🧪 Split: train={train_idx.size}, test={test_idx.size}")

    # TIME depth (real)
    time_values = pts_raw[:, 0]
    depth_time = depth_from_variable(time_values, K=DEPTH_K, method=TIME_DEPTH_METHOD, reverse=TIME_DEPTH_REVERSE)

    # SPACE depths (3 alternatives computed once from full point cloud)
    space_depths = compute_spatial_depths(pts)
    print("\n📍 Spatial depth alternatives:", list(space_depths.keys()))

    # ---------------------------
    # REAL: TIME
    # ---------------------------
    print("\n" + "="*72)
    print("REAL: TIME (train grid -> select -> test eval -> permutation)")
    time_train_df = grid_eval(pts[train_idx], depth_time[train_idx], label="time_train_real")
    best_time = select_best(time_train_df, target="phi")
    print("Best TIME (train):", best_time)

    time_test_best = eval_on_subset(pts[test_idx], depth_time[test_idx], best_time["family"], best_time["alpha"])
    print("TIME TEST (best):", time_test_best)

    perm_time = permutation_test(pts[test_idx], depth_time[test_idx], best_time["family"], best_time["alpha"])
    print(f"TIME permutation (TEST): err={perm_time['base_err']:.6f}, p≈{perm_time['p_value']:.5f}")

    # ---------------------------
    # REAL: SPACE (alternatives)
    # ---------------------------
    space_rows = []
    print("\n" + "="*72)
    print("REAL: SPACE alternatives (each: train select -> test eval -> permutation)")
    for name, dspace in space_depths.items():
        tr_df = grid_eval(pts[train_idx], dspace[train_idx], label=f"space_train_real_{name}")
        best = select_best(tr_df, target="phi")
        te = eval_on_subset(pts[test_idx], dspace[test_idx], best["family"], best["alpha"])
        perm = permutation_test(pts[test_idx], dspace[test_idx], best["family"], best["alpha"])
        print(f"\n[SPACE={name}] Best(train) family={best['family']} alpha={best['alpha']}")
        print(f"  TEST median={te['median']:.6f}, err_phi={te['err_phi']:.6f}, perm_p≈{perm['p_value']:.5f}")
        space_rows.append(dict(
            space_depth=name,
            best_family=best["family"],
            best_alpha=best["alpha"],
            test_median=te["median"],
            test_err_phi=te["err_phi"],
            perm_p=perm["p_value"]
        ))

    space_summary = pd.DataFrame(space_rows).sort_values("test_err_phi")
    print("\nSPACE SUMMARY (sorted by test_err_phi):")
    print(space_summary)

    # Save real grids
    grid_csv = "grid_train_real_time.csv"
    time_train_df.to_csv(grid_csv, index=False)
    print(f"\n✅ Saved: {grid_csv}")

    space_csv = "space_summary_real.csv"
    space_summary.to_csv(space_csv, index=False)
    print(f"✅ Saved: {space_csv}")

    # ---------------------------
    # SURROGATES (TIME): gaussian / phase / iaaft
    # ---------------------------
    print("\n" + "="*72)
    print(f"SURROGATES (TIME): {N_SURR} Gaussian + {N_SURR} Phase + {N_SURR} IAAFT")
    surr_gauss = run_surrogates_time("gaussian", pts_raw, pts, train_idx, test_idx, rng_seed=111)
    surr_phase = run_surrogates_time("phase", pts_raw, pts, train_idx, test_idx, rng_seed=222)
    surr_iaaft = run_surrogates_time("iaaft", pts_raw, pts, train_idx, test_idx, rng_seed=333)

    surr_all = pd.concat([surr_gauss, surr_phase, surr_iaaft], ignore_index=True)
    surr_csv = "surrogates_time_results_v2.csv"
    surr_all.to_csv(surr_csv, index=False)
    print(f"✅ Saved: {surr_csv}")

    # Empirical p-values (real time vs each surrogate family)
    real_time_err = time_test_best["err_phi"]
    p_gauss = empirical_p(real_time_err, surr_gauss["test_err_phi"].dropna().to_numpy())
    p_phase = empirical_p(real_time_err, surr_phase["test_err_phi"].dropna().to_numpy())
    p_iaaft = empirical_p(real_time_err, surr_iaaft["test_err_phi"].dropna().to_numpy())

    print("\n" + "="*72)
    print("SURROGATE COMPARISON (TIME, TEST)")
    print(f"Real TIME err_phi = {real_time_err:.6f}")
    print(f"Empirical p vs Gaussian: {p_gauss:.5f}")
    print(f"Empirical p vs Phase:   {p_phase:.5f}")
    print(f"Empirical p vs IAAFT:   {p_iaaft:.5f}")

    plot_hist_and_cdf(real_time_err,
                      {"gaussian": surr_gauss["test_err_phi"].dropna().to_numpy(),
                       "phase": surr_phase["test_err_phi"].dropna().to_numpy(),
                       "iaaft": surr_iaaft["test_err_phi"].dropna().to_numpy()},
                      title="TIME (TEST): real vs surrogates (incl. IAAFT)")

    # Final summary
    summary = pd.DataFrame([
        dict(embedding="time",
             best_family=best_time["family"],
             best_alpha=best_time["alpha"],
             test_median=time_test_best["median"],
             test_err_phi=time_test_best["err_phi"],
             perm_p=perm_time["p_value"],
             surr_p_gauss=p_gauss,
             surr_p_phase=p_phase,
             surr_p_iaaft=p_iaaft),
    ])
    summary_csv = "summary_time_real_v2.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\n✅ Saved: {summary_csv}")
    print("\nSUMMARY (TIME, TEST ONLY):")
    print(summary)

    # Download outputs
    print("\nDownload outputs:")
    for fn in [grid_csv, space_csv, surr_csv, summary_csv]:
        try:
            files.download(fn)
        except Exception:
            pass

if __name__ == "__main__":
    main()

