# ============================================
# GrammarTree 9.1 - τ_ij (Reynolds-stress prototype)
# 2D NS + synthetic τ_ij teacher
# Grid: 32x32, Nt=1000 (T=0.1)
# 标量 ν_t → 张量 τ_ij 基函数 [S, I1*S, δ I1]
# ============================================

import os, math
os.environ["PYTHONHASHSEED"] = "0"
os.environ["JULIA_NUM_THREADS"] = "4"

!pip install -q numpy sympy matplotlib pysr torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from pysr import PySRRegressor  # 只在 DO_PYSR=True 时用

%matplotlib inline
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["figure.dpi"] = 120

DO_PYSR = False          # 如需 PySR 审计，把它改成 True
CORE_MODE = "core+residual"  # core-only / residual-only / core+residual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("[Init] Using device:", device)
print("[Init] Random seed set to", SEED)
print("[Init] GrammarTree 9.1 τ_ij header OK.\n")

# -------------------------
# 1. Grid & helpers
# -------------------------
Nx, Ny = 32, 32
x = torch.linspace(0.0, 1.0, Nx, dtype=DTYPE)
y = torch.linspace(0.0, 1.0, Ny, dtype=DTYPE)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
dt = 1e-4
Nt = 1000
T = Nt * dt

print(f"[Grid] Nx={Nx}, Ny={Ny}, dx={dx:.6f}, dy={dy:.6f}, dt={dt:.1e}, Nt={Nt}, T={T:.3f}")

X, Y = torch.meshgrid(x, y, indexing="ij")
def to_dev(t):
    return t.to(device=device, dtype=DTYPE)

X = to_dev(X)
Y = to_dev(Y)

def roll2d(u, sx, sy):
    return torch.roll(torch.roll(u, shifts=sx, dims=-2), shifts=sy, dims=-1)

def grad2d(u, dx, dy):
    dudx = (roll2d(u, -1, 0) - roll2d(u,  1, 0)) / (2.0 * dx)
    dudy = (roll2d(u,  0,-1) - roll2d(u,  0, 1)) / (2.0 * dy)
    return dudx, dudy

def laplace2d(u, dx, dy):
    u_xx = (roll2d(u, -1, 0) - 2.0*u + roll2d(u, 1, 0)) / (dx*dx)
    u_yy = (roll2d(u,  0,-1) - 2.0*u + roll2d(u, 0, 1)) / (dy*dy)
    return u_xx + u_yy

def divergence(v, dx, dy):
    vx, vy = v[0], v[1]
    dvx_dx, _      = grad2d(vx, dx, dy)
    _,      dvy_dy = grad2d(vy, dx, dy)
    return dvx_dx + dvy_dy

# -------------------------
# 2. SCCT-like stats for NS
# -------------------------
def scct_stats_vec_ns(v):
    vx, vy = v[0], v[1]
    mag2 = vx*vx + vy*vy
    mag  = torch.sqrt(mag2 + 1e-12)

    phi2 = torch.mean(mag2)
    phi4 = torch.mean(mag2*mag2)
    M    = torch.mean(vx)

    flat = mag.reshape(-1)
    vmax = torch.max(flat)
    eps  = 1e-12
    if vmax < eps:
        H = torch.tensor(0.0, dtype=DTYPE, device=device)
    else:
        normed = flat / (vmax + eps)
        hist   = torch.histc(normed, bins=64, min=0.0, max=1.0)
        probs  = hist / (torch.sum(hist) + eps)
        H      = -torch.sum(probs * torch.log(probs + eps))

    div = divergence(v, dx, dy)
    div_rms = torch.sqrt(torch.mean(div*div))
    return phi2, H, M, phi4, div_rms

# -------------------------
# 3. Teacher PDE: NS + synthetic τ_ij
#    core: ν_mol Δv
#    τ_ij^teacher = -2 ν_t(I1) S_ij, ν_t clamp 防止爆炸
# -------------------------
nu_mol      = 0.05    # 分子粘性 (core)
nu0_turb    = 0.002   # 更温和一点的涡粘性
nu1_turb    = 0.001
nu_t_maxcap = 0.03    # 最大涡粘性上限，防止过大导致数值不稳定

def compute_S_and_I1(v):
    vx, vy = v[0], v[1]
    dvx_dx, dvx_dy = grad2d(vx, dx, dy)
    dvy_dx, dvy_dy = grad2d(vy, dx, dy)

    Sxx = dvx_dx
    Syy = dvy_dy
    Sxy = 0.5 * (dvx_dy + dvy_dx)
    Syx = Sxy

    I1 = Sxx*Sxx + 2.0*Sxy*Syx + Syy*Syy
    return Sxx, Sxy, Syx, Syy, I1

def tau_teacher(v):
    Sxx, Sxy, Syx, Syy, I1 = compute_S_and_I1(v)
    nu_t = nu0_turb + nu1_turb * I1
    nu_t = torch.clamp(nu_t, 0.0, nu_t_maxcap)  # 防止过大导致爆掉
    coeff = -2.0 * nu_t
    tau_xx = coeff * Sxx
    tau_xy = coeff * Sxy
    tau_yx = coeff * Syx
    tau_yy = coeff * Syy
    return tau_xx, tau_xy, tau_yx, tau_yy

def div_tau_from_components(tau_xx, tau_xy, tau_yx, tau_yy):
    dtxx_dx, _  = grad2d(tau_xx, dx, dy)
    _, dtxy_dy  = grad2d(tau_xy, dx, dy)
    dtyx_dx, _  = grad2d(tau_yx, dx, dy)
    _, dtyy_dy  = grad2d(tau_yy, dx, dy)
    div_x = dtxx_dx + dtxy_dy
    div_y = dtyx_dx + dtyy_dy
    return torch.stack([div_x, div_y], dim=0)

def teacher_rhs(v):
    vx, vy = v[0], v[1]
    dvx_dx, dvx_dy = grad2d(vx, dx, dy)
    dvy_dx, dvy_dy = grad2d(vy, dx, dy)

    adv_x = vx*dvx_dx + vy*dvx_dy
    adv_y = vx*dvy_dx + vy*dvy_dy

    lap_vx = laplace2d(vx, dx, dy)
    lap_vy = laplace2d(vy, dx, dy)

    tau_xx, tau_xy, tau_yx, tau_yy = tau_teacher(v)
    div_tau = div_tau_from_components(tau_xx, tau_xy, tau_yx, tau_yy)

    rhs_x = -adv_x + nu_mol * lap_vx + div_tau[0]
    rhs_y = -adv_y + nu_mol * lap_vy + div_tau[1]
    return torch.stack([rhs_x, rhs_y], dim=0)

def core_rhs(v):
    vx, vy = v[0], v[1]
    dvx_dx, dvx_dy = grad2d(vx, dx, dy)
    dvy_dx, dvy_dy = grad2d(vy, dx, dy)

    adv_x = vx*dvx_dx + vy*dvx_dy
    adv_y = vx*dvy_dx + vy*dvy_dy

    lap_vx = laplace2d(vx, dx, dy)
    lap_vy = laplace2d(vy, dx, dy)

    rhs_x = -adv_x + nu_mol * lap_vx
    rhs_y = -adv_y + nu_mol * lap_vy
    return torch.stack([rhs_x, rhs_y], dim=0)

def euler_step(v, rhs_fun):
    return v + dt * rhs_fun(v)

def simulate_teacher(v0, Nt):
    v = v0.clone()
    traj  = torch.zeros((Nt, 2, Nx, Ny), dtype=DTYPE, device=device)
    phi2s = torch.zeros(Nt, dtype=DTYPE, device=device)
    Hs    = torch.zeros(Nt, dtype=DTYPE, device=device)
    Ms    = torch.zeros(Nt, dtype=DTYPE, device=device)
    phi4s = torch.zeros(Nt, dtype=DTYPE, device=device)
    divs  = torch.zeros(Nt, dtype=DTYPE, device=device)

    for n in range(Nt):
        v = euler_step(v, teacher_rhs)
        if not torch.isfinite(v).all():
            print(f"[Teacher] non-finite detected at step {n}, clipping.")
            v = torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)

        traj[n] = v
        phi2, H, M, phi4, div_rms = scct_stats_vec_ns(v)
        phi2s[n] = phi2
        Hs[n]    = H
        Ms[n]    = M
        phi4s[n] = phi4
        divs[n]  = div_rms

    return traj, phi2s, Hs, Ms, phi4s, divs

# -------------------------
# 4. Multi-IC teacher data
# -------------------------
def make_ic(seed_offset=0):
    rng = np.random.RandomState(SEED + seed_offset)
    k1 = 2.0 * math.pi
    k2 = 4.0 * math.pi
    phase1 = rng.rand() * 2.0 * math.pi
    phase2 = rng.rand() * 2.0 * math.pi

    vx0 = 0.2 * torch.sin(k1 * X + phase1) * torch.cos(k2 * Y + phase2)
    vy0 = -0.2 * torch.cos(k1 * X + phase2) * torch.sin(k2 * Y + phase1)
    noise_x = 0.02 * torch.randn_like(vx0)
    noise_y = 0.02 * torch.randn_like(vy0)
    return torch.stack([vx0 + noise_x, vy0 + noise_y], dim=0)

n_ic = 3
teacher_stats = []

print("[Teacher] Generating multi-IC NS+τ teacher trajectories...")
for i in range(n_ic):
    v0 = make_ic(seed_offset=10*i)
    v0 = to_dev(v0)
    traj, phi2s, Hs, Ms, phi4s, divs = simulate_teacher(v0, Nt)
    stats = {
        "v0": v0,
        "traj": traj,
        "phi2": phi2s,
        "H": Hs,
        "M": Ms,
        "phi4": phi4s,
        "div": divs,
    }
    teacher_stats.append(stats)
    print(f"[Teacher] IC #{i}: Φ²(T)={phi2s[-1].item():.3e}, "
          f"H(T)={Hs[-1].item():.3f}, M(T)={Ms[-1].item():+.3e}, "
          f"<Φ²>_meta={phi2s[:200].mean().item():.3e}, "
          f"<div>_rms={divs[:200].mean().item():.3e}")
print()

# -------------------------
# 5. GrammarTree 9.1 τ_ij bases
#    [S, I1*S, δ I1]
# -------------------------
TAU_TERMS = ["S", "I1*S", "δ I1"]
n_terms = len(TAU_TERMS)
print(f"[GrammarTree 9.1] tensor τ_ij bases = {TAU_TERMS}\n")

def tau_basis_features(v):
    Sxx, Sxy, Syx, Syy, I1 = compute_S_and_I1(v)

    # Basis 0: S
    tau0_xx, tau0_xy, tau0_yx, tau0_yy = Sxx, Sxy, Syx, Syy
    # Basis 1: I1 * S
    tau1_xx = I1 * Sxx
    tau1_xy = I1 * Sxy
    tau1_yx = I1 * Syx
    tau1_yy = I1 * Syy
    # Basis 2: δ I1
    zero = torch.zeros_like(I1)
    tau2_xx, tau2_xy, tau2_yx, tau2_yy = I1, zero, zero, I1

    # 修正 stack_tau：先 stack 行，再 stack 整个 2x2
    def stack_tau(tx, txy, tyx, tyy):
        row0 = torch.stack([tx,  txy], dim=0)   # [2,Nx,Ny]
        row1 = torch.stack([tyx, tyy], dim=0)   # [2,Nx,Ny]
        return torch.stack([row0, row1], dim=0) # [2,2,Nx,Ny]

    B0 = stack_tau(tau0_xx, tau0_xy, tau0_yx, tau0_yy)
    B1 = stack_tau(tau1_xx, tau1_xy, tau1_yx, tau1_yy)
    B2 = stack_tau(tau2_xx, tau2_xy, tau2_yx, tau2_yy)

    feats = torch.stack([B0, B1, B2], dim=0)      # [n_terms,2,2,Nx,Ny]
    return feats

def div_tau_tensor(tau):
    # tau: [2,2,Nx,Ny], tau[i,j] = τ_ij(field)
    tau_xx = tau[0,0]
    tau_xy = tau[0,1]
    tau_yx = tau[1,0]
    tau_yy = tau[1,1]
    return div_tau_from_components(tau_xx, tau_xy, tau_yx, tau_yy)

# -------------------------
# 6. GrammarTree 9.1 model
# -------------------------
class GrammarTree91Tau(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.w     = nn.Parameter(torch.zeros(n_terms, dtype=DTYPE))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        if mask is None:
            mask = torch.ones(n_terms, dtype=DTYPE)
        self.register_buffer("mask", mask)

    def tau_residual(self, v):
        feats = tau_basis_features(v)   # [n_terms,2,2,Nx,Ny]
        eff   = self.gamma * self.w * self.mask   # [n_terms]
        eff_v = eff.view(n_terms, 1, 1, 1, 1)
        tau   = torch.sum(eff_v * feats, dim=0)   # [2,2,Nx,Ny]
        return tau

    def residual_div(self, v):
        tau = self.tau_residual(v)
        return div_tau_tensor(tau)   # [2,Nx,Ny]

    def pde_rhs(self, v):
        if CORE_MODE == "core-only":
            return core_rhs(v)
        elif CORE_MODE == "residual-only":
            return self.residual_div(v)
        elif CORE_MODE == "core+residual":
            return core_rhs(v) + self.residual_div(v)
        else:
            raise ValueError(f"Unknown CORE_MODE={CORE_MODE}")

    def step(self, v):
        return v + dt * self.pde_rhs(v)

    def simulate(self, v0, Nt):
        v = v0.clone()
        for _ in range(Nt):
            v = self.step(v)
        phi2, H, M, phi4, div_rms = scct_stats_vec_ns(v)
        return v, phi2, H, M, phi4, div_rms

    def simulate_phi2_traj(self, v0, steps):
        v = v0.clone()
        phi2s = []
        for _ in range(steps):
            v = self.step(v)
            phi2, _, _, _, _ = scct_stats_vec_ns(v)
            phi2s.append(phi2)
        return torch.stack(phi2s, dim=0)

# -------------------------
# 7. Rollout error
# -------------------------
def rollout_error(model, v0, steps):
    v_t = v0.clone()
    v_m = v0.clone()
    errs = []
    with torch.no_grad():
        for _ in range(steps):
            v_t = euler_step(v_t, teacher_rhs)
            v_m = model.step(v_m)
            errs.append(torch.mean((v_t - v_m)**2).item())
    return np.array(errs)

# -------------------------
# 8. Stage 1 training
# -------------------------
def train_stage1(epochs=120):
    model = GrammarTree91Tau().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-3)

    lambda_phi2_base = 5e-2
    lambda_H_base    = 5e-2
    lambda_M_base    = 1e-2
    lambda_phi4_base = 1e-2
    lambda_div_base  = 5e-2

    lambda_gamma  = 1e-5
    lambda_w_base = torch.tensor([5e-5, 5e-5, 5e-5],
                                 dtype=DTYPE, device=device)

    print("[Stage 1] Training GrammarTree 9.1 (τ_ij, multi-IC, SCCT+div, CORE_MODE=core+residual)...")
    for ep in range(1, epochs+1):
        opt.zero_grad()
        t = ep / epochs
        lam_phi2 = lambda_phi2_base * (0.5 + 0.5*t)
        lam_H    = lambda_H_base    * (0.5 + 0.5*t)
        lam_M    = lambda_M_base    * (0.5 + 0.5*t)
        lam_phi4 = lambda_phi4_base * (0.5 + 0.5*t)
        lam_div  = lambda_div_base  * (0.5 + 0.5*t)

        loss_total = 0.0
        for ic in range(n_ic):
            stats     = teacher_stats[ic]
            v0        = stats["v0"]
            traj_true = stats["traj"]
            phi2_true = stats["phi2"]
            H_true    = stats["H"]
            M_true    = stats["M"]
            phi4_true = stats["phi4"]
            div_true  = stats["div"]

            vT_true     = traj_true[-1]
            phi2_true_T = phi2_true[-1]
            H_true_T    = H_true[-1]
            M_true_T    = M_true[-1]
            phi4_true_T = phi4_true[-1]
            div_true_T  = div_true[-1]

            vT_pred, phi2_pred_T, H_pred_T, M_pred_T, phi4_pred_T, div_pred_T = model.simulate(v0, Nt)
            misfit = F.mse_loss(vT_pred, vT_true)

            loss_ic = misfit \
                + lam_phi2 * torch.abs(phi2_pred_T - phi2_true_T) \
                + lam_H    * torch.abs(H_pred_T    - H_true_T)    \
                + lam_M    * torch.abs(M_pred_T    - M_true_T)    \
                + lam_phi4 * torch.abs(phi4_pred_T - phi4_true_T) \
                + lam_div  * torch.abs(div_pred_T  - div_true_T)

            phi2_pred_traj = model.simulate_phi2_traj(v0, 200)
            phi2_meta_true = phi2_true[:200].mean()
            phi2_meta_pred = phi2_pred_traj.mean()
            loss_meta      = (phi2_meta_pred - phi2_meta_true)**2
            loss_ic        = loss_ic + 1e-1 * loss_meta

            loss_total = loss_total + loss_ic

        g = model.gamma
        w = model.w
        eff_w    = w * model.mask
        l1_w     = torch.sum(lambda_w_base * torch.abs(eff_w))
        l1_gamma = lambda_gamma * torch.abs(g)
        loss     = loss_total + l1_w + l1_gamma

        loss.backward()
        opt.step()

    with torch.no_grad():
        stats0     = teacher_stats[0]
        v0         = stats0["v0"]
        traj_true  = stats0["traj"]
        phi2_true  = stats0["phi2"]
        H_true     = stats0["H"]
        div_true   = stats0["div"]

        vT_true = traj_true[-1]
        vT_pred, phi2_pred_T, H_pred_T, M_pred_T, phi4_pred_T, div_pred_T = model.simulate(v0, Nt)
        misfit_final     = F.mse_loss(vT_pred, vT_true).item()
        phi2_meta_model  = model.simulate_phi2_traj(v0, 200).mean().item()
        eff              = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 1][Summary] (IC #0) misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}, div_rms(T)={:.3e}".format(
        misfit_final, phi2_pred_T.item(), H_pred_T.item(), div_pred_T.item()))
    print("  <Φ²>_meta(model, IC #0) = {:.3e}".format(phi2_meta_model))
    print("  γ(Stage 1) = {:+.3e}".format(model.gamma.item()))
    for i, name in enumerate(TAU_TERMS):
        print(f"  eff({name:6s} idx {i}) = {eff[i]:+.6e}")
    print()

    return model

model_stage1 = train_stage1()

# -------------------------
# 9. Adaptive pruning
# -------------------------
def adaptive_pruning_mask(model, floor_threshold=1e-4, factor=0.4):
    with torch.no_grad():
        eff     = (model.gamma * model.w * model.mask).detach().cpu().numpy()
        abs_eff = np.abs(eff)
        med     = np.median(abs_eff)
        if med < floor_threshold:
            tau = floor_threshold
        else:
            tau = max(floor_threshold, factor*med)
        keep = abs_eff >= tau
    return torch.tensor(keep.astype(np.float32), device=device), tau, eff

mask_stage2, tau_prune, eff_stage1 = adaptive_pruning_mask(model_stage1)
kept_indices = np.where(mask_stage2.detach().cpu().numpy() > 0.5)[0].tolist()
kept_terms   = [TAU_TERMS[i] for i in kept_indices]
lambda_k     = len(kept_indices) / len(TAU_TERMS)

print("[Stage 2] Adaptive pruning threshold τ = {:.3e}".format(tau_prune))
print("  kept indices =", kept_indices)
print("  kept terms   =", kept_terms)
print("  λ_k = {:.3f} ({}/{})".format(lambda_k, len(kept_indices), len(TAU_TERMS)))
print()

# -------------------------
# 10. Stage 2 training (pruned)
# -------------------------
def train_stage2(mask, epochs=80):
    model = GrammarTree91Tau(mask=mask).to(device)
    with torch.no_grad():
        model.w.copy_(model_stage1.w.detach())
        model.gamma.copy_(model_stage1.gamma.detach())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    lambda_phi2 = 5e-2
    lambda_H    = 5e-2
    lambda_M    = 1e-2
    lambda_phi4 = 1e-2
    lambda_div  = 5e-2

    lambda_gamma  = 1e-5
    lambda_w_base = torch.tensor([5e-5, 5e-5, 5e-5],
                                 dtype=DTYPE, device=device)

    print("[Stage 2] Refining pruned GrammarTree 9.1 (τ_ij, multi-IC, CORE_MODE=core+residual)...")
    for ep in range(1, epochs+1):
        opt.zero_grad()
        loss_total = 0.0
        for ic in range(n_ic):
            stats     = teacher_stats[ic]
            v0        = stats["v0"]
            traj_true = stats["traj"]
            phi2_true = stats["phi2"]
            H_true    = stats["H"]
            M_true    = stats["M"]
            phi4_true = stats["phi4"]
            div_true  = stats["div"]

            vT_true     = traj_true[-1]
            phi2_true_T = phi2_true[-1]
            H_true_T    = H_true[-1]
            M_true_T    = M_true[-1]
            phi4_true_T = phi4_true[-1]
            div_true_T  = div_true[-1]

            vT_pred, phi2_pred_T, H_pred_T, M_pred_T, phi4_pred_T, div_pred_T = model.simulate(v0, Nt)
            misfit = F.mse_loss(vT_pred, vT_true)

            loss_ic = misfit \
                + lambda_phi2 * torch.abs(phi2_pred_T - phi2_true_T) \
                + lambda_H    * torch.abs(H_pred_T    - H_true_T)    \
                + lambda_M    * torch.abs(M_pred_T    - M_true_T)    \
                + lambda_phi4 * torch.abs(phi4_pred_T - phi4_true_T) \
                + lambda_div  * torch.abs(div_pred_T  - div_true_T)

            loss_total = loss_total + loss_ic

        g = model.gamma
        w = model.w
        eff_w    = w * model.mask
        l1_w     = torch.sum(lambda_w_base * torch.abs(eff_w))
        l1_gamma = lambda_gamma * torch.abs(g)
        loss     = loss_total + l1_w + l1_gamma

        loss.backward()
        opt.step()

    with torch.no_grad():
        stats0     = teacher_stats[0]
        v0         = stats0["v0"]
        traj_true  = stats0["traj"]
        phi2_true  = stats0["phi2"]
        H_true     = stats0["H"]
        div_true   = stats0["div"]

        vT_true = traj_true[-1]
        vT_pred, phi2_pred_T, H_pred_T, M_pred_T, phi4_pred_T, div_pred_T = model.simulate(v0, Nt)
        misfit_final    = F.mse_loss(vT_pred, vT_true).item()
        phi2_meta_model = model.simulate_phi2_traj(v0, 200).mean().item()
        eff             = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 2][Summary] (IC #0) misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}, div_rms(T)={:.3e}".format(
        misfit_final, phi2_pred_T.item(), H_pred_T.item(), div_pred_T.item()))
    print("  <Φ²>_meta(model, IC #0) = {:.3e}".format(phi2_meta_model))
    print("  γ(Stage 2) = {:+.3e}".format(model.gamma.item()))
    for i, name in enumerate(TAU_TERMS):
        print(f"  eff({name:6s} idx {i}) = {eff[i]:+.6e}")
    print()

    return model, eff

model_stage2, eff_stage2 = train_stage2(mask_stage2)

# -------------------------
# 11. Evaluation
# -------------------------
err_arr = rollout_error(model_stage2, teacher_stats[0]["v0"], steps=200)
phi2_teacher_early = teacher_stats[0]["phi2"][:200]
phi2_model_early   = model_stage2.simulate_phi2_traj(teacher_stats[0]["v0"], 200)

print("[Eval] Mean L2 orbit error over [0,{:.3f}] = {:.3e}".format(
    200*dt, err_arr.mean()))
print("[Eval] Early-time Φ² teacher≈{:.3e}, GT9.1≈{:.3e}, ratio≈{:.3f}".format(
    phi2_teacher_early.mean().item(),
    phi2_model_early.mean().item(),
    phi2_model_early.mean().item() / (phi2_teacher_early.mean().item() + 1e-12)))
print()

print("[GrammarTree 9.1] Effective τ_ij coefficients (S, I1*S, δ I1):")
for i, name in enumerate(TAU_TERMS):
    print(f"  eff({name:6s} idx {i}) Stage1 = {eff_stage1[i]:+.6e}, Stage2 = {eff_stage2[i]:+.6e}")
print()

# -------------------------
# 12. (可选) Stage 3: PySR audit on τ_xx
# -------------------------
if DO_PYSR:
    print("[Stage 3] Building dataset for PySR on τ_xx vs (Sxx, I1)...")
    X_list, R_list = [], []
    with torch.no_grad():
        for ic in range(n_ic):
            traj = teacher_stats[ic]["traj"]
            for n in range(0, 200):
                v = traj[n]
                Sxx, Sxy, Syx, Syy, I1 = compute_S_and_I1(v)
                tau_xx, tau_xy, tau_yx, tau_yy = tau_teacher(v)
                for i in range(Nx):
                    for j in range(Ny):
                        X_list.append([
                            float(Sxx[i,j].item()),
                            float(I1[i,j].item()),
                        ])
                        R_list.append(float(tau_xx[i,j].item()))
    X = np.array(X_list, dtype=np.float64)
    R = np.array(R_list, dtype=np.float64)
    print(f"[Stage 3] Dataset shapes: X={X.shape}, R={R.shape}")

    model_pysr = PySRRegressor(
        niterations=800,
        populations=20,
        population_size=40,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=15,
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        parallelism="serial",
        random_state=0,
    )
    model_pysr.fit(X, R, variable_names=["Sxx", "I1"])
    eqs = model_pysr.equations_
    best_row = eqs.sort_values("score", ascending=False).iloc[0]
    row_idx  = int(best_row.name)

    print("\n[Stage 3] PySR best equation for τ_xx:")
    print("  equation  :", best_row["equation"])
    print(f"  loss      = {best_row['loss']:.3e}")
    print(f"  complexity= {int(best_row['complexity'])}")

    sym_list = model_pysr.sympy()
    sym_best = sym_list[row_idx]
    print("[Stage 3] Sympy form:", sym_best)

print("\n[Summary] GrammarTree 9.1 (τ_ij tensor RANS prototype, Nt=1000, 32x32) finished.")

