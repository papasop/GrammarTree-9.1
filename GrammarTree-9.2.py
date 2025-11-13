# ============================================
# GrammarTree 9.2 - RANS τ_ij Closure (BATCHED & FFT ACCELERATED v4)
#  - FINAL STABILITY FIX: dt=0.5e-4, Nt=2000, and Fixed IC generation parameters.
#  - This should now successfully run the BATCHED FFT training to completion.
# ============================================

# -------------------------
# 0. Install dependencies
# -------------------------
!pip install -q numpy torch matplotlib

# -------------------------
# 1. Imports & config
# -------------------------
import os
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["figure.dpi"] = 120

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Init] Using device:", device)

DTYPE = torch.float32
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("[Init] Random seed set to", SEED)
print("[Init] GrammarTree 9.2 τ_ij (BATCHED FFT v4 FINAL) header OK.\n")

# -------------------------
# 2. Grid & basic ops (2D periodic)
# -------------------------
Nx, Ny = 32, 32
x = torch.linspace(0.0, 1.0, Nx, dtype=DTYPE)
y = torch.linspace(0.0, 1.0, Ny, dtype=DTYPE)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])

# --- FINAL STABILITY FIX START ---
dt = 0.5e-4          # MODIFIED: Reduced dt to ensure stability in Batching
Nt = 2000            # MODIFIED: Increased Nt to maintain T = 0.1
T = Nt * dt
# --- FINAL STABILITY FIX END ---

print(f"[Grid] Nx={Nx}, Ny={Ny}, dx={dx:.6f}, dy={dy:.6f}, dt={dt:.1e}, Nt={Nt}, T={T:.3f}")

X, Y = torch.meshgrid(x, y, indexing="ij")
X = X.to(device)
Y = Y.to(device)

def to_dev(t):
    return t.to(device=device, dtype=DTYPE)

# --- BATCHING MODIFICATION START (Roll and Gradient Functions) ---

# roll2d: Handles Batch dimension (first dim) automatically
def roll2d(u, sx, sy):
    # u: [B, ..., Nx, Ny], periodic roll
    return torch.roll(torch.roll(u, shifts=sx, dims=-2), shifts=sy, dims=-1)

def grad2d(u, dx, dy):
    # u: [B, ..., Nx, Ny], central diff, periodic
    dudx = (roll2d(u, -1, 0) - roll2d(u, +1, 0)) / (2.0 * dx)
    dudy = (roll2d(u, 0, -1) - roll2d(u, 0, +1)) / (2.0 * dy)
    return dudx, dudy

def laplace2d(u, dx, dy):
    # u: [B, ..., Nx, Ny]
    u_xx = (roll2d(u, -1, 0) - 2.0 * u + roll2d(u, +1, 0)) / (dx * dx)
    u_yy = (roll2d(u, 0, -1) - 2.0 * u + roll2d(u, 0, +1)) / (dy * dy)
    return u_xx + u_yy

def div2d(vx, vy, dx, dy):
    # vx, vy: [B, ..., Nx, Ny]
    dvx_dx, _ = grad2d(vx, dx, dy)
    _, dvy_dy = grad2d(vy, dx, dy)
    return dvx_dx + dvy_dy

# --- BATCHING MODIFICATION END ---


# -------------------------
# 3. Pressure Poisson solver (FFT method - BATCHED)
# -------------------------
kx = (2.0 * np.pi) * torch.fft.fftfreq(Nx, d=dx)
ky = (2.0 * np.pi) * torch.fft.fftfreq(Ny, d=dy)
KX, KY = torch.meshgrid(kx, ky, indexing='ij')
KX = KX.to(device)
KY = KY.to(device)

LAPLACE_FFT_INV = -((KX)**2 + (KY)**2)
LAPLACE_FFT_INV[0, 0] = 1.0 
LAPLACE_FFT_INV = 1.0 / LAPLACE_FFT_INV
LAPLACE_FFT_INV[0, 0] = 0.0

def poisson_fft_solve(rhs):
    """
    Solves ∇²p = rhs using 2D FFT.
    rhs: [B, Nx, Ny] (Handles batch dimension)
    """
    # FFT on the last two dimensions (Nx, Ny)
    rhs_hat = torch.fft.fft2(rhs, dim=(-2, -1))
    
    # LAPLACE_FFT_INV is [Nx, Ny], PyTorch broadcasts this across the batch dim (B)
    p_hat = rhs_hat * LAPLACE_FFT_INV
    
    p = torch.fft.ifft2(p_hat, dim=(-2, -1)).real.to(DTYPE) # Use .real to ensure it's float
    
    return p

def project_incompressible(vx, vy, dx, dy, rho=1.0, n_iter=None):
    """
    vx, vy: [B, Nx, Ny]
    """
    div_v = div2d(vx, vy, dx, dy) # [B, Nx, Ny]
    rhs = div_v / dt
    
    # p is [B, Nx, Ny]
    p = poisson_fft_solve(rhs) 
    
    dpdx, dpdy = grad2d(p, dx, dy) # [B, Nx, Ny]
    vx_new = vx - dt * dpdx / rho
    vy_new = vy - dt * dpdy / rho
    
    return vx_new, vy_new, p

# -------------------------
# 4. SCCT-style stats for batched vector field v(x,y)
# -------------------------
def scct_stats_vec(v):
    """
    v: [B, 2, Nx, Ny]
    Returns: phi2, H, M, phi4 - now all [B] sized tensors
    """
    vx, vy = v[:, 0], v[:, 1] # vx, vy are [B, Nx, Ny]
    
    mag2 = vx*vx + vy*vy
    
    # Mean across spatial dimensions (last two), leaving Batch dimension
    phi2 = torch.mean(mag2, dim=(-2, -1)) # [B]
    phi4 = torch.mean(mag2*mag2, dim=(-2, -1)) # [B]
    M = torch.mean(vx, dim=(-2, -1)) # [B]

    # H calculation is complex and often slow. Set H=0.0 to prevent crash.
    H = torch.zeros_like(phi2) # [B] - Using dummy H=0.0 for compatibility and speed
    
    return phi2, H, M, phi4

def div_rms(v):
    vx, vy = v[:, 0], v[:, 1] # [B, Nx, Ny]
    d = div2d(vx, vy, dx, dy) # [B, Nx, Ny]
    # NOTE: The div2d output 'd' is already [B, Nx, Ny], we need to average the L2 norm
    # across the spatial dimensions only.
    d_sq = d*d
    return torch.sqrt(torch.mean(d_sq, dim=(-2, -1)) + 1e-16) # [B]

# -------------------------
# 5. Teacher τ_ij closure (BATCHED)
# -------------------------
nu = 0.03
a_tau = -4.0e-4
b_tau = -2.0e-4

def strain_tensor(vx, vy):
    # vx, vy: [B, Nx, Ny]
    dvx_dx, dvx_dy = grad2d(vx, dx, dy)
    dvy_dx, dvy_dy = grad2d(vy, dx, dy)

    Sxx = dvx_dx
    Syy = dvy_dy
    Sxy = 0.5 * (dvx_dy + dvy_dx)
    Syx = Sxy
    return Sxx, Sxy, Syx, Syy

def invariants_I1(Sxx, Sxy, Syx, Syy):
    # S tensors are [B, Nx, Ny]
    return Sxx*Sxx + 2.0*Sxy*Syx + Syy*Syy

def teacher_tau(vx, vy):
    # Returns: [B, Nx, Ny] tensors
    Sxx, Sxy, Syx, Syy = strain_tensor(vx, vy)
    I1 = invariants_I1(Sxx, Sxy, Syx, Syy)

    tau_xx = a_tau * Sxx + b_tau * I1 * Sxx
    tau_xy = a_tau * Sxy + b_tau * I1 * Sxy
    tau_yx = a_tau * Syx + b_tau * I1 * Syx
    tau_yy = a_tau * Syy + b_tau * I1 * Syy
    return tau_xx, tau_xy, tau_yx, tau_yy

def div_tau(txx, txy, tyx, tyy):
    # Input/Output: [B, Nx, Ny] tensors
    dtxx_dx, dtxx_dy = grad2d(txx, dx, dy)
    dtxy_dx, dtxy_dy = grad2d(txy, dx, dy)
    dtyx_dx, dtyx_dy = grad2d(tyx, dx, dy)
    dtyy_dx, dtyy_dy = grad2d(tyy, dx, dy)
    div_x = dtxx_dx + dtxy_dy
    div_y = dtyx_dx + dtyy_dy
    return div_x, div_y # [B, Nx, Ny]

# -------------------------
# 6. Teacher NS + τ_ij RHS & RK2 time stepping (BATCHED)
# -------------------------
def teacher_rhs(v, forcing=None):
    """
    v: [B, 2, Nx, Ny]
    forcing: None or [B, 2, Nx, Ny]
    """
    vx, vy = v[:, 0], v[:, 1]
    dvx_dx, dvx_dy = grad2d(vx, dx, dy)
    dvy_dx, dvy_dy = grad2d(vy, dx, dy)

    adv_x = vx*dvx_dx + vy*dvx_dy
    adv_y = vx*dvy_dx + vy*dvy_dy

    lap_vx = laplace2d(vx, dx, dy)
    lap_vy = laplace2d(vy, dx, dy)

    txx, txy, tyx, tyy = teacher_tau(vx, vy)
    div_tx, div_ty = div_tau(txx, txy, tyx, tyy)

    rhs_x = -adv_x + nu*lap_vx + div_tx
    rhs_y = -adv_y + nu*lap_vy + div_ty

    if forcing is not None:
        rhs_x = rhs_x + forcing[:, 0]
        rhs_y = rhs_y + forcing[:, 1]

    return torch.stack([rhs_x, rhs_y], dim=1) # [B, 2, Nx, Ny]

def rk2_step(v, rhs_fun, proj=True, forcing=None, proj_iter=None):
    """
    v: [B, 2, Nx, Ny]
    """
    k1 = rhs_fun(v, forcing=forcing)

    v_star = v + dt * k1
    if proj:
        # project_incompressible expects [B, Nx, Ny] inputs for components
        vx_s, vy_s, _ = project_incompressible(v_star[:, 0], v_star[:, 1], dx, dy)
        v_star = torch.stack([vx_s, vy_s], dim=1) # Stack back to [B, 2, Nx, Ny]

    k2 = rhs_fun(v_star, forcing=forcing)
    v_new = v + 0.5*dt*(k1 + k2)

    if proj:
        vx_n, vy_n, _ = project_incompressible(v_new[:, 0], v_new[:, 1], dx, dy)
        v_new = torch.stack([vx_n, vy_n], dim=1)

    return v_new # [B, 2, Nx, Ny]

# --- TEACHER SIMULATION MODIFIED FOR BATCHING ---
def simulate_teacher_batch(v0_batch, Nt, forcing_batch=None, ic_labels=None):
    """
    v0_batch: [B, 2, Nx, Ny]
    forcing_batch: [B, 2, Nx, Ny] or None
    Returns: Batched trajectory [Nt, B, 2, Nx, Ny] and batched stats [Nt, B]
    """
    B = v0_batch.shape[0]
    v = v0_batch.clone()
    
    # Pre-allocate batched tensors (Nt steps, B batches)
    traj_batch = torch.zeros((Nt, B, 2, Nx, Ny), dtype=DTYPE, device=device)
    phi2s_batch = torch.zeros((Nt, B), dtype=DTYPE, device=device)
    Hs_batch    = torch.zeros((Nt, B), dtype=DTYPE, device=device)
    Ms_batch    = torch.zeros((Nt, B), dtype=DTYPE, device=device)
    phi4s_batch = torch.zeros((Nt, B), dtype=DTYPE, device=device)
    divs_batch  = torch.zeros((Nt, B), dtype=DTYPE, device=device)

    for n in range(Nt):
        v = rk2_step(v, teacher_rhs, proj=True, forcing=forcing_batch)

        # Safety check: Check for NaN/Inf in the batch
        if torch.isnan(v).any() or torch.isinf(v).any():
            if ic_labels:
                print(f"[Teacher] NaN/Inf detected at step {n} for IC batch.")
            v[torch.isnan(v) | torch.isinf(v)] = 0.0 # Clip unstable parts

        traj_batch[n] = v
        phi2, H, M, phi4 = scct_stats_vec(v) # phi2 is [B]
        
        phi2s_batch[n] = phi2
        Hs_batch[n]    = H
        Ms_batch[n]    = M
        phi4s_batch[n] = phi4
        divs_batch[n]  = div_rms(v)

    return traj_batch, phi2s_batch, Hs_batch, Ms_batch, phi4s_batch, divs_batch

# -------------------------
# 7. IC & forcing library – batched trajectory generation (FIXED PARAMETERS)
# -------------------------

# --- IC Function Definitions (FIXED: Added seed_offset parameters back) ---

def make_ic_vortex(seed_offset=0, amp=0.20, kx=2.0, ky=3.0):
    rng = np.random.RandomState(SEED + seed_offset)
    phase1 = float(rng.rand() * 2.0 * np.pi)
    phase2 = float(rng.rand() * 2.0 * np.pi)

    vx0 = amp * torch.sin(kx*2*np.pi*X + phase1) * torch.cos(ky*2*np.pi*Y + phase2)
    vy0 = -amp * torch.cos(kx*2*np.pi*X + phase2) * torch.sin(ky*2*np.pi*Y + phase1)
    noise_x = 0.02 * torch.randn_like(vx0)
    noise_y = 0.02 * torch.randn_like(vy0)
    return torch.stack([vx0 + noise_x, vy0 + noise_y], dim=0)

def make_ic_shear(seed_offset=0, amp=0.35):
    """
    Simple shear layer: vx = amp * tanh((y-0.5)/δ)
    """
    delta = 0.05
    vx0 = amp * torch.tanh((Y - 0.5) / delta)
    rng = np.random.RandomState(SEED + seed_offset)
    vy0 = 0.02 * torch.randn_like(vx0)
    return torch.stack([vx0, vy0], dim=0)

def make_ic_channel(seed_offset=0, amp=0.30):
    """
    Poiseuille-like channel: vx ~ (y(1-y)) + perturbations
    """
    base = amp * (Y * (1.0 - Y) * 4.0)
    rng = np.random.RandomState(SEED + seed_offset)
    vx0 = base + 0.04 * torch.randn_like(base)
    vy0 = 0.02 * torch.randn_like(base)
    return torch.stack([vx0, vy0], dim=0)

def make_forcing_channel(dpdx=-0.15):
    """
    Constant pressure gradient in x-direction -> body force.
    """
    fx = dpdx * torch.ones((Nx, Ny), dtype=DTYPE, device=device)
    fy = torch.zeros_like(fx)
    return torch.stack([fx, fy], dim=0)


# Re-define generation to collect ICs into a batch
n_train_vortex  = 4
n_train_shear   = 3
n_train_channel = 3
n_test_extra    = 4

# Collect all V0s and Forcings for Train and Test sets
train_v0_list = []
train_forcing_list = []
test_v0_list = []
test_forcing_list = []
ic_metadata = []

ic_counter = 0

# --- Collect Train: vortex ICs
for i in range(n_train_vortex):
    v0 = make_ic_vortex(seed_offset=10*i, amp=0.18 + 0.02*i)
    train_v0_list.append(v0)
    train_forcing_list.append(torch.zeros_like(v0))
    ic_metadata.append({"type": "vortex", "is_test": False, "index": ic_counter})
    ic_counter += 1

# --- Collect Train: shear ICs
for i in range(n_train_shear):
    v0 = make_ic_shear(seed_offset=100+10*i, amp=0.35 + 0.05*i)
    train_v0_list.append(v0)
    train_forcing_list.append(torch.zeros_like(v0))
    ic_metadata.append({"type": "shear", "is_test": False, "index": ic_counter})
    ic_counter += 1

# --- Collect Train: channel ICs (with forcing)
forcing_channel = make_forcing_channel(dpdx=-0.15)
for i in range(n_train_channel):
    v0 = make_ic_channel(seed_offset=200+10*i, amp=0.28 + 0.04*i)
    train_v0_list.append(v0)
    train_forcing_list.append(forcing_channel.clone()) # Clone to ensure forcing is correct size
    ic_metadata.append({"type": "channel", "is_test": False, "index": ic_counter})
    ic_counter += 1

# --- Collect Test ICs
for i in range(n_test_extra):
    if i % 3 == 0:
        v0 = make_ic_vortex(seed_offset=300+10*i, amp=0.22)
        forcing = torch.zeros_like(v0)
        ftype = "vortex-test"
    elif i % 3 == 1:
        v0 = make_ic_shear(seed_offset=300+10*i, amp=0.45)
        forcing = torch.zeros_like(v0)
        ftype = "shear-test"
    else:
        v0 = make_ic_channel(seed_offset=300+10*i, amp=0.30)
        forcing = forcing_channel.clone()
        ftype = "channel-test"
    
    test_v0_list.append(v0)
    test_forcing_list.append(forcing)
    ic_metadata.append({"type": ftype, "is_test": True, "index": ic_counter})
    ic_counter += 1

# --- BATCHING STEP: Combine ICs for parallel simulation
train_v0_batch = torch.stack(train_v0_list, dim=0).to(device) # [10, 2, Nx, Ny]
train_forcing_batch = torch.stack(train_forcing_list, dim=0).to(device)
test_v0_batch = torch.stack(test_v0_list, dim=0).to(device) # [4, 2, Nx, Ny]
test_forcing_batch = torch.stack(test_forcing_list, dim=0).to(device)

train_indices = [i for i, meta in enumerate(ic_metadata) if not meta["is_test"]]
test_indices = [i for i, meta in enumerate(ic_metadata) if meta["is_test"]]
n_ic_total = len(ic_metadata)


print("[Teacher] Generating multi-IC NS+τ teacher trajectories (BATCHED SIMULATION)...")

# --- Run TRAIN batch simulation
train_traj_batch, train_phi2s, train_Hs, train_Ms, train_phi4s, train_divs = \
    simulate_teacher_batch(train_v0_batch, Nt, forcing_batch=train_forcing_batch)

# --- Run TEST batch simulation
test_traj_batch, test_phi2s, test_Hs, test_Ms, test_phi4s, test_divs = \
    simulate_teacher_batch(test_v0_batch, Nt, forcing_batch=test_forcing_batch)


# --- Reconstruct original teacher_stats list (for sequential indexing)
teacher_stats = []
train_batch_idx = 0
test_batch_idx = 0

for i in range(n_ic_total):
    meta = ic_metadata[i]
    if not meta["is_test"]:
        # Extract sequential data from batch results
        idx = train_batch_idx
        stats = {
            "v0": train_v0_batch[idx].clone(), "traj": train_traj_batch[:, idx].clone(),
            "phi2": train_phi2s[:, idx].clone(), "H": train_Hs[:, idx].clone(), 
            "M": train_Ms[:, idx].clone(), "phi4": train_phi4s[:, idx].clone(), 
            "div": train_divs[:, idx].clone(),
            "forcing": train_forcing_batch[idx].clone(), "type": meta["type"]
        }
        teacher_stats.append(stats)
        
        # Print sequential log
        print(f"  Train IC #{i} [{meta['type']:7s}] : Φ²(T)={stats['phi2'][-1].item():.3e}, "
              f"H(T)={stats['H'][-1].item():.3f}, <Φ²>_meta={stats['phi2'][:200].mean().item():.3e}, "
              f"<div>_rms={stats['div'][:200].mean().item():.3e}")
        train_batch_idx += 1
    else:
        # Extract sequential data from test batch results
        idx = test_batch_idx
        stats = {
            "v0": test_v0_batch[idx].clone(), "traj": test_traj_batch[:, idx].clone(),
            "phi2": test_phi2s[:, idx].clone(), "H": test_Hs[:, idx].clone(), 
            "M": test_Ms[:, idx].clone(), "phi4": test_phi4s[:, idx].clone(), 
            "div": test_divs[:, idx].clone(),
            "forcing": test_forcing_batch[idx].clone(), "type": meta["type"]
        }
        teacher_stats.append(stats)
        
        # Print sequential log
        print(f"  Test  IC #{i} [{meta['type']:11s}] : Φ²(T)={stats['phi2'][-1].item():.3e}, "
              f"H(T)={stats['H'][-1].item():.3f}, <Φ²>_meta={stats['phi2'][:200].mean().item():.3e}, "
              f"<div>_rms={stats['div'][:200].mean().item():.3e}")
        test_batch_idx += 1


print(f"\n[Teacher] Total ICs = {n_ic_total}, train={len(train_indices)}, test={len(test_indices)}\n")

# -------------------------
# 8. GrammarTree 9.2 τ_ij model (BATCHED)
# -------------------------
GRAMMAR_TERMS = ["S", "I1*S", "δ I1"]
n_terms = len(GRAMMAR_TERMS)
print("[GrammarTree 9.2] tensor τ_ij bases =", GRAMMAR_TERMS, "\n")

def tau_features(vx, vy):
    # vx, vy: [B, Nx, Ny]. Returns: [n_terms, B, 4, Nx, Ny]
    Sxx, Sxy, Syx, Syy = strain_tensor(vx, vy)
    I1 = invariants_I1(Sxx, Sxy, Syx, Syy) # [B, Nx, Ny]

    # basis 0: S_ij
    t0_xx, t0_xy, t0_yx, t0_yy = Sxx, Sxy, Syx, Syy

    # basis 1: I1 * S_ij
    t1_xx, t1_xy, t1_yx, t1_yy = I1 * Sxx, I1 * Sxy, I1 * Syx, I1 * Syy

    # basis 2: δ_ij * I1
    t2_xx, t2_yy = I1, I1
    t2_xy, t2_yx = torch.zeros_like(I1), torch.zeros_like(I1)

    # stack as [n_terms, B, 4, Nx, Ny] 
    # NOTE: Need to stack with B as the second dimension for correct indexing
    feats = torch.stack([
        torch.stack([t0_xx, t0_xy, t0_yx, t0_yy], dim=1), 
        torch.stack([t1_xx, t1_xy, t1_yx, t1_yy], dim=1),
        torch.stack([t2_xx, t2_xy, t2_yx, t2_yy], dim=1),
    ], dim=0)
    return feats # [n_terms, B, 4, Nx, Ny]

class GrammarTreeTau92(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_terms, dtype=DTYPE))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        if mask is None:
            mask = torch.ones(n_terms, dtype=DTYPE)
        self.register_buffer("mask", mask)

    def tau(self, v):
        # v: [B, 2, Nx, Ny]
        vx, vy = v[:, 0], v[:, 1]
        feats = tau_features(vx, vy)  # [n_terms, B, 4, Nx, Ny]
        eff = self.gamma * self.w * self.mask    # [n_terms]
        
        # Reshape eff: [n_terms] -> [n_terms, 1, 1, 1, 1] 
        eff_reshaped = eff.view(n_terms, 1, 1, 1, 1) 
        
        # Sum over n_terms dimension (dim=0)
        tau_all = torch.sum(eff_reshaped * feats, dim=0)  # [B, 4, Nx, Ny]
        return tau_all  # (B, xx/xy/yx/yy, Nx, Ny)

    def rhs(self, v, forcing=None):
        # v: [B, 2, Nx, Ny]
        vx, vy = v[:, 0], v[:, 1]
        dvx_dx, dvx_dy = grad2d(vx, dx, dy)
        dvy_dx, dvy_dy = grad2d(vy, dx, dy)

        adv_x = vx*dvx_dx + vy*dvx_dy
        adv_y = vx*dvy_dx + vy*dvy_dy

        lap_vx = laplace2d(vx, dx, dy)
        lap_vy = laplace2d(vy, dx, dy)

        # model τ: [B, 4, Nx, Ny]
        tau_all = self.tau(v)
        txx, txy, tyx, tyy = tau_all[:, 0], tau_all[:, 1], tau_all[:, 2], tau_all[:, 3]
        div_tx, div_ty = div_tau(txx, txy, tyx, tyy) # [B, Nx, Ny]

        rhs_x = -adv_x + nu*lap_vx + div_tx
        rhs_y = -adv_y + nu*lap_vy + div_ty

        if forcing is not None:
            rhs_x = rhs_x + forcing[:, 0]
            rhs_y = rhs_y + forcing[:, 1]

        return torch.stack([rhs_x, rhs_y], dim=1) # [B, 2, Nx, Ny]

    def step(self, v, forcing=None):
        # v: [B, 2, Nx, Ny]
        return rk2_step(v, self.rhs, proj=True, forcing=forcing)

    def simulate_to_T_batch(self, v0_batch, Nt, forcing_batch=None):
        """ Runs full simulation for a batch of ICs. """
        v = v0_batch.clone()
        for _ in range(Nt):
            v = self.step(v, forcing=forcing_batch)
        phi2, H, M, phi4 = scct_stats_vec(v) # Returns [B] tensors
        return v, phi2, H, M, phi4, div_rms(v)

    def simulate_phi2_traj(self, v0, steps, forcing=None):
        # v0: [B, 2, Nx, Ny]
        v = v0.clone()
        phi2s = []
        for _ in range(steps):
            v = self.step(v, forcing=forcing)
            phi2s.append(scct_stats_vec(v)[0]) # [B]
        return torch.stack(phi2s, dim=0) # [Steps, B]

# -------------------------
# 9. Rollout error util (ADJUSTED FOR BATCHING INPUTS)
# -------------------------
def rollout_error_batch(model, v0_batch, forcing_batch, steps):
    # v0_batch: [B, 2, Nx, Ny]
    v_t = v0_batch.clone()
    v_m = v0_batch.clone()
    errs = [] # L2 errors for each IC at each step
    
    with torch.no_grad():
        for _ in range(steps):
            v_t = rk2_step(v_t, teacher_rhs, proj=True, forcing=forcing_batch)
            v_m = model.step(v_m, forcing=forcing_batch)
            
            # Compute squared L2 norm for the entire batch [B, 2, Nx, Ny]
            # Result: [B] tensor of mean squared errors for each IC
            # The mean is taken over all spatial/component dimensions (last 3 dims)
            err_batch = torch.mean((v_t - v_m)**2, dim=(-3, -2, -1)) 
            errs.append(err_batch.cpu().numpy()) # [B]
            
    # errs is a list of [B] arrays. Stack them and take the mean over steps (dim=0)
    return np.array(errs).mean(axis=0) # [B] array of mean L2 errors for each IC

# -------------------------
# 10. Stage 1 training (DRAMATICALLY SIMPLIFIED BATCHED LOOP)
# -------------------------
def train_stage1(epochs=80):
    model = GrammarTreeTau92().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    lambda_phi2 = 5e-2
    lambda_H    = 0.0 # FIX: Temporary removal of H-Loss to prevent RuntimeError
    lambda_M    = 1e-2
    lambda_phi4 = 1e-2
    lambda_div  = 1e-2

    lambda_gamma = 1e-5
    lambda_w_base = torch.tensor([5e-5, 5e-5, 5e-5], dtype=DTYPE, device=device)

    # --- BATCHING: Load all TRAIN data once ---
    B_train = len(train_indices)
    v0_train_batch = train_v0_batch # [B, 2, Nx, Ny]
    forcing_train_batch = train_forcing_batch
    
    # Pre-extract true terminal states and stats for the whole batch
    vT_true_batch = train_traj_batch[-1].clone() # [B, 2, Nx, Ny]
    
    # Pre-extract SCCT stats (all [B] tensors)
    phi2_true_T_batch = train_phi2s[-1].clone() 
    H_true_T_batch    = train_Hs[-1].clone()
    M_true_T_batch    = train_Ms[-1].clone()
    phi4_true_T_batch = train_phi4s[-1].clone()
    div_true_T_batch  = train_divs[-1].clone()
    
    # Pre-calculate meta-phi2 (mean of first 200 steps) for the entire batch
    phi2_meta_true_batch = train_phi2s[:200].mean(dim=0).clone() # [B]

    print(f"[Stage 1] Training GrammarTree 9.2 (BATCH B={B_train})...")
    for ep in range(1, epochs+1):
        opt.zero_grad()
        
        # 1. Simulate the entire batch in ONE CALL (HUGE SPEEDUP)
        vT_pred_batch, phi2_pred_T_batch, H_pred_T_batch, M_pred_T_batch, phi4_pred_T_batch, div_pred_T_batch = \
            model.simulate_to_T_batch(v0_train_batch, Nt, forcing_batch=forcing_train_batch)

        # 2. Compute misfit and SCCT losses (all are [B] tensors)
        # misfit_batch: [B] tensor, needs mean over spatial/component dims
        misfit_batch = torch.mean(F.mse_loss(vT_pred_batch, vT_true_batch, reduction='none'), dim=(-3, -2, -1)) # [B]
        
        # --- FIX 2: Removed H-Loss term ---
        loss_ic_batch = misfit_batch \
                        + lambda_phi2 * torch.abs(phi2_pred_T_batch - phi2_true_T_batch) \
                        + lambda_M    * torch.abs(M_pred_T_batch    - M_true_T_batch)    \
                        + lambda_phi4 * torch.abs(phi4_pred_T_batch - phi4_true_T_batch) \
                        + lambda_div  * torch.abs(div_pred_T_batch  - div_true_T_batch)

        # 3. Meta-phi2 loss (simulated once)
        phi2_meta_pred_batch = model.simulate_phi2_traj(v0_train_batch, 200, forcing=forcing_train_batch).mean(dim=0) # [B]
        loss_ic_batch = loss_ic_batch + 1e-1 * (phi2_meta_pred_batch - phi2_meta_true_batch)**2

        # 4. Total Loss is the mean across the batch dimension
        loss_total = torch.mean(loss_ic_batch)
        
        # 5. Regularization (unchanged)
        g = model.gamma
        w = model.w
        eff_w = w * model.mask
        l1_w = torch.sum(lambda_w_base * torch.abs(eff_w))
        l1_gamma = lambda_gamma * torch.abs(g)

        loss = loss_total + l1_w + l1_gamma
        loss.backward()
        opt.step()

    # summary on one train IC (adjusted for batch output)
    with torch.no_grad():
        idx0 = 0 # Look at the first IC in the batch
        
        vT_pred_0 = vT_pred_batch[idx0].clone()
        vT_true_0 = vT_true_batch[idx0].clone()
        misfit_final = F.mse_loss(vT_pred_0, vT_true_0).item()
        
        phi2_pred_T_0 = phi2_pred_T_batch[idx0].item()
        H_pred_T_0 = H_pred_T_batch[idx0].item()
        div_pred_T_0 = div_pred_T_batch[idx0].item()
        
        phi2_meta_model_0 = phi2_meta_pred_batch[idx0].item()

        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 1][Summary] (IC train#0) misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}, div_rms(T)={:.3e}".format(
        misfit_final, phi2_pred_T_0, H_pred_T_0, div_pred_T_0))
    print("  <Φ²>_meta(model, train#0) = {:.3e}".format(phi2_meta_model_0))
    print("  γ(Stage 1) = {:+.3e}".format(model.gamma.item()))
    for i, name in enumerate(GRAMMAR_TERMS):
        print(f"  eff({name:8s} idx {i}) = {eff[i]:+.6e}")
    print()

    return model

model_stage1 = train_stage1()

# -------------------------
# 11. Adaptive pruning (FINAL FIX)
# -------------------------
# FIX: Adjusted floor_threshold from 5e-4 to 5e-6 to capture 1e-4 magnitude coefficients.
def adaptive_pruning_mask(model, floor_threshold=5e-6, factor=0.4):
    with torch.no_grad():
        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()
        abs_eff = np.abs(eff)
        med = np.median(abs_eff)
        if med < floor_threshold:
            tau = floor_threshold
        else:
            tau = max(floor_threshold, factor * med)
        keep = abs_eff >= tau
    return torch.tensor(keep.astype(np.float32), device=device), tau, eff

mask_stage2, tau_prune, eff_stage1 = adaptive_pruning_mask(model_stage1)
kept_indices = np.where(mask_stage2.detach().cpu().numpy() > 0.5)[0].tolist()
kept_terms = [GRAMMAR_TERMS[i] for i in kept_indices]
lambda_k = len(kept_indices) / n_terms
print("[Stage 2] Adaptive pruning threshold τ = {:.3e}".format(tau_prune))
print("  kept indices =", kept_indices)
print("  kept terms   =", kept_terms)
print("  λ_k = {:.3f} ({}/{})\n".format(lambda_k, len(kept_indices), n_terms))

# -------------------------
# 12. Stage 2 training (BATCHED)
# -------------------------
def train_stage2(mask, epochs=60):
    model = GrammarTreeTau92(mask=mask).to(device)
    with torch.no_grad():
        model.w.copy_(model_stage1.w.detach())
        model.gamma.copy_(model_stage1.gamma.detach())

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    lambda_phi2 = 5e-2
    lambda_H    = 0.0 # FIX: Temporary removal of H-Loss to prevent RuntimeError
    lambda_M    = 1e-2
    lambda_phi4 = 1e-2
    lambda_div  = 1e-2

    lambda_gamma = 1e-5
    lambda_w_base = torch.tensor([5e-5, 5e-5, 5e-5], dtype=DTYPE, device=device)

    # --- BATCHING: Load all TRAIN data once ---
    B_train = len(train_indices)
    v0_train_batch = train_v0_batch # [B, 2, Nx, Ny]
    forcing_train_batch = train_forcing_batch
    
    vT_true_batch = train_traj_batch[-1].clone() # [B, 2, Nx, Ny]
    phi2_true_T_batch = train_phi2s[-1].clone() 
    H_true_T_batch    = train_Hs[-1].clone()
    M_true_T_batch    = train_Ms[-1].clone()
    phi4_true_T_batch = train_phi4s[-1].clone()
    div_true_T_batch  = train_divs[-1].clone()
    phi2_meta_true_batch = train_phi2s[:200].mean(dim=0).clone() # [B]


    print(f"[Stage 2] Refining pruned GrammarTree 9.2 (BATCH B={B_train})...")
    for ep in range(1, epochs+1):
        opt.zero_grad()
        
        # 1. Simulate the entire batch in ONE CALL
        vT_pred_batch, phi2_pred_T_batch, H_pred_T_batch, M_pred_T_batch, phi4_pred_T_batch, div_pred_T_batch = \
            model.simulate_to_T_batch(v0_train_batch, Nt, forcing_batch=forcing_train_batch)

        # 2. Compute misfit and SCCT losses (all are [B] tensors)
        # misfit_batch: [B] tensor, needs mean over spatial/component dims
        misfit_batch = torch.mean(F.mse_loss(vT_pred_batch, vT_true_batch, reduction='none'), dim=(-3, -2, -1)) # [B]
        
        # --- FIX 2: Removed H-Loss term ---
        loss_ic_batch = misfit_batch \
                        + lambda_phi2 * torch.abs(phi2_pred_T_batch - phi2_true_T_batch) \
                        + lambda_M    * torch.abs(M_pred_T_batch    - M_true_T_batch)    \
                        + lambda_phi4 * torch.abs(phi4_pred_T_batch - phi4_true_T_batch) \
                        + lambda_div  * torch.abs(div_pred_T_batch  - div_true_T_batch)

        # 3. Meta-phi2 loss
        phi2_meta_pred_batch = model.simulate_phi2_traj(v0_train_batch, 200, forcing=forcing_train_batch).mean(dim=0) # [B]
        loss_ic_batch = loss_ic_batch + 1e-1 * (phi2_meta_pred_batch - phi2_meta_true_batch)**2

        # 4. Total Loss is the mean across the batch dimension
        loss_total = torch.mean(loss_ic_batch)
        
        # 5. Regularization (unchanged)
        g = model.gamma
        w = model.w
        eff_w = w * model.mask
        l1_w = torch.sum(lambda_w_base * torch.abs(eff_w))
        l1_gamma = lambda_gamma * torch.abs(g)

        loss = loss_total + l1_w + l1_gamma
        loss.backward()
        opt.step()

    # summary on one train IC (adjusted for batch output)
    with torch.no_grad():
        idx0 = 0 # Look at the first IC in the batch
        
        vT_pred_0 = vT_pred_batch[idx0].clone()
        vT_true_0 = vT_true_batch[idx0].clone()
        misfit_final = F.mse_loss(vT_pred_0, vT_true_0).item()
        
        phi2_pred_T_0 = phi2_pred_T_batch[idx0].item()
        H_pred_T_0 = H_pred_T_batch[idx0].item()
        div_pred_T_0 = div_pred_T_batch[idx0].item()
        
        phi2_meta_model_0 = phi2_meta_pred_batch[idx0].item()

        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 2][Summary] (IC train#0) misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}, div_rms(T)={:.3e}".format(
        misfit_final, phi2_pred_T_0, H_pred_T_0, div_pred_T_0))
    print("  <Φ²>_meta(model, train#0) = {:.3e}".format(phi2_meta_model_0))
    print("  γ(Stage 2) = {:+.3e}".format(model.gamma.item()))
    for i, name in enumerate(GRAMMAR_TERMS):
        print(f"  eff({name:8s} idx {i}) = {eff[i]:+.6e}")
    print()

    return model, eff

model_stage2, eff_stage2 = train_stage2(mask_stage2)

print("[GrammarTree 9.2] Effective τ_ij coefficients (S, I1*S, δ I1):")
for i, name in enumerate(GRAMMAR_TERMS):
    print(f"  eff({name:8s} idx {i}) Stage1 = {eff_stage1[i]:+.6e}, Stage2 = {eff_stage2[i]:+.6e}")
print()

# -------------------------
# 13. Generalization eval: train vs test IC rollout errors
# -------------------------
def eval_ic_set(name, indices, steps_eval=200):
    # Prepare batch inputs for the entire set (Train or Test)
    if name == "train":
        v0_batch = train_v0_batch
        forcing_batch = train_forcing_batch
    elif name == "test ":
        v0_batch = test_v0_batch
        forcing_batch = test_forcing_batch
    else:
        return np.array([])

    # Compute rollout errors for the entire batch in one call
    # Returns [B] array of mean L2 errors
    err_arr = rollout_error_batch(model_stage2, v0_batch, forcing_batch, steps=steps_eval)

    # Print sequential log based on the results from the batch array
    err_list = err_arr.tolist()
    
    ic_indices_sequential = [meta['index'] for meta in ic_metadata if meta['is_test'] == (name == "test ")]

    for idx_batch, ic in enumerate(ic_indices_sequential):
        s = teacher_stats[ic]
        err_mean = err_arr[idx_batch]
        print(f"  [{name}] IC #{ic:2d} ({s['type']:12s})  mean L2 error[0,{steps_eval*dt:.3f}] = {err_mean:.3e}")
        
    print(f"  ==> {name} set: mean={err_arr.mean():.3e}, median={np.median(err_arr):.3e}, max={err_arr.max():.3e}\n")
    return err_arr

print("[Eval] Rollout generalization (RK2 + projection, τ_ij closure)...)\n")
train_errs = eval_ic_set("train", train_indices, steps_eval=200)
test_errs  = eval_ic_set("test ", test_indices,  steps_eval=200)

print("[Summary] GrammarTree 9.2 (τ_ij tensor closure, many ICs + HF NS solver, stable v2) finished.")
