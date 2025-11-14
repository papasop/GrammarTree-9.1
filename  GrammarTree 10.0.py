# ============================================
# GrammarTree 10.0 - τ_ij Boundary Test (3D MINIMAL RESOURCE CHALLENGE)
#  - GOAL: Test if the 2-term structure holds under the full complexity of 3D turbulence.
#  - CHANGE: D=3, Nx=Ny=Nz=16 (Minimal resolution).
#  - BASES: Expanded 5 bases (S, I1*S, δI1, S_sq, R_sq) - 3D tensor forms.
# ============================================

# -------------------------
# 0. Install dependencies
# -------------------------
!pip install -q numpy torch matplotlib

# -------------------------
# 1. Imports & config
# -------------------------
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["figure.dpi"] = 120

# Setup device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("[Init] Using device:", device)
print("[Init] Random seed set to", SEED)
print("[Init] GrammarTree 10.0 τ_ij (3D MINIMAL) header OK.\n")

# -------------------------
# 2. Grid & basic ops (3D periodic)
# -------------------------
Nx, Ny, Nz = 16, 16, 16 # <<< MINIMAL 3D RESOLUTION
x = torch.linspace(0.0, 1.0, Nx, dtype=DTYPE)
y = torch.linspace(0.0, 1.0, Ny, dtype=DTYPE)
z = torch.linspace(0.0, 1.0, Nz, dtype=DTYPE)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
dz = float(z[1] - z[0])

# --- Stability Configuration (HIGH RE) ---
nu = 0.01 
dt = 5e-05
Nt = 1000          # T = 0.05 (Shorter integration time)
T = Nt * dt
# --------------------------------------------------------

print(f"[Grid] Nx={Nx}, Ny={Ny}, Nz={Nz}, dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}, dt={dt:.1e}, Nt={Nt}, T={T:.3f}, nu={nu}")

X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
X = X.to(device)
Y = Y.to(device)
Z = Z.to(device)

def roll3d(u, sx, sy, sz):
    # u: [..., Nx, Ny, Nz]
    return torch.roll(torch.roll(torch.roll(u, shifts=sx, dims=-3), shifts=sy, dims=-2), shifts=sz, dims=-1)

def grad3d(u, dx, dy, dz):
    dudx = (roll3d(u, -1, 0, 0) - roll3d(u, +1, 0, 0)) / (2.0 * dx)
    dudy = (roll3d(u, 0, -1, 0) - roll3d(u, 0, +1, 0)) / (2.0 * dy)
    dudz = (roll3d(u, 0, 0, -1) - roll3d(u, 0, 0, +1)) / (2.0 * dz)
    return dudx, dudy, dudz

def laplace3d(u, dx, dy, dz):
    u_xx = (roll3d(u, -1, 0, 0) - 2.0*u + roll3d(u, +1, 0, 0)) / (dx*dx)
    u_yy = (roll3d(u, 0, -1, 0) - 2.0*u + roll3d(u, 0, +1, 0)) / (dy*dy)
    u_zz = (roll3d(u, 0, 0, -1) - 2.0*u + roll3d(u, 0, 0, +1)) / (dz*dz)
    return u_xx + u_yy + u_zz

def div3d(vx, vy, vz, dx, dy, dz):
    dvx_dx, _, _ = grad3d(vx, dx, dy, dz)
    _, dvy_dy, _ = grad3d(vy, dx, dy, dz)
    _, _, dvz_dz = grad3d(vz, dx, dy, dz)
    return dvx_dx + dvy_dy + dvz_dz

# -------------------------
# 3. Pressure Poisson (FFT, batched)
# -------------------------
kx = (2.0 * np.pi) * torch.fft.fftfreq(Nx, d=dx)
ky = (2.0 * np.pi) * torch.fft.fftfreq(Ny, d=dy)
kz = (2.0 * np.pi) * torch.fft.fftfreq(Nz, d=dz)
KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
KX = KX.to(device)
KY = KY.to(device)
KZ = KZ.to(device)

# 3D Laplacian Inverse Kernel
LAPLACE_FFT_INV = - (KX**2 + KY**2 + KZ**2)
LAPLACE_FFT_INV[0,0,0] = 1.0
LAPLACE_FFT_INV = 1.0 / LAPLACE_FFT_INV
LAPLACE_FFT_INV[0,0,0] = 0.0

def poisson_fft_solve(rhs):
    # Dims are (-3, -2, -1) for 3D FFT
    rhs_hat = torch.fft.fftn(rhs, dim=(-3,-2,-1))
    p_hat = rhs_hat * LAPLACE_FFT_INV
    p = torch.fft.ifftn(p_hat, dim=(-3,-2,-1)).real.to(DTYPE)
    return p

def project_incompressible(vx, vy, vz, dx, dy, dz, rho=1.0):
    div_v = div3d(vx, vy, vz, dx, dy, dz)
    rhs = div_v / dt
    p = poisson_fft_solve(rhs)
    dpdx, dpdy, dpdz = grad3d(p, dx, dy, dz)
    vx_new = vx - dt * dpdx / rho
    vy_new = vy - dt * dpdy / rho
    vz_new = vz - dt * dpdz / rho
    return vx_new, vy_new, vz_new, p

# -------------------------
# 4. SCCT stats (with batch entropy H)
# -------------------------
def scct_stats_vec(v, nbins=32):
    vx, vy, vz = v[:,0], v[:,1], v[:,2]
    mag2 = vx*vx + vy*vy + vz*vz
    mag  = torch.sqrt(mag2 + 1e-16)

    phi2 = mag2.mean(dim=(-3,-2,-1))
    phi4 = (mag2*mag2).mean(dim=(-3,-2,-1))
    M = vx.mean(dim=(-3,-2,-1))
    
    # --- Batch Entropy H ---
    B, Nx_, Ny_, Nz_ = mag.shape
    flat = mag.view(B, -1)
    mmin = flat.min(dim=1, keepdim=True).values
    mmax = flat.max(dim=1, keepdim=True).values
    mspan = (mmax - mmin + 1e-8)
    norm = (flat - mmin) / mspan
    idx = torch.clamp((norm * nbins).long(), 0, nbins-1)
    hist = torch.zeros(B, nbins, device=flat.device, dtype=DTYPE)
    hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=DTYPE))
    p = hist / (Nx_*Ny_*Nz_ + 1e-8)
    H = - (p * (p + 1e-12).log()).sum(dim=1)

    return phi2, H, M, phi4

def div_rms(v):
    vx, vy, vz = v[:,0], v[:,1], v[:,2]
    d = div3d(vx, vy, vz, dx, dy, dz)
    d_sq = d*d
    return torch.sqrt(d_sq.mean(dim=(-3,-2,-1)) + 1e-16)

# -------------------------
# 5. Teacher τ_ij closure (3D Tensor Ops)
# -------------------------
a_tau = -4.0e-4
b_tau = -2.0e-4

# Indices for 3D symmetric stress tensor (6 independent components)
# (xx, yy, zz, xy, xz, yz)
TENSOR_COMPONENTS = 6 

def velocity_gradients(vx, vy, vz):
    dvx_dx, dvx_dy, dvx_dz = grad3d(vx, dx, dy, dz)
    dvy_dx, dvy_dy, dvy_dz = grad3d(vy, dx, dy, dz)
    dvz_dx, dvz_dy, dvz_dz = grad3d(vz, dx, dy, dz)
    return dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz

def strain_tensor(grads):
    (dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz) = grads
    Sxx = dvx_dx
    Syy = dvy_dy
    Szz = dvz_dz
    Sxy = 0.5 * (dvx_dy + dvy_dx)
    Sxz = 0.5 * (dvx_dz + dvz_dx)
    Syz = 0.5 * (dvy_dz + dvz_dy)
    # Return 6 unique components
    return Sxx, Syy, Szz, Sxy, Sxz, Syz

def rotation_tensor(grads):
    (dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz) = grads
    Rxx = torch.zeros_like(dvx_dx)
    Ryy = torch.zeros_like(dvy_dy)
    Rzz = torch.zeros_like(dvz_dz)
    Rxy = 0.5 * (dvx_dy - dvy_dx)
    Rxz = 0.5 * (dvx_dz - dvz_dx)
    Ryz = 0.5 * (dvy_dz - dvz_dy)
    # Return 6 unique components (R is skew-symm, R_yx = -R_xy, etc.)
    return Rxx, Ryy, Rzz, Rxy, Rxz, Ryz # 3 main, 3 off-diag (upper triangle)

def invariants_I1(Sxx, Syy, Szz, Sxy, Sxz, Syz):
    # I1 = trace(S^2)
    return Sxx*Sxx + Syy*Syy + Szz*Szz + 2.0*(Sxy*Sxy + Sxz*Sxz + Syz*Syz)

def teacher_tau(v):
    grads = velocity_gradients(v[:,0], v[:,1], v[:,2])
    Sxx, Syy, Szz, Sxy, Sxz, Syz = strain_tensor(grads)
    I1 = invariants_I1(Sxx, Syy, Szz, Sxy, Sxz, Syz)
    
    # Linear + Non-linear closure (3D)
    tau_xx = a_tau*Sxx + b_tau*I1*Sxx
    tau_yy = a_tau*Syy + b_tau*I1*Syy
    tau_zz = a_tau*Szz + b_tau*I1*Szz
    tau_xy = a_tau*Sxy + b_tau*I1*Sxy
    tau_xz = a_tau*Sxz + b_tau*I1*Sxz
    tau_yz = a_tau*Syz + b_tau*I1*Syz

    return tau_xx, tau_yy, tau_zz, tau_xy, tau_xz, tau_yz

def div_tau(tau_comps):
    txx, tyy, tzz, txy, txz, tyz = tau_comps
    
    # Reconstruct the full symmetric tensor components for gradient calculation
    tyx, tzx, tzy = txy, txz, tyz 
    
    # x-component of divergence: ∂_x τ_xx + ∂_y τ_yx + ∂_z τ_zx
    dtxx_dx, _, _ = grad3d(txx, dx, dy, dz)
    _, dtyx_dy, _ = grad3d(tyx, dx, dy, dz)
    _, _, dtzx_dz = grad3d(tzx, dx, dy, dz)
    div_x = dtxx_dx + dtyx_dy + dtzx_dz
    
    # y-component of divergence: ∂_x τ_xy + ∂_y τ_yy + ∂_z τ_zy
    dtxy_dx, _, _ = grad3d(txy, dx, dy, dz)
    _, dtyy_dy, _ = grad3d(tyy, dx, dy, dz)
    _, _, dtzy_dz = grad3d(tzy, dx, dy, dz)
    div_y = dtxy_dx + dtyy_dy + dtzy_dz
    
    # z-component of divergence: ∂_x τ_xz + ∂_y τ_yz + ∂_z τ_zz
    dtxz_dx, _, _ = grad3d(txz, dx, dy, dz)
    _, dtyz_dy, _ = grad3d(tyz, dx, dy, dz)
    _, _, dtzz_dz = grad3d(tzz, dx, dy, dz)
    div_z = dtxz_dx + dtyz_dy + dtzz_dz
    
    return div_x, div_y, div_z

# -------------------------
# 6. Teacher NS+τ RHS & RK2 (3D batched)
# -------------------------
def teacher_rhs(v, forcing=None):
    vx, vy, vz = v[:,0], v[:,1], v[:,2]
    grads = velocity_gradients(vx, vy, vz)
    (dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz) = grads

    adv_x = vx*dvx_dx + vy*dvx_dy + vz*dvx_dz
    adv_y = vx*dvy_dx + vy*dvy_dy + vz*dvy_dz
    adv_z = vx*dvz_dx + vy*dvz_dy + vz*dvz_dz

    lap_vx = laplace3d(vx, dx, dy, dz)
    lap_vy = laplace3d(vy, dx, dy, dz)
    lap_vz = laplace3d(vz, dx, dy, dz)

    tau_comps = teacher_tau(v)
    div_tx, div_ty, div_tz = div_tau(tau_comps)

    rhs_x = -adv_x + nu*lap_vx + div_tx
    rhs_y = -adv_y + nu*lap_vy + div_ty
    rhs_z = -adv_z + nu*lap_vz + div_tz
    
    if forcing is not None:
        rhs_x = rhs_x + forcing[:,0]
        rhs_y = rhs_y + forcing[:,1]
        rhs_z = rhs_z + forcing[:,2]
        
    return torch.stack([rhs_x, rhs_y, rhs_z], dim=1)

def rk2_step(v, rhs_fun, proj=True, forcing=None):
    k1 = rhs_fun(v, forcing=forcing)
    v_star = v + dt*k1
    if proj:
        vx_s, vy_s, vz_s, _ = project_incompressible(v_star[:,0], v_star[:,1], v_star[:,2], dx, dy, dz)
        v_star = torch.stack([vx_s, vy_s, vz_s], dim=1)
    k2 = rhs_fun(v_star, forcing=forcing)
    v_new = v + 0.5*dt*(k1 + k2)
    if proj:
        vx_n, vy_n, vz_n, _ = project_incompressible(v_new[:,0], v_new[:,1], v_new[:,2], dx, dy, dz)
        v_new = torch.stack([vx_n, vy_n, vz_n], dim=1)
    return v_new

def simulate_teacher_batch(v0_batch, Nt, forcing_batch=None):
    B = v0_batch.shape[0]
    v = v0_batch.clone()
    # Trajectory dimension: [Nt, B, 3, Nx, Ny, Nz]
    traj    = torch.zeros((Nt,B,3,Nx,Ny,Nz), dtype=DTYPE, device=device)
    phi2s   = torch.zeros((Nt,B), dtype=DTYPE, device=device)
    Hs      = torch.zeros((Nt,B), dtype=DTYPE, device=device)
    divs    = torch.zeros((Nt,B), dtype=DTYPE, device=device)
    
    print(f"  [Sim] Running {B} ICs for {Nt} steps ({Nx}³ grid)...")
    for n in range(Nt):
        v = rk2_step(v, teacher_rhs, proj=True, forcing=forcing_batch)
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"  [Sim] NaN/Inf detected at step {n}!")
            break
        
        # Save only last snapshot for resource optimization
        if n == Nt-1:
            traj[-1] = v
        
        # Only compute stats for key points to save time
        if n % 100 == 0 or n == Nt-1:
            phi2,H,M,phi4 = scct_stats_vec(v)
            phi2s[n] = phi2
            Hs[n]    = H
            divs[n]  = div_rms(v)
            
    return traj, phi2s, Hs, divs

# -------------------------
# 7. IC & forcing library (3D TGV Focus)
# -------------------------
def make_ic_tgv_3d(seed_offset=0, amp=0.5):
    """3D Taylor-Green Vortex (TGV) IC"""
    torch.manual_seed(SEED + seed_offset)
    k = 2.0*np.pi 
    vx = amp * torch.sin(k*X) * torch.cos(k*Y) * torch.cos(k*Z)
    vy = -amp * torch.cos(k*X) * torch.sin(k*Y) * torch.cos(k*Z)
    vz = torch.zeros_like(vx) # Zero initial vz for simpler TGV
    
    # Add small random noise for transition
    noise = 0.05 * torch.randn_like(vx) 
    vz += noise # Introduce initial 3D perturbation

    # Ensure incompressibility after noise (important for stability)
    v_st = torch.stack([vx, vy, vz], dim=1).transpose(0, 1) # [3, Nx, Ny, Nz]
    vx_proj, vy_proj, vz_proj, _ = project_incompressible(v_st[0], v_st[1], v_st[2], dx, dy, dz)
    return torch.stack([vx_proj, vy_proj, vz_proj], dim=0)

# --- Assemble train / test batches (3D TGV) ---
n_train_tgv_3d  = 4
n_test_tgv_3d   = 2

train_v0_list, train_Force_list = [], []
test_v0_list,  test_Force_list  = [], []
ic_metadata = []
ic_counter  = 0

# train TGV 3D (4 ICs)
for i in range(n_train_tgv_3d):
    v0 = make_ic_tgv_3d(seed_offset=10*i, amp=0.45+0.05*i)
    train_v0_list.append(v0)
    train_Force_list.append(torch.zeros_like(v0))
    ic_metadata.append({"type":"TGV-3D", "is_test":False, "index":ic_counter})
    ic_counter += 1

# test TGV 3D (2 ICs)
for i in range(n_test_tgv_3d):
    v0 = make_ic_tgv_3d(seed_offset=100+10*i, amp=0.55+0.05*i)
    test_v0_list.append(v0)
    test_Force_list.append(torch.zeros_like(v0))
    ic_metadata.append({"type":"TGV-3D-test", "is_test":True, "index":ic_counter})
    ic_counter += 1

train_v0_batch = torch.stack(train_v0_list, dim=0).to(device)
train_Force_batch  = torch.stack(train_Force_list,  dim=0).to(device)
test_v0_batch  = torch.stack(test_v0_list,  dim=0).to(device)
test_Force_batch   = torch.stack(test_Force_list,   dim=0).to(device)

train_indices = [i for i,m in enumerate(ic_metadata) if not m["is_test"]]
test_indices  = [i for i,m in enumerate(ic_metadata) if m["is_test"]]
n_ic_total    = len(ic_metadata)

print("[Teacher] Generating batched NS+τ trajectories (train/test)...")
t0 = time.time()
train_traj, train_phi2, train_H, train_div = simulate_teacher_batch(train_v0_batch, Nt, train_Force_batch)
test_traj,  test_phi2,  test_H,  test_div  = simulate_teacher_batch(test_v0_batch,  Nt, test_Force_batch)
t1 = time.time()

print(f"  [train      ] Φ²(T)={train_phi2[-1].mean().item():.3e}, H(T)={train_H[-1].mean().item():.3f}, "
      f"<div>_rms={train_div[-1].mean().item():.3e}")
print(f"  [test       ] Φ²(T)={test_phi2[-1].mean().item():.3e}, H(T)={test_H[-1].mean().item():.3f}, "
      f"<div>_rms={test_div[-1].mean().item():.3e}")
print(f"[Teacher] Done in {t1-t0:.2f}s, train={len(train_indices)}, test={len(test_indices)}\n")

# --- Pre-extract GT data for loss function ---
vT_true_batch = train_traj[-1].clone()
phi2_T_gt = train_phi2[-1].clone()
H_T_gt    = train_H[-1].clone()
div_T_gt  = train_div[-1].clone()
# Note: phi2_meta is skipped for minimal 3D resource

# -------------------------
# 8. GrammarTree 10.0 τ_ij model (EXPANDED BASES - 3D)
# -------------------------
# Same 5 conceptual bases, now with 3D tensor components (6 comps: xx,yy,zz,xy,xz,yz)
GRAMMAR_TERMS = ["S", "I1*S", "δ I1", "S_sq", "R_sq"] 
n_terms = len(GRAMMAR_TERMS)
print("[GrammarTree 10.0] tensor τ_ij bases (3D) =", GRAMMAR_TERMS, "\n")

def matrix_sq_sym_3d(Axx, Ayy, Azz, Axy, Axz, Ayz):
    """
    3x3 对称张量 A 的平方 C = A @ A 的 6 个独立分量：
    Cxx, Cyy, Czz, Cxy, Cxz, Cyz
    约定：Axy = Ayx, Axz = Azx, Ayz = Azy
    """
    Cxx = Axx*Axx + Axy*Axy + Axz*Axz
    Cyy = Axy*Axy + Ayy*Ayy + Ayz*Ayz
    Czz = Axz*Axz + Ayz*Ayz + Azz*Azz
    
    Cxy = Axx*Axy + Axy*Ayy + Axz*Ayz
    Cxz = Axx*Axz + Axy*Ayz + Axz*Azz
    Cyz = Axy*Axz + Ayy*Ayz + Ayz*Azz
    
    return Cxx, Cyy, Czz, Cxy, Cxz, Cyz

def matrix_sq_skew_3d(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz):
    """
    3x3 反对称张量 R 的平方 C = R @ R 的 6 个独立分量。
    约定：Rxx=Ryy=Rzz=0（这里保留参数以接口统一），
    R_yx = -R_xy, R_zx = -R_xz, R_zy = -R_yz。
    """
    Cxx = -(Rxy*Rxy + Rxz*Rxz)
    Cyy = -(Rxy*Rxy + Ryz*Ryz)
    Czz = -(Rxz*Rxz + Ryz*Ryz)
    
    Cxy = -Rxz*Ryz
    Cxz =  Rxy*Ryz
    Cyz = -Rxy*Rxz
    
    return Cxx, Cyy, Czz, Cxy, Cxz, Cyz

def tau_features(v):
    grads = velocity_gradients(v[:,0], v[:,1], v[:,2])
    Sxx, Syy, Szz, Sxy, Sxz, Syz = strain_tensor(grads)
    Rxx, Ryy, Rzz, Rxy, Rxz, Ryz = rotation_tensor(grads) # Full 6 components

    I1 = invariants_I1(Sxx, Syy, Szz, Sxy, Sxz, Syz)
    z  = torch.zeros_like(I1)
    
    # T0: S_ij
    f0 = torch.stack([Sxx, Syy, Szz, Sxy, Sxz, Syz], dim=1) # [B, 6, Nx, Ny, Nz]
    
    # T1: I1*S_ij
    f1 = torch.stack([I1*Sxx, I1*Syy, I1*Szz, I1*Sxy, I1*Sxz, I1*Syz], dim=1)
    
    # T2: δ_ij * I1
    f2 = torch.stack([I1, I1, I1, z, z, z], dim=1)
    
    # T3: S^2 (6 unique components, 对称张量平方)
    S2_xx, S2_yy, S2_zz, S2_xy, S2_xz, S2_yz = matrix_sq_sym_3d(Sxx, Syy, Szz, Sxy, Sxz, Syz)
    f3 = torch.stack([S2_xx, S2_yy, S2_zz, S2_xy, S2_xz, S2_yz], dim=1)

    # T4: R^2 (6 unique components, 反对称张量平方)
    R2_xx, R2_yy, R2_zz, R2_xy, R2_xz, R2_yz = matrix_sq_skew_3d(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz)
    f4 = torch.stack([R2_xx, R2_yy, R2_zz, R2_xy, R2_xz, R2_yz], dim=1)

    feats = torch.stack([f0,f1,f2,f3,f4], dim=0)  # [n_terms, B, 6, Nx, Ny, Nz]
    return feats


class GrammarTreeTau100(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_terms, dtype=DTYPE))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        if mask is None:
            mask = torch.ones(n_terms, dtype=DTYPE)
        self.register_buffer("mask", mask)

    def tau(self, v):
        feats = tau_features(v) # [n_terms, B, 6, Nx, Ny, Nz]
        eff   = self.gamma * self.w * self.mask
        eff_r = eff.view(n_terms,1,1,1,1,1)
        tau_all = torch.sum(eff_r * feats, dim=0) # [B, 6, Nx, Ny, Nz]
        return tau_all

    def rhs(self, v, forcing=None):
        vx, vy, vz = v[:,0], v[:,1], v[:,2]
        grads = velocity_gradients(vx, vy, vz)
        (dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz) = grads

        adv_x = vx*dvx_dx + vy*dvx_dy + vz*dvx_dz
        adv_y = vx*dvy_dx + vy*dvy_dy + vz*dvy_dz
        adv_z = vx*dvz_dx + vy*dvz_dy + vz*dvz_dz

        lap_vx = laplace3d(vx, dx, dy, dz)
        lap_vy = laplace3d(vy, dx, dy, dz)
        lap_vz = laplace3d(vz, dx, dy, dz)
        
        tau_all = self.tau(v)
        # Components: (xx, yy, zz, xy, xz, yz)
        tau_comps = tau_all[:,0], tau_all[:,1], tau_all[:,2], tau_all[:,3], tau_all[:,4], tau_all[:,5]
        div_tx, div_ty, div_tz = div_tau(tau_comps)

        rhs_x = -adv_x + nu*lap_vx + div_tx
        rhs_y = -adv_y + nu*lap_vy + div_ty
        rhs_z = -adv_z + nu*lap_vz + div_tz
        
        if forcing is not None:
            rhs_x = rhs_x + forcing[:,0]
            rhs_y = rhs_y + forcing[:,1]
            rhs_z = rhs_z + forcing[:,2]

        return torch.stack([rhs_x,rhs_y,rhs_z], dim=1)

    def step(self, v, forcing=None):
        return rk2_step(v, self.rhs, proj=True, forcing=forcing)

    def simulate_to_T_batch(self, v0_batch, Nt, forcing_batch=None):
        v = v0_batch.clone()
        # Only last snapshot is needed for loss calculation
        for n in range(Nt):
            v = self.step(v, forcing=forcing_batch)
        phi2,H,M,phi4 = scct_stats_vec(v)
        return v, phi2, H, M, phi4, div_rms(v)


# -------------------------
# 9. Rollout error (batched) - 3D
# -------------------------
def rollout_error_batch(model, v0_batch, forcing_batch, steps):
    v_t = v0_batch.clone()
    v_m = v0_batch.clone()
    errs = []
    with torch.no_grad():
        for _ in range(steps):
            v_t = rk2_step(v_t, teacher_rhs, proj=True, forcing=forcing_batch)
            v_m = model.step(v_m, forcing=forcing_batch)
            # MSE over all 3 velocity components and the 3D domain
            err_batch = ((v_t - v_m)**2).mean(dim=(-4,-3,-2,-1))
            errs.append(err_batch.cpu().numpy())
    return np.array(errs).mean(axis=0)

# -------------------------
# 10. Stage 1 (Training)
# -------------------------
def train_stage1(epochs=60): # Reduced epochs due to 3D time
    model = GrammarTreeTau100().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Adjusted loss weights for 3D TGV (no M constraint)
    lambda_phi2 = 5e-2
    lambda_H    = 1e-3
    lambda_div  = 1e-2

    lambda_gamma  = 1e-5
    lambda_w_base = torch.tensor([5e-5]*n_terms, dtype=DTYPE, device=device) 

    v0_train = train_v0_batch
    Force_train  = train_Force_batch 

    print(f"[Stage 1] Training GrammarTree 10.0-3D (B={v0_train.shape[0]})...")
    for ep in range(1, epochs+1):
        opt.zero_grad()
        # Simulate only to T (reduced Nt)
        vT_pred, phi2_T, H_T, M_T, phi4_T, div_T = model.simulate_to_T_batch(v0_train, Nt, Force_train)
        misfit_batch = ((vT_pred - vT_true_batch)**2).mean(dim=(-4,-3,-2,-1)) # 3D MSE

        loss_ic = misfit_batch \
                  + lambda_phi2 * torch.abs(phi2_T - phi2_T_gt) \
                  + lambda_H    * torch.abs(H_T    - H_T_gt)    \
                  + lambda_div  * torch.abs(div_T  - div_T_gt)

        loss_total = loss_ic.mean()
        g  = model.gamma
        w  = model.w
        eff_w = w * model.mask
        l1_w = torch.sum(lambda_w_base * torch.abs(eff_w))
        l1_g = lambda_gamma * torch.abs(g)

        loss = loss_total + l1_w + l1_g
        loss.backward()
        opt.step()
        
    # Summary
    with torch.no_grad():
        vT_pred, phi2_T, H_T, M_T, phi4_T, div_T = model.simulate_to_T_batch(v0_train, Nt, Force_train)
        misfit_final = ((vT_pred[0] - vT_true_batch[0])**2).mean().item()
        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 1][Summary] (IC train#0) misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}, div_rms(T)={:.3e}".format(
        misfit_final, phi2_T[0].item(), H_T[0].item(), div_T[0].item()))
    print("  γ(Stage 1) = {:+.3e}".format(model.gamma.item()))
    for i,name in enumerate(GRAMMAR_TERMS):
        print(f"  eff({name:8s} idx {i}) = {eff[i]:+.6e}")
    print()

    return model

# -------------------------
# 11. Pruning 
# -------------------------
def adaptive_pruning_mask(model, floor_threshold=1.0e-05, factor=0.4): 
    with torch.no_grad():
        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()
        abs_eff = np.abs(eff)
        med = np.median(abs_eff)
        tau = max(floor_threshold, factor*med)
        keep = abs_eff >= tau
    return torch.tensor(keep.astype(np.float32), device=device), tau, eff

model_stage1 = train_stage1()
mask_stage2, tau_prune, eff_stage1 = adaptive_pruning_mask(model_stage1)
kept_idx   = np.where(mask_stage2.cpu().numpy()>0.5)[0].tolist()
kept_terms = [GRAMMAR_TERMS[i] for i in kept_idx]
lambda_k   = len(kept_idx) / n_terms
print("[Stage 2] Adaptive pruning τ = {:.3e}".format(tau_prune))
print("  kept indices =", kept_idx)
print("  kept terms   =", kept_terms)
print("  λ_k = {:.3f} ({}/{})\n".format(lambda_k, len(kept_idx), n_terms))

# -------------------------
# 12. Stage 2 (refine pruned model)
# -------------------------
def train_stage2(mask, epochs=40): # Further reduced epochs
    model = GrammarTreeTau100(mask=mask).to(device)
    with torch.no_grad():
        model.w.copy_(model_stage1.w.detach())
        model.gamma.copy_(model_stage1.gamma.detach())

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    lambda_phi2 = 5e-2
    lambda_H    = 1e-3
    lambda_div  = 1e-2

    lambda_gamma  = 1e-5
    lambda_w_base = torch.tensor([5e-5]*n_terms, dtype=DTYPE, device=device)

    v0_train = train_v0_batch
    Force_train  = train_Force_batch 

    print(f"[Stage 2] Refining pruned GrammarTree 10.0-3D (B={v0_train.shape[0]})...")
    for ep in range(1, epochs+1):
        opt.zero_grad()
        vT_pred, phi2_T, H_T, M_T, phi4_T, div_T = model.simulate_to_T_batch(v0_train, Nt, Force_train)
        misfit_batch = ((vT_pred - vT_true_batch)**2).mean(dim=(-4,-3,-2,-1))

        loss_ic = misfit_batch \
                  + lambda_phi2 * torch.abs(phi2_T - phi2_T_gt) \
                  + lambda_H    * torch.abs(H_T    - H_T_gt)    \
                  + lambda_div  * torch.abs(div_T  - div_T_gt)

        loss_total = loss_ic.mean()
        g = model.gamma
        w = model.w
        eff_w = w*model.mask
        l1_w = torch.sum(lambda_w_base * torch.abs(eff_w))
        l1_g = lambda_gamma * torch.abs(g)
        loss = loss_total + l1_w + l1_g
        loss.backward()
        opt.step()

    # Summary
    with torch.no_grad():
        vT_pred, phi2_T, H_T, M_T, phi4_T, div_T = model.simulate_to_T_batch(v0_train, Nt, Force_train)
        misfit_final = ((vT_pred[0] - vT_true_batch[0])**2).mean().item()
        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 2][Summary] (IC train#0) misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}, div_rms(T)={:.3e}".format(
        misfit_final, phi2_T[0].item(), H_T[0].item(), div_T[0].item()))
    print("  γ(Stage 2) = {:+.3e}".format(model.gamma.item()))
    for i,name in enumerate(GRAMMAR_TERMS):
        print(f"  eff({name:8s} idx {i}) = {eff[i]:+.6e}")
    print()
    return model, eff_stage1, eff

model_stage2, eff_stage1, eff_stage2 = train_stage2(mask_stage2)

print("[GrammarTree 10.0-3D] Effective τ_ij coefficients (S, I1*S, δ I1, S_sq, R_sq):")
for i,name in enumerate(GRAMMAR_TERMS):
    print(f"  eff({name:8s} idx {i}) Stage1 = {eff_stage1[i]:+.6e}, Stage2 = {eff_stage2[i]:+.6e}")
print()

# -------------------------
# 13. Generalization eval - 3D
# -------------------------
def eval_ic_set(name, steps_eval=100): # Shortened eval steps
    if name == "train":
        v0_batch = train_v0_batch
        Force_batch  = train_Force_batch
        is_test_flag = False
    else:
        v0_batch = test_v0_batch
        Force_batch  = test_Force_batch
        is_test_flag = True

    err_arr = rollout_error_batch(model_stage2, v0_batch, Force_batch, steps_eval)
    ic_indices = [i for i,m in enumerate(ic_metadata) if m["is_test"]==is_test_flag]
    
    print(f"[Eval] {name} set rollout (steps={steps_eval}):")
    for i_batch, ic in enumerate(ic_indices):
        s = next(m for m in ic_metadata if m["index"]==ic)
        print(f"  [{name}] IC #{ic:2d} ({s['type']:12s})  mean L2 error[0,{steps_eval*dt:.3f}] = {err_arr[i_batch]:.3e}")
    
    err_arr_full = np.array(err_arr)
    print(f"  ==> {name} set: mean={err_arr_full.mean():.3e}, median={np.median(err_arr_full):.3e}, max={err_arr_full.max():.3e}\n")
    return err_arr_full

print("[Eval] Rollout generalization (RK2 + projection, τ_ij closure, 3D)...\n")
train_errs = eval_ic_set("train", steps_eval=100)
test_errs  = eval_ic_set("test ", steps_eval=100)

print("[Summary] GrammarTree 10.0-3D (τ_ij tensor closure + FINAL DIMENSIONAL CHALLENGE) finished.")
