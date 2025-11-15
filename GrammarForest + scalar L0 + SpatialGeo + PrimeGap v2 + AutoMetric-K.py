# ============================================================
# GrammarForest + scalar L0 + SpatialGeo + PrimeGap v2 + AutoMetric-K
# 专注：TGV 3D, TwoTerm teacher, L0 闭合, 时间几何 PrimeGap 正则
# ============================================================

# -------------------------
# 0. 基本依赖
# -------------------------
!pip install -q torch numpy matplotlib

import time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["figure.dpi"] = 120

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
SEED   = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("[Init] Device:", device)
print("[Init] Seed:", SEED)

# -------------------------
# 1. 网格 & 基本数值算子 (3D periodic)
# -------------------------
Nx = Ny = Nz = 16

x = torch.linspace(0.0, 1.0, Nx, dtype=DTYPE)
y = torch.linspace(0.0, 1.0, Ny, dtype=DTYPE)
z = torch.linspace(0.0, 1.0, Nz, dtype=DTYPE)

dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
dz = float(z[1] - z[0])

dt = 5e-5
Nt = 600

print(f"[Grid] Nx={Nx}, Ny={Ny}, Nz={Nz}, dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}, dt={dt}, Nt={Nt}")

X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
X = X.to(device)
Y = Y.to(device)
Z = Z.to(device)

def roll3d(u, sx, sy, sz):
    return torch.roll(torch.roll(torch.roll(u, sx, -3), sy, -2), sz, -1)

def grad3d(u, dx, dy, dz):
    dudx = (roll3d(u, -1, 0, 0) - roll3d(u, +1, 0, 0)) / (2.0 * dx)
    dudy = (roll3d(u,  0,-1, 0) - roll3d(u,  0,+1, 0)) / (2.0 * dy)
    dudz = (roll3d(u,  0, 0,-1) - roll3d(u,  0, 0,+1)) / (2.0 * dz)
    return dudx, dudy, dudz

def laplace3d(u, dx, dy, dz):
    u_xx = (roll3d(u, -1, 0, 0) - 2*u + roll3d(u, +1, 0, 0)) / (dx*dx)
    u_yy = (roll3d(u,  0,-1, 0) - 2*u + roll3d(u,  0,+1, 0)) / (dy*dy)
    u_zz = (roll3d(u,  0, 0,-1) - 2*u + roll3d(u,  0, 0,+1)) / (dz*dz)
    return u_xx + u_yy + u_zz

def div3d(vx, vy, vz, dx, dy, dz):
    dvx_dx, _, _ = grad3d(vx, dx, dy, dz)
    _, dvy_dy, _ = grad3d(vy, dx, dy, dz)
    _, _, dvz_dz = grad3d(vz, dx, dy, dz)
    return dvx_dx + dvy_dy + dvz_dz

# -------------------------
# 2. FFT 压力投影 (incompressible)
# -------------------------
kx = (2.0*math.pi)*torch.fft.fftfreq(Nx, d=dx)
ky = (2.0*math.pi)*torch.fft.fftfreq(Ny, d=dy)
kz = (2.0*math.pi)*torch.fft.fftfreq(Nz, d=dz)

KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
KX = KX.to(device); KY = KY.to(device); KZ = KZ.to(device)

LAPLACE_FFT_INV = - (KX**2 + KY**2 + KZ**2)
LAPLACE_FFT_INV[0,0,0] = 1.0
LAPLACE_FFT_INV = 1.0 / LAPLACE_FFT_INV
LAPLACE_FFT_INV[0,0,0] = 0.0

def poisson_fft_solve(rhs):
    rhs_hat = torch.fft.fftn(rhs, dim=(-3,-2,-1))
    p_hat   = rhs_hat * LAPLACE_FFT_INV
    p = torch.fft.ifftn(p_hat, dim=(-3,-2,-1)).real.to(DTYPE)
    return p

def project_incompressible(vx, vy, vz, dx, dy, dz, dt, rho=1.0):
    div_v = div3d(vx, vy, vz, dx, dy, dz)
    rhs = div_v / dt
    p = poisson_fft_solve(rhs)
    dpdx, dpdy, dpdz = grad3d(p, dx, dy, dz)
    vx_new = vx - dt*dpdx/rho
    vy_new = vy - dt*dpdy/rho
    vz_new = vz - dt*dpdz/rho
    return vx_new, vy_new, vz_new, p

# -------------------------
# 3. SCCT 指标：Φ², H, K
# -------------------------
def scct_stats_vec(v, dx, dy, dz, nbins=32):
    """
    v: [B,3,Nx,Ny,Nz]
    return: phi2[B], H[B]
    """
    vx, vy, vz = v[:,0], v[:,1], v[:,2]
    mag2 = vx*vx + vy*vy + vz*vz
    mag  = torch.sqrt(mag2 + 1e-16)
    phi2 = mag2.mean(dim=(-3,-2,-1))  # [B]

    B, Nx_, Ny_, Nz_ = mag.shape
    flat = mag.view(B, -1)

    mmin = flat.min(dim=1, keepdim=True).values
    mmax = flat.max(dim=1, keepdim=True).values
    span = (mmax - mmin + 1e-8)
    norm = (flat - mmin)/span

    idx = torch.clamp((norm*nbins).long(), 0, nbins-1)
    hist = torch.zeros(B, nbins, device=flat.device, dtype=DTYPE)
    hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=DTYPE))
    p = hist / (Nx_*Ny_*Nz_ + 1e-8)
    H = - (p * (p + 1e-12).log()).sum(dim=1)
    return phi2, H

def K_from_phi2_H(phi2, H):
    return phi2 / (H + 1e-8)

# -------------------------
# 4. L0：标量场算子 (graph Laplacian on |u|)
# -------------------------
def build_L0_scalar_operator(Nx, Ny, Nz, radius=1, device=device):
    """
    标量 L0: 在标量节点上构造带 radius 邻域的图 Laplacian: L = deg - A
    然后 L0 = L - B (轻微对角)
    """
    print("[L0-scalar] Build graph operator ...")
    V = Nx*Ny*Nz

    def idx3(i,j,k):
        return i + Nx*(j + Ny*k)

    edges_i = []
    edges_j = []

    # radius-邻域 (曼哈顿距离 <= radius)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                v = idx3(i,j,k)
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        for dk in range(-radius, radius+1):
                            if di==0 and dj==0 and dk==0:
                                continue
                            if abs(di)+abs(dj)+abs(dk) > radius:
                                continue
                            ni = (i+di) % Nx
                            nj = (j+dj) % Ny
                            nk = (k+dk) % Nz
                            u = idx3(ni,nj,nk)
                            edges_i.append(v)
                            edges_j.append(u)

    E = len(edges_i)
    edges_i = torch.tensor(edges_i, dtype=torch.long, device=device)
    edges_j = torch.tensor(edges_j, dtype=torch.long, device=device)

    L = torch.zeros(V, V, dtype=DTYPE, device=device)
    # Laplacian: L = deg - A
    for e in range(E):
        i = edges_i[e].item()
        j = edges_j[e].item()
        L[i,i] += 1.0
        L[i,j] -= 1.0

    B = 1e-6 * torch.eye(V, dtype=DTYPE, device=device)
    L0 = L - B
    print(f"[L0-scalar] shape: {L0.shape}, radius={radius}, E={E}")
    return L0

L0_scalar_mat = build_L0_scalar_operator(Nx,Ny,Nz,radius=2,device=device)

def L0_scalar_op(field):
    """
    field: [B,Nx,Ny,Nz] scalar field (e.g. |u|)
    返回 χ_rms
    """
    B = field.shape[0]
    V = Nx*Ny*Nz
    flat = field.view(B, V)             # [B,V]
    Lu   = flat @ L0_scalar_mat.T       # [B,V]
    chi  = Lu                           # χ = L0 u
    chi_norm = torch.sqrt((chi*chi).mean(dim=1) + 1e-20)  # [B]
    return chi_norm.mean()

# -------------------------
# 5. 梯度张量 & 不变量
# -------------------------
def velocity_gradients(vx, vy, vz, dx, dy, dz):
    dvx_dx, dvx_dy, dvx_dz = grad3d(vx, dx, dy, dz)
    dvy_dx, dvy_dy, dvy_dz = grad3d(vy, dx, dy, dz)
    dvz_dx, dvz_dy, dvz_dz = grad3d(vz, dx, dy, dz)
    return dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz

def strain_tensor(grads):
    dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz = grads
    Sxx = dvx_dx
    Syy = dvy_dy
    Szz = dvz_dz
    Sxy = 0.5*(dvx_dy + dvy_dx)
    Sxz = 0.5*(dvx_dz + dvz_dx)
    Syz = 0.5*(dvy_dz + dvz_dy)
    return Sxx,Syy,Szz,Sxy,Sxz,Syz

def rotation_tensor(grads):
    dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz = grads
    Rxx = torch.zeros_like(dvx_dx)
    Ryy = torch.zeros_like(dvy_dy)
    Rzz = torch.zeros_like(dvz_dz)
    Rxy = 0.5*(dvx_dy - dvy_dx)
    Rxz = 0.5*(dvx_dz - dvz_dx)
    Ryz = 0.5*(dvy_dz - dvz_dy)
    return Rxx,Ryy,Rzz,Rxy,Rxz,Ryz

def invariant_I1(Sxx,Syy,Szz,Sxy,Sxz,Syz):
    return Sxx**2 + Syy**2 + Szz**2 + 2*(Sxy**2 + Sxz**2 + Syz**2)

def matrix_sq_sym_3d(Axx,Ayy,Azz,Axy,Axz,Ayz):
    Cxx = Axx*Axx + Axy*Axy + Axz*Axz
    Cyy = Axy*Axy + Ayy*Ayy + Ayz*Ayz
    Czz = Axz*Axz + Ayz*Ayz + Azz*Azz
    Cxy = Axx*Axy + Axy*Ayy + Axz*Ayz
    Cxz = Axx*Axz + Axy*Ayz + Axz*Azz
    Cyz = Axy*Axz + Ayy*Ayz + Ayz*Azz
    return Cxx,Cyy,Czz,Cxy,Cxz,Cyz

def matrix_sq_skew_3d(Rxx,Ryy,Rzz,Rxy,Rxz,Ryz):
    Cxx = -(Rxy*Rxy + Rxz*Rxz)
    Cyy = -(Rxy*Rxy + Ryz*Ryz)
    Czz = -(Rxz*Rxz + Ryz*Ryz)
    Cxy = -Rxz*Ryz
    Cxz =  Rxy*Ryz
    Cyz = -Rxy*Rxz
    return Cxx,Cyy,Czz,Cxy,Cxz,Cyz

# -------------------------
# 6. Teacher: TwoTermInvariant
# -------------------------
class TeacherTwoTermInvariant:
    def __init__(self, a_tau=-4e-4, b_tau=-2e-4):
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.name  = "TwoTermInvariant"

    def tau(self, v, dx, dy, dz):
        vx, vy, vz = v[:,0], v[:,1], v[:,2]
        grads = velocity_gradients(vx,vy,vz,dx,dy,dz)
        Sxx,Syy,Szz,Sxy,Sxz,Syz = strain_tensor(grads)
        I1 = invariant_I1(Sxx,Syy,Szz,Sxy,Sxz,Syz)
        a, b = self.a_tau, self.b_tau
        tau_xx = a*Sxx + b*I1*Sxx
        tau_yy = a*Syy + b*I1*Syy
        tau_zz = a*Szz + b*I1*Szz
        tau_xy = a*Sxy + b*I1*Sxy
        tau_xz = a*Sxz + b*I1*Sxz
        tau_yz = a*Syz + b*I1*Syz
        return tau_xx,tau_yy,tau_zz,tau_xy,tau_xz,tau_yz

    def rhs(self, v, nu, dx, dy, dz, forcing=None):
        vx, vy, vz = v[:,0], v[:,1], v[:,2]
        grads = velocity_gradients(vx,vy,vz,dx,dy,dz)
        dvx_dx,dvx_dy,dvx_dz,dvy_dx,dvy_dy,dvy_dz,dvz_dx,dvz_dy,dvz_dz = grads

        adv_x = vx*dvx_dx + vy*dvx_dy + vz*dvx_dz
        adv_y = vx*dvy_dx + vy*dvy_dy + vz*dvy_dz
        adv_z = vx*dvz_dx + vy*dvz_dy + vz*dvz_dz

        lap_vx = laplace3d(vx,dx,dy,dz)
        lap_vy = laplace3d(vy,dx,dy,dz)
        lap_vz = laplace3d(vz,dx,dy,dz)

        tau_xx,tau_yy,tau_zz,tau_xy,tau_xz,tau_yz = self.tau(v,dx,dy,dz)

        def div_tau_comp(txx,tyy,tzz,txy,txz,tyz):
            tyx,tzx,tzy = txy,txz,tyz
            dtxx_dx, _, _ = grad3d(txx,dx,dy,dz)
            _, dtyx_dy, _ = grad3d(tyx,dx,dy,dz)
            _, _, dtzx_dz = grad3d(tzx,dx,dy,dz)
            div_x = dtxx_dx + dtyx_dy + dtzx_dz

            dtxy_dx, _, _ = grad3d(txy,dx,dy,dz)
            _, dtyy_dy, _ = grad3d(tyy,dx,dy,dz)
            _, _, dtzy_dz = grad3d(tzy,dx,dy,dz)
            div_y = dtxy_dx + dtyy_dy + dtzy_dz

            dtxz_dx, _, _ = grad3d(txz,dx,dy,dz)
            _, dtyz_dy, _ = grad3d(tyz,dx,dy,dz)
            _, _, dtzz_dz = grad3d(tzz,dx,dy,dz)
            div_z = dtxz_dx + dtyz_dy + dtzz_dz
            return div_x,div_y,div_z

        div_tx,div_ty,div_tz = div_tau_comp(
            tau_xx,tau_yy,tau_zz,tau_xy,tau_xz,tau_yz
        )

        rhs_x = -adv_x + nu*lap_vx + div_tx
        rhs_y = -adv_y + nu*lap_vy + div_ty
        rhs_z = -adv_z + nu*lap_vz + div_tz

        if forcing is not None:
            rhs_x = rhs_x + forcing[:,0]
            rhs_y = rhs_y + forcing[:,1]
            rhs_z = rhs_z + forcing[:,2]

        return torch.stack([rhs_x,rhs_y,rhs_z], dim=1)

def rk2_step(v, rhs_fun, dt, proj=True, forcing=None):
    k1 = rhs_fun(v, forcing=forcing)
    v_star = v + dt*k1
    if proj:
        vx_s,vy_s,vz_s,_ = project_incompressible(
            v_star[:,0],v_star[:,1],v_star[:,2],
            dx,dy,dz,dt
        )
        v_star = torch.stack([vx_s,vy_s,vz_s], dim=1)
    k2 = rhs_fun(v_star, forcing=forcing)
    v_new = v + 0.5*dt*(k1 + k2)
    if proj:
        vx_n,vy_n,vz_n,_ = project_incompressible(
            v_new[:,0],v_new[:,1],v_new[:,2],
            dx,dy,dz,dt
        )
        v_new = torch.stack([vx_n,vy_n,vz_n], dim=1)
    return v_new

def simulate_teacher(teacher, v0_batch, nu, dt, Nt, forcing=None,
                     record_ts=False):
    """
    Teacher 演化:
      - 使用 TwoTermInvariant 闭合
      - 用 scalar L0(|u|) 计算 χ_ref
    """
    B = v0_batch.shape[0]
    v = v0_batch.clone()

    if record_ts:
        phi2_ts = []
        H_ts    = []
        K_ts    = []
        chi_ts  = []

    for n in range(Nt):
        v = rk2_step(
            v,
            rhs_fun=lambda v_, forcing=None: teacher.rhs(
                v_, nu, dx, dy, dz, forcing=forcing
            ),
            dt=dt,
            proj=True,
            forcing=forcing,
        )
        if record_ts:
            phi2, H = scct_stats_vec(v,dx,dy,dz)
            K       = K_from_phi2_H(phi2,H)

            vx,vy,vz = v[:,0],v[:,1],v[:,2]
            mag = torch.sqrt(vx*vx + vy*vy + vz*vz + 1e-16)  # [B,Nx,Ny,Nz]
            chi = L0_scalar_op(mag)

            phi2_ts.append(phi2.mean().item())
            H_ts.append(H.mean().item())
            K_ts.append(K.mean().item())
            chi_ts.append(chi.item())

    if record_ts:
        return v, (phi2_ts, H_ts, K_ts, chi_ts)
    else:
        return v

# -------------------------
# 7. GrammarForest τ_ij 模型
# -------------------------
class GrammarForestTau(nn.Module):
    """
    主干: a1*S + a2*I1S
    forest: S2, R2, SR-RS, S3
    """
    def __init__(self):
        super().__init__()
        # 主干两项
        self.a1 = nn.Parameter(torch.tensor(-4e-4, dtype=DTYPE))
        self.a2 = nn.Parameter(torch.tensor(-2e-4, dtype=DTYPE))
        # forest 权重
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        self.w_forest = nn.Parameter(torch.zeros(4, dtype=DTYPE))

    def _tau_terms(self, v):
        vx,vy,vz = v[:,0], v[:,1], v[:,2]
        grads = velocity_gradients(vx,vy,vz,dx,dy,dz)
        Sxx,Syy,Szz,Sxy,Sxz,Syz = strain_tensor(grads)
        Rxx,Ryy,Rzz,Rxy,Rxz,Ryz = rotation_tensor(grads)
        I1 = invariant_I1(Sxx,Syy,Szz,Sxy,Sxz,Syz)

        # S
        S = torch.stack([Sxx,Syy,Szz,Sxy,Sxz,Syz], dim=1)
        # I1S
        I1S = torch.stack(
            [I1*Sxx,I1*Syy,I1*Szz,I1*Sxy,I1*Sxz,I1*Syz], dim=1
        )
        # S2
        S2 = matrix_sq_sym_3d(Sxx,Syy,Szz,Sxy,Sxz,Syz)
        S2 = torch.stack(S2, dim=1)
        # R2
        R2 = matrix_sq_skew_3d(Rxx,Ryy,Rzz,Rxy,Rxz,Ryz)
        R2 = torch.stack(R2, dim=1)

        # SR - RS (简化版)
        SR_xx = Sxx*Rxx + Sxy*Rxy + Sxz*Rxz
        SR_yy = Sxy*Rxy + Syy*Ryy + Syz*Ryz
        SR_zz = Sxz*Rxz + Syz*Ryz + Szz*Rzz
        SR_xy = Sxx*Rxy + Sxy*Ryy + Sxz*Ryz
        SR_xz = Sxx*Rxz + Sxy*Ryz + Sxz*Rzz
        SR_yz = Sxy*Rxz + Syy*Ryz + Syz*Rzz

        RS_xx = Rxx*Sxx + Rxy*Sxy + Rxz*Sxz
        RS_yy = Rxy*Sxy + Ryy*Syy + Ryz*Syz
        RS_zz = Rxz*Sxz + Ryz*Syz + Rzz*Szz
        RS_xy = Rxx*Sxy + Rxy*Syy + Rxz*Syz
        RS_xz = Rxx*Sxz + Rxy*Syz + Rxz*Szz
        RS_yz = Rxy*Sxz + Ryy*Syz + Ryz*Szz

        Dxx = SR_xx - RS_xx
        Dyy = SR_yy - RS_yy
        Dzz = SR_zz - RS_zz
        Dxy = SR_xy - RS_xy
        Dxz = SR_xz - RS_xz
        Dyz = SR_yz - RS_yz
        SRmRS = torch.stack([Dxx,Dyy,Dzz,Dxy,Dxz,Dyz], dim=1)

        # S3: (S^2)·S
        S2x,S2y,S2z,S2xy,S2xz,S2yz = matrix_sq_sym_3d(Sxx,Syy,Szz,Sxy,Sxz,Syz)
        S3x  = S2x*Sxx
        S3y  = S2y*Syy
        S3z  = S2z*Szz
        S3xy = S2xy*Sxy
        S3xz = S2xz*Sxz
        S3yz = S2yz*Syz
        S3 = torch.stack([S3x,S3y,S3z,S3xy,S3xz,S3yz], dim=1)

        return S, I1S, S2, R2, SRmRS, S3

    def tau(self, v):
        S, I1S, S2, R2, SRmRS, S3 = self._tau_terms(v)
        a1 = self.a1
        a2 = self.a2
        tau_main = a1*S + a2*I1S
        forest_terms = [S2, R2, SRmRS, S3]
        forest = 0.0
        for i,term in enumerate(forest_terms):
            forest = forest + self.w_forest[i]*term
        tau_all = tau_main + self.gamma*forest
        return tau_all

    def rhs(self, v, nu, dx, dy, dz, forcing=None):
        vx,vy,vz = v[:,0], v[:,1], v[:,2]
        grads = velocity_gradients(vx,vy,vz,dx,dy,dz)
        dvx_dx,dvx_dy,dvx_dz,dvy_dx,dvy_dy,dvy_dz,dvz_dx,dvz_dy,dvz_dz = grads

        adv_x = vx*dvx_dx + vy*dvx_dy + vz*dvx_dz
        adv_y = vx*dvy_dx + vy*dvy_dy + vz*dvy_dz
        adv_z = vx*dvz_dx + vy*dvz_dy + vz*dvz_dz

        lap_vx = laplace3d(vx,dx,dy,dz)
        lap_vy = laplace3d(vy,dx,dy,dz)
        lap_vz = laplace3d(vz,dx,dy,dz)

        tau = self.tau(v)
        txx,tyy,tzz,txy,txz,tyz = tau[:,0],tau[:,1],tau[:,2],tau[:,3],tau[:,4],tau[:,5]
        tyx,tzx,tzy = txy,txz,tyz

        dtxx_dx,_,_ = grad3d(txx,dx,dy,dz)
        _,dtyx_dy,_ = grad3d(tyx,dx,dy,dz)
        _,_,dtzx_dz = grad3d(tzx,dx,dy,dz)
        div_x = dtxx_dx + dtyx_dy + dtzx_dz

        dtxy_dx,_,_ = grad3d(txy,dx,dy,dz)
        _,dtyy_dy,_ = grad3d(tyy,dx,dy,dz)
        _,_,dtzy_dz = grad3d(tzy,dx,dy,dz)
        div_y = dtxy_dx + dtyy_dy + dtzy_dz

        dtxz_dx,_,_ = grad3d(txz,dx,dy,dz)
        _,dtyz_dy,_ = grad3d(tyz,dx,dy,dz)
        _,_,dtzz_dz = grad3d(tzz,dx,dy,dz)
        div_z = dtxz_dx + dtyz_dy + dtzz_dz

        rhs_x = -adv_x + nu*lap_vx + div_x
        rhs_y = -adv_y + nu*lap_vy + div_y
        rhs_z = -adv_z + nu*lap_vz + div_z

        if forcing is not None:
            rhs_x = rhs_x + forcing[:,0]
            rhs_y = rhs_y + forcing[:,1]
            rhs_z = rhs_z + forcing[:,2]

        return torch.stack([rhs_x,rhs_y,rhs_z], dim=1)

    def step(self, v, nu, dt, forcing=None):
        return rk2_step(
            v,
            rhs_fun=lambda v_, forcing=None: self.rhs(
                v_, nu, dx, dy, dz, forcing=forcing
            ),
            dt=dt,
            proj=True,
            forcing=forcing,
        )

# -------------------------
# 8. PrimeGap v2 时间几何正则
# -------------------------
def primegap_v2(K_ts):
    """
    K_ts: 1D torch tensor (T,)
    """
    K_ts = K_ts.flatten()
    if K_ts.numel() < 4:
        return torch.tensor(0.0, dtype=DTYPE, device=K_ts.device)

    d1 = K_ts[1:] - K_ts[:-1]
    d2 = d1[1:] - d1[:-1]
    curv = (d2**2).mean()

    # 高频惩罚
    K_centered = K_ts - K_ts.mean()
    K_fft = torch.fft.rfft(K_centered)
    freqs = torch.arange(K_fft.shape[0], dtype=DTYPE, device=K_ts.device)
    freq_penalty = ((freqs**2) * (K_fft.abs()**2)).sum() / (freqs.shape[0]**2 + 1e-8)

    return curv + 1e-4*freq_penalty

# -------------------------
# 9. SpatialGeo: τ 场空间几何正则
# -------------------------
def spatial_geo_loss_tau(tau):
    """
    tau: [B,6,Nx,Ny,Nz]
    简单版：惩罚 τ 的空间梯度能量
    """
    B = tau.shape[0]
    loss = 0.0
    for c in range(6):
        field = tau[:,c]
        gx,gy,gz = grad3d(field,dx,dy,dz)
        loss = loss + (gx*gx + gy*gy + gz*gz).mean()
    return loss / 6.0

# -------------------------
# 10. TGV 初始条件 / 数据集
# -------------------------
def make_ic_tgv(amp, dt, seed_offset=0):
    torch.manual_seed(SEED+seed_offset)
    k = 2.0*math.pi
    vx = amp * torch.sin(k*X)*torch.cos(k*Y)*torch.cos(k*Z)
    vy = -amp* torch.cos(k*X)*torch.sin(k*Y)*torch.cos(k*Z)
    vz = torch.zeros_like(vx)
    vz = vz + 0.05*torch.randn_like(vx)
    v0 = torch.stack([vx,vy,vz], dim=0)
    vx_p,vy_p,vz_p,_ = project_incompressible(v0[0],v0[1],v0[2],dx,dy,dz,dt)
    return torch.stack([vx_p,vy_p,vz_p], dim=0)

def build_dataset_TGV(n_train=4, amp_range=(0.45,0.60)):
    amps = np.linspace(amp_range[0], amp_range[1], n_train)
    v0_list = []
    for i,a in enumerate(amps):
        v0 = make_ic_tgv(float(a), dt, seed_offset=10*i)
        v0_list.append(v0)
    v0_batch = torch.stack(v0_list, dim=0).to(device)
    F_batch  = torch.zeros_like(v0_batch)
    return v0_batch, F_batch, amps

# -------------------------
# 11. 训练 GrammarForestTau (traj + struct + L0 + PrimeGap + SpatialGeo)
# -------------------------
def train_grammarforest_TGV_extended():
    nu = 0.01
    teacher = TeacherTwoTermInvariant(a_tau=-4e-4, b_tau=-2e-4)
    v0_batch, F_batch, amps = build_dataset_TGV(
        n_train=4, amp_range=(0.45,0.60)
    )

    print("=======================================================")
    print(f"[Experiment] flow_type=TGV, amp_range≈[{amps[0]:.3f},{amps[-1]:.3f}]")

    t0 = time.time()
    vT_teacher, (phi2_ts_T,H_ts_T,K_ts_T,chi_ts_T) = simulate_teacher(
        teacher, v0_batch, nu, dt, Nt, forcing=F_batch, record_ts=True
    )
    t1 = time.time()

    phi2_T, H_T = scct_stats_vec(vT_teacher,dx,dy,dz)
    K_T = K_from_phi2_H(phi2_T,H_T)
    vxT,vyT,vzT = vT_teacher[:,0],vT_teacher[:,1],vT_teacher[:,2]
    magT = torch.sqrt(vxT*vxT+vyT*vyT+vzT*vzT + 1e-16)
    chi_ref = L0_scalar_op(magT)

    print(f"[Teacher:{teacher.name}] done in {t1-t0:.2f}s, Φ²(T)≈{phi2_T.mean().item():.3e}, H(T)≈{H_T.mean().item():.3f}")
    print(f"[L0] teacher χ_ref(T) ≈ {chi_ref.item():.3e}")

    model = GrammarForestTau().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=2e-3)

    # loss 权重
    w_traj    = 1.0
    w_struct  = 1e-1
    w_L0      = 1e-1
    w_PG      = 1e-2
    w_spatial = 1e-3

    print("[Stage] Training GrammarForestTau (traj + struct + L0 + PrimeGap v2 + SpatialGeo)...")
    log_epochs = [1,5,10,15,20,25,30,35,40]
    max_ep = 40

    for ep in range(1,max_ep+1):
        opt.zero_grad()

        # rollout to T
        v = v0_batch.clone()
        phi2_ts = []
        H_ts    = []
        K_ts    = []

        for n in range(Nt):
            v = model.step(v,nu,dt,forcing=F_batch)
            if n % 10 == 0:
                phi2,H = scct_stats_vec(v,dx,dy,dz)
                K  = K_from_phi2_H(phi2,H)
                phi2_ts.append(phi2.mean())
                H_ts.append(H.mean())
                K_ts.append(K.mean())

        phi2_ts = torch.stack(phi2_ts)
        H_ts    = torch.stack(H_ts)
        K_ts    = torch.stack(K_ts)

        vT_model = v
        phi2_M, H_M = scct_stats_vec(vT_model,dx,dy,dz)
        K_M = K_from_phi2_H(phi2_M,H_M)

        vxM,vyM,vzM = vT_model[:,0],vT_model[:,1],vT_model[:,2]
        magM = torch.sqrt(vxM*vxM+vyM*vyM+vzM*vzM + 1e-16)
        chi_M = L0_scalar_op(magM)

        tau_M = model.tau(vT_model)  # [B,6,Nx,Ny,Nz]
        loss_spatial = spatial_geo_loss_tau(tau_M)

        misfit = ((vT_model - vT_teacher)**2).mean()
        loss_traj = misfit
        loss_struct = (phi2_M - phi2_T).abs().mean() + (H_M - H_T).abs().mean()
        loss_L0 = (chi_M - chi_ref).abs()
        loss_PG = primegap_v2(K_ts)

        loss = (w_traj*loss_traj +
                w_struct*loss_struct +
                w_L0*loss_L0 +
                w_PG*loss_PG +
                w_spatial*loss_spatial)

        loss.backward()
        opt.step()

        if ep in log_epochs:
            with torch.no_grad():
                a1 = model.a1.item()
                a2 = model.a2.item()
                ratio = abs(a2)/max(abs(a1),1e-12)
                print(f"[ep {ep:02d}] "
                      f"loss_traj={loss_traj.item():.3e}, "
                      f"loss_struct={loss_struct.item():.3e}, "
                      f"loss_L0={loss_L0.item():.3e}, "
                      f"loss_PG={loss_PG.item():.3e}, "
                      f"loss_spatial={loss_spatial.item():.3e}, "
                      f"a1={a1:+.3e}, a2={a2:+.3e}, |a2|/|a1|={ratio:.3f}")

    # ---- 训练后 summary ----
    with torch.no_grad():
        v = v0_batch.clone()
        for n in range(Nt):
            v = model.step(v,nu,dt,forcing=F_batch)
        vT_model = v
        phi2_M, H_M = scct_stats_vec(vT_model,dx,dy,dz)
        K_M = K_from_phi2_H(phi2_M,H_M)
        vxM,vyM,vzM = vT_model[:,0],vT_model[:,1],vT_model[:,2]
        magM = torch.sqrt(vxM*vxM+vyM*vyM+vzM*vzM + 1e-16)
        chi_M = L0_scalar_op(magM)
        misfit_final = ((vT_model - vT_teacher)**2).mean().item()
        a1 = model.a1.item()
        a2 = model.a2.item()
        ratio = abs(a2)/max(abs(a1),1e-12)

    print("\n[Stage][Summary]")
    print(f"  misfit≈{misfit_final:.3e}")
    print(f"  a1 (S)   = {a1:+.6e}")
    print(f"  a2 (I1S) = {a2:+.6e}")
    print(f"  |a2|/|a1| = {ratio:.3f}")
    print(f"  Φ²(T)_model = {phi2_M.mean().item():.3e}, H(T)_model = {H_M.mean().item():.3f}")
    print(f"  [L0] model χ_rms(T) ≈ {chi_M.item():.3e}, teacher χ_ref(T) ≈ {chi_ref.item():.3e}")

    # =======================================================
    # Eval：收集时间序列做 AutoMetric-K
    # =======================================================
    Nt_eval = 300
    print("=======================================================")
    print(f"[Eval] rollout Nt_eval={Nt_eval} for flow_type=TGV ...")

    with torch.no_grad():
        v_t = v0_batch.clone()
        v_m = v0_batch.clone()
        phi2_ts = []; H_ts = []; K_ts = []; chi_ts = []; err_ts = []

        for n in range(Nt_eval):
            v_t = rk2_step(
                v_t,
                rhs_fun=lambda v_, forcing=None: teacher.rhs(
                    v_, nu, dx, dy, dz, forcing=F_batch
                ),
                dt=dt, proj=True, forcing=F_batch
            )
            v_m = model.step(v_m,nu,dt,forcing=F_batch)

            phi2,H = scct_stats_vec(v_m,dx,dy,dz)
            K      = K_from_phi2_H(phi2,H)
            vxM,vyM,vzM = v_m[:,0],v_m[:,1],v_m[:,2]
            magM = torch.sqrt(vxM*vxM+vyM*vyM+vzM*vzM + 1e-16)
            chi  = L0_scalar_op(magM)
            err  = ((v_m - v_t)**2).mean()

            phi2_ts.append(phi2.mean().item())
            H_ts.append(H.mean().item())
            K_ts.append(K.mean().item())
            chi_ts.append(chi.item())
            err_ts.append(err.item())

    phi2_ts = np.array(phi2_ts)
    H_ts    = np.array(H_ts)
    K_ts    = np.array(K_ts)
    chi_ts  = np.array(chi_ts)
    err_ts  = np.array(err_ts)

    PG_eval = primegap_v2(torch.from_numpy(K_ts).to(device,DTYPE)).item()
    print(f"\n[PrimeGap] prime-gap-style loss(K) (eval, v2) ≈ {PG_eval:.3e}")

    # AutoMetric-K
    def corr(x,y):
        x = x - x.mean()
        y = y - y.mean()
        num = (x*y).sum()
        den = math.sqrt((x*x).sum()*(y*y).sum() + 1e-20)
        return float(num/den)

    metrics = {}
    eps = 1e-8
    metrics["phi2_over_H"] = phi2_ts/(H_ts+eps)
    metrics["KH"]          = K_ts*H_ts
    metrics["K_over_H"]    = K_ts/(H_ts+eps)
    metrics["sqrt_phi2H"]  = np.sqrt(np.maximum(phi2_ts*H_ts,0))
    metrics["phi2H"]       = phi2_ts*H_ts
    metrics["Kphi2"]       = K_ts*phi2_ts
    metrics["chi"]         = chi_ts
    metrics["K_over_phi2"] = K_ts/(phi2_ts+eps)
    metrics["K_sq"]        = K_ts**2
    metrics["K_over_chi"]  = K_ts/(chi_ts+eps)

    corrs = []
    for name,arr in metrics.items():
        corrs.append((name, corr(arr, err_ts)))
    corrs_sorted = sorted(corrs, key=lambda t: -abs(t[1]))

    print("\n[AutoMetric] Top metrics by |corr| with err(t):")
    for name,cv in corrs_sorted[:10]:
        print(f"  {name:15s}: corr={cv:+.3f}")

    ts = {
        "phi2_ts":phi2_ts,
        "H_ts":H_ts,
        "K_ts":K_ts,
        "chi_ts":chi_ts,
        "err_ts":err_ts,
    }
    return model, ts

# -------------------------
# 12. 一键运行 demo
# -------------------------
model_ext, ts_ext = train_grammarforest_TGV_extended()
print("\n[Done] GrammarForestTau + scalar L0 + SpatialGeo + PrimeGap v2 + AutoMetric-K demo 完成。")
