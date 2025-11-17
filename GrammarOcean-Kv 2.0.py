# ============================================================
# GrammarOcean-Kv 2.0: 终极结构正则化 (L0 强化)
# 目标：利用高权重 L0 (w_struct=10.0) 突破 R_hat=0.500 陷阱
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cpu")
torch.set_default_dtype(torch.float32)

print("Using device:", device)

# ============================================================
# Part A. Vertical grid + L0 算符 + 几何指标 R_hat(Kv)
# ============================================================

def build_vertical_grid(Nz=64, H=4000.0):
    """1D vertical column: z from 0 (surface) to H (bottom)."""
    z = np.linspace(0.0, H, Nz)
    dz = z[1] - z[0]
    return z, dz

def build_L0_1d(Nz, dz, we=1.0, b0=0.0):
    """1D 版 L0 算符，用于平滑 Kv(z). **已修复 float32 类型问题**"""
    D_np = np.zeros((Nz - 1, Nz), dtype=np.float32)
    for i in range(Nz - 1):
        D_np[i, i] = -1.0 / dz
        D_np[i, i + 1] = 1.0 / dz
    
    D = torch.from_numpy(D_np).to(torch.float32) 
    W_E = we * torch.eye(Nz - 1, dtype=torch.float32) 
    B = b0 * torch.eye(Nz, dtype=torch.float32) 

    L0 = D.T @ W_E @ D - B
    return L0

def geometric_R_hat(Kv, z, H):
    """几何指标 R_hat(Kv): (上半层混合 / 总混合)"""
    Nz = Kv.shape[0]
    dz = H / (Nz - 1)
    Kv_top = Kv[:Nz // 2]
    Kv_all = Kv
    num = torch.sum(Kv_top) * dz
    den = torch.sum(Kv_all) * dz + 1e-12 
    R_hat = num / den
    return R_hat

# ============================================================
# Part B. 1D Column Diffusion Model
# ============================================================

def diffusion_step(T, Kv, dz, dt):
    """显式时间步进: dT/dt = d/dz ( Kv * dT/dz )"""
    Nz = T.shape[0]
    
    T_ext = torch.zeros(Nz + 2, dtype=T.dtype, device=T.device)
    Kv_ext = torch.zeros(Nz + 2, dtype=T.dtype, device=T.device)

    T_ext[1:-1] = T
    T_ext[0] = T[0]
    T_ext[-1] = T[-1]

    Kv_ext[1:-1] = Kv
    Kv_ext[0] = Kv[0]
    Kv_ext[-1] = Kv[-1]

    dTdz = (T_ext[2:] - T_ext[:-2]) / (2.0 * dz) 
    F = -Kv * dTdz
    
    F_ext = torch.zeros(Nz + 2, dtype=T.dtype, device=T.device)
    F_ext[1:-1] = F
    F_ext[0] = F[0]
    F_ext[-1] = F[-1]

    dFdz = (F_ext[2:] - F_ext[:-2]) / (2.0 * dz)

    dTdt = -dFdz 
    T_new = T + dt * dTdt
    return T_new

def run_column(T_init, Kv, dz, dt, Nt):
    """积分温度剖面 T(t,z)"""
    T = T_init.clone()
    for _ in range(Nt):
        T = diffusion_step(T, Kv, dz, dt)
    return T

# ============================================================
# Part C. GrammarOcean-Kv 2.0 训练框架核心
# ============================================================

# ---------- 网格 & 算符 ----------
Nz = 64
H = 4000.0
z_np, dz = build_vertical_grid(Nz=Nz, H=H)
z_torch = torch.from_numpy(z_np).to(device).to(torch.float32) 

L0 = build_L0_1d(Nz, dz, we=1.0, b0=0.0).to(device)
print("L0 shape:", L0.shape)

# ---------- 构造一个“真值” Kv_true (非均匀剖面) ----------
Kv_true_np = 1e-5 * np.ones(Nz, dtype=np.float32) 
Kv_true_np[:10] += 5e-4 
Kv_true_np[-10:] += 1e-4 
Kv_true = torch.from_numpy(Kv_true_np).to(device)

# ---------- 初始温度剖面 & Teacher 轨迹生成 ----------
T_surface = 20.0
T_bottom = 2.0
T_init_np = T_surface + (T_bottom - T_surface) * (z_np / H)
T_init = torch.from_numpy(T_init_np.astype(np.float32)).to(device)

dt = 3600.0 
Nt = 24 * 30 
print("Running teacher simulation with Kv_true ...")
with torch.no_grad():
    T_teacher = run_column(T_init, Kv_true, dz, dt, Nt)

# ---------- 几何目标 R_geo ----------
R_geo_target = 0.305 
R_geo_target_t = torch.tensor(R_geo_target, dtype=torch.float32, device=device)

# ---------- 定义待学习的 Kv(z) 参数 (回归 64 个点) ----------
# 初始化为常数 + 小扰动
Kv_param = nn.Parameter(1e-5 * torch.ones(Nz, dtype=torch.float32, device=device))

optimizer = optim.Adam([Kv_param], lr=1e-2)

# Loss 权重 (GrammarOcean-Kv 2.0 权重配置)
w_data = 0.1     # 数据拟合权重 (低优先级)
w_geo  = 50.0    # 暴力强化几何约束 (最高优先级)
w_struct = 10.0  # <--- 修复：增强结构平滑权重，间接支持几何约束

def compute_losses(Kv, T_init, T_teacher, z, H, L0, dz, dt, Nt):
    """
    综合损失: L_data + L_geo + L_struct
    """
    Kv_eff = torch.nn.functional.softplus(Kv) # 确保 Kv 非负

    T_model = run_column(T_init, Kv_eff, dz, dt, Nt)

    L_data = torch.mean((T_model - T_teacher)**2)
    R_hat_val = geometric_R_hat(Kv_eff, z, H)
    L_geo = (R_hat_val - R_geo_target_t)**2
    L_struct = torch.mean((L0 @ Kv_eff)**2) 

    return L_data, L_geo, L_struct, T_model, Kv_eff, R_hat_val

# ---------- 训练循环 ----------

n_epochs = 200
print_interval = 20

for ep in range(1, n_epochs + 1):
    optimizer.zero_grad()

    L_data, L_geo, L_struct, T_model, Kv_eff, R_hat_val = compute_losses(
        Kv_param, T_init, T_teacher, z_torch, H, L0, dz, dt, Nt
    )

    loss = w_data * L_data + w_geo * L_geo + w_struct * L_struct
    loss.backward()
    optimizer.step()

    if ep % print_interval == 0 or ep == 1:
        print(f"[ep {ep:03d}] "
              f"loss={loss.item():.3e}, "
              f"L_data={L_data.item():.3e}, "
              f"L_geo={L_geo.item():.3e}, "
              f"L_struct={L_struct.item():.3e}, "
              f"R_hat={R_hat_val.item():.3f}")

# ============================================================
# 可视化
# ============================================================

Kv_learned = Kv_eff.detach().cpu().numpy()
T_model_np = T_model.detach().cpu().numpy()

final_R_hat = geometric_R_hat(torch.from_numpy(Kv_learned).to(device), z_torch, H).item()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(T_init_np, z_np, label="T_init")
plt.gca().invert_yaxis()
plt.xlabel("Temperature (°C)")
plt.ylabel("Depth (m)")
plt.title("Initial T(z)")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(T_teacher.detach().cpu().numpy(), z_np, label="Teacher", lw=2)
plt.plot(T_model_np, z_np, label="Model (Kv learned)", lw=2, ls="--")
plt.gca().invert_yaxis()
plt.xlabel("Temperature (°C)")
plt.title("T(z) after mixing")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(Kv_true_np, z_np, label="Kv_true (Target Shape)")
plt.plot(Kv_learned, z_np, label=f"Kv_learned (3.0 Final)", ls="--")
plt.gca().invert_yaxis()
plt.xlabel("Kv (m²/s)")
plt.title(f"Kv(z), R_hat≈{final_R_hat:.3f} (Target 0.305)")
plt.legend()

plt.tight_layout()
plt.show()
