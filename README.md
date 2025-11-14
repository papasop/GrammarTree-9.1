📘 GrammarTree 10.0 — AI-Discovered Invariant Closure for 3D Navier–Stokes

Trajectory-Level Accuracy · Nonlinear Invariant Dissipation · Minimal 3D Turbulence

🔥 Overview

This repository contains the official implementation of GrammarTree 10.0, a physics-informed symbolic regression framework that discovers compact invariant closures for 3D Navier–Stokes turbulence.

The key scientific result:

AI reveals that the dominant Reynolds-stress dissipation pathway in 3D turbulence is not the classical linear strain channel 
𝑆
𝑖
𝑗
S
ij
	​

, but a nonlinear invariant-amplified channel 
(
𝐼
1
𝑆
)
𝑖
𝑗
(I
1
	​

S)
ij
	​

.

This finding arises from a five-term invariant closure that achieves 
10
−
11
10
−11
 rollout accuracy on minimal-resolution 3D Taylor–Green vortex (TGV) trajectories.

📌 Key Findings
🧩 1. AI-Discovered 5-Term Invariant Closure
𝜏
𝑖
𝑗
=
𝑎
1
𝑆
𝑖
𝑗
+
𝑎
2
(
𝐼
1
𝑆
)
𝑖
𝑗
+
𝑎
3
(
𝐼
2
Ω
)
𝑖
𝑗
+
𝑎
4
𝑆
𝑖
𝑗
2
+
𝑎
5
Ω
𝑖
𝑗
2
.
τ
ij
	​

=a
1
	​

S
ij
	​

+a
2
	​

(I
1
	​

S)
ij
	​

+a
3
	​

(I
2
	​

Ω)
ij
	​

+a
4
	​

S
ij
2
	​

+a
5
	​

Ω
ij
2
	​

.

Final coefficients after global scaling:

a1 = -0.2221
a2 = -0.3900
a3 = -0.0924
a4 = -0.00635
a5 = -0.00677

🔥 2. Dominance of the Nonlinear Invariant Channel

The magnitude ratio:

∣
𝑎
2
∣
∣
𝑎
1
∣
≈
1.76
>
1
∣a
1
	​

∣
∣a
2
	​

∣
	​

≈1.76>1

indicates:

Invariant-amplified dissipation dominates over linear eddy-viscosity dissipation.

🌀 3. Trajectory-Level Accuracy (Not Just Fitting Coefficients)

On six 3D TGV flows (minimal 16³ resolution):

Mean rollout MSE ≈ 
10
−
11
10
−11

Divergence RMS ≈ 
2
×
10
−
4
2×10
−4

Stable over 1000 steps

This supports the interpretation of the closure as a trajectory-centric approximate law, not merely a regression fit.



https://zenodo.org/records/17606775
GrammarTree 10.0: An AI-Discovered Nonlinear Invariant Closure for 3D Navier–Stokes with Trajectory-Level Accuracy
