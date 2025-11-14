

# GrammarTree 10.0 — AI Discovered Nonlinear Invariant Closure for 3D Navier–Stokes

This repository contains the official implementation of GrammarTree 10.0, a physics informed symbolic regression system that discovers compact invariant closures for 3D Navier–Stokes turbulence.

Main scientific finding:
AI shows that the dominant dissipation channel is not the classical linear strain S, but a nonlinear invariant amplified channel I1*S.
This overturns the traditional assumption “eddy viscosity ~ S”.

A five term invariant closure discovered by GrammarTree 10.0 achieves trajectory level accuracy on minimal 3D Taylor Green vortex simulations.

---

Key Findings

1. Five Term Invariant Closure
   tau_ij = a1 S + a2 (I1*S) + a3 (delta I1) + a4 S_sq + a5 R_sq

Final coefficients:
a1 = -0.2221
a2 = -0.3900
a3 = -0.0924
a4 = -0.00635
a5 = -0.00677

2. Nonlinear Invariant Dominance
   Absolute ratio |a2| / |a1| ≈ 1.76 > 1

Conclusion: The nonlinear I1*S channel contributes more than classical S. Linear eddy viscosity is not the main actor anymore.

3. Trajectory Level Accuracy
   Rollout mean squared error: around 1e-11
   Divergence RMS: around 2e-4
   Stable for 1000 time steps on 3D Taylor Green vortex
   Validated on 6 trajectories (4 training, 2 testing)

---

Repository Structure

gt10/                GrammarTree 10.0 core code
ns3d/                3D Navier–Stokes solver and teacher model
data/                Training and testing trajectories
scripts/             Training, pruning, rollout evaluation scripts
logs/                Reproducible training logs
examples/            Minimal usage examples

---

Usage

Install:
pip install -r requirements.txt

Train GrammarTree 10.0:
python scripts/train_minimal_3d.py

Run rollout evaluation:
python scripts/eval_rollout.py

---

Method Summary

GrammarTree 10.0 combines:

* Symbolic regression with invariant tensor bases
* Physics informed constraints
* Adaptive pruning with lambda_k sparsity
* Trajectory matching on GPU
* Global gamma normalization

Teacher simulations:

* Domain [0,1]^3 periodic
* Grid 16 x 16 x 16
* Time step dt = 5e-5
* Viscosity nu = 0.01
* 6 Taylor Green vortex trajectories

---

Representative Training Log (excerpt)

Stage 1 misfit ≈ 3.455e-09
Stage 2 misfit ≈ 2.227e-09
Selected terms: S, I1*S, delta I1, S_sq, R_sq
Coefficient ratio |a2|/|a1| = 1.76
Rollout error = 1e-11

---


License

MIT License

---

Acknowledgements

This project introduces the first AI discovered nonlinear invariant dominance law for 3D Navier–Stokes turbulence. It provides a compact and physically meaningful closure capable of trajectory level accuracy.



https://zenodo.org/records/17606775
GrammarTree 10.0: An AI-Discovered Nonlinear Invariant Closure for 3D Navier–Stokes with Trajectory-Level Accuracy
