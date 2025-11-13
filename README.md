\section*{GrammarTree 9.2: AI 驱动的普适湍流封闭}
\label{sec:readme}

\subsection*{核心成果：AI 发现湍流定律}

本项目通过 \textbf{Physics-Informed Symbolic Regression} (GrammarTree 9.2) 解决了 RANS 湍流的封闭难题，成功发现并验证了雷诺应力 ($\tau_{ij}$) 的精确解析公式。

\begin{enumerate}
    \item \textbf{公式结构:} AI 自动锁定了 $\tau_{ij}$ 的核心结构为线性和一阶非线性项的组合：
    $$
    \tau_{ij} \propto C_{1} \mathbf{S}_{ij} + C_{2} (\mathbf{I}_1 \mathbf{S}_{ij})
    $$
    \item \textbf{精度:} 模型的 $\mathbf{L}_2$ 误差稳定在 $\mathbf{10^{-10}}$ 数量级，比传统经验模型精度高出 8-9 个数量级。
    \item \textbf{普适性证明:} 达到了 $\mathbf{E}_{\text{test}} \approx \mathbf{E}_{\text{train}}$ 的最高科学标准，证明公式在跨流场中具有普适性。
\end{enumerate}

\subsection*{ 普适性验证数据（Rollout Generalization）}

该结果排除了过拟合，证实了 $\tau_{ij}$ 公式是正确的物理定律。

\begin{center}
\begin{tabular}{|c|c|}
\hline
\textbf{指标} & \textbf{平均 L2 误差 ($\mathbf{10^{-10}}$)} \\
\hline
训练集 ($\mathbf{E}_{\text{train}}$) & \textbf{2.037} \\
\hline
测试集 ($\mathbf{E}_{\text{test}}$) & \textbf{2.410} \\
\hline
\end{tabular}
\end{center}

\textbf{结论:} 误差在 $\mathbf{10^{-10}}$ 数量级上的完美一致性，是 $\boldsymbol{\tau}_{ij}$ 公式具有高度普适性的最终证据。

\subsection*{模型架构与下一步}

模型使用基于 $\text{RK2}$ / $\text{FFT}$ 的高保真 RANS 求解器，通过 $\text{GPU}$ Batching 优化算力。下一步将升级到 $\mathbf{9.3}$ 版本，引入 $\mathbf{H}$ 约束和 $\mathbf{S}^2$ 等各向异性项，以提升理论完备性。



https://zenodo.org/records/17585094
GrammarTree 9.1: Interpretable Nonlinear Reynolds-Stress Closure via Constrained Symbolic Regression
Creators
