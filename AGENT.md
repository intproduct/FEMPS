# FEMPS / Fermionic Functional Tensor Network 项目最高总体规划

**项目状态：** 正式立项 / Research & Method Development
**文档级别：** Project Master Plan（最高计划；所有阶段计划、Codex 任务、实验计划与论文草稿均应服从本文件）
**工作名称：** FEMPS — Functional Exterior Matrix Product State
**名称状态：** 暂定。正式论文投稿前必须再次做名称与方法查新，避免与既有术语冲突。
**核心目标：** 在 first quantization 的连续坐标 / functional-basis 框架下，为费米多体 Schrödinger 方程建立一种严格保持全反对称性、避免 ordinary particle-TT 指数级统计复杂度、并可进行可微分变分优化的新型张量网络表示与算法。

**当前最高优先级（2026-09-01 修订）：** 构建一个可运行、可验证、可复现的 FEMPS 求解器，使其能在实际计算资源内对非平凡相互作用连续费米体系完成范数、能量、梯度、变分优化和 (D\)–\(\chi\) 独立收敛分析。参数量少、形式优美或新增外代数分类均不能替代算法闭环和物理 benchmark。

当前成功必须由以下事实共同支持：

1. 单 Slater、有限多行列式和至少一类非平凡关联态能够嵌入；
2. 范数、一体与二体期望及梯度能够精确计算或带受控误差估计；
3. 至少一个相互作用 benchmark 能稳定优化并呈现可解释的系统收敛；
4. 每次近似或截断显式报告 antisymmetry residual、误差或方差、时间和峰值内存；
5. 算法不显式枚举全部 virtual paths 或完整反对称 coefficient tensor；
6. ordinary particle TT、Slater/CI、exact diagonalization 以及条件允许时的 second-quantized DMRG 只作为清楚标名的比较对象。

---

## 0. 项目总纲

本项目从 Hong–Xiao–Hu–Ji–Ran 2022 的 Functional Tensor Network（FTN）框架出发。2201 工作的核心思想是：对连续多变量波函数选择正交局域函数基，将连续 Schrödinger 方程转化为有限维 coefficient tensor 上的算符作用与变分问题，再使用 MPS/TN 压缩 coefficient tensor，并通过张量收缩和自动微分完成能量优化。

2201 明确指出该方法可推广到 fermionic tensor networks / electronic Schrödinger equation，但没有给出这种推广。直接推广会自然得到

$$
\Psi(x_1,\dots,x_N)
=\sum_{s_1\dots s_N} C_{s_1\dots s_N}
\prod_{i=1}^N\phi_{s_i}(x_i),
\qquad C\in\Lambda^N V_D,
$$

其中 \(V_D\) 是截断的一粒子 functional basis 空间。若继续把 \(C\) 当 ordinary particle-site TT/MPS，则本项目现有数学研究已经表明：精确 TT bond rank 等于 exterior contraction/unfolding rank；任何非零完全反对称 \(N\)-tensor 均存在不可忽略的组合学 rank 下界；对单 Slater determinant，中间 particle Schmidt rank 为

$$
\binom{N}{\lfloor N/2\rfloor},
$$

且其非零 Schmidt singular values 完全等权。因此，ordinary particle-TT 不仅在 exact representation 上指数增长，对单 Slater determinant 的高精度近似同样无法通过普通 SVD truncation 消除这一统计复杂度。

因此，本项目不以“优化 ordinary antisymmetric TT”为主要方向，而提出更根本的问题：

> **能否在保持 2201 first-quantized continuous functional solver 核心结构的同时，把 fermionic exchange statistics 作为固定的 exterior/structural carrier 解析处理，而让可变 TN bond 只承担真正的 many-body correlation complexity？**

本项目把这一候选路线暂称为 **FEMPS / Functional Exterior Matrix Product State**。已经闭合的 no-go、7 维和 8 维四形式结果作为独立、可复核的数学成果保存；高维 rank spectrum 不再独立持续占用主线资源。只有某个维度或秩问题能够直接决定 FEMPS 的表达能力、收缩复杂度、规范结构、截断规则或物理 benchmark 设计时，才通过新的 ADR 重新激活。

---

# 1. 项目必须回答的核心科学问题

## Q1. Ordinary particle Functional TT 是否存在结构性 no-go？

需要把当前认识正式化为严格定理体系，而不是停留在直觉。

### Q1.1 Exact rank no-go

对

$$
C\in\Lambda^N V
$$

证明和整理

$$
r_k^{TT}(C)=\operatorname{rank}C_k(C),
$$

并给出严格 lower bounds、full-support/concise 条件下更强下界以及可实现性结果。

### Q1.2 Symmetry-preserving truncation no-go

证明以下形式的推论：若某 ordinary TT approximation \(\widetilde C\neq0\) 在任一 particle cut 上

$$
r_k(\widetilde C)<\binom Nk,
$$

则

$$
\widetilde C\notin\Lambda^N V,
$$

即该近似不可能保持严格全反对称性。

需要区分：

* ordinary TT-SVD truncation；
* structure-preserving antisymmetric tensor approximation；
* 是否保持块内反对称性；
* 是否保持完整 \(S_N\) sign representation。

### Q1.3 Approximate no-go for Slater determinant

对正交 Slater determinant

$$
\Phi=u_1\wedge\cdots\wedge u_N
$$

严格写出 \(k|(N-k)\) particle Schmidt decomposition，并证明

$$
\lambda_I=\binom Nk^{-1/2}.
$$

因此最佳 rank-\(r\) approximation 满足

$$
\epsilon_r^2=1-\frac{r}{\binom Nk},
$$

即给定 \(L^2\) 误差 \(\epsilon\)，必须

$$
r\ge (1-\epsilon^2)\binom Nk.
$$

这一结果应成为 FEMPS 论文最重要的理论 motivation 之一。

### Q1.4 一般反对称张量的 approximate-rank 问题

不可过度声称所有 antisymmetric states 都有 flat spectrum。需要明确研究：

* 哪些张量族存在谱衰减；
* generic/full-support 情况的 singular-value statistics；
* exact rank、border rank、approximate TT rank 的关系；
* 是否存在与 fermionic correlation 有关的 sharper lower bounds。

---

## Q2. FEMPS 应该表示什么？其 bond dimension 的物理/数学意义是什么？

当前候选定义：令

$$
A^{[j]}\in\mathrm{Mat}_{\chi_{j-1}\times\chi_j}(V_D),
$$

即每个矩阵元素本身是一粒子 functional state：

$$
a^{[j]}_{\alpha\beta}(x)
=\sum_s A^{[j]s}_{\alpha\beta}\phi_s(x).
$$

定义 matrix-wedge product

$$
(A\wedge B)_{\alpha\gamma}
=\sum_\beta A_{\alpha\beta}\wedge B_{\beta\gamma},
$$

并定义

$$
C_{\mathrm{FEMPS}}
=\left[A^{[1]}\wedge A^{[2]}\wedge\cdots\wedge A^{[N]}\right]_{11}
\in\Lambda^N V_D.
$$

等价展开为

$$
C=\sum_{\alpha_1\dots\alpha_{N-1}}
a^{[1]}_{1\alpha_1}\wedge
a^{[2]}_{\alpha_1\alpha_2}\wedge\cdots\wedge
a^{[N]}_{\alpha_{N-1}1}.
$$

必须回答：

1. 这个 ansatz 是否严格定义良好、结合律是否足够保证链式表示一致；
2. \(\chi=1\) 是否恰好覆盖 decomposable \(N\)-forms / single Slater determinants；
3. ordinary sum of \(R\) Slater determinants 如何嵌入 FEMPS；
4. FEMPS bond rank 与 Slater rank、secant rank、ordinary TT rank 的严格关系；
5. FEMPS 表示是否存在 gauge redundancy；
6. 是否存在 canonical form；
7. 是否能定义 gauge-independent 的 correlation multiplicity spectrum；
8. 是否可使 single Slater determinant 的 reduced/correlation bond dimension 为 1，同时把 \(\binom Nk\) statistics multiplicity 解析地放进 exterior structural sector；
9. 这一新 spectrum 能否与某种 fermionic correlation entropy 对应；在没有严格证明以前，禁止把它直接称为 entanglement entropy。

FEMPS 的核心哲学不是“降低同一个 Schmidt rank”，而是**改变表示范畴和复杂度度量**：ordinary TT 对 particle bipartition 的 total Schmidt space 计费；FEMPS 试图把 exchange-statistics carrier 与真正的 correlation multiplicity 分开。

---

## Q3. FEMPS 如何在已知 exact-contraction 障碍下成为可用算法？

固定小键维 generic FEMPS exact squared-norm contraction 已有条件困难性结果，因此“通用 FEMPS 的多项式时间精确收缩”不再是默认成功条件，也不得假设新恒等式必然绕过该障碍。项目必须在小系统物理实验和复杂度审计后选择一条主路线，并至多保留一条备用路线：

1. 可精确多项式收缩、可系统增加表达能力的受限 FEMPS 子类；
2. 报告误差、方差和稳定性的近似/随机收缩；
3. 保持 first-quantized continuous functional state 定义的 carrier–correlation 重构。

参数量

$$
O(ND\chi^2)
$$

不等于算法复杂度也是多项式。若 norm 或 Hamiltonian expectation 必须显式枚举 \(\chi^{N-1}\) virtual paths，则 FEMPS 只能成为数学 ansatz，而不能成为 scalable solver。

必须研究：

### Q3.1 Norm

对

$$
\Psi=\sum_{\boldsymbol\alpha}
u_1(\boldsymbol\alpha)\wedge\cdots\wedge u_N(\boldsymbol\alpha)
$$

研究

$$
\langle\Psi|\Psi\rangle
=\sum_{\boldsymbol\alpha,\boldsymbol\beta}
\det S(\boldsymbol\alpha,\boldsymbol\beta)
$$

能否利用以下结构重排：

* Cauchy–Binet identities；
* exterior algebra transfer operators；
* compound matrices；
* determinant lemmas；
* Grassmann integral representation；
* generalized Wick theorem；
* symmetry-carrier / multiplicity factorization。

### Q3.2 One-body operator

对

$$
\hat H_1=\sum_i h(i)
$$

建立不 materialize \(C_{s_1\dots s_N}\) 的收缩公式。

### Q3.3 Two-body operator

对

$$
\hat H_2=\sum_{i<j}V(i,j)
$$

研究 Slater–Condon / generalized Wick / exterior transfer 形式能否与 virtual-chain contraction 兼容。

### Q3.4 Complexity theorem

目标必须给出显式复杂度

$$
T(N,D,\chi),\qquad M(N,D,\chi),
$$

而不是只说“polynomial”。

### Algorithm Recovery Gate

候选路线必须预先给出 (T(N,D,\chi)\)、峰值内存、适用结构、误差/方差控制、AD 方式和 antisymmetry residual 定义。先完成 (N=2\) materialization equivalence 与梯度检查，再进入 (N=4\)。若主路线不能通过非平凡相互作用 benchmark，则转向预先登记的备用路线或明确命名的替代一阶量子化方法；不得用更多纯数学结果替代失败判定。

---

## Q4. FEMPS 是否真正构成 2201 Functional TN 的 fermionic completion？

必须始终保留下列“2201 DNA”：

1. **First quantization**；
2. 基本变量仍是连续 particle coordinates \(x_i\)，而不是 orbital occupation；
3. 每个 particle coordinate 使用局部 functional basis \(\phi_s(x)\)；
4. 微分、坐标与相互作用算符转化为 functional-basis operator matrices/tensors；
5. 通过 TN/exterior contraction 直接计算 norm 与能量；
6. 使用 automatic differentiation 直接优化 trial wavefunction parameters；
7. functional basis truncation \(D\) 和 correlation complexity \(\chi\) 是两个独立且可收敛检查的控制参数；
8. 不以“先转成 occupation-number Fock MPS”作为核心实现。

如果最终算法变成

$$
\text{functional orbitals}
\rightarrow
\text{occupation strings}
\rightarrow
\text{standard fMPS/DMRG},
$$

则项目已经失去主要方法创新，应视为失败/旁支，不得包装成 FEMPS 主结果。

Grassmann algebra、determinant identities、Pfaffian、JW/parity 或其他 fermionic tools 可以作为**contraction engine**，但不能改变上述 state representation 与 solver definition。

---

## Q5. FEMPS 与已有工作的创新边界是什么？

必须主动寻找最危险 prior art，而非仅寻找支持材料。

### 必须持续比较的近邻

1. Hong et al. 2022 — Functional TN for continuous many-body Schrödinger equation；
2. Li & Waintal 2026 — first-quantized MPS，通过重新处理 antisymmetry 降低 first-quantized entanglement；
3. Li & Chan 2016 — Hilbert-space MPS / many-electron Hilbert-space renormalization；
4. standard quantum-chemistry DMRG / second-quantized fMPS；
5. Grassmann tensor networks / graded fermionic tensor networks；
6. Slater determinant / CI / AGP / Pfaffian / multideterminant methods；
7. Beylkin–Mohlenkamp–Pérez — continuous wavefunction as unconstrained sums of Slater determinants；
8. antisymmetry-preserving low-rank tensor approximation；
9. symmetry-adapted MPS 的 structural tensor × degeneracy tensor 思想；
10. tensor-decomposed fermionic backflow；
11. exterior algebra / Grassmannian secant-rank literature；
12. 任意 antisymmetric tensor contraction ranks / Hilbert functions / alternating forms literature。

### 暂定 novelty claim

以下仅作为研究目标，不可现在写成最终论文事实：

> **A first-quantized continuous functional tensor-network ansatz in which matrix-valued one-particle functions are contracted through an exterior structural layer, so that exact fermionic antisymmetry is enforced by construction while the optimizable virtual multiplicity represents correlations beyond exchange statistics; together with either a polynomially contractible structured subclass or a controlled approximate contraction method for norm and one-/two-body operators, enabling direct AD variational solution of continuous fermionic Schrödinger equations without conversion to an occupation-number MPS.**

只有在 Q3 和持续查新同时通过后，才能把它升级为正式 priority claim。

---

# 2. 六条并行研究工作流

## Workstream A — 已封存的反对称张量 TT/Schmidt 谱数学理论

此工作流保存既有 exact-rank、Slater 平坦谱、固定小键维复杂性以及 7/8 维四形式成果，但默认处于 **parked** 状态，不再独立持续。

### A1. 当前已有结构整理

* exact TT-SVD rank = contiguous unfolding rank；
* antisymmetric case = exterior contraction rank；
* contraction ranks = exterior Artinian Gorenstein Hilbert function；
* complementary-cut symmetry；
* generic / Hodge dual / Lefschetz 结构；
* support / concise 条件；
* \(p=2\)、\(p=3\)、余维二分类；
* \(p=4\) extremal rank program。

### A2. 四形式成果边界

固定 \(\mu_4(7)=\mu_4(8)=12\) 的当前精确检查点。16D rank 22/23 分支保留为明确开放问题，不再开展无直接算法用途的有限域、浮点或 orbit/chart 搜索。

所有计算机辅助证明必须：

* 保留 exact arithmetic / finite-field / Gröbner certificates；
* 保存所有 orbit/chart coverage metadata；
* 可以从 clean environment 重跑；
* 明确区分 exploratory numerics 与 proof certificates；
* 严格记录 base field（\(\mathbb R\)、\(\mathbb C\)、\(\mathbb Q\)、有限域）及跨域推理。

### A3. 保留的 no-go 论文任务

优先完成：

1. Slater flat particle-Schmidt spectrum theorem；
2. best rank-\(r\) approximation error corollary；
3. low ordinary-TT rank implies loss of strict antisymmetry corollary；
4. structure-preserving approximation 与 TT truncation 的区别；
5. 一般 antisymmetric tensors approximate rank 的开放问题。

### A4. 重新激活条件与论文归属

* 若数学结果形成独立完整定理体系：单独发表；
* 若成果主要作为 FEMPS motivation/no-go：主文理论节 + 长附录；
* 已有四形式结果可作为独立数学成果投稿，FEMPS 文中只引用直接相关的 no-go theorem；
* 新的高维问题必须在 ADR 中写明它阻塞的具体算法、复杂度或物理判据，才可重新进入 active plan。

不得为了“合成一篇大论文”强行牺牲两个方向各自的逻辑完整性。

---

## Workstream B — FEMPS algebra / canonical theory

目标：回答“FEMPS 到底是什么”，先于大规模代码实现。

### B1. 严格定义

* matrix-valued one-forms；
* wedge matrix multiplication；
* boundary conditions；
* open-chain form；
* complex/real field；
* normalization conventions；
* spin/orbital internal degrees of freedom 的处理。

### B2. Expressivity

严格证明或反例验证：

* \(\chi=1\) = decomposable form；
* any finite Slater sum can embed into finite-\(\chi\) FEMPS；
* \(\chi_{\wedge}\le\) Slater rank 的条件；
* FEMPS variety 与 Grassmannian secant varieties 的关系；
* 是否存在 FEMPS 无法有效表示但 low Slater-rank 可表示的病态情况；
* generic FEMPS parameter count 与 gauge quotient dimension。

### B3. Gauge / canonical form

研究：

* 普通 MPS gauge 是否仍成立；
* wedge-valued cores 是否存在额外 gauge；
* 是否可以 left/right canonicalize；
* reduced correlation spectrum 是否唯一；
* canonicalization 是否需要使用 exterior inner product / compound transfer operator。

### B4. Statistics carrier × correlation multiplicity

尝试建立更强形式

$$
\mathcal B_k
\cong
\mathcal S_k^{\rm fermion}
\otimes
\mathbb C^{\chi_k^{\rm corr}},
$$

其中 \(\mathcal S_k^{\rm fermion}\) 为固定 statistics structural sector，\(\chi_k^{\rm corr}\) 才是实际储存与截断对象。

目标 sanity condition：

$$
\chi_k^{\rm corr}(\text{single Slater})=1.
$$

---

## Workstream C — Exterior transfer / contraction calculus

这是项目最高优先级理论工程任务。

### C0. 小规模 symbolic derivation

只做

$$
N=2,3,4,\qquad \chi=1,2
$$

先手工/符号推导，不立即追求 GPU。

### C1. Norm contraction

比较至少四条路径：

1. explicit determinant-path sum（reference truth，只用于小规模）；
2. Cauchy–Binet / compound-matrix recurrence；
3. Grassmann auxiliary integral；
4. exterior transfer operator / Fock-lifted transfer。

### C2. One-body matrix elements

建立与 2201 functional operator matrices 的接口：

$$
h_{ss'}
=
\langle\phi_s|\hat h|\phi_{s'}\rangle.
$$

### C3. Two-body matrix elements

接口：

$$
V_{s_1s_2;s'_1s'_2}
=
\langle\phi_{s_1}\phi_{s_2}|
\hat V|
\phi_{s'_1}\phi_{s'_2}\rangle.
$$

研究 operator low-rank factorization / density fitting / separable expansions 是否必要。

### C4. Complexity benchmark

每个候选 contraction algorithm 必须输出：

* asymptotic FLOPs；
* peak memory；
* dependence on \(N,D,\chi\)；
* numerical stability；
* AD friendliness；
* GPU vectorization potential。

### Recovery route selection

只允许一条主路线和至多一条备用路线。当前主路线由新的 Algorithm and Physics Recovery ADR 决定；其他候选只做有界审计，不得平行无限扩张。精确、近似和随机算法使用各自的成功判据，不再把 generic polynomial exact contraction 设为唯一入口。

---

## Workstream D — 基于现有 lattice AD TN 包的数值实现

原则：

> **最大程度复用张量、AD、优化、设备、dtype、checkpoint、benchmark 基础设施；不复制成熟代码。**

### D1. 与现有 lattice AD TN 包的关系

推荐保持 lattice AD TN 包为独立 upstream package，不要复制一份进入 FEMPS 仓库。

开发期：

```text
workspace/
├── latticeTN/        # existing package, editable install
└── femps/            # new project
```

开发环境使用 editable dependency；CI/release 使用明确 tag/commit pin。

FEMPS 只新增 continuum/exterior-specific layer。

### D2. 需要复用的能力

* tensor backend；
* complex dtype；
* CPU/GPU device abstraction；
* automatic differentiation；
* optimizer wrappers；
* contraction helpers；
* MPS/MPO diagnostics；
* checkpoint / resume；
* deterministic seeds；
* benchmark harness；
* logging / artifact export。

### D3. 新模块建议

```text
src/femps/
├── basis/
│   ├── harmonic.py
│   ├── laguerre.py
│   ├── quadrature.py
│   └── operators.py
├── exterior/
│   ├── wedge.py
│   ├── forms.py
│   ├── contractions.py
│   ├── compound.py
│   └── grassmann_backend.py
├── states/
│   ├── slater.py
│   ├── antisymmetric_full.py
│   ├── femps.py
│   └── ordered_sector.py
├── hamiltonians/
│   ├── harmonic_fermions.py
│   ├── harmonic_interaction.py
│   └── soft_coulomb.py
├── algorithms/
│   ├── norm.py
│   ├── expectation.py
│   ├── optimize.py
│   └── canonicalize.py
├── diagnostics/
│   ├── antisymmetry.py
│   ├── schmidt.py
│   ├── ranks.py
│   └── complexity.py
└── benchmarks/
```

其中 `grassmann_backend.py` 若存在，只能是 backend，不能定义 FEMPS state 的概念本身。

纯数学证明/证书代码建议隔离：

```text
math/
├── antisymmetric_tt/
├── four_forms/
├── certificates/
├── symbolic/
└── README.md
```

不要把 proof certificate pipeline 与 production solver 混在同一模块中。

---

## Workstream E — 物理 benchmark

所有 benchmark 必须统一报告：变分能量与参考误差、能量方差或估计不确定度、范数误差、antisymmetry residual、(D\) 与 \(\chi\) 独立收敛、wall time、峰值内存和优化稳定性。exact diagonalization、Slater/CI、ordinary particle TT 是必需的小系统比较；second-quantized DMRG 在条件允许时作为外部基准，且不得改称 FEMPS。

### E0. 全量 exact reference（小 N）

对极小 \(N,D\) 显式 materialize \(C\) 与 Hamiltonian，作为一切新算法的 truth oracle。

### E1. \(N=2\) noninteracting spinless fermionic harmonic oscillator

验证：

* exact antisymmetry；
* Slater state；
* energy；
* full tensor vs FEMPS norm；
* AD gradient；
* \(\chi_{\rm corr}=1\) sanity check。

### E2. \(N=2\) analytically solvable interacting harmonic fermions

选可分离 center-of-mass / relative-coordinate 的 harmonic interaction，要求 exact energy 与波函数结构可用于 regression。

### E3. \(N=4\) noninteracting fermions

这是与当前 four-form 数学最直接的连接。

要求同时计算：

* ordinary particle-TT ranks \((1,4,6,4,1)\)；
* flat Schmidt spectrum；
* ordinary TT truncation error；
* FEMPS reduced bond \(\chi=1\)。

该实验应成为论文最具说服力的 representation-complexity demonstration 之一。

**顺序门：** E1、E2、E3 全部通过后，才能将 E4 或更大系统作为主结果。不得因已有历史实验而跳过用当前求解器和统一记录格式重新验证前三关。

### E4. \(N=4\) interacting harmonic fermions

扫描 interaction strength：

* energy error；
* support dimension；
* ordinary contraction ranks；
* FEMPS \(\chi\)；
* correlation spectrum（若已建立）；
* optimization stability。

### E5. \(N>4\) scaling

目标不是一开始追求大系统，而是验证

$$
\text{ordinary particle TT complexity}
\quad\text{vs}\quad
\text{FEMPS correlation complexity}
$$

是否真正分离。

### E6. electronic-like interaction

最后进入 soft-Coulomb / long-range pair interaction；只有此前 contraction 与 benchmark 均稳定后才进入。

---

## Workstream F — 持续查新与论文定位

建立 `references/` 研究知识库，而不是只在聊天记录中保存文献。

建议：

```text
references/
├── references.bib
├── reading_list.md
├── novelty_matrix.md
├── notes/
│   ├── 2201_FTN.md
│   ├── li_waintal_2026.md
│   ├── li_chan_HSMPS.md
│   ├── antisymmetric_low_rank.md
│   ├── grassmann_TN.md
│   └── ...
└── local_pdfs/
```

其中 `local_pdfs/` 默认加入 `.gitignore`。

### `novelty_matrix.md` 必须至少包含

| Work | Quantization | Site meaning | Antisymmetry mechanism | Continuous functional basis? | State ansatz | Contraction | 与 FEMPS 重叠 | 剩余创新 |
| ---- | ------------ | ------------ | ---------------------- | ---------------------------- | ------------ | ----------- | ---------- | ---- |

每出现一篇危险近邻，先更新该表，再继续 claim originality。

---

# 3. 阶段计划与硬性里程碑

## Current recovery stage — FEMPS Algorithm and Physics Recovery

本阶段取代高维四形式搜索成为唯一 active 主计划。目标是先完成算法可行性审计和路线 ADR，再交付一个不枚举全部 virtual paths/完整反对称系数张量的最小求解器。求解器必须支持连续 functional basis、Slater/多行列式初始化、一/二体算符、范数/能量/梯度、checkpoint 与确定性种子、antisymmetry residual，以及 (D\)–\(\chi\) 独立扫描。

当前首选是可精确收缩的受限 matrix-wedge FEMPS：用一个全局守恒 virtual label 表示 (K=\chi\) 个非分支 Slater paths，按 (K^2\) 个 determinant/Slater–Condon transition 计算 observable。它严格包含单 Slater，并随 (K\) 系统扩展到非正交多行列式。备用路线仅为带非渐近误差/方差和反对称残差的 VMC；在主路线没有完成 E1--E4 之前，不启动通用 FEMPS VMC 的大规模开发。

Phase 32 已完成相互作用 N=6 soft-Coulomb 的独立 \(D\) 与 \(K\) 收敛。Phase 33 在严格 value/gradient parity 下批处理 transition/factor 轴，使注册的 CPU kernel 相对参考实现加速 34.926 倍；ADR 0022 接纳 Blackwell 而保留 CPU 默认。Phase 34 的预注册 truth-free 自适应 D12 K=4→5→6 lineage 又把同基组 CI 误差从 1.04729e-4 降至 3.20214e-5、方差降至 3.72621e-4，并比同预算 cold K6 低 2.44109e-4；选择阶段只读取 factorized transition 和条件数，CI 在三次优化冻结后才构造。Phase 35 的三条新候选池 lineage 全部通过：K6 能量展宽为 4.87701e-6，最大同基组 CI 误差为 3.76345e-5，最大方差为 4.54299e-4。六个预测与实际增长决定均为 continue，因此候选池稳定性成立，但没有观察到 stop 事件，自动停止规则不得接纳，K 必须继续由外部上限控制。唯一 active 数值任务为 Phase 36：把已验证的增长流程收束为具有确定种子序列、逐级 checkpoint 和强制 `max_K` 的公共自适应求解器 API；不得扩大 N、D 或重启高维形式秩搜索。

旧 Phase 0--7 条目保留为历史设计与已完成工作的索引；若与本 recovery stage 冲突，以当前阶段和最新 ADR 为准。

## Phase 0 — 项目固化与基线复现

**目标：** 建仓、接入已有 lattice AD TN、复现 2201 风格 bosonic/可区分 oscillator functional benchmark，并建立最小 fermion full-tensor reference。

**必须完成：**

* GitHub repo；
* `AGENTS.md`；
* 本 Master Plan；
* `ARCHITECTURE.md`；
* `references.bib` + `novelty_matrix.md`；
* CI；
* exact small-\(N\) test harness；
* latticeTN dependency pin；
* deterministic environment lock。

**出口条件：** 新项目能在 clean checkout 中一条命令完成 tests；2201 式 functional operator matrix 与 AD optimization 跑通。

---

## Phase 1 — No-Go 理论闭合

**目标：** 把 ordinary antisymmetric particle TT 的问题写成可投稿的严格 theorem set。

**必须完成：**

* exact rank theorem；
* antisymmetry-preserving rank floor corollary；
* Slater flat Schmidt theorem；
* approximate TT lower bound；
* small-\(N\) numerical verification；
* 与 antisymmetry-preserving approximation literature 的差异说明。

**出口条件：** 形成独立 `docs/theory/no_go.md` + LaTeX theorem draft。

---

## Phase 2 — FEMPS definition & small-N algebra

**目标：** 证明候选 ansatz 不是记号游戏。

**必须完成：**

* formal definition；
* \(\chi=1\) theorem；
* finite Slater sum embedding；
* \(N=2\) exact characterization；
* gauge analysis 初版；
* \(N=2,3,4\) explicit reference implementation。

**禁止：** 这一阶段不要为了 benchmark 分数跳过 algebraic definition。

---

## Phase 3 — Contraction Gate

**目标：** 回答 FEMPS 是否算法上可活。

**必须完成：**

* norm 2–3 种独立实现交叉验证；
* one-body expectation；
* two-body expectation minimal prototype；
* complexity model；
* AD gradient check。

### Gate A 判定

**PASS：**

存在非平凡 \(\chi>1\) family 的 polynomial exact contraction，继续 Phase 4。

**CONDITIONAL：**

generic exponential，但有物理上有意义、systematically improvable 的 polynomial subclass，转向该 subclass。

**FAIL：**

仅 \(\chi=1\) 或 trivial family 可算，FEMPS 作为 solver 暂停，转向 ordered-sector FTN 或新的 carrier factorization，同时保留数学 ansatz 研究。

---

## Phase 4 — Functional FEMPS solver

**目标：** 真正接回 2201。

实现：

* HO basis；
* derivative / \(x\) / \(x^2\) operator matrices；
* one-/two-body functional operators；
* FEMPS energy functional；
* normalization；
* AD optimization；
* checkpoint/resume；
* \(D\) 与 \(\chi\) 两维收敛扫描。

**出口条件：** E1/E2 benchmark 达到高精度并通过全量 reference。

---

## Phase 5 — Four-fermion science benchmark

**目标：** 第一次把数学 rank theory、FEMPS representation、连续多体物理统一在一个实验中。

核心图表候选：

1. Slater ordinary TT flat singular spectrum；
2. ordinary TT required rank vs \(N\)；
3. FEMPS \(\chi_{\rm corr}\) vs ordinary particle rank；
4. interacting \(N=4\) energy convergence in \((D,\chi)\)；
5. ordinary exterior contraction ranks vs interaction；
6. physical states 在当前 four-form extremal geometry 中的位置。

---

## Phase 6 — Scaling & competing representations

至少比较：

1. direct antisymmetric ordinary TT（仅能做到小 \(N\)，作为 no-go control）；
2. FEMPS；
3. ordered-sector / Weyl-chamber FTN（若实现）；
4. exact diagonalization / CI small-\(N\) reference；
5. 必要时加入 second-quantized MPS 作为外部性能 reference，但不得把项目退化成 QC-DMRG 实现。

指标：

* energy error；
* norm error；
* antisymmetry violation；
* memory；
* wall time；
* effective bond complexity；
* basis size \(D\)；
* interaction strength；
* gradient/optimization stability。

---

## Phase 7 — 论文决策

### Paper A：FEMPS / Fermionic FTN 主文

成立条件：

* representation 有明确新意；
* contraction Gate PASS/CONDITIONAL；
* 至少有 \(N=2,N=4\) interacting continuum benchmarks；
* 与 2201、Li–Waintal、HS-MPS、Grassmann TN 等差异清楚；
* no-go theorem 有完整理论支持。

### Paper B：Antisymmetric TT rank mathematics

若四形式和一般 spectrum 结果足够完整，则独立投稿；FEMPS 只引用核心 theorem。

### 合并条件

只有数学结果能够直接服务 FEMPS 的核心理论，而篇幅仍可控制时才合并。不要把大量 orbit/Gröbner 证明塞进主物理论文正文。

---

# 4. 测试与科学合规要求

本项目最重要的测试不是一般软件 coverage，而是**数学与物理不变量测试**。

## L0 — 普通单元测试

* shapes；
* dtype/device；
* basis orthogonality；
* wedge signs；
* matrix-wedge associativity；
* determinant / Pfaffian helpers；
* serialization。

## L1 — Property-based tests

随机小张量验证：

* \(u\wedge v=-v\wedge u\)；
* repeated vector wedge \(=0\)；
* permutation sign；
* associativity；
* gauge invariance（若理论要求）；
* full antisymmetry under all/transposition samples。

## L2 — Full-materialization equivalence

对小 \(N,D,\chi\)：

$$
\text{FEMPS direct contraction}
=
\text{explicit antisymmetric coefficient tensor}.
$$

至少双精度下达到设定 tolerance。

## L3 — Exact physics tests

* noninteracting fermionic HO energy；
* analytically solvable interacting HO；
* Slater overlap determinant；
* one-/two-body matrix elements；
* comparison with exact diagonalization。

## L4 — AD gradient tests

每个新 contraction primitive：

* autograd vs finite difference；
* complex gradient convention；
* CPU vs GPU；
* float64/complex128 reference。

## L5 — Regression

固定 seed 与 benchmark，保存：

* energy；
* norm；
* gradients；
* rank diagnostics；
* wall time range；
* peak memory。

## L6 — Mathematical certificate tests

对纯数学结论：

* exact integer/rational arithmetic；
* finite-field prime recorded；
* chart count exact；
* certificate hash；
* independent verifier；
* 不允许浮点 rank 判断代替证明。

---

# 5. GitHub 与 Codex 工程管理规范

## 5.1 GitHub 足够作为主代码管理平台

采用单一 FEMPS 主仓库 + 已有 latticeTN upstream repo。

建议：

* `main`：始终可测试、可复现；
* feature branches：`feat/...`、`theory/...`、`bench/...`、`fix/...`；
* 所有非平凡修改通过 PR；
* 每个 PR 必须写科学目标、修改内容、验证命令、结果、已知限制；
* milestones 对应 Phase 0–7；
* Issues 对应可独立验收任务；
* release tags 对应可复现论文/报告节点。

Codex 可以自主工作，但不得把“agent 说 tests passed”当证据；CI/日志/输出 artifact 才是证据。

## 5.2 `AGENTS.md`

保持短小，主要内容：

1. 本项目的三条红线；
2. 项目文档入口；
3. 测试命令；
4. 代码架构索引；
5. 哪些结论是 theorem / conjecture / numerical evidence；
6. 禁止把 exploratory numerics 写成 proof；
7. 禁止绕过 physics compliance test；
8. 每完成任务更新 active plan 与 changelog。

详细科学背景不要全部塞进 `AGENTS.md`，应指向 `docs/`。

## 5.3 推荐 docs 结构

```text
docs/
├── MASTER_PLAN.md
├── ARCHITECTURE.md
├── THEORY_STATUS.md
├── NOVELTY_STATUS.md
├── TESTING.md
├── design/
├── theory/
├── experiments/
├── exec-plans/
│   ├── active/
│   └── completed/
└── decisions/
```

`decisions/` 使用 ADR 风格记录重要选择，例如：

* 为什么不用 direct antisymmetric TT；
* 为什么 Grassmann 只作为 backend；
* FEMPS state definition 版本变化；
* Gate A 判定。

---

# 6. “代码图 / Code Graph”是否要上？

## 结论

**第一阶段不要把重型 Code Property Graph 系统作为项目依赖。**

代码图有三类概念，不能混用：

1. **GitHub Dependency Graph**：跟踪包依赖和供应链，不是代码逻辑图；建议开启。
2. **CodeQL / static analysis database**：找安全问题和一部分代码错误；Python repo 可开启，但它不能证明 tensor contraction 或物理实现正确。
3. **Code Property Graph（Joern 等）**：融合 AST/control-flow/data-flow，适合大代码库程序分析、漏洞挖掘和复杂静态查询；本项目前期性价比不高。

### 本项目真正需要的“图”

建议维护一个轻量 architecture/module graph，例如 Mermaid：

```text
basis -> functional operators -> exterior algebra
                               -> FEMPS state

FEMPS state -> contraction engine -> AD optimizer
Hamiltonian -----------------------> expectation
reference full tensor ------------> validation
math certificates ----------------> theorem artifacts
```

将其放在 `ARCHITECTURE.md` 并随架构改变更新。

### 何时再上 Sourcegraph / code intelligence

满足任一条件再引入：

* FEMPS + latticeTN 跨多个仓库频繁修改；
* Python/C++/CUDA 多语言实现出现；
* 函数/类引用图人工难以跟踪；
* Codex 经常误判跨模块影响。

此时优先使用 code navigation / reference graph，而不是为“测试”引入 CPG。

---

# 7. 参考文献管理：必须做

不要把 PDF 散放在根目录，也不要依赖聊天历史。

建议 Git 跟踪：

* `references.bib`；
* DOI/arXiv/期刊信息；
* 每篇文献一份简短 note；
* novelty matrix；
* 与项目结论的关系。

PDF 默认放 `references/local_pdfs/` 并 `.gitignore`，避免 repo 膨胀与版权问题。需要多人同步大文件时再考虑独立文献管理工具或 Git LFS，不要把 GitHub 当 PDF 数据库。

### 第一批必须纳入的文献

1. Hong, Xiao, Hu, Ji, Ran, **Functional Tensor Network Solving Many-body Schrödinger Equation**, Phys. Rev. B 105, 165116 (2022), arXiv:2201.12823.
2. Oseledets, **Tensor-Train Decomposition**, SIAM J. Sci. Comput. 33, 2295–2317 (2011), DOI 10.1137/090752286.
3. Li, Waintal, **Matrix Product States and First Quantization**, Phys. Rev. Lett. 136, 116503 (2026), DOI 10.1103/5fx2-rsf8.
4. Li, Chan, **Hilbert space renormalization for the many-electron problem**, J. Chem. Phys. 144, 084103 (2016), DOI 10.1063/1.4942174.
5. Begović Kovač, Kressner, **Structure-Preserving Low Multilinear Rank Approximation of Antisymmetric Tensors**, SIAM J. Matrix Anal. Appl. 38, 967–983 (2017), DOI 10.1137/16M106618X.
6. Beylkin, Mohlenkamp, Pérez, **Approximating a wavefunction as an unconstrained sum of Slater determinants**, J. Math. Phys. 49, 032107 (2008), DOI 10.1063/1.2873123.
7. Grassmann tensor network / graded fermionic TN 代表性工作，重点区分 path-integral/coherent-state formalism 与本项目 first-quantized functional solver。
8. symmetry-adapted tensor networks，尤其 structural tensor × degeneracy tensor 的经典工作。
9. Slater/AGP/Pfaffian/CI 与 fermionic entanglement / particle RDM 文献。
10. alternating forms、exterior Gorenstein Hilbert functions、Grassmannian secant varieties 与当前四形式 rank 研究所使用的全部分类文献。
11. tensor-decomposed fermionic backflow 等近年 determinant carrier + tensorized correlation 工作。

`references.bib` 的地位应与 source code 类似：任何论文 claim 必须能追溯到其中的文献与 `novelty_matrix.md`。

---

# 8. 项目成功与失败标准

## 最小成功

即使通用 FEMPS solver 最终失败，本项目仍应如实保留：

1. rigorous ordinary particle-TT fermionic no-go theorem；
2. Slater approximate no-go；
3. 已经闭合的低维数学结果；
4. 明确说明为什么 2201 的 naive fermionic extension 不 scalable；
5. 受限子类、受控近似和替代一阶量子化路线的可复现实验性判定。

这已经可以形成有价值的理论成果。

## 方法学成功

同时满足：

1. FEMPS 或其结构化子类严格保持 antisymmetry；
2. single Slater reduced correlation bond = 1；
3. 非平凡 correlated states systematically improvable；
4. norm + one-/two-body expectation 可精确多项式收缩，或有可测误差/方差的受控估计；
5. 与 2201 functional operator calculus 无缝衔接；
6. AD variational solver 稳定；
7. 至少一个 interacting continuum benchmark 稳定优化，能量随 (D\) 或 \(\chi\) 呈系统收敛；
8. novelty audit 未发现等价已有方法。
9. 全过程报告 antisymmetry residual、时间和峰值内存，且不枚举全部 virtual paths 或完整反对称 coefficient tensor；
10. 至少展示一项 FEMPS 结构带来的实际优势或清晰且诚实的取舍。

## 强成功

进一步证明/展示：

* correlation multiplicity spectrum 有 canonical/gauge-independent 定义；
* 可以安全截断该 spectrum 而不破坏 exact antisymmetry；
* 与 genuine fermionic correlation measure 存在数学联系；
* scaling 明显优于 direct particle TT；
* 扩展到 spinful / 2D/3D continuum fermions 或 realistic electronic interaction。

---

# 9. Codex 的默认执行原则

所有 Codex 任务必须遵守：

1. **Theory before optimization**：未证明结构正确前，不用工程技巧掩盖数学问题。
2. **Small exact first**：新算法必须先在可全量展开的小 \(N,D\) 上与 reference 完全一致。
3. **No silent symmetry breaking**：任何近似/截断必须显式报告 antisymmetry residual。
4. **No floating proof**：数学不存在性/秩结论不能依赖浮点随机试验。
5. **No second-quantization drift**：若实现逐渐退化成 occupation-MPS，应立即标记为旁支。
6. **No novelty assumption**：每出现新数学结构或算法名，先查最危险 prior art。
7. **Separate evidence levels**：theorem / exact certificate / numerical evidence / conjecture 必须在代码、文档和图表中明确标记。
8. **Reproducibility**：每个重要结果对应 commit SHA、config、seed、environment、raw output。
9. **Physics compliance over speed**：性能优化不得改变定义或绕过物理约束。
10. **Master Plan governs**：若阶段任务与本文件冲突，先更新本文件/ADR，再修改实现。
11. **Algorithm/physics priority**：除非直接阻塞当前求解器闭环，高维分类或开放式纯数学搜索不得取得主线资源。
12. **One primary route**：候选路线先做有界复杂度与小系统审计，随后只推进 ADR 选定的主路线和至多一条备用路线。

---

# 10. 项目启动后的第一批具体任务

## Task 001 — Repository bootstrap

建立 GitHub repo、CI、`AGENTS.md`、docs、references、latticeTN editable integration。

## Task 002 — 2201 functional baseline

使用现有 lattice AD TN backend 复现最小连续 harmonic functional-basis solver，证明基础设施无障碍。

## Task 003 — Fermionic no-go theorem package

将 exact rank、Slater flat spectrum、approximate lower bound 写成可运行验证 + LaTeX theorem draft。

## Task 004 — Explicit antisymmetric reference engine

只服务小 \(N\)：wedge/full coefficient materialization、particle unfoldings、Schmidt spectrum、antisymmetry residual。

## Task 005 — FEMPS algebra prototype

实现 matrix-valued one-forms、matrix-wedge multiplication、\(\chi=1\)、small-\(\chi\) materialization。

## Task 006 — Norm contraction research sprint

独立实现 determinant-path baseline、Cauchy–Binet/compound、Grassmann 或 exterior-transfer 候选，并给出复杂度审计。

## Task 007 — Novelty audit v1

完成 HS-MPS、Li–Waintal、Grassmann TN、Slater-sum、symmetry-adapted MPS、backflow 等 novelty matrix。

## Task 008 — Gate A report

输出正式报告：FEMPS contraction 是否可行；若不完全可行，给出可收缩子类或 pivot 方案。

历史 Gate A 已经否定 generic polynomial exact contraction。后续按 Algorithm Recovery Gate：先通过当前受限/近似路线的 E1--E3，再开启 E4。

---

# 11. 项目最终科学叙事（暂定）

若项目按最理想路线成功，论文的核心故事应当是：

1. 2201 的 functional MPS 为连续 many-body Schrödinger equation 提供了直接 first-quantized TN solver，但 fermionic extension 尚未解决；
2. 直接要求 coefficient tensor 全反对称并做 ordinary particle TT 会产生结构性指数 bond；
3. 这一问题不仅是 exact rank：single Slater determinant 的 particle Schmidt spectrum 已经完全平坦，因此 ordinary SVD truncation 无法提供高精度压缩；
4. 因此 exchange statistics 不能作为 ordinary particle-TT correlation 来存储；
5. FEMPS 把 antisymmetry 放进 exterior structural layer，把可变 virtual multiplicity 留给 correlations beyond exchange；
6. 建立受限结构的 polynomial contraction calculus，或具有明确误差/方差控制的近似 contraction；
7. 将其与 2201 functional-basis operator framework 和 AD 结合；
8. 在连续 fermionic harmonic/interacting systems 中验证能量、收敛、复杂度与严格 antisymmetry；
9. 与 direct TT、ordered-sector first quantization、HS-MPS 和 second-quantized approaches 明确比较。

如果第 6 步无法在非平凡相互作用 benchmark 上成立，则不得强行保持该叙事；项目应转为：

> **no-go + 已封存的外代数数学成果 + 明确命名的受限子类或替代 first-quantized method**

而不是通过工程绕过来制造一个“看起来能跑”的 FEMPS。

---

# 12. 一句话项目原则

> **本项目不是试图让普通 TT 更聪明地近似一个本来就具有指数 particle Schmidt rank 的全反对称张量，而是要重新设计 first-quantized Functional Tensor Network 的费米表示，使交换统计成为解析结构，让张量网络的可变自由度只为真实多体相关性付费。**
