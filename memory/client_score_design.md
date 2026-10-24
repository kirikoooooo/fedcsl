# 面向联邦多尺度学习的客户端--尺度分配理论

## 1. 问题设定

考虑包含 $N$ 个客户端和 $R$ 个尺度分支的联邦学习系统。服务端在每轮开始前为客户端 $c$ 分配尺度集合；若客户端被分配尺度 $r$，则仅训练并上传该尺度专属分支参数。记二元变量 $x_{c,r}\in\{0,1\}$ 表示这一分配关系，尺度 $r$ 的覆盖集合及覆盖人数分别为

$$
C_r=\{c:x_{c,r}=1\},\qquad n_r=|C_r|=\sum_{c=1}^N x_{c,r}.
$$

目标是在异构显存预算下，同时保留客户端数据的局部周期适配性、限制尺度选择诱发的通信量，并避免任一尺度因训练客户端过少而出现高方差更新。

## 2. 基本假设

**假设 1（显存可加性）**：尺度 $r$ 在一次局部训练中产生的边际显存成本为 $g_r>0$；同一客户端训练多个尺度时，总边际成本由 $\sum_r g_rx_{c,r}$ 近似。客户端 $c$ 的显存预算为 $B_c$。

**假设 2（局部价值可比性）**：$q_{c,r}$ 表示尺度 $r$ 对客户端 $c$ 的周期匹配得分。$q_{c,r}$ 仅在固定客户端 $c$ 内具有相对可比性；客户端间的绝对量级可能不同。

**假设 3（通信成本可预知）**：尺度专属状态字典的单向传输字节数为 $p_r>0$，且可在分配前由模型结构确定。共享参数的传输量与 $x_{c,r}$ 无关，因此可从分配目标中剔除。

**假设 4（梯度无偏、独立且方差有界）**：对尺度 $r$，客户端 $c$ 的随机梯度 $g_c(r)$ 满足

$$
\mathbb E[g_c(r)]=\nabla L_r(w_r),\qquad
\mathbb E\|g_c(r)-\nabla L_r(w_r)\|^2\le\sigma_r^2,
$$

且不同客户端的梯度噪声相互独立。

**假设 5（光滑性）**：全局目标 $F(w)=\sum_{r=1}^R L_r(w_r)$ 是 $L$-光滑的，局部聚合步长满足 $\eta\le 1/L$。

## 3. 分配准则与主要结论

### 3.1 资源归一化的局部价值

定义原始单位显存价值和客户端内归一化价值为

$$
u_{c,r}=\frac{q_{c,r}}{\max(g_r,\varepsilon)},\qquad
v_{c,r}=\frac{u_{c,r}-\min_su_{c,s}}
{\max_su_{c,s}-\min_su_{c,s}},
$$

其中 $\varepsilon>0$ 防止除零。若分母为零，则令全部 $v_{c,r}=0$，表示该客户端对所有尺度无可区分的相对偏好。

**结论 1（尺度排序的局部有效性）**：对固定客户端，$v_{c,r}$ 保持 $u_{c,r}$ 的排序，因而可作为“周期适配性--显存开销”权衡下的相对选择价值；逐客户端归一化避免高分布幅度客户端主导全局目标。

### 3.2 通信感知效用

令归一化通信成本为

$$
a_r=\frac{p_r}{\max_s p_s}\in(0,1].
$$

给定通信权重 $\mu\ge0$，定义客户端--尺度对的净效用

$$
h_{c,r}=v_{c,r}-\mu a_r.
$$

**结论 2（通信偏好）**：对于同一客户端的两个候选尺度，若其 NSV 相同，则净效用更高者必为通信成本更低者；当 $\mu=0$ 时，净效用退化为 NSV，通信优化不再改变分配偏好。若下发和上传参数量相同，使用单向 $p_r$ 或双向 $2p_r$ 仅相差可被权重吸收的常数倍。

### 3.3 统一优化模型

给定理想覆盖区间 $[T_{\mathrm{lo}},T_{\mathrm{hi}}]$，引入非负松弛变量 $s_r^-$ 与 $s_r^+$。分配由下列混合整数线性规划确定：

$$
\begin{aligned}
\max_{x,s^-,s^+}\quad
&\sum_{c=1}^N\sum_{r=1}^R h_{c,r}x_{c,r}
-\lambda\sum_{r=1}^R(s_r^-+s_r^+)\\
\mathrm{s.t.}\quad
&\sum_{r=1}^R g_rx_{c,r}\le B_c, &&\forall c,\\
&n_r+s_r^-\ge T_{\mathrm{lo}}, &&\forall r,\\
&n_r-s_r^+\le T_{\mathrm{hi}}, &&\forall r,\\
&x_{c,r}\in\{0,1\},\quad s_r^-,s_r^+\ge0.
\end{aligned}
\tag{1}
$$

其中 $\lambda\ge0$ 控制覆盖均衡与局部净效用之间的权衡。

**结论 3（可行性与硬资源保证）**：模型 (1) 对任意预算 $B_c$ 均可行，因为覆盖约束可由 $s_r^-$、$s_r^+$ 松弛；但任意可行解都严格满足每个客户端的显存预算。故预算不足时，模型输出的是硬件约束下净效用最大的、覆盖偏离最小的分配，而非不可行解。

**结论 4（松弛的几何意义）**：在任一最优解中，覆盖惩罚等于 $n_r$ 到区间 $[T_{\mathrm{lo}},T_{\mathrm{hi}}]$ 的 $\ell_1$ 距离。因此 $\lambda$ 对每一个单位的覆盖不足或过度覆盖施加相同边际代价。

### 3.4 覆盖目标与权重参数化

若没有先验目标，可使用平均尺度显存 $\bar g=R^{-1}\sum_rg_r$ 估计全体客户端可提供的有效容量：

$$
E=\sum_{c=1}^N\min\left(\frac{B_c}{\bar g},R\right),\qquad
T=\max\left(2,\left\lfloor\frac{E}{R}\right\rceil\right).
$$

对容忍度 $\tau\in\mathbb N^+$，取

$$
T_{\mathrm{lo}}=\max(1,T-\tau),\qquad T_{\mathrm{hi}}=T+\tau.
$$

为使用有界的用户参数表示覆盖偏好，可令 $\rho\in[0,1)$ 并设置

$$
\lambda=\frac{\rho}{1-\rho}.
\tag{2}
$$

**结论 5（参数解释）**：$\rho=0$ 对应无覆盖偏好，$\rho=0.5$ 对应单位覆盖偏离与单位归一化效用同量级，且 $\rho\to1$ 时覆盖偏离的相对代价发散。式 (2) 是便于调参的单调重参数化，而非普适的统计概率解释。

### 3.5 覆盖均衡的方差保证

服务端在尺度 $r$ 上使用平均梯度

$$
\widehat g(r)=\frac{1}{n_r}\sum_{c\in C_r}g_c(r),\qquad n_r\ge1.
$$

**定理 1（覆盖下界的方差界）**：在假设 4 下，$\widehat g(r)$ 是无偏估计，且

$$
\operatorname{Var}[\widehat g(r)]\le\frac{\sigma_r^2}{n_r}.
\tag{3}
$$

若某轮满足 $n_r\ge T_{\mathrm{lo}}\ge1$，则

$$
\sum_{r=1}^R\operatorname{Var}[\widehat g(r)]
\le\frac{R\sigma_{\max}^2}{T_{\mathrm{lo}}},
\qquad\sigma_{\max}=\max_r\sigma_r.
\tag{4}
$$

**推论 1**：覆盖下界防止单个尺度因客户端数过少而拥有无界的方差上界。覆盖约束为软约束时，式 (4) 在目标区间未被违反的尺度上成立；不可行情况下，松弛量刻画了保证缺失的程度。

### 3.6 均匀覆盖的最优性与近似性

在固定总分配容量 $M=\sum_rn_r$ 下，考虑最小化方差上界 $\sum_r\sigma_r^2/n_r$。

**定理 2（最优覆盖比例）**：连续松弛下，最优解为

$$
n_r^*=\frac{\sigma_r}{\sum_s\sigma_s}M.
\tag{5}
$$

特别地，当各尺度具有相同梯度标准差时，均匀覆盖 $n_r=M/R$ 是最优解。

**定理 3（均匀覆盖的近似界）**：令 $V_{\mathrm{uni}}$ 与 $V_*$ 分别为均匀覆盖和式 (5) 对应的方差和。则

$$
\frac{V_{\mathrm{uni}}}{V_*}
=\frac{\mathbb E[\sigma_r^2]}{\mathbb E[\sigma_r]^2}
=1+\frac{\operatorname{Var}[\sigma_r]}{\mathbb E[\sigma_r]^2}.
\tag{6}
$$

若 $\kappa=\sigma_{\max}/\sigma_{\min}$，则该比值不超过 $(1+\kappa)^2/(4\kappa)$。

**推论 2**：均匀覆盖并非在异方差情形下严格最优，但当尺度间方差异质性有限时具有与尺度数无关的近似保证。对共享主体结构、仅改变 shapelet 长度的尺度分支，这为采用覆盖区间提供理论依据。

### 3.7 收敛含义

**定理 4（覆盖约束下的 SGD 收敛项）**：在假设 4--5 下，若每轮 $n_r\ge T_{\mathrm{lo}}$，则经过 $K$ 轮更新有

$$
\frac1K\sum_{k=0}^{K-1}\mathbb E\|\nabla F(w^{(k)})\|^2
\le\frac{2(F(w^{(0)})-F^*)}{\eta K}
+\frac{\eta L R\sigma_{\max}^2}{T_{\mathrm{lo}}}.
\tag{7}
$$

**结论 6**：提高覆盖下界会直接降低收敛界中的噪声平台；但覆盖下界的增大受客户端预算和通信代价制约，这正是模型 (1) 同时建模三者的原因。

---

# 附录 A. 输入构造与模型细节

## A.1 NSV 与通信量的计算

实践中，$q_{c,r}$ 可由 `_precompute_client_scale_scores` 给出的 beta-weighted 周期匹配得分取得，$g_r$ 可由 `_collect_scale_memory_table` 在 `batch_size=32` 下测得的训练峰值 CUDA 内存增量（MB）取得或从缓存加载。$p_r$ 由 `model.scale_state_dict(r)` 中每个张量的 `numel() * element_size()` 累加计算；此状态字典应与 `FedCSL.scale_state_dict` 的聚合/上传状态一致。float32 参数的 `element_size()` 为 4。

共享 encoder backbone 等参数对所有客户端均会传输，故其成本对 $x$ 为常数，不影响模型 (1) 的最优解。

## A.2 覆盖松弛的证明

固定 $x$ 后，因 $s_r^-$ 和 $s_r^+$ 在目标中具有负系数，最优松弛变量取其最小可行值：

$$
s_r^{-*}=\max(0,T_{\mathrm{lo}}-n_r),\qquad
s_r^{+*}=\max(0,n_r-T_{\mathrm{hi}}).
$$

代回目标可得

$$
s_r^{-*}+s_r^{+*}
=d_1(n_r,[T_{\mathrm{lo}},T_{\mathrm{hi}}])
=\max(0,T_{\mathrm{lo}}-n_r)+\max(0,n_r-T_{\mathrm{hi}}),
$$

即结论 4。

## A.3 大惩罚系数的充分条件

下述结论仅讨论覆盖硬约束可行的情形。令 $H(x)=\sum_{c,r}h_{c,r}x_{c,r}$，并定义硬资源约束可行集合上的效用跨度

$$
\Delta_H=\max_xH(x)-\min_xH(x).
$$

由于覆盖偏离为整数，任一违反覆盖区间的解至少产生一个单位惩罚。若 $\lambda>\Delta_H$，任何有覆盖偏离的解均不会优于任一同时满足硬资源与覆盖区间约束的解。因此，所有惩罚问题的最优解均满足覆盖区间。

这是精确罚函数的充分条件。若 $v_{c,r}\in[0,1]$ 且 $\mu\ge0$，保守地有 $\Delta_H\le NR(1+\mu)$，故 $\lambda>NR(1+\mu)$ 是可用但通常偏松的界。它也说明了使用有限大数近似硬约束必须同时考虑通信权重，不能只以 $NR$ 为界。

## A.4 参数量感知覆盖的扩展

若每个参数坐标的梯度噪声二阶矩相近且坐标近似独立，则尺度参数量 $d_r$ 与方差满足 $\sigma_r^2\propto d_r\propto p_r$。由式 (5) 可得

$$
n_r^*=\frac{\sqrt{p_r}}{\sum_s\sqrt{p_s}}M.
$$

可据此在模型 (1) 中加入建议性的密度项

$$
-\beta\sum_r\frac{a_r}{n_r},\qquad n_r\ge1.
$$

该项偏好为大参数尺度增加覆盖，且边际收益随 $n_r$ 以 $1/n_r^2$ 衰减。它是对当前线性模型的扩展，而非模型 (1) 的必要组成。其精确 MILP 表示可使用阈值变量 $y_{r,k}=\mathbf1[n_r\ge k]$：

$$
\frac1{n_r}=\sum_{k=1}^N\frac{1}{k(k+1)}y_{r,k},\qquad
y_{r,k}\ge y_{r,k+1},\quad\sum_ky_{r,k}=n_r.
$$

---

# 附录 B. 主要定理的证明

## B.1 定理 1 的证明

由假设 4 和线性期望，$\mathbb E[\widehat g(r)]=\nabla L_r(w_r)$。客户端噪声独立时，

$$
\operatorname{Var}[\widehat g(r)]
=\frac1{n_r^2}\sum_{c\in C_r}\operatorname{Var}[g_c(r)]
\le\frac{\sigma_r^2}{n_r},
$$

即式 (3)。再由 $n_r\ge T_{\mathrm{lo}}$ 与 $\sigma_r\le\sigma_{\max}$ 对 $r$ 求和，即得式 (4)。

## B.2 定理 2 与定理 3 的证明

将 $n_r$ 连续松弛，构造拉格朗日函数

$$
\mathcal L(n,\nu)=\sum_r\frac{\sigma_r^2}{n_r}+\nu\left(\sum_rn_r-M\right).
$$

一阶条件 $-\sigma_r^2/n_r^2+\nu=0$ 表明 $n_r\propto\sigma_r$；结合容量约束可得式 (5)。代入后：

$$
V_*=\frac{(\sum_r\sigma_r)^2}{M},\qquad
V_{\mathrm{uni}}=\frac{R\sum_r\sigma_r^2}{M}.
$$

两式相除即得式 (6)。对 $\sigma_r\in[\sigma_{\min},\sigma_{\max}]$ 应用 Kantorovich 不等式，得到

$$
\frac{V_{\mathrm{uni}}}{V_*}
\le\frac{(\sigma_{\min}+\sigma_{\max})^2}
{4\sigma_{\min}\sigma_{\max}}
=\frac{(1+\kappa)^2}{4\kappa}.
$$

## B.3 定理 4 的证明

由 $L$-光滑性及 $\eta\le1/L$，对一步随机梯度更新有

$$
\mathbb E[F(w^{(k+1)})]
\le F(w^{(k)})-\frac\eta2\|\nabla F(w^{(k)})\|^2
+\frac{L\eta^2}{2}\sum_r\operatorname{Var}[\widehat g^{(k)}(r)].
$$

将该不等式从 $k=0$ 累加至 $K-1$，并使用式 (4)，得到

$$
\frac1K\sum_{k=0}^{K-1}\mathbb E\|\nabla F(w^{(k)})\|^2
\le\frac{2(F(w^{(0)})-F^*)}{\eta K}
+\eta L\frac{R\sigma_{\max}^2}{T_{\mathrm{lo}}},
$$

即式 (7)。
