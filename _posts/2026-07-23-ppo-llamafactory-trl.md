---
title: "适配 PPO 流程梳理LLaMA-Factory（TRL）实现"
date: 2026-07-23 19:30:00 +0800
categories: [强化学习, LLM微调]
tags: [PPO, TRL, LLaMA-Factory, RLHF, 大模型对齐]
math: true
---

# 适配 PPO 流程梳理LLaMA-Factory（TRL）实现

## 一、模型组成
1. **Actor（新策略）**：可训练，初始化权重 = SFT Reference模型；Transformer主干，输出token分布
2. **Old Policy**：**无独立网络**；采样生成response时缓存当前Actor的 $\log\pi_{\text{old}}$，同一批样本多次梯度更新全程固定
3. **Reference Model（SFT对照模型）**：永久冻结，用于计算逐token近似KL惩罚
4. **RM奖励模型**：永久冻结，输入完整`prompt+response`，输出单一终端标量奖励 $R_{\text{rm}}$
5. **Critic价值模型**：与Actor共享Transformer主干，附加独立ValueHead，输出逐token状态价值 $V(s_t)$

## 二、采样 Rollout 流程
1. Prompt送入**Actor（新策略）**自回归生成response；同步记录采样时刻 $\boldsymbol{\log\pi_{\text{old}}}$、逐token价值 $V(s_1),V(s_2),V(s_3)$
2. 冻结Reference模型前向传播，得到每个token对应的 $\log\pi_{\text{ref}}(a_t|s_t)$
3. `prompt+response`送入RM，得到整条序列终端奖励 $R_{\text{rm}}$

### 推演样例预设参数
```
序列长度 T=3
超参：
γ = 0.99    折扣因子
λ = 0.95    GAE λ
β = 0.05    KL惩罚系数
ε = 0.2     PPO clip阈值
c = 0.5     Critic损失权重

采样得到数据：
V(s₁)=0.30, V(s₂)=0.50, V(s₃)=0.70, V(s₄)=0.0（序列终止位置价值=0）
R_rm = 2.0
logπ_new = [-0.80, -0.60, -0.40]
logπ_old = [-0.82, -0.58, -0.43]
logπ_ref = [-1.20, -0.90, -0.70]
```

## 三、训练完整计算链路（附带分步数值演算）
### 1. 逐token近似KL（TRL默认单样本估计）
公式：
$$
\text{KL}_t \approx \log\pi_{\text{new}}(a_t|s_t) - \log\pi_{\text{ref}}(a_t|s_t)
$$

演算：
$$
\begin{align}
\text{KL}_1 &= -0.80 - (-1.20) = 0.40 \\
\text{KL}_2 &= -0.60 - (-0.90) = 0.30 \\
\text{KL}_3 &= -0.40 - (-0.70) = 0.30 \\
\end{align}
$$

### 2. 构造带KL约束的有效奖励 $\tilde{r}_t$
基础原始奖励规则：
$$
\begin{cases}
r_t = 0, & t=1,2,\dots,T-1 \\
r_T = R_{\text{rm}}, & t=T
\end{cases}
$$

有效奖励：
$$
\tilde{r}_t = r_t - \beta\cdot\text{KL}_t
$$

演算：
$$
\begin{align}
\tilde{r}_1 &= 0 - 0.05 \times 0.40 = -0.02 \\
\tilde{r}_2 &= 0 - 0.05 \times 0.30 = -0.015 \\
\tilde{r}_3 &= 2.0 - 0.05 \times 0.30 = 1.985 \\
\end{align}
$$

### 3. GAE广义优势估计（从后向前递推）
公式：
$$
\begin{align}
\delta_t &= \tilde{r}_t + \gamma V(s_{t+1}) - V(s_t) \\
A_t &= \delta_t + \gamma\lambda \cdot A_{t+1}
\end{align}
$$

边界条件：序列结束 $A_{T+1}=0$

逐位演算：
- t=3
$$
\begin{align}
\delta_3 &= \tilde{r}_3 + \gamma V(s_4) - V(s_3) = 1.985 + 0.99\times0 - 0.70 = 1.285 \\
A_3 &= \delta_3 + \gamma\lambda\cdot A_4 = 1.285 + 0 = 1.285
\end{align}
$$

- t=2
$$
\begin{align}
\delta_2 &= \tilde{r}_2 + \gamma V(s_3) - V(s_2) = -0.015 + 0.99\times0.70 - 0.50 = 0.178 \\
A_2 &= \delta_2 + \gamma\lambda\cdot A_3 = 0.178 + 0.99\times0.95\times1.285 \approx 1.388
\end{align}
$$

- t=1
$$
\begin{align}
\delta_1 &= \tilde{r}_1 + \gamma V(s_2) - V(s_1) = -0.02 + 0.99\times0.50 - 0.30 = 0.175 \\
A_1 &= \delta_1 + \gamma\lambda\cdot A_2 = 0.175 + 0.99\times0.95\times1.388 \approx 1.480
\end{align}
$$

> 工程可选：对batch内所有token的优势做标准化 $A_{\text{norm}} = \frac{A-\text{mean}(A)}{\text{std}(A)}$，稳定训练

### 4. Actor Clip Loss 计算
概率比率：
$$
r_t = \frac{\pi_{\text{new}}}{\pi_{\text{old}}} = \exp\big(\log\pi_{\text{new}} - \log\pi_{\text{old}}\big)
$$

Clip目标项：
$$
L_{\text{clip},t}=-\min\big(r_t A_t,\text{clip}(r_t,1-\epsilon,1+\epsilon)\cdot A_t\big)
$$

演算：
$$
\begin{align}
r_1 &= \exp(-0.80 - (-0.82)) = e^{0.02} \approx 1.0202 \\
r_2 &= \exp(-0.60 - (-0.58)) = e^{-0.02} \approx 0.9802 \\
r_3 &= \exp(-0.40 - (-0.43)) = e^{0.03} \approx 1.0305 \\
\end{align}
$$
$\epsilon=0.2$，clip区间 $[0.8,1.2]$，三个比率均落在区间内，不触发截断：
$$
\begin{align}
L_{\text{clip},1} &= -\min(1.0202\times1.480,\ 1.0202\times1.480) \approx -1.510 \\
L_{\text{clip},2} &= -\min(0.9802\times1.388,\ 0.9802\times1.388) \approx -1.361 \\
L_{\text{clip},3} &= -\min(1.0305\times1.285,\ 1.0305\times1.285) \approx -1.324 \\
\end{align}
$$
batch取均值得到最终 $L_{\text{clip}}$

### 5. Critic价值损失（MSE）
价值目标（target，计算后执行`detach()`切断梯度）
$$
G_t = A_t + V(s_t)
$$

演算：
$$
\begin{align}
G_1 &= 1.480 + 0.30 = 1.780 \\
G_2 &= 1.388 + 0.50 = 1.888 \\
G_3 &= 1.285 + 0.70 = 1.985 \\
\end{align}
$$

Critic损失：
$$
L_{\text{critic},t} = \big(V(s_t) - G_t\big)^2
$$
$$
\begin{align}
L_{\text{critic},1} &= (0.30 - 1.780)^2 = 2.1904 \\
L_{\text{critic},2} &= (0.50 - 1.888)^2 = 1.9265 \\
L_{\text{critic},3} &= (0.70 - 1.985)^2 = 1.6512 \\
\end{align}
$$
batch取均值得到 $L_{\text{critic}}$

### 6. 总损失
$$
L_{\text{total}}=L_{\text{clip}}+c\cdot L_{\text{critic}}
$$

## 四、参数更新规则
- Reference Model、Reward Model：全程冻结，不参与梯度更新
- $\log\pi_{\text{old}}$：采样rollout阶段缓存，本轮样本多次`ppo_epoch`更新中保持固定；框架不存在独立Old Actor网络
- Actor主干 + Critic ValueHead：联合反向传播更新
这样对吗
