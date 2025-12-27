---
title: LLM-RL-DPO
date: 2025-12-09
description: DPO (Direct Preference Optimization) 直接偏好优化
draft: false
categories:
  - 博客
tags:
  - 大模型
  - 强化学习
slug: dpo
katex: true
math: true
---
{{< figure src="/images/ppo_full.png" alt="ppo" width="720" >}}

{{< figure src="/images/dpo_full.png" alt="dpo_full" width="720" >}}

### KL 散度（Kullback–Leibler Divergence）

$$
KL(P \parallel Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}
$$

**含义**
P 分布相对于 Q 分布的相似程度。

**性质**

- KL 散度的值 **大于等于 0**。
- P 和 Q 越相似，KL 散度越接近 0。
- 如果 P 和 Q 分布完全一致，则 **KL 散度 = 0**。
- 注意：
  $$
  KL(P \parallel Q) \neq KL(Q \parallel P)
  $$

**KL 散度大于等于 0 的直观理解**
**直观理解**：KL 散度是一个非负数，因为我们在比较两个分布时，
只有在完全一致时，它们之间的“差异”才为 0。

$$
KL(P \parallel Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}
$$

示例

|   变量   | P(x) | Q(x) |
| :------: | :--: | :--: |
|  x = 0  | 0.2 | 0.8 |
|  x = 1  | 0.8 | 0.2 |
| 计算过程 |      |      |

$$
KL(P \parallel Q) = 0.2 \times \log \frac{0.2}{0.8} + 0.8 \times \log \frac{0.8}{0.2}
$$

直观理解

- KL 散度衡量 **P 分布相对于 Q 分布的差异程度**。
- 由于对数函数的凸性，KL 散度始终 **大于等于 0**。
- 当 P 和 Q 分布完全一致时，KL 散度 **等于 0**。
- 若 P 与 Q 差异越大，KL 散度值越大。

## Bradley–Terry 模型（成对比较模型）

#### 基本思想

用于描述多个选手（或元素）之间的**成对胜负概率**。
假设每个元素 $i$ 都有一个“实力参数”  $\alpha_i > 0$。

定义：

$$
P(i > j) = \frac{\alpha_i}{\alpha_i + \alpha_j}
$$

其中：

- $\alpha_i$：第 $i$ 个元素的真实实力；
- $P(i > j)$：第 $i$ 个元素战胜第 $j$ 个元素的概率。

#### 对数似然函数（Maximum Likelihood Estimation）

模型给出每次对战的概率：

$$
P(i>j)= \frac{\alpha_i}{\alpha_i + \alpha_j}
$$

我们可以计算出“在当前参数 $\alpha$下，**所有比赛结果同时发生的概率**：

$$
L(\boldsymbol{\alpha}) = \prod_{\text{所有比赛}} P(\text{胜方} > \text{负方})
$$

这就是所谓的 **似然函数（Likelihood Function）**。它衡量了：

> 给定参数 $\alpha$，观测到这些比赛结果的可能性有多大”。

根据观测数据构建似然函数并取对数（假设观测到的对战结果如下表）：

| 对战双方 | 胜方 | 败方 | 胜场数 |
| :------: | :--: | :--: | :----: |
|  A vs B  |  A  |  B  |   8   |
|  A vs B  |  B  |  A  |   4   |
|  A vs C  |  A  |  C  |   3   |
|  A vs C  |  C  |  A  |   5   |

$$
\ln L =
8 \ln\left(\frac{\alpha_A}{\alpha_A + \alpha_B}\right)+ 4 \ln\left(\frac{\alpha_B}{\alpha_A + \alpha_B}\right)+ 3 \ln\left(\frac{\alpha_A}{\alpha_A + \alpha_C}\right)+ 5 \ln\left(\frac{\alpha_C}{\alpha_A + \alpha_C}\right)
$$

通过最大化对数似然，可以求得各选手的相对实力参数。

#### 参数估计结果

$$
\alpha_A = 1, \quad
\alpha_B = \frac{1}{2}, \quad
\alpha_C = \frac{5}{3}
$$

指定$\alpha_A$ 的值，可的得到剩下两个的值

#### 推导新的对战概率

例如计算 $P(B > C)$：

$$
P(B > C) = \frac{\alpha_B}{\alpha_B + \alpha_C}
= \frac{1/2}{1/2 + 5/3}
\approx 0.23
$$

**直观理解：**

- 模型通过比较胜负记录估计出每个选手的“潜在实力”；
- 实力越大，战胜其他人的概率越高；
- 可推广用于比赛预测、排序、推荐系统等任务。

#### 一般的 Loss 函数（Bradley–Terry 模型）

模型目标是最大化对数似然，对应的最小化损失函数（负对数似然）为：

$$
Loss = - \mathbb{E}_{(a_x, a_y) \sim D}
\left[
\ln \frac{a_x}{a_x + a_y}
\right]
$$

### 强化学习中的比较建模（Pairwise Preference in RLHF）

在强化学习中：

- 大模型的输入是 **prompt**，记作 $x$；
- 模型的输出（回答）是 **response**，记作 $y$；
- 回答 $y$ 的好坏（即“得分”或“实力”）由 **Reward 模型** $r(x, y)$ 进行评估。

#### 比较两个回答的优劣概率

对于同一个输入 $x$，有两个不同回答 $y_1$ 和 $y_2$。

原始形式为：

$$
P(y_1 \succ y_2) = \frac{r(x, y_1)}{r(x, y_1) + r(x, y_2)}
$$

由于 $r(x, y)$ 可能返回负数，因此引入指数函数，使概率始终为正：

$$
P(y_1 \succ y_2) =
\frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))}
$$

**说明**

- $r(x, y)$：Reward 模型给出的得分（越大表示回答越好）；
- $P(y_1 \succ y_2)$：模型预测回答 $y_1$ 优于 $y_2$ 的概率；
- 该形式本质上是 **Bradley–Terry 模型** 在强化学习（RLHF）中的应用。

**偏好概率建模**

对于同一输入 $x$，Reward 模型给出两个回答 $y_w$（获胜回答）和 $y_l$（失败回答）的得分：

$$
P(y_w \succ y_l) =
\frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}
$$

Sigmoid 函数定义为：

$$
\sigma(x) = \frac{1}{1 + \exp(-x)}
$$

损失函数（Loss Function）
Reward 模型的目标是最小化负对数似然（Negative Log-Likelihood）：

$$
Loss = -\mathbb{E}_{(x, y_w, y_l) \sim D}
\left[
\ln
\frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}
\right]
$$

等价地：

$$
Loss = - \mathbb{E}_{(x, y_w, y_l) \sim D}
\left[
\ln
\frac{1}{1 + \exp(r(x, y_l) - r(x, y_w))}
\right]
$$

最终可写为简洁的 **Sigmoid 形式**：

$$
Loss = - \mathbb{E}_{(x, y_w, y_l) \sim D}
\left[
\ln \sigma(r(x, y_w) - r(x, y_l))
\right]
$$

### DPO 的训练目标（Direct Preference Optimization）

#### 基本定义

- 奖励函数：$r(x, y)$，其中 $x$ 为 **prompt**，$y$ 为 **response**
- 基准模型（reference model）：$\pi_{\text{ref}}(y|x)$
- 训练模型（policy model）：$\pi(y|x)$

DPO 的目标是在获得高奖励的同时，使新模型不偏离参考模型：

$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi(y|x)}[r(x,y)] - \beta \mathbb{D}_{\mathrm{KL}}\big(\pi(y|x) \parallel \pi_{\mathrm{ref}}(y|x)\big)
$$

- 第一项：希望得到尽可能多的奖励；
- 第二项：限制新模型与基准模型的分布差距；
- $\beta$：超参数，用于平衡两者。

#### 目标函数的等价形式推导

将 KL 散度展开：

$$
\max_{\pi} \mathbb{E}_{x, y \sim \pi} [r(x, y)] - \beta \mathbb{E}_{x, y \sim \pi} \left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

等价地（加负号取最小值且两项同时除以$\beta$）：

$$
\min_{\pi} \mathbb{E}_{x, y \sim \pi} \left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta} r(x, y)\right]
$$

将减法项写成对数形式

$$
= \min_{\pi} \; \mathbb{E}_{x, y \sim \pi} \left[ \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \log \exp\left(\frac{1}{\beta} r(x, y)\right) \right]
$$

合并为单个对数项

$$
= \min_{\pi} \; \mathbb{E}_{x, y \sim \pi} \left[ \log \frac{\pi(y|x)} {\pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)} \right]
$$

引入归一化常数 $Z(x)$

$$
= \min_{\pi} \; \mathbb{E}_{x, y \sim \pi} \left[ \log \frac{\pi(y|x)} {\pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right) \frac{1}{Z(x)} Z(x)} \right]
$$

拆出常数项

$$
=\min_{\pi} \; \mathbb{E}_{x, y \sim \pi} \left[ \log \frac{\pi(y|x)}{\frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)} -\log Z(x) \right]
$$

#### 求解最优策略分布

定义：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

其中：

$$
Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

表示归一化常数，是我们自己定义的，这么定义的原因是为了能和后面的式子消去，简化式子。

> **解释**：最优策略 $\pi^*(y|x)$ 是在参考模型分布上，通过奖励函数加权后的归一化分布。

代入 $\pi^*(y|x)$，目标可写为：

$$
\min_{\pi} \mathbb{E}_{x \sim D} \left[ D_{KL}(\pi(y|x) \parallel \pi^*(y|x)) \right]
$$

因此最优时：

$$
\pi(y|x) = \pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

#### 奖励函数反求形式

由上式反推奖励：
由 DPO 的最优策略定义：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp \left(\frac{1}{\beta} r(x, y)\right)
$$

可得：

$$
\exp\left(\frac{1}{\beta} r(x, y)\right) = \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} Z(x)
$$

两边取对数并乘以 $\beta$，得到：

$$
r(x, y) = \beta \ln\left( \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} Z(x) \right)
$$

即：

$$
r(x, y) = \beta \ln \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \ln Z(x)
$$

> 解释：
> 奖励函数可以理解为模型输出相对参考模型的“对数优势”。
> 它衡量了当前模型相对参考模型在特定回答上的改进幅度

#### 与偏好建模的联系

在成对偏好比较中，假设有更优回答 $y_w$ 与较差回答 $y_l$，则最终的损失函数为：

$$
-\ln \sigma(r(x, y_w) - r(x, y_l)) = - \ln \sigma\left( \beta \ln \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \ln \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
$$

> 这表明 DPO 的优化目标实际上是将 **Reward 模型的比较学习**
> 转换为对 **策略分布比值** 的直接优化。

## 总结

DPO的核心洞察在于原始强化学习问题存在解析最优解，表明最优策略与奖励函数存在一一映射关系。DPO将此关系反解后代入Bradley-Terry偏好模型，将对奖励函数的似然最大化，等价地转化为直接对策略的似然最大化。因此，优化DPO损失函数即是在直接寻找那个能同时最大化人类偏好概率且满足最优解形式的策略，避免了先用偏好数据拟合奖励模型再进行强化学习过程寻找最优策略

---

### 参考论文

- [DPO](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)

