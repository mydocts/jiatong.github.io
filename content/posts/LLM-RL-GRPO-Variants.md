---
title: LLM-RL-GRPO-and its Variants
date: 2025-12-14
description: PPO 的改进算法及各类变体 (DAPO, DRPO, GFPO, GRPO-Training-free)
draft: false
categories:
  - 博客
tags:
  - 大模型
  - 强化学习
slug: grpo-variants
katex: true
math: true
---
 **GRPO是针对LLM的一种改进PPO算法**

# 回顾前置知识：

## Policy Gradient

$$
\nabla_\theta J(\theta)
\approx 
\frac{1}{N} 
\sum_{n=1}^{N} 
\sum_{t=1}^{T_n} 
R(\tau^{(n)}) 
\nabla_\theta 
\log P_\theta(a_t^{(n)} \mid s_t^{(n)})
$$

---

 **符号解释**

| 符号                                                     | 含义                                                                                                                                                     |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\nabla_\theta J(\theta)$                              | **目标函数（期望回报）关于参数$\theta$的梯度**，即策略更新的方向。                                                                               |
| $N$                                                    | **采样的轨迹数量（episodes）**，即从环境中采样的完整回合数。                                                                                       |
| $T_n$                                                  | 第$n$条轨迹的时间步数（episode的长度）。                                                                                                               |
| $\tau^{(n)}$                                           | 第$n$条**轨迹（trajectory）**：`<br>` $\tau^{(n)} = (s_1^{(n)}, a_1^{(n)}, r_1^{(n)}, \dots, s_{T_n}^{(n)}, a_{T_n}^{(n)}, r_{T_n}^{(n)})$。 |
| $R(\tau^{(n)})$                                        | **整条轨迹的总回报（return）**，通常为折扣累计奖励：`<br>` $R(\tau^{(n)}) = \sum_{t=1}^{T_n}\gamma^{t-1}r_t^{(n)}$。                           |
| $\nabla_\theta \log P_\theta(a_t^{(n)}\mid s_t^{(n)})$ | **策略梯度项**，表示策略在状态$s_t^{(n)}$下选择动作$a_t^{(n)}$的对数概率的梯度，用于指导参数更新。                                             |
| $P_\theta(a_t^{(n)}\mid s_t^{(n)})$                    | **参数化策略（policy）**，给定状态$s_t^{(n)}$时采取动作$a_t^{(n)}$的概率，由参数$\theta$控制。                                               |

---

 **直观理解**
这个公式的含义是：如果一条轨迹带来的总奖励$R(\tau)$很高，那么模型应该调整参数$\theta$，让在这条轨迹中采取的动作$a_t$的概率更高；反之则降低这些动作的概率。

## 考虑动作影响时效

Discounted Return（折扣回报）

$$
R_t^{(n)} = \sum_{t'=t}^{T_n} \gamma^{\,t'-t} r_{t'}^{(n)}
$$

符号解释

| 符号             | 含义                                                                                                      |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| $t$、$t'$    | 时间步索引；$t'$ 是求和变量，从当前时间步 $t$ 开始一直到 $T_n$。                                    |
| $\gamma$       | **折扣因子（discount factor）**，$0 < \gamma \le 1$。它控制未来奖励的重要性：越小表示越“短视”。 |
| $r_{t'}^{(n)}$ | 第$n$ 条轨迹在时间步 $t'$ 获得的即时奖励（immediate reward）。                                        |

---

 **直观理解**
这个公式计算的是从某个时间步 $t$ 开始，到轨迹结束的所有未来奖励的加权和。
靠近当前的奖励权重大（因为$\gamma^{0}=1$），越往后的奖励会被折扣（因为$\gamma^{t'-t}$会变小）。

---

 **与策略梯度公式的关系**
在策略梯度中，$R_t^{(n)}$ 通常替代 $R(\tau^{(n)})$ 用作加权项，使得每个时间步的梯度更新都考虑该时刻之后的未来回报：

$$
\nabla_\theta J(\theta)
\approx 
\frac{1}{N} 
\sum_{n=1}^{N} 
\sum_{t=1}^{T_n} 
R_t^{(n)} 
\nabla_\theta 
\log P_\theta(a_t^{(n)} \mid s_t^{(n)})
$$

## 考虑到动作的相对优势

  Policy Gradient with Baseline

$$
\nabla_\theta J(\theta)
\approx 
\frac{1}{N} 
\sum_{n=1}^{N} 
\sum_{t=1}^{T_n} 
\big(R_t^{(n)} - B(s_t^{(n)})\big)
\nabla_\theta 
\log P_\theta(a_t^{(n)} \mid s_t^{(n)})
$$

---

符号解释

| 符号                                                     | 含义                                                                                                       |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| $B(s_t^{(n)})$                                         | **基线函数（baseline function）**，通常是状态价值函数$V_\pi(s_t^{(n)})$，用于减小方差。            |
| $\nabla_\theta \log P_\theta(a_t^{(n)}\mid s_t^{(n)})$ | **策略梯度项**，表示在状态$s_t^{(n)}$下采取动作$a_t^{(n)}$的对数概率的梯度。                     |
| $P_\theta(a_t^{(n)}\mid s_t^{(n)})$                    | **参数化策略（policy）**，给定状态$s_t^{(n)}$时采取动作$a_t^{(n)}$的概率，由参数$\theta$控制。 |

---

**直观理解**与原始 REINFORCE 相比，这个版本在计算梯度时引入了一个**基线项 $B(s_t^{(n)})$**，用于减少方差但不引入偏差。它衡量“当前动作的回报比期望高多少”：

- 如果 $R_t^{(n)} > B(s_t^{(n)})$，说明该动作比平均表现好 → 提高它的概率；
- 如果 $R_t^{(n)} < B(s_t^{(n)})$，说明该动作表现差 → 降低它的概率。

---

**备注**
若将 $A_t^{(n)} = R_t^{(n)} - B(s_t^{(n)})$ 记作**优势函数（advantage function）**，
公式可简写为：

$$
\nabla_\theta J(\theta)
\approx 
\frac{1}{N} 
\sum_{n=1}^{N} 
\sum_{t=1}^{T_n} 
A_t^{(n)}
\nabla_\theta 
\log P_\theta(a_t^{(n)} \mid s_t^{(n)})
$$

## 引入优势函数

#### 优势函数定义

$$
A_\theta(s, a) = Q_\theta(s, a) - V_\theta(s)
$$

在状态 $s$ 下，执行动作 $a$，相比平均水平（状态价值）**能带来多少额外优势**。

---

各符号解释

| 符号               | 含义                                                                                                              |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| $A_\theta(s, a)$ | **优势函数（Advantage Function）**，衡量在状态 $s$ 下执行动作 $a$ 比平均表现（$V_\theta(s)$）好多少。 |
| $Q_\theta(s, a)$ | **动作价值函数（Action-Value Function）**：在状态 $s$ 下执行动作 $a$ 后，期望获得的累计回报。           |
| $V_\theta(s)$    | **状态价值函数（State-Value Function）**：在状态 $s$ 下，依据当前策略期望的回报。                         |
| $\theta$         | 策略参数，控制策略$\pi_\theta$ 或价值函数的参数化形式。                                                         |

---

 **直观理解**

- $Q_\theta(s,a)$ 衡量“执行某动作后能得到的回报”；
- $V_\theta(s)$ 衡量“在该状态下平均能得到的回报”；
- 二者之差 $A_\theta(s,a)$ 衡量“这个动作比平均动作好多少”。

当 $A_\theta(s,a) > 0$ 时，该动作优于平均水平，应提升其概率；
当 $A_\theta(s,a) < 0$ 时，该动作劣于平均，应降低其概率。

---

#### Advantage Policy Gradient 公式

$$
\nabla_\theta J(\theta)
\approx 
\frac{1}{N} 
\sum_{n=1}^{N} 
\sum_{t=1}^{T_n} 
A_\theta(s_t^{(n)}, a_t^{(n)})
\nabla_\theta 
\log P_\theta(a_t^{(n)} \mid s_t^{(n)})
$$

---

符号解释

| 符号                                        | 含义                                                                |
| ------------------------------------------- | ------------------------------------------------------------------- |
| $\nabla_\theta J(\theta)$                 | 期望回报关于参数$\theta$ 的梯度，即策略改进方向。                 |
| $A_\theta(s_t^{(n)}, a_t^{(n)})$          | 第$n$ 条轨迹第 $t$ 步的优势值，衡量该动作相对于平均水平的好坏。 |
| $\log P_\theta(a_t^{(n)} \mid s_t^{(n)})$ | 策略函数在状态$s_t^{(n)}$ 下选择动作 $a_t^{(n)}$ 的对数概率。   |
| $N$                                       | 采样轨迹数量（episodes）。                                          |
| $T_n$                                     | 第$n$ 条轨迹的时间步数。                                          |

---

## GAE 优势函数（Generalized Advantage Estimation）

$$
A_\theta(s_t, a_t) = Q_\theta(s_t, a_t) - V_\theta(s_t)
$$

$$
Q_\theta(s_t, a_t) = r_t + \gamma V_\theta(s_{t+1})
$$

$$
A_\theta(s_t, a_t) = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)
$$

$$
V_\theta(s_{t+1}) \approx r_{t+1} + \gamma V_\theta(s_{t+2})
$$

<div align="left">

$$
\begin{aligned}
A_\theta^{1}(s_t, a_t) &= r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t) \\
A_\theta^{2}(s_t, a_t) &= r_t + \gamma r_{t+1} + \gamma^2 V_\theta(s_{t+2}) - V_\theta(s_t) \\
A_\theta^{3}(s_t, a_t) &= r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V_\theta(s_{t+3}) - V_\theta(s_t) \\
A_\theta^{T}(s_t, a_t) &= r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots + \gamma^{T} r_T - V_\theta(s_t)
\end{aligned}
$$

</div>

$$
A_\theta^{GAE}(s_t, a_t) = (1 - \lambda)(A_\theta^{1} + \lambda A_\theta^{2} + \lambda^2 A_\theta^{3} + \cdots)
$$

# GRPO

## GRPO算法计算每个Token的Advantage

**GRPO**：Group Relative Policy Optimization
**中文名称**：群体相对策略优化

#### 示例输入（Prompt及生成结果）

同一个问题，模型会输出三个结果

| 什么 | 是 | 数据库 | ? | → | 数据库 | 用于 | 存储 |   数据   |    。    |      |    |      |
| :--: | :-: | :----: | :-: | :-: | :----: | :--: | :--: | :------: | :------: | ---- | -- | ---- |
| 什么 | 是 | 数据库 | ? | → | 数据库 |  是  | 一个 | 有组织的 | 数据集合 |      |    |      |
| 什么 | 是 | 数据库 | ? | → | 数据库 |  是  | 用来 |   高效   |   存取   | 数据 | 的 | 软件 |

#### 奖励（Reward）

对每个句子打一个分

$$
r_1 = 3.8, \quad r_2 = 5.2, \quad r_3 = 6.1
$$

 **标准化奖励（Standardized Reward）**

$$
\tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
$$

$$
\tilde{r}_1 = -1.06, \quad \tilde{r}_2 = 0.14, \quad \tilde{r}_3 = 0.92
$$

**每个Token的Advantage表**

| Token序列                             | Advantage值 |
| :------------------------------------ | :---------: |
| 数据库 用于 存储 数据 。              |    -1.06    |
| 数据库 是 一个 有组织的 数据集合      |    0.14    |
| 数据库 是 用来 高效 存取 数据 的 软件 |    0.92    |

这里代表着每一个token都是Advantage值，例如‘数据库’ ’用于‘ ’存储‘ ’数据‘ 的token都是 -1.

#### 目标函数

$$
J_{GRPO} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{T_n}\sum_{t=1}^{T_n}
\min \Bigg(
\textcolor{red}{A_{\theta'}^{GRPO}(s_n^t, a_n^t)
\frac{P_\theta(a_n^t|s_n^t)}{P_{\theta'}(a_n^t|s_n^t)}},
clip\left(
\frac{P_\theta(a_n^t|s_n^t)}{P_{\theta'}(a_n^t|s_n^t)},
1-\varepsilon, 1+\varepsilon
\right)
A_{\theta'}^{GRPO}(s_n^t, a_n^t)
\Bigg) - \beta KL(P_\theta, P_{ref})
$$

#### 点评

PPO在计算token的优势的时候，Reward模型只把得分到最后一个token里，其他得分都为0，然后用KL散度乘以一个系数加上刚才的得分 得到Reward对整体句子中每一个token的打分，但是对于大模型回答来说，我们评判的是整体的回答，而不是像游戏中那样在某个场景下只关注当下的动作，因此用KL散度乘系数再加上Reward对最后一个token打分这种计算方式，对于评估大模型生成的回答质量实在有些牵强

简洁来说，PPO只关注最后一个token的优势值，而我们想关注的是一句话作为整体的每个token的优势值。

GRPO提供了一种不需要训练状态价值网络，就可以估算每个token优势值的方法，而且这个方法更适合大模型生成强化学习这个场景

实际代码中，Clip和KL散度用一个就行了。

---

### 其他变体

- [GRPO](https://arxiv.org/abs/2402.03300) - DeepSeekMath (February 2024)
- [DAPO](https://arxiv.org/abs/2503.14476) - Decoupled Clip and Dynamic Sampling Policy Optimization (March 2025)
- [Dr.DRPO](https://arxiv.org/abs/2510.04474) - Decoupled Reward Policy Optimization (October 2025)
- [GFPO](https://arxiv.org/abs/2508.09726) - Group Filtered Policy Optimization (August 2025)
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) - DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (December 2025)
- [Training-Free GRPO](https://arxiv.org/abs/2510.08191) - Tencent: Training-Free Group Relative Policy Optimization (October 2025)

---

### 附录：不同规模模型的推荐学习率参考

以下表格整理了在不同模型规模下，**SFT**（指令微调）与 **DPO**（偏好优化）的推荐学习率区间。数据主要基于 LoRA/QLoRA 微调场景。

| 模型规模 (Model Scale) | SFT 推荐 LR (LoRA)     | DPO 推荐 LR            | 备注 (Notes)                                                                          |
| :--------------------- | :--------------------- | :--------------------- | :------------------------------------------------------------------------------------ |
| **0.5B - 1.8B**  | **2e-4 — 5e-4** | **5e-6 — 1e-5** | 模型较小，相对“坚韧”，可以使用较大的步长快速收敛。                                  |
| **7B - 8B**      | **5e-5 — 1e-4** | **1e-6 — 3e-6** | 主流模型规模（如 Llama 3, Qwen 2.5）。**5e-7** 通常是下限，过低会导致收敛极慢。 |
| **14B - 32B**    | **1e-5 — 5e-5** | **5e-7 — 1e-6** | 参数量较大，即使是 LoRA 也需要非常小心，LR 过大容易导致灾难性遗忘。                   |
| **70B+**         | **5e-6 — 1e-5** | **1e-7 — 5e-7** | 极小的步长，通常配合极大的 Batch Size 进行训练。                                      |

> **关键指标参考**：
>
> * **GRPO**: 学习率通常建议 **≤ 1e-5** (Full) 或 **5e-5** (LoRA)，Beta 推荐 **0.001**。
> * **DPO 监控**: 重点关注 `rewards/chosen`、`rewards/rejected` 和 `rewards/margins`。
