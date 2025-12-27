---
title: LLM-RL-PPO
date: 2025-12-06
description: PPO（Proximal Policy Optimization) 近端策略优化
draft: false
categories:
  - 博客
tags:
  - 大模型
  - 强化学习
slug: ppo
math: true
katex: true
---
{{< figure src="/images/ppo_1.png" alt="前置基础" width="720" >}}

## 基础概念I

* 如果你不熟悉强化学习，学习ppo了解这些基础知识就足够了

**Action Space**: 可选择的动作，比如 {left, up, right}

**Policy**: 策略函数，输入 State，输出 Action 的概率分布。一般用 π 表示。

* $\pi(left|s_t) = 0.1$,在状态$s_t$下，采取的动作为left的概率为0.1
* $\pi(up|s_t) = 0.2$ , $\pi(right|s_t) = 0.7$, 这个状态下的 所有动作 概率之和 为1

**Trajectory**: 轨迹，用$τ$表示，一连串状态和动作的序列。又称Episode, Rollout。 ${s0,a0,s1,a1,…}$

$s_{t+1}=f(st,at)$ 确定状态转移

$s_{t+1}=P(⋅|st,at)$随机状态转移

**Return**: 回报，从当前时间点到游戏结束的 Reward 的累积和。

**期望**：每个可能结果的概率与其结果值的乘积之和

$\mathbb{E}(x)_{x \sim p(x)} = \sum x \cdot p(x) \approx \frac{1}{n} \sum_{i=1}^n x, \quad x \sim p(x)$

$\mathbb{E}[f(\tau)] \approx \frac{1}{N} \sum_{n=1}^N f(\tau^n)$

> [!NOTE]
> 运用的是蒙特卡洛思想，从分布p(x)中随机采样n次求平均，n趋于无穷时样本平均会趋近于期望值/大数定律，期望可以用样本平均近似

**轨迹概率**：一条轨迹 $\tau = (s_0, a_0, s_1, a_1, \dots, s_T, a_T)$的概率是：  $P_\theta(\tau) = P(s_0, a_0, s_1, a_1, \dots, s_T, a_T)$
**概率论基本公式**： $P(x_1, x_2, \dots, x_n) = \prod_{i=1}^n P(x_i \mid x_1, \dots, x_{i-1})$
把它用在轨迹上，就可以逐步展开：  $P_\theta(\tau) = P(s_0) \cdot P(a_0 \mid s_0) \cdot P(s_1 \mid s_0, a_0) \cdot P(a_1 \mid s_1,a_0,s_0) \cdot \dots$
在强化学习里，环境通常假设是 马尔可夫决策过程 (MDP)，即：

* 下一状态只依赖于当前状态和动作： $P(s_{t+1} \mid s_0, a_0, \dots, s_t, a_t) = P(s_{t+1} \mid s_t, a_t)$
* 动作只依赖于当前状态： $P(a_t \mid s_0, a_0, \dots, s_t) = \pi_\theta(a_t \mid s_t)$
* 于是上面的展开可以简化为： $\  P_\theta(\tau) = P(s_0) \prod_{t=0}^T \pi_\theta(a_t \mid s_t) \, P(s_{t+1} \mid s_t, a_t)$
* $\pi_\theta(a_n^t | s_n^t)$：策略在状态 $s_n^t$下选择动作 $a_n^t$ 的概率（依赖参数 $\theta$）。
* $P(s_{n}^{t+1}|s_n^t, a_n^t)$：环境的转移概率（不依赖 $\theta$）。

> [!IMPORTANT]
> **目标：**
> 训练一个Policy神经网络 $π_{\theta}$ ,在所有的状态S下，给出相应的Action，得到的Return的期望最大
> or
> 训练一个Policy神经网络 $\pi_{\theta}$ ,在所有Trajectory中，得到的Return的期望最大

## PPO原理-part1

我们将刚才的目标翻译成数学表达式，就是希望
$\mathbb{E}[R(\tau)]_{\tau \sim P_\theta(\tau)} = \sum_{\tau} R(\tau) P_\theta(\tau)$
越大越好

* $P_\theta(\tau)$ :在策略$\pi_\theta$下，采样到轨迹 $\tau$ 的概率。
* $R(\tau)$：轨迹 $\tau$ 的总回报（Return）。

> [!NOTE]
> 为什么$R(\tau)$不受$\theta$影响？
> $R(τ)=∑_{t=0}^{T−1}γ^{t}r(s_t,a_t)$ 因为一旦一条轨迹被确定下来，这条轨迹的回报R是由每个状态下的每个动作的得分之和决定的，又因为越远的动作影响对当前状态影响越小，所以要乘上一个衰减因子$\gamma$

我们只能改变神经网络里的参数 θ ,不能改变Reward，所以对 θ求梯度

$$
\nabla \mathbb{E}[R(\tau)]_{\tau \sim P_{\theta}(\tau)}
$$

$$
= \nabla \sum_{\tau} R(\tau) P_\theta(\tau)
$$

$$
= \sum_{\tau} R(\tau) \nabla P_\theta(\tau)
$$

$$
= \sum_{\tau} R(\tau) \nabla P_\theta(\tau) \frac{P_\theta(\tau)}{P_\theta(\tau)}
$$

$$
= \sum_{\tau} P_\theta(\tau) R(\tau) \frac{\nabla P_\theta(\tau)}{P_\theta(\tau)}
$$

$$
\approx \frac{1}{N} \sum_{n=1}^N R(\tau^n) \frac{\nabla P_\theta(\tau^n)}{P_\theta(\tau^n)}
$$

> [!TIP]
> $\tau^n$表示第n条轨迹（trajectory）;多次采样取平均

$$
= \frac{1}{N} \sum_{n=1}^N R(\tau^n) \nabla \log P_\theta(\tau^n)
$$

> [!TIP]
> 因为$\nabla \log f(x) = \frac{\nabla f(x)}{f(x)}$

$$
= \frac{1}{N} \sum_{n=1}^N R(\tau^n) \nabla \log \prod_{t=1}^{T_n} P_\theta(a_n^t | s_n^t)
$$

> [!TIP]
> 这里是轨迹分解，因为对 $\theta$  求梯度只关注 $P_\theta$ ,就是前面基础知识中的 $\pi_\theta$,环境转移概率不包含 $\theta$ ,故当作常数，求梯度就没了

$$
= \frac{1}{N} \sum_{n=1}^N R(\tau^n) \sum_{t=1}^{T_n} \nabla \log P_\theta(a_n^t | s_n^t)
$$

$$
= \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} R(\tau^n)\,\nabla \log P_\theta(a_n^t | s_n^t)
$$

到这里，我们求出了对于所有可能的Trajectory，期望的最大梯度，用个梯度去更新神经网络参数，就是 Policy gradient梯度策略算法

## PPO原理-part2

$$
\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} R(\tau^n)\,\nabla \log P_\theta(a_n^t | s_n^t)
$$

刚刚我们得到的这个式子，很明显有两点可以改进

* 应该看在$S_n$状态下采取动作$a_n$之后的Reward，而不是整个Trajectory（轨迹）的Reward，因为动作action只能影响后面的
* action是只会影响后面的动作，但是影响会逐步衰减

由此我们修改Reward公式为

$$
R(\tau^n) \to \sum_{t'=t}^{T_n} \gamma^{t'-t}r_{t'}^n = R_t^n
$$

 表示从当前t时刻开始到最后的累积Reward， 

$\gamma$

 是衰减因子，表示离当前动作越远，在当前状态下采取当前动作的Reward越小，核心就是想突出当前状态下采取当前动作对于Reward的影响.

另一个值得注意的点是**局势**也会影响算法的稳定性，比如说在好的局势下，采取什么动作都会得分（顺风局当赢的感觉），但是这样就偏离我们的初衷——我们想知道在某种状态下采取哪些动作更好，模型需要明确区分哪个动作“更好”

因此我们希望让相对好的action的Reward得分增加，相对差的action的Reward减少，这样会加快训练速度。

由此我们减去一个Baseline

$$
\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} \bigl(R_t^n - B(s_n^t)\bigr) \nabla \log P_\theta(a_n^t \mid s_n^t)
$$

## 基础概念II

强化学习中有几个重要的定义，可以简化上述式子

$\textbf{Action-Value Function}(动作价值函数)$
 $Q_\theta(s,a) 在 state \ \ s 下，做出 Action \ \ a的回报的期望$

$\textbf{State-Value Function} (状态价值函数)$
$V_\theta(s) 在 state \ \ s 下，回报的期望$。

$\textbf{Advantage Function}(优势函数)$
$A_\theta(s,a) = Q_\theta(s,a) - V_\theta(s) \quad$
$在 state \ \ s 下，做出 Action \ \ a，比其他动作能带来多少优势。$于是原式被转化为：

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n}  A_\theta(s_n^t, a_n^t) \nabla \log P_\theta(a_n^t | s_n^t)
$$

现在我们得到了理论上表达式，但是如何表示实际采样的呢？

一次采样：

 $Q_\theta(s_t, a) = r_t + \gamma \cdot V_\theta(s_{t+1})$

 $A_\theta(s_t, a) = r_t + \gamma \cdot V_\theta(s_{t+1}) - V_\theta(s_t)$

 $V_\theta(s_{t+1}) \approx r_{t+1} + \gamma \cdot V_\theta(s_{t+2})$

> [!TIP]
> 上面我们分别对动作价值函数和状态价值函数进行了一次采样，对于动作价值函数Q来说，采用的 动作是固定的，因此可以用等号来表示，对于状态价值函数，采取的动作a还没有固定，因此用约等于表示一次采样

多次采样，虽然会增大方差，但是会减少偏差

$$
A_\theta^1(s_t, a) = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)
$$

$$
A_\theta^2(s_t, a) = r_t + \gamma r_{t+1} + \gamma^2 V_\theta(s_{t+2}) - V_\theta(s_t)
$$

$$
A_\theta^3(s_t, a) = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V_\theta(s_{t+3}) - V_\theta(s_t)
$$

$$
\vdots
$$

$$
A_\theta^T(s_t, a) = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots + \gamma^T r_T - V_\theta(s_t)
$$

定义新函数，简化表示

$$
\delta_t^V = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)
$$

$$
\delta_{t+1}^V = r_{t+1} + \gamma V_\theta(s_{t+2}) - V_\theta(s_{t+1})
$$

$$
A_\theta^1(s_t, a) = \delta_t^V
$$

$$
A_\theta^2(s_t, a) = \delta_t^V + \gamma \delta_{t+1}^V
$$

$$
A_\theta^3(s_t, a) = \delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V
$$

$$
\vdots
$$

所以算法（比如 GAE）会综合使用多个步长 k 的估计，构建出一个平衡版本（平衡方差和偏差）：
多次采样，并乘上对应衰减权重

$$
A_\theta^{GAE}(s_t, a) = (1 - \lambda)\big(A_\theta^1 + \lambda A_\theta^2 + \lambda^2 A_\theta^3 + \cdots \big)
$$

$$
\lambda = 0.9: \quad A_\theta^{GAE} = 0.1 A_\theta^1 + 0.09 A_\theta^2 + 0.081 A_\theta^3 + \cdots
$$

$$
= (1 - \lambda)\big(\delta_t^V + \lambda(\delta_t^V + \gamma \delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V) + \cdots \big)
$$

$$
= (1 - \lambda)\big(\delta_t^V(1 + \lambda + \lambda^2 + \cdots) + \gamma \delta_{t+1}^V (\lambda + \lambda^2 + \cdots) + \cdots \big)
$$

$$
= (1 - \lambda)\big(\delta_t^V \tfrac{1}{1-\lambda} + \gamma \delta_{t+1}^V \tfrac{\lambda}{1-\lambda} + \cdots \big)
$$

$$
= \sum_{b=0}^{\infty} (\gamma \lambda)^b \, \delta_{t+b}^V
$$

最后得到这三个表达式：

$$
\delta_t^V = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)
$$

$$
A_\theta^{GAE}(s_t, a) = \sum_{b=0}^{\infty} (\gamma \lambda)^b \, \delta_{t+b}^V
$$

$$
\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} A_\theta^{GAE}(s_n^t, a_n^t) \nabla \log P_\theta(a_n^t \mid s_n^t)
$$

## PPO原理-part3

{{< figure src="/images/ppo_2.png" alt="前置基础" width="720" >}}
通过运行模型来采集数据，这样就会导致采集数据时间过长，这也是ppo需要解决的问题,对于此我们采用重要性采样来解决

### 重要性采样

$$
\mathbb{E}(f(x))_{x \sim p(x)} = \sum_x f(x) p(x)
$$

$$
= \sum_x f(x) p(x) \frac{q(x)}{q(x)}
$$

$$
= \sum_x f(x) \frac{p(x)}{q(x)} q(x)
$$

$$
= \mathbb{E}\!\left(f(x) \frac{p(x)}{q(x)}\right)_{x \sim q(x)}
$$

$$
\approx \frac{1}{N} \sum_{n=1}^N f(x) \frac{p(x)}{q(x)}, \quad x \sim q(x)
$$

由此我们可以变换我们的公式，由On-Policy 转向Off-Policy

$$
\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n}  A_{\theta}^{GAE}(s_n^t, a_n^t) \nabla \log P_\theta(a_n^t \mid s_n^t)
$$

$$
= \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t \mid s_n^t)}{P_{\theta'}(a_n^t \mid s_n^t)} \nabla \log P_\theta(a_n^t \mid s_n^t)
$$

$$
= \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{\nabla P_\theta(a_n^t \mid s_n^t)}{P_{\theta'}(a_n^t \mid s_n^t)}
$$

这里我们的做法是用旧策略(off-policy)的优势函数乘一个比例系数去模拟新的策略函数(on-policy)

最后去掉梯度我们可以得到最终的loss函数（中间对$\theta$求梯度是为了消去不影响$theta$的状态转移概率）

$$
\text{Loss} = -\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t \mid s_n^t)}{P_{\theta'}(a_n^t \mid s_n^t)}
$$

注意：旧策略与新策略之间的分布不能差距过大，否则很难学到有用的经验

可以采用KL散度进行约束或者clip函数（限定那个新旧两种策略之间的比例在1左右，不能相差过大）

$$
\text{Loss}_{ppo_1} = -\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t \mid s_n^t)}{P_{\theta'}(a_n^t \mid s_n^t)} + \beta \, KL(P_\theta, P_{\theta'})
$$

$$
\text{Loss}_{ppo_2} = -\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} \min \Bigg( A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t \mid s_n^t)}{P_{\theta'}(a_n^t \mid s_n^t)} , \text{clip}\left( \frac{P_\theta(a_n^t \mid s_n^t)}{P_{\theta'}(a_n^t \mid s_n^t)}, 1-\epsilon, 1+\epsilon \right) A_{\theta'}^{GAE}(s_n^t, a_n^t) \Bigg)
$$

---

### 参考论文

- [PPO](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization Algorithms (Schulman et al., 2017)

