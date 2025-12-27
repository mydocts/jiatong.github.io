# 核心强化学习算法代码总结 (`core_algos.py`)

该文件是 **verl** 框架中强化学习 (RL) 核心算法和工具函数的集合，主要针对大语言模型 (LLM) 对齐任务中的**策略梯度 (Policy Gradient)** 方法（如 PPO, GRPO, RLOO 等）。

## 1. 概览
`core_algos.py` 提供了以下功能的实现：
- **KL 散度控制器 (KL Controllers)**：平衡新策略与参考策略（Reference Policy）之间的距离。
- **优势估计器 (Advantage Estimators)**：实现多种计算“动作优劣”的方法。
- **策略损失函数 (Policy Loss)**：更新 Actor（策略网络）的各种目标函数。
- **价值损失函数 (Value Loss)**：更新 Critic（价值网络）的目标函数。
- **奖励工具 (Reward Utilities)**：计算包含 KL 惩罚的 Token 级奖励。

---

## 2. 核心组件

### A. KL 控制器 (KL Controllers)
用于在训练过程中动态或静态地调整 KL 惩罚系数。
- **`AdaptiveKLController`**：实现自适应 KL 控制。根据当前的 KL 散度与目标值（target）的差距来自动调整系数，确保训练稳定。
- **`FixedKLController`**：使用固定的 KL 系数。

### B. 优势估计器 (Advantage Estimators)
优势估计是减少策略梯度方差的关键。文件中实现了多种算法：
- **`compute_gae_advantage_return`**：标准的 **GAE (广义优势估计)**。依赖价值网络计算 TD 误差。
- **`compute_grpo_outcome_advantage`**：**GRPO (群体相对策略优化)**。这是 DeepSeek 等框架常用的方法，通过同一 Prompt 下的一组采样样本的奖励进行归一化计算优势，**不需要 Critic 网络**。
- **`compute_rloo_outcome_advantage`**：**RLOO (Reinforce Leave-One-Out)**。通过留一法计算基线。
- **`compute_remax_outcome_advantage`**：ReMax 算法。
- **`compute_opo_outcome_advantage`**：在线偏好优化 (OPO)。

### C. 策略损失函数 (`Actor`)
计算用于更新策略网络的损失值。
- **`compute_policy_loss`**：标准的 **PPO 裁剪目标函数 (Clipped Objective)**。防止策略在一轮迭代中变化过大。
- **`compute_policy_loss_reinforce`**：标准的 REINFORCE 损失。
- **`compute_policy_loss_gspo`**：GSPO 损失函数。
- **`compute_policy_loss_gpg`**：组策略梯度损失。

### D. 价值损失函数 (`Critic`)
- **`compute_value_loss`**：计算预测价值与实际回报（Returns）之间的均方误差 (MSE)，通常包含裁剪（clip）逻辑以增强训练稳定性。

---

## 3. 核心逻辑流程

1.  **计算奖励 (Reward Calculation)**：`compute_rewards` 函数将奖励模型给出的原始分数（Scores）扣除 KL 惩罚，得到最终用于训练的 Token 级奖励。
2.  **计算优势 (Advantage Calculation)**：根据配置（如 GAE 或 GRPO），为每个 Token 或序列计算优势值 $A_t$。
3.  **计算损失并更新**：
    *   **Actor Loss**：利用优势值和对数概率计算梯度。
    *   **Critic Loss**：更新价值网络，使其更好地预测未来回报。

---

## 4. 设计模式
- **注册机制 (Registry System)**：通过装饰器（如 `@register_policy_loss`, `@register_adv_est`）注册不同的损失函数和估算器。这使得开发者可以通过配置文件灵活切换算法。
- **向量化 (Vectorization)**：大量使用 PyTorch 的张量操作，针对 GPU 并行计算进行了高度优化。

---

## 5. 快速代码查阅索引
- **文件开头**：算法注册表与 KL 控制器。
- **200-750 行**：密集的优势估计器实现（GAE, GRPO, RLOO 等）。
- **840-1400 行**：各种策略损失函数（$L_{CLIP}$, $L_{PG}$）。
- **结尾部分**：价值损失、KL 惩罚数学逻辑及数据重采样逻辑。
