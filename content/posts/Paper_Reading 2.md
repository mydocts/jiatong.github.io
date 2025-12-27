---
title: Paper Reading 2
date: 2025-11-15T10:00:00+08:00
draft: false
categories:
  - 论文阅读
tags:
  - 论文阅读
slug: paper2
description: 读的一些论文
katex: true
math: true
---
## Lost In the Middle

我们在这项工作中观察到的 U 形曲线，与心理学中的一个现象有关——**系列位置效应（serial-position effect）**（Ebbinghaus，1913；Murdock Jr，1962）。这一效应指出：在让人们自由回忆一串列表元素的实验中，人类往往**最容易记住列表中的第一个和最后一个元素**。

系列位置效应在研究人类如何形成短期记忆和长期记忆方面具有重要意义。

在语言模型中观察到类似系列位置效应的现象是有些令人惊讶的，因为 Transformer 语言模型底层的**自注意力机制在理论上能够同等地从上下文中的任何位置检索信息**。

## Positional Biases Shift as Inputs Approach Context Window 

{{< figure src="/images/paper_reading/pb_1.png" alt="" width="720" >}}

论文提出了一个关键变量：
#### Relative Input Length (输入长度 ÷ 模型最大 context window)**
$$
L_{rel}=\frac{L_{input}}{L_{max}}$$衡量输入文本长度与上下文窗口大小比值

并首次系统研究：
1. LiM 何时出现
2. 何时消失
3. 消失后会出现什么新现象
4. 检索与推理之间的关系（推理是否依赖检索？）
    
最终得到三个主要发现：

#### 主要发现 1：LiM 在所有模型中都存在（但只在 L_rel ≤ 0.5 时）

大量实验显示：
**当输入占模型 Context Window 的 0%～50% 时：**
- LiM 非常明显
- 开头与结尾准确率高
- 中间显著下降
当输入超过 50% Context Window（L_rel > 0.5）后：**
- **LiM 开始变弱**
- 最终完全消失

这解释了为什么：
- 短输入研究 → 发现 LiM
- 超长 context LLM（100k tokens） → 看不到 LiM

它们研究的其实是不同 $L_{rel}$ 区间。

#### 主要发现 2：Primacy 急剧下降导致 LiM 消失

当输入接近 context window 上限时：

- **Recency bias（尾部信息偏置）仍然稳定甚至增强**
- **Primacy bias（开头的信息）急剧下降**

最终变成：

> **模型基本只擅长处理“接近结尾的位置” → 出现 Distance-based Bias**

 即：

距离结尾越近，表现越好；距离越远越差。

这也是为什么：

> 长文本 LLM 更依赖“把重要内容放在结尾”（如 RAG 系统会将 evidence 放在尾部）

同时

位置准确率排序变成：

`Last > Middle > First`

文章认为这是 **context window 压力导致的记忆衰减模式**，非常类似于 transformer 的 attention 衰减。


---

#### 主要发现 3：推理成功强烈依赖于“检索成功”（Retrieval → Reasoning）

作者首次构造 **Retrieval–Reasoning Minimal Pairs**：

每个 reasoning 问题都有一个对应的 retrieval 问题， retrieval 问题的答案被包含在文中一个明确句子里。

若检索失败 (RT=0)，推理仍正确的概率只有：**35%–52%**
若检索成功 (RT=1)，推理正确率直接升至：**72%–82%**

也就是说：

> **推理的位置信偏置几乎完全继承于检索偏置。  
> 只要检索成功，推理偏置就几乎消失。**

## On the Emergence of Position Bias in Transformers

背景: LIM
动机：分析原因
方法: 图论(graph-theoretic)框架，定量分析，不同掩码／位置编码方案进行了定性              或定量分析，看哪些情况下会出现LIM
结果: 
- 如果模型使用某些掩码模式（比如典型的自回归 Transformer 中的因果掩码／过去不能看未来），那么序列最开头或最结尾的 token 往往具有更强的“接收／聚合”信息能力（因为路径更短、更直接、注意力度更集中）。
    
- 中间位置的 token 由于可能既要“接受”前面很多信息、又要“传递”给后面很多，反而在“影晌力”或“被关注程度”上处于劣势。换句话说，模型在结构上对中段位置天然更弱。
    
- 位置编码和掩码设计是关键：如果设计得当，可以减弱这种偏向。但在很多标准模型设定下，这种偏向“自动”产生。也就是说，并不是训练数据本身唯一原因：模型结构／掩码本身也“内建”了偏向。
    
- 该偏向在实际任务中可能带来问题：例如，当你希望模型关注长序列中的**中段**细节（如法律文档中间页、对话中的某次中间转折）时，这种结构偏向可能会使模型“忘记”中段内容，或者中段内容被弱化。报导中称：“研究已显示大型语言模型在长文档或会话中倾向于强调开头和结尾信息，而忽视中间。”

总结: LIM是由Transformer的因果掩码机制天然造成的，系统误差


## LOST IN THE MIDDLE: AN EMERGENT PROPERTY FROM INFORMATION RETRIEVAL DEMANDS IN LLMS

{{< figure src="/images/paper_reading/LIM_Prop_1.png" alt="" width="720" >}}
作者指出：

他们提出一个新的视角：

> “lost in the middle 是训练时的信息检索任务需求 + 模型架构共同导致的 **适应性行为**，而不是缺陷。”

他们借鉴了心理学中的 human memory：
- Free recall → 前面更容易记
- Running span → 后面更容易记  
- 因此形成 U 型曲线。
#### 论文的结论: 
Lost-in-the-middle arises because：
1. **Long-term retrieval demand → primacy**（靠 attention sinks + causal mask）
2. **Short-term retrieval demand → recency**
3. **Both demands coexist in pre-training → U-shaped behavior**
→ 这不是 flaw，而是 optimal adaptation under constraints

{{< figure src="/images/paper_reading/LIM_Prop_2.png" alt="" width="720" >}}


#### **大模型的表示能力更强**

更强的网络能学会：

- 更好地压缩长距离依赖
    
- 更好地关联中间 token 与 query
    
- 对中间内容的 embedding 表示质量更高
#### **为什么 next-token 预测会自然产生 primacy + recency？**

训练目标是：

> **预测下一个 token（next-token prediction）。**

这个目标天然引入两个偏置：
（1）Recency（短程）偏置来自语言本身
- 下一个词最依赖最近几个词
- 模型学到“最近的信息最重要”
- → 强 Recency（末尾 recall 强）
（2）Primacy（长程）偏置来自模型结构
- 因果 mask + sink 模型强化序列开头
- 模型在 long-context QA 时必须能检索任何远处 token
- 所以开头被当成“长期记忆 anchor”
因此：

> **next-token prediction 自然生成 Primacy + Recency 并最终形成 U 形曲线（lost-in-the-middle）。**

#### **如何通过架构或 sink 干预改变 LITM？**

论文及相关研究告诉我们几种方式能改变 lost-in-the-middle。

(1）弱化或去掉 attention sink

方法包括：

- 给 BOS token 做 dropout（论文实验）
- 修改 embedding 初始化
- 删除 sink heads
- 添加 anti-sink loss（抵消 ingoing attention）

论文发现：

> 去掉 sink → Primacy 消失，但 Recency 保留。

---

（2）改变模型架构（从 autoregressive → bidirectional）

如：

- T5
- BERT
- Mamba（序列建模但无强因果 mask）

结果：

> bidirectional 模型不产生 lost-in-the-middle。

(3) 改变位置编码或引入 RoPE scaling

更弱的位置偏置 → 减弱 Primacy。
(4）改变训练任务（Free Recall vs RS 的比例）

论文中的 combined task 说明：

- 增加 RS → 强调结尾 → 减弱 Primacy
    
- 增加 FR → 强调开头 → 增强 Primacy

（5）使用 memory-augmented 架构**



## Identifying and Evaluating Inactive Heads in Pretrained LLMs

#### 一、核心问题

这篇论文研究了大型语言模型（LLM）中 “注意力头”（attention heads） 的**非活跃**（inactive／dormant）现象。具体来说：
- 在 Transformer 架构中，多头注意力（multi-head attention）通过多个 “头（head）” 并行计算，让模型从不同角度关注输入序列。
- 然而，已有研究发现一些注意力头的行为并不像“真正起作用”那样：例如它们的大部分注意力集中到第一个 token 或者某些“sink”（汇聚点）token 上，而这些 token 在语义上却可能并不重要。即所谓 “attention sinks” 现象。 
- 本文提出：这些表现 “像在但是没用” 的头，可以视为“非活跃注意力头”（dormant attention heads）或“非活跃头”（inactive heads）。研究的目标是：**如何识别这些头？它们确实对模型输出贡献很小吗？**如果是的话，是否可将它们移除或屏蔽，从而减少计算冗余。
#### 主要发现

论文报道了多个有趣结果，以下为摘要：

- 在所测试的模型上（多个预训练 LLM 家族），“非活跃头”占比并非极低：作者发现平均超过 **12% 的注意力头**可以被视为非活跃，而屏蔽它们后，模型在某些任务（如 MMLU）上的准确率仍可维持在原模型水平以内（例如下降不到 1%） 。 
- 值得注意的是，用传统 “First Token” 方法（只看第一个 token 注意力聚焦）去识别非活跃头，会**低估**其比例：说白了，仅靠“看它把注意力投给第一个 token”这种指标，漏检很多非活跃头。作者指出，这种方法平均会漏掉 7% 以上的头。 [
- 在不同模型规模之间、或者预训练 vs 微调之间，注意力头行为有显著差异：例如微调 (finetuning) 对注意力头是否活跃的改变很小，说明很多头的“非活跃状态”在预训练阶段就已形成。
- 输入文本特征也会影响头是否被判为非活跃：换句话说，一个头可能在某些输入上活跃、在其他输入上非活跃，状态有依赖于具体输入。

**不同的情况，注意力头的活跃程度不同（对指定问题的解答能力不同）**



## MOTIF: Modular Thinking via Reinforcement Fine‑tuning in LLMs

因为 LLM 有 **context 长度限制**，一次性写太长的思维会丢失注意力。  
论文目的就是：  
**通过多轮，让模型相当于“超长思考，但分段执行”。**
{{< figure src="/images/paper_reading/GRPO_1.png" alt="" width="720" >}}
{{< figure src="/images/paper_reading/GRPO_2.png" alt="" width="720" >}}
{{< figure src="/images/paper_reading/GRPO_3.png" alt="" width="720" >}}



## Table

| Paper                                                                               | Motivation  | Method               | Task                         | Datasets                            | Models                                                                                  | Metrics         | Publication Date | Baselines | Detailed Methds | Others |
| :---------------------------------------------------------------------------------- | :---------- | :------------------- | :--------------------------- | :---------------------------------- | :-------------------------------------------------------------------------------------- | :-------------- | :--------------- | :-------- | --------------- | ------ |
| Positional Biases Shift as Inputs Approach Context Window Limits                    | LIM并未总是出现   | 引入$L_{rel}$          | Non  Reasoning and Reasoning | MonoRel, PIR, RuleTaker, BoxTracker | Llama-3.1-70B, Llama-3.3-70B, Llama-3-70B,Mistral-Small-24B,  Qwen-2.5-32B, Gemma-2-27B | Acc,            | 2025.7           |           |                 |        |
| LOST IN THE MIDDLE: AN EMERGENT PROPERTY FROM INFORMATION RETRIEVAL DEMANDS IN LLMS | 并非缺陷，而是一种适应 |                      | Non Reasoning                | 人工构造                                | GPT-2 Small,GPT-2 large,Llama 3.2B-1B; RNN(autoreg), T5 (双向)                            | Recall          | 2025.10          |           |                 |        |
| On the Emergence of Position Bias in  Transformers                                  | Why LIM     | 图论(grah-theoretic)框架 | Non Reasoning                | 自己合成的可控数据集                          | 简化的自注意力网络+MLP                                                                           | 自定义：准确率差距       | 2025.2           |           |                 |        |
| MOTIF                                                                               | 推理长度受上下文限制  | 改进GRPO               | Reasoning                    | GSM8K, MATH500, AIME2024            | Qwen2.5‑3B‑Instruct                                                                     | pass@1 accuracy | 2025.7           |           |                 |        |
