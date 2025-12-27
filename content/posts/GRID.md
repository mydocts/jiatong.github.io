---
title: GRID
date: 2025-10-01T10:00:00+08:00
draft: false
categories:
  - 论文阅读
tags:
  - 生成式推荐系统
slug: grid
description: GRID —— 一个易于使用、灵活高效的开源平台，用于快速原型化基于 SID 的 GR 方法。
katex: true
math: true
---
## 1.前言

GR 利用生成模型的进步，实现两种主要方式：
* 直接生成用户感兴趣物品的文本内容 
*  **或者从预训练模型中提取语义表示（semantic representations），以编码开放世界的知识**，本文中的模型是后者。 

## 2.语义ID（Semantic IDs）

#### 2.1语义ID是什么

* 促成 **GR（Generative Recommender）** 成功的关键因素之一是 **语义 ID（SID, Semantic ID）**。它将连续的语义表示（例如来自大型语言模型的向量表示）**转换为离散的 ID 序列**。

#### 2.2传统ID与语义ID对比
* 传统的ID特征是为用户/物品分配独一无二，但不提供信息的ID，这些ID被映射到嵌入向量中，主要用于捕获协同过滤信号
* 语义ID将连续的语义表示(例如，来自大模型语言)转换为离散的ID序列；这样能够同时利用预训练基础模型中编码的语义知识以及用户-物品交互历史中的编码的协同信号。
* 当两个物品的 SID 有重叠时，这种重叠在原理上反映了它们的**语义相似性**；同时，下一步的监督学习（next-item supervision）使模型能够学习**跨 SID 的协同过滤信号**
#### 2.3怎么得到语义ID
* 首先通过模态编码器（modality encoder，例如大型语言模型 LLMs）提取语义表示；
* 然后使用量化器（quantizer）将连续嵌入压缩为离散稀疏 ID。
* 常见的基于量化的标记器包括：
- **RQ-VAE** ，**RVQ** ，**Residual K-Means** 。
#### 2.4如何将语义ID用于生成式推荐
TIGER 首次将 Transformer 应用于推荐任务，
用于预测物品的语义 ID（SID），并借鉴了文档检索（document retrieval）中生成式推荐的思想 。后续研究在多个方向上改进了 SID 的训练：

- 通过协同过滤信号增强学习 ；
- 引入分布式平衡机制（distributional balancing）；
- 使用更强大的大型语言模型或多模态编码器（multimodal encoders）。

## 3.GRID架构
我们考虑一个用户集合 $\mathcal{U}$，每个用户与一个物品集合 $\mathcal{I}$ 中的项目交互。每个物品 $i \in \mathcal{I}$ 都有相关的语义特征 $f_i$，包括但不限于文本和图像。  
每个用户 $u \in \mathcal{U}$ 都有一个交互序列，其长度为 $L_u$，记作 $S_u = [i_u^1, i_u^2, \dots, i_u^{L_u}]$。  
一般性地，模态编码器（modality encoder，例如 LLM 或 VLM）将特征 $f_i$ 转换为 $d$ 维的表示 $h_i \in \mathbb{R}^d$。  
生成式推荐（GR）的目标是解决序列推荐问题：给定一个用户的交互序列 $S_u$，生成一个候选物品集合，使得这些物品是用户下一步最可能交互的内容（即 $i_u^{L_u+1}$）。

{{< figure src="/images/GRID.png" alt="" width="720" >}}

####  3.1架构：先标记化（Tokenization）后生成（Generation）**

GRID 将基于语义ID的生成式推荐划分为两个阶段：

1. **标记化阶段（Tokenization）**  
    将物品特征映射成嵌入向量（即 $h_i$），再量化为语义ID（SID）。
    
2. **生成阶段（Generation）**  
    使用所有物品的SID，探索不同的生成模型结构（例如 Transformer-based 模型），以生成下一个物品的SID。  
    GRID 在每个阶段都提供了灵活的可配置实现。
#### 3.2**语义ID标记化（Semantic ID Tokenization）**

SID 标记化首先通过预训练的模态编码器 $E(\cdot)$ 将物品的语义特征映射到嵌入空间，然后通过分层结构将这些嵌入量化为稀疏的语义ID。  
这种层次化组织的SID可以通过不同层级的前缀（prefix）灵活控制粒度。  
形式化地，给定 $h_i$，量化器（Tokenizer）$Tokenizer(\cdot): \mathbb{R}^d \to {0, 1, \dots, V}^L$ 将嵌入 $h_i$ 映射为ID序列：  
$SID_i=Tokenizer(h_i)=[SID_i^1,SID_i^2,…,SID_i^L]$
其中 $V$ 表示每层的词汇大小，$L$ 表示层数。

GRID 提供了可插拔的模块化设计，方便用户自定义模型或直接使用 HuggingFace 上的模型。  
目前支持三种主要量化器：
- **Residual Mini-Batch K-Means (RK-Means)** 
- **Residual Vector Quantization (R-VQ)** 
- **Residual Quantized Variational Autoencoder (RQ-VAE)** 
#### **3.3下一项生成（Next Item Generation）**

当所有物品都生成了SID后，针对每个用户序列，GR 框架会使用序列模型生成用户最可能交互的候选SID。  
在 GRID 中，支持多种生成结构：
- **Encoder-Decoder** 模型；
- **Decoder-only（仅解码器）** 模型；
- 可灵活配置的 Transformer 层数、头数、MoE 专家结构等。
用户可以导入 HuggingFace 上公开的架构或自定义自己的结构。  
默认训练目标是基于 **下一token预测（Next-token prediction）** 的生成任务，并采用 **滑动窗口增强（sliding window augmentation）。

推理阶段使用带 KV-cache 的 beam search，支持调节 beam width，确保生成的SID是合法的。  
GRID 还内置了多种技巧，比如：
- **用户token机制（user token）**；
- **去重（de-duplication）**；
- **避免SID冲突**。

## **4.GRID 实验结果总结**

#### **总体设置**
- 数据集：Amazon 5-core（Beauty, Sports, Toys）
- 评估方式：最后一个交互用于测试，倒数第二个用于验证，其余用于训练。
- 模型：Flan-T5-Large / XL / XXL 提取语义嵌入。
- 标记化算法：RK-Means、R-VQ、RQ-VAE。
- 生成模型：采用 Transformer 架构（共8层，encoder 4层 + decoder 4层）。
- 评估指标：Recall@K 与 NDCG@K（K = 5, 10）。
#### **4.1 语义ID标记化（Semantic ID Tokenization）**
#### **标记化算法（Tokenizer Algorithm）**
**比较 RQ-VAE、RK-Means、R-VQ 三者。**
	**结果：**
	-**RQ-VAE 虽常被采用，但需同时训练自编码器与量化器，复杂度高。
	-**RK-Means 与 R-VQ 表现更优**，即使在计算量更低的情况下也能获得更好推荐结果。
	- 说明复杂的 RQ-VAE 性价比不高。
**语义编码器规模（Semantic Encoder Size）**
- 测试 Flan-T5-Large (780M) → XL (3B) → XXL (11B)。
	- **结果：** 模型参数扩大14倍，性能提升却非常有限。
	- **结论：** 当前 GR with SID 体系**未充分利用更大LLM的语义知识**，性能主要受其它模块影响。
#### **标记器层数与维度（SID Tokenizer Dimension）**
* 调整残差层数 L 与每层token数量 W。默认配置 (L, W) = (3, 256)。
- **结果：**
	* 默认配置效果最佳；
    -  层数过多会降低性能——更多层虽可传递更多语义信息，但会造成SID序列不稳定。
    - **存在“语义信息量 vs. 序列可学习性”的权衡。**

#### **4.2 生成式推荐（Generative Recommendation）**

实验从五个角度研究了 GR with SID 的生成阶段设计。

*  **用户token数量（User Tokens）**
    - 每个用户的SID序列前可加入一个用户token。
    - **结果：**
        - 更大的用户词汇表**并不提升性能**；
        - 完全去掉用户token（即不加个性化标识）反而效果最佳。
        - **结论：** 当前 GR with SID 框架中的用户token设计**未实现个性化目标**。
* **模型架构：Encoder-Decoder vs. Decoder-only**
    - 对比 Transformer 编解码器与仅解码器架构。
    - **结果：**
        - **Decoder-only 明显性能更差。**
        - Encoder-Decoder 模型更能捕捉用户长期行为的上下文信息。
        - **说明 Encoder 层的上下文建模对生成式推荐至关重要。**
*  **数据增强（Data Augmentation）**
    - 使用滑动窗口（sliding window）生成多样化序列样本。
    - **结果：**
        - 适当的数据增强显著提高泛化能力、减少过拟合；
        - 模型能更好预测多样化和稀疏样本中的下一交互项。
        - **结论：** 数据增强是提升 GR 性能的关键手段。
* **SID去重（De-duplication）**
    - 比较两种去重策略：
        1. **TIGER策略：** 在SID末尾加数字；
        2. **随机选择策略：** 当碰撞时随机选一个。
    - **结果：**
        - 两种方式效果接近，TIGER略好但计算代价更高。
        - TIGER方法需全局SID分布信息，不适合大规模数据。
        - **结论：** 简单随机去重在实际场景更高效。
* **束搜索（Beam Search）策略**
    - 对比 **约束式** 与 **非约束式** beam search。
    - **结果：**
        - 性能差距极小；
        - **非约束式搜索更快、计算更省**；
        - 说明SID生成任务本身的模式约束足以保证生成质量，无需显式约束。            

## **5. 结论（Conclusion）**

- **主要发现：**
    1. 过去被认为“关键”的组件（如复杂量化器、RQ-VAE、自定义LLM、大量用户token）其实可以被更简单、高效的设计替代；
    2. 相反，一些被忽视的因素（如 **Encoder-Decoder结构** 与 **数据增强**）却是性能提升的关键；
    3. 这些结果为理解 GR with SID 的真正性能驱动因素提供了新的视角。
- **GRID 的价值：**
    - 通过开源的、统一的实验平台，系统揭示了 GR with SID 的核心设计权衡；
    - **GRID 框架** 能作为标准基准（benchmark）和实验工具，加速后续研究与验证。