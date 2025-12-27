---
title: HSTU
date: 2025-10-24
description: 基于万亿参数序列转换器的工业界生成式推荐系统
draft: false
categories:
  - 论文阅读
tags:
  - 生成式推荐系统
slug: hstu
katex: true
math: true
---
## DLRM（深度学习推荐系统）：
一个经典的，标准的，符合工业界趋势的深度学习推荐系统模型应该如何构建？

* Bottom MLP：处理连续数值特征。
* EmbeddingLookup：将离散类别特征映射到向量空间。
* Feature Interaction Layer: 融合 dense 与 sparse 特征。
* Top MLP：进行最终预测（如点击率）。

DLRM 成为推荐系统工业界（如 Meta、Tencent、ByteDance）广泛采用的标准架构之一，因为它兼顾：
- 高效的稀疏特征处理能力；
- 强大的特征交互建模能力；
- 在分布式训练与推理环境下的可扩展性。

{{< figure src="/images/hstu/dlrm_1.png" alt="" width="720" >}}

过去的十年间，在DLRM框架内不断地迭代、发展新模型
- Feature interactions (FMs, DCN, Autolnt, DHEN/Wukong, MaskNet, ...)
- Multi-task learning (MMoE, ESMM, PLE, ...)
- Sequential (sub-)modules (one-stage DIN, BST, hybrid UBM, SIM, ...)
- Debiasing (off-policy correction / REINFORCE, IPW / CLRec, ...)
- Beyond two-tower settings (multi-interest / MIND, beam search / "generative retrieval" / TDM, OTM, DR, learned similarities / MoL, ...) .
## But 生成式模型的出现打破了传统的模型发展思路…

Many explored use cases in RecSys:

* In-context Learning (e.g., LLMRank, ...)
- Instruction Tuning (e.g., M6-Rec, TALLRec, ...)
- Transfer Learning utilizing World Knowledge (e.g., NoteLLM, ...) .

**现有问题和挑战**

* 推荐系统中的特征缺乏明确的结构。
* 十亿级词汇动态系统 vs 100K级自然语言静态系统
* 计算成本是实现大规模序列模型的主要”卡脖子”问题。
	* GR-24已经直追LLaMa-2的运算规模。

## DLRMs + Generative Models: How do we get the best of both worlds?
Classical recommendation models - DLRMs - vs LLMs
Pros of LLMs

- Replace feature engineering, to the extent capable by language;
- World knowledge benefits cold-start scenarios;
- Scale with compute.

Pros of DLRMs

- Leverage vast number of human-engineered features;
- Concise representations — efficient and support very long context sizes;
- Scale with (in-domain recommendation) data.

解决方案一重新构建工业化的推荐系统

#### - 将用户行为视为首要模态

将点击/跳过/停留/搜索等动作记为token;

与图像/视频/文本/元数据融合而不丢失信息。
#### - 规范化特征空间与编码

在sequence-to-sequence这种序列化视图下统一检索与排序
#### - 消除架构中一些不好掌控的冗余

#### -工业级就绪性与可扩展性

* 随数据与计算资源扩展；

* 对动态词汇表[持续新增项]及长历史数据具备泛化能力和鲁邦性。
### 本文的核心贡献

- 统一的生成式推荐器 [GR]。部署在核心产品上；一个序列模型同时完成召回和排序，取代了长期以来依赖大量特征的推荐管线。
- HSTU 编码器。针对高基数、非平稳数据流的新架构；在质量上超过 Transformer，并且在序列长度 L = 8192 时训练速度比 FlashAttention2 快 15.2 倍。
- M-FALCON 算法。在成千上万候选项之间共享计算 [约 700 倍摊销]，即使模型大了 285 倍，也能实现约 2.48 倍的吞吐率提升。
- 公开数据集实验。在 MovieLens / Amazon Reviews 上，相比强序列模型基线（如 SASRec），NDCG@10 指标提升约 20\% - 66\%
- 线上A/B测试。端到端〔召回+排序〕提升最高 +18.6% [排序 +12.4%, 召回 +6.2%]。
- 大规模训练。训练了 1.5 万亿参数 [Trillion-Parameter] 的 GR；年计算量提升约 1000 倍，可与 GPT-3/LLaMA-2 相媲美；首次在推荐系统中观察到类似 LLM 的 scaling law。
    

## DLRMs + Generative Models => Generative Recommenders

#### 从DLRM到GR

{{< figure src="/images/hstu/dlrm_2.png" alt="" width="720" >}}

用户和商品的交互序列当作主线

地理位置为辅助序列1，慢变化属性，辅助变量

用户加入的购物社区2，慢变化属性，辅助变量

像点击率这种快速变化,不能真实地反应用户兴趣，不再显示输入，而是让模型自己从历史时序序列中[生成]相同的信息

作者将所有的正负反馈行为组织成序列，这个序列中既包含item_id, user_id,也包含稀疏特征，交互行为类型等，而摒弃了数值型特征，构造了生成式建模所需要的统一输入格式，这也印证了actions speak louder


{{< figure src="/images/hstu/GR_1.png" alt="" width="720" >}}

​
* 简单说：GR里面，排序就是一个预测动作，检索就是预测内容，而这两个都是通过同一个序列模型来完成，在检索中，输入是动作和内容的交互序列
* 检索: 如果动作是正反馈，输出的候选内容加入候选集；如果是负反馈，则输出为空，所以检索的本质是根据用户的历史行为生成新的候选内容
* 排序：输入的是历史的内容和用户的动作序列，最后一位是内容，叫做内容位，模型在内容位预测用户会采取什么动作，比如点击，喜欢等等。所以排序的本质是在已有的候选内容上预测用户的排序分布

{{< figure src="/images/hstu/GR_2.png" alt="" width="720" >}}
{{< figure src="/images/hstu/GR_3.png" alt="" width="720" >}}



{{< figure src="/images/hstu/GR_4.png" alt="" width="720" >}}

监督策略：只对正反馈监督，负反馈标记为空，只保留有意义的交互，减少冗余

GR在同一条交互里序列完成检索，也就是动作位生成内容，再加上排序，也就是内容位生成内容，并且保持最新的动作和内容能够与历史快速交互，GR并不是每次曝光就发送一条样本，而是在会话末或者关键发送一条样本，在样本减少的情况下还能保存丰富的信息量。在传统的DLRM，每曝光一个item，就会生成一个独立的训练样本，这样的问题是样本特别多，而且很多计算是重复，因为用户的历史行为在不同样本会被反复算一遍。GR不会在每次曝光时就立刻生成样本，而是在一个关键时刻，比如一次推荐对话时间结束，把前面的交互序列打包成一个训练样本，这样在减少算力消耗的同时，每个样本的监督信号更密集，更干净， 
同时通过更密集/更精确的每轮监督以及候选模型与历史数据间的目标感知交互，保持甚至提升质量。由公式可以看到，计算量下降了一个量级
$$O(N^3 d + N^2 d^2) t \ \ -> \ \ O(N^{2}d + Nd^{2})$$

## HSTU(Hierarchical Sequential Transduction Unit)

 为了让GR模型在工业界大规模推荐系统中实现高可扩展性处理海量非稳态的词表和数据

DLRM中的三个主要阶段：特征提取、特征交互作用以及表示的转换。
{{< figure src="/images/hstu/hstu_1.png" alt="" width="720" >}}

HSTU每个层包含三个主要的子层：

* pointwise投影层：信息压缩与重构
	* Pointwise Projection Layer 在传统的 Q, K, V 投影之外引入了额外的 U 投影，用于建模用户长期历史（long-term user representation）。
	* 它将用户的长序列行为压缩成较低维的语义表示，从而在与目标（target item）交互时，能更有效地筛选和增强关键信息。
* pointwise空间聚合层:  局部信息聚合
	* Pointwise Aggregation Layer 使用“聚合注意力（aggregation attention）”取代传统 softmax 注意力，不再进行概率归一化，从而避免了 softmax 的稀释效应。
	* 这种改进能更充分地保留输入信息，使模型在面对动态变化或长尾分布的行为词表时更加稳定。
* pointwise转换层: 非线性变换与偏置调整
	* Pointwise Transformation Layer 在非线性变换（如 ReLU/GELU）中引入了偏置项 $r_{ab}$，以对 token 之间的关系进行细粒度建模。
	* 该层增强了模型的表达灵活性，使其能更好地捕捉行为序列中的复杂交互模式。


# 工程优化
## 稀疏性优化

#### 高效注意力内核 [GPU内核]
将注意力机制的计算重构为一组分组矩阵乘法（Grouped GEMM），并通过融合内核（Fused Kernel）实现一次 GPU 调用完成所有步骤（QKᵀ、Softmax、乘 V），从而减少显存访问和内核启动开销，使模型在 GPU 上的计算吞吐量提升约 2–5 倍。

#### 算法增强稀疏性：随机长度 [SL]

用户行为在多时间尺度上呈现重复性

通过从完整历史记录中提取子序列进行训练，人工增强数据稀疏性
​
>[!NOTE]内存是工业级scaling非常重要的瓶颈!
#### 精简层级与融合操作：

HSTU将非注意力线性层从6层削减至2层

使每层内存消耗降至约14d [bf16]

该设计可构建深度提升2倍以上的模型。

#### 嵌入/优化器内存：

针对十亿级词汇表，采用行向AdamW算法，使嵌入参数的
HBM占用从约12B缩减至约2B。
* 对于十亿级 embedding 表，普通 AdamW 会因维护元素级动量状态导致显存占用极高。
* 采用“**行向 AdamW**”后，将状态按行聚合，大幅压缩优化器状态存储，使嵌入参数在 GPU 高速显存（HBM）中的占用从约 12B 降至 2B，提高了训练可扩展性和内存效率
## 并行优化

#### 通过成本摊销实现推理扩展

在排序阶段，我们面临数万个候选项目。作者提出名为M-FALCON[微批量快速注意力缓存操作]的算法，可在输入序列长度为n时，对m个候选项执行单次推理。
* M-FALCON 的共享计算原理：通过将用户历史序列的中间特征（K,V）缓存起来，让所有候选 item 复用这份表示，仅计算目标 item 的交叉注意力，并采用微批并行方式统一处理，从而实现“计算一次，多次使用”的推理加速。

他们成功部署了复杂度提升285倍的目标感知交叉注意力模型，在相同推理计算能力下实现了1.5倍的吞吐量提升。
#### Microbatched-Fast Attention Leveraging Cachable Operations
{{< figure src="/images/hstu/hstu_2.png" alt="" width="720" >}}

## 数据集

MovieLens

该非商业性数据集由明尼苏达大学计算机科学与工程学院的GroupLens项目团队创建，旨在支持研究工作。数据集包含多个电影评分数据子集，已成为推荐系统领域的经典数据集。

Amazon Book Reviews

亚马逊提供的产品数据集包含商品评论及元数据，涵盖1996年5月至2014年7月期间的1.428亿条评论。

{{< figure src="/images/hstu/hstu_3.png" alt="" width="720" >}}
{{< figure src="/images/hstu/hstu_4.png" alt="" width="720" >}}
{{< figure src="/images/hstu/hstu_5.png" alt="" width="720" >}}


## 工程优化的有效性

* 随机序列长度[SL]的有效性：长度为4096的序列可以减少80%的token。即使 $\alpha$ 变大，可有效减少的幅度也不会下降很多。同时NE指标并没有因此变差，变差幅度不超过0.2%

* 相比FlashAttention优化的 Transformers，HSTU在训练和推理阶段效率能分别提升15.2x、5.6x倍。

* 此外，由于内存的各种优化和节省，相比于Transformers，HSTU可以再叠深2层网络。



{{< figure src="/images/hstu/hstu_6.png" alt="" width="720" >}}

>[!TIP]NE指标下降明显，意味着线上指标效果提升明显
## Generative Recommenders vs DLRMs

* 传统DLRM里用了大量的特征，如果把DLRM的特征做消融，仅仅保留GR里用到的那些，那么模型效果会大打折扣，看离线指标降了很多。一方面说明传统DLRM没有特征不行，另一方面表明了新架构的优势。

* 在GR中只考虑item的属性(content-based)，召回效果奇差，比baseline也低不少，说明用户行为中蕴含的高基数、高阶信息建模的重要性。

* 即使模型复杂度比基线高出285倍，当候选序列为1024时，GR仍能实现1.5倍吞吐量；选序列为16384时，吞吐量提升至2.48倍。

* 在大型工业环境中，DLRM模型在特定计算和参数配置条件下会达到质量饱和点。
	而GRs体现出来了更强劲的、在大模型领域出现过的scaling能力。

* 推荐系统中的幂律扩展趋势则无法显现算力越大，效果越好这种趋势


**总结：**
作者提出一种统一的生成式推荐模型 [GR]，以实践证明“行动胜于言辞”。性能表现：在传统公开测试集和行业真实流数据集上均取得显著提升，其NDCG@10指标较经典SASRec模型提升 20.3\% - 65.8\% 。相较于经多次迭代优化的DLRMs基准模型，该模型在线召回率与精确度实现 18.6\% 的相对提升。首次在核心产品线中取代传统深度推荐模型——而这些基于海量异构特征的模型曾主导推荐领域近十年。


**长远的意义**
* 通过降低推荐、搜索和广告系统对海量异构特征的依赖，这些系统既能提升用户体验，又能更注重隐私保护。 
* 传统推荐系统常基于用户短期行为和偏好生成推荐，这可能导致用户接触到与其长期目标不符的内容。
* 该方法还使平台激励机制与用户价值更趋一致，能够真正识别用户潜在需求而非平台推广导向。