---
title: Paper Reading
date: 2025-11-11T10:00:00+08:00
draft: false
categories:
  - 论文阅读
tags:
  - 论文阅读
slug: paper
description: 读的一些论文
katex: true
math: true
---
#### 基于历史信息生成时，对不同位置信息的关注不同 
* Lost In the Middle
#### 注意力层面：校准/差分/选择性注意，削弱“首尾优势”；  
 * Found In the Middle
 * DIFF
#### **训练数据**：位置无关/信息密集型合成任务，让模型学会“任何位置都可能关键”；
* Never Lost in the Model
* Fill in the Model
#### **提示工程/预处理**：压缩与**重排**把关键信息放到模型更易注意的位置，或在对话中**状态化总结与提醒**

* LongLLMLingua
## Lost In the Middle

目标信息的位置会显著影响模型推理的表现，当目标信息在上下文的开头和结尾时，模型表现好；在中间位置时，模型表现较差。

{{< figure src="/images/paper_reading/LIM_1_1.png" style="height:100px; width:auto;" >}}

微调是否对这个现象有影响？-并没有

{{< figure src="/images/paper_reading/LIM_3.png" style="height:100px; width:auto;" >}}

越多上下文越好？ 并非，取决于下游任务,本文的实验就早早饱和


## Found in the Middle
三件事
* 发现了造成lost-in-the-middle problem的原因
* 引入注意力矫正机制: found-in-the-middle
* 这个机制同时提升RAG能力
{{< figure src="/images/paper_reading/FITM_1.png" alt="" width="560" >}}

之前方法：
重排文档的相关性——需要额外监督信号或专门微调
且未从根本上解决

全文假设:
位置注意力偏差可能导致模型过度依赖输入首尾内容（无论其实际相关性如何），从而引发该现象。

理论上来说注意力机制应该对Gold doc的注意力更高，但是实际上对开头和结尾的注意力更高，所以提出了这个假设。本文通过引入注意力矫正机制（抑制对首尾的注意力），来进行实验，如果矫正后消除了偏差，得到了期望的答案，就说明两个事情
1.假设成立
2.让注意力不再依赖位置，而只取决于真实相关性。

建模注意力
简化建模,rel()代表prompt和第k个文档的相关性，bias(k)代表第k个位置的偏差

$$ \operatorname {A t t n} \left(x ^ {\text {d o c}}, k\right) = \operatorname {r e l} \left(x ^ {\text {d o c}}\right) + \operatorname {b i a s} (k) + \epsilon \tag {2} $$

引入一个“虚拟文档$x_{dum}$​ 以获得仅含位置偏差的基线注意力，然后通过相减去除偏差：

$$Calibrated Attention=Attn(x_{doc},k)−Attn(x_{dum},k)$$
再进行归一化
$$ \operatorname {a t t n} _ {\text {calibrated}} \left(x _ {k, i} ^ {\text {d o c}}\right) = \tag {5}
 \frac {\alpha_ {k}}{\mathrm {A t t n} _ {\mathrm {o r i g i n a l}} (x _ {k} ^ {\mathrm {d o c}})} \cdot \mathrm {a t t n} _ {\mathrm {o r i g i n a l}} (x _ {k, i} ^ {\mathrm {d o c}}) \cdot C, $$
## Never Lost in the Middle
* 背景：Lost In the Middle
* 方法:  作者提出一个特别的训练任务 **PAM QA（Position-Agnostic Multi-step QA）**  
		让模型学会在长文档里**精确定位目标信息**，无论它在开头、中间还是结尾。
* 效果:  非常惊人：模型在“中间有答案”的测试中几乎不掉分，甚至能做到 **中间、尾部都不再迷路**。

{{< figure src="/images/paper_reading/NLIM.png" alt="" width="720" >}}


## Fill in the Model
Make Your LLM Fully Utilize the Context

背景： Lost In the Middle
方法：IN2训练 （Information-Intensive Training）

	用一句人话总结：
	让模型在训练时反复做“从长文本的随机位置取关键信息”的任务。  让它知道：重要信息什么位置都有
{{< figure src="/images/paper_reading/FITM_1.png" alt="" width="720" >}}

{{< figure src="/images/paper_reading/FITM_2.png" alt="" width="720" >}}
#### ① **微粒度信息感知**（Fine-grained Information Awareness）
做法：
1. 从文章中 **随机裁剪一个 128-token 的小段**（例如一小段描述、一句话等）
2. 用 GPT-4 生成一个问题，其答案只在这个小段里
3. 把这个小段和很多其他随机无关的段落混在一起，组成非常长的文本（4K–32K token）
4. 让模型在这个长文本里找答案
通俗理解：
> **在 3 万字的文章里找“那一句话”。**
这训练模型：  
✔ 任何位置都可能有关键句  
✔ 必须在噪音中找信号

#### ② **跨段落推理（多跳问题）**（Integration & Reasoning）
步骤类似，但不同的是：

- 不是一个小段，而是 **两个或更多小段**
- 问题需要结合多个段落的信息来回答
    
举例（类比）：

> “段落 A 说张三出生地，段落 C 说他的现居地，请问他从哪里搬到哪里？”  
> → 答案需要读两个不同位置的段落。

目的是训练模型实现真正的“跨长文推理能力”。
#### ③ **合成长文本的方法**

长文本长度从 **4K → 32K** 平均分布  
目的是让模型习惯不同长度的上下文。

还保留了 **10% 的短文本任务**，以防它忘记短文本能力。

## LongLLMLingua
背景：
Long context scenarios下，LLM有higher computational cost, performance reduction, and position bias 的问题
方法：
{{< figure src="/images/paper_reading/LongLLM.png" alt="" width="720" >}}

## Differential Transformer
{{< figure src="/images/paper_reading/DT.png" alt="" width="720" >}}

## Disentangling Memory and Reasoning Ability in Large Language Models

背景: 推理阶段中没有明确区分memory和reasoning，会导致knowledge forgetting
方法：加入special token进行训练
实验：
* Shuffle Token Ablation：随机打乱 token（检验 token 的必要性)——效果下降
* Token 数量 Ablation：**4**/6/8 个 Token 哪个最好
* Prompt Ablation：去掉 prompt 中对 memory/reason 的要求 ——效果下降
{{< figure src="/images/paper_reading/special.png" alt="" width="720" >}}


## Table

| Paper                                                       | Motivation                             | Method                                    | Task | Experiment                                            | Datasets                                                                                            | Models                                                                                              | Metrics                                                                    | Publication Date | Baselines                       | Detailed Methds | Others |
| :---------------------------------------------------------- | :------------------------------------- | :---------------------------------------- | :--- | ----------------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------- | :--------------- | :------------------------------ | --------------- | ------ |
| Lost In the Middle                                          | How well LM use longer text            |                                           |      | Multi-document question answering; Key Value Retrival | NaturalQuestions-Open+干扰文档(用 Contriever 检索系统（基于 MS-MARCO 微调))获取；人工合成数据集                             | MPT-30B-Instruct;GPT-3.5-Turbo, GPT-3.5-Turbo(16K),Claude-1.3 and Claude-1.3(100k);LongChat-13B(6k) | Acc（as taken from annotations appear in the predicted output）              | 2023             |                                 |                 |        |
| Found In the Middle                                         | Improve the lost-in-the-middle problem | Calibrate Attention                       |      |                                                       | NaturalQuestion；SynthWiki                                                                           | Vicuna-7b-v1.5-16k，Tulu‑2‑7Bss                                                                      | Recall                                                                     | 2024             |                                 |                 |        |
| Disentangling Memory and Reasoning Ability in LLM           | knowledge forgetting                   | data-generationw with special token;train |      | Ablation Study                                        | StrategyQA,CommonsenseQA ,TruthfulQA.                                                               | LLaMA,Qwen,GPT-4o                                                                                   | Acc                                                                        | 2025             | zero shot,CoT,Lora, Lora+prompt |                 |        |
| Never Lost In the Middle                                    | Improve the lost-in-the-middle problem | PAM-QA                                    |      |                                                       | DuReader2.0,WebCPM                                                                                  | Baichuan,longchat, Improve the lost-in-the-middle problem                                           | Acc                                                                        | 2024             |                                 |                 |        |
| Make Your LLM Fully Utilize the Context(Fill in the Middle) | Improve the lost-in-the-middle problem | IN2 (Information-Intensive Training）      |      |                                                       | NarrativeQA,GovReport....                                                                           | Mistral, Llama,  GPT-4 Turbo, Claude, Gemini                                                        | Acc, F1 score,                                                             | 2024             |                                 |                 |        |
| LongLLMLingua                                               |                                        | Four Methods                              |      |                                                       | NarrativeQA, HotpotQA, TriviaQA, PassageRetrieval-en, Needle-in-a-Haystack (RULER), LooGLE, MuSiQue | LLama,Mistral                                                                                       | Perplexity, Exact Math,F1 score,Recall, Acc,Token Reduction, Latency, Cost | 2024             |                                 |                 |        |
| DIFF                                                        |                                        | subtract noise                            |      |                                                       | StableLM-3B-4E1T, Book corpus,Needle-In-A-Haystack, Multi-Needle Retrieval....                      | DIFF                                                                                                | LM- loss, Acc,activation outliers                                          | 2025             |                                 |                 |        |