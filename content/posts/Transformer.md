---
title: Transformer
date: 2025-09-17
description: 一切的开始
draft: false
categories:
  - 博客
tags:
  - 基础知识
slug: transformer
katex: true
math: true
---
# Transformer
{{< figure src="/images/transformer/architecture.png" alt="" width="720" >}}

## Embedding
下面的示例实现一个最简词嵌入层，附带缩放确保数值稳定。

```python
import torch  # 引入 PyTorch 用于张量运算
import torch.nn as nn  # 导入神经网络模块方便搭建组件
class SimpleEmbedding(nn.Module):  # 定义词嵌入层类
    def __init__(self, vocab_size: int, d_model: int):  # 接收词表大小和向量维度
        super().__init__()  # 初始化父类确保模块注册
        self.embed = nn.Embedding(vocab_size, d_model)  # 创建可学习的词嵌入矩阵
        self.scale = d_model ** 0.5  # 保存缩放系数防止数值过小
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # 定义前向传播
        return self.embed(tokens) * self.scale  # 查表并放大嵌入向量
```

## Muti_Head Attention
下面的类展示如何把输入拆成多头并执行自注意力。

```python
import torch  # 引入 PyTorch 支撑矩阵运算
import torch.nn as nn  # 导入 nn 模块方便线性层创建
class SimpleMultiHeadAttention(nn.Module):  # 定义多头注意力层
    def __init__(self, d_model: int, num_heads: int):  # 接收特征维度和头数
        super().__init__()  # 调用父类初始化模块
        assert d_model % num_heads == 0  # 断言能均分每个头的维度
        self.d_model = d_model  # 保存特征总维度
        self.num_heads = num_heads  # 保存头数
        self.head_dim = d_model // num_heads  # 计算每个头的维度
        self.q_proj = nn.Linear(d_model, d_model)  # 定义查询映射层
        self.k_proj = nn.Linear(d_model, d_model)  # 定义键映射层
        self.v_proj = nn.Linear(d_model, d_model)  # 定义值映射层
        self.o_proj = nn.Linear(d_model, d_model)  # 定义输出映射层
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向函数执行自注意力
        batch_size, seq_len, _ = x.shape  # 读取批大小和序列长度
        q = self.q_proj(x)  # 线性映射生成查询
        k = self.k_proj(x)  # 线性映射生成键
        v = self.v_proj(x)  # 线性映射生成值
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # 重塑查询形状并交换维度
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # 重塑键形状并交换维度
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # 重塑值形状并交换维度
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # 计算缩放点积注意力分数
        weights = torch.softmax(scores, dim=-1)  # 对最后一维做 softmax 获得注意力权重
        context = torch.matmul(weights, v)  # 加权求和值向量
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # 合并头并恢复原尺寸
        return self.o_proj(context)  # 输出线性层映射后的结果
```

## Encoder
这里的编码器层由自注意力和前馈网络组成。

```python
import torch  # 引入 PyTorch 支撑张量操作
import torch.nn as nn  # 导入 nn 模块构建层
class SimpleEncoderLayer(nn.Module):  # 定义基本的 Transformer 编码器层
    def __init__(self, d_model: int, num_heads: int, d_ff: int):  # 接收核心超参数
        super().__init__()  # 初始化父类
        self.self_attn = SimpleMultiHeadAttention(d_model, num_heads)  # 自注意力子层
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))  # 前馈网络子层
        self.norm1 = nn.LayerNorm(d_model)  # 第一层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二层归一化
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义编码器层前向传播
        attn_out = self.self_attn(self.norm1(x))  # 对归一化后的输入做自注意力
        x = x + attn_out  # 残差连接叠加注意力输出
        ffn_out = self.ffn(self.norm2(x))  # 对归一化后的结果做前馈网络
        return x + ffn_out  # 残差连接返回最终输出
```

## Decoder
下面的解码器层包含自注意力和交叉注意力两个部分。

```python
import torch  # 引入 PyTorch 以执行张量计算
import torch.nn as nn  # 导入 nn 模块搭建子层
class SimpleDecoderLayer(nn.Module):  # 定义基本的 Transformer 解码器层
    def __init__(self, d_model: int, num_heads: int, d_ff: int):  # 初始化解码器参数
        super().__init__()  # 初始化父类模块
        self.self_attn = SimpleMultiHeadAttention(d_model, num_heads)  # 自注意力处理目标序列
        self.cross_attn = SimpleCrossAttention(d_model, num_heads)  # 交叉注意力融合编码器信息
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))  # 前馈网络细化特征
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力前归一化
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力前归一化
        self.norm3 = nn.LayerNorm(d_model)  # 前馈网络前归一化
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:  # 前向传播接收目标输入与编码器输出
        attn_out = self.self_attn(self.norm1(x))  # 对归一化后的目标序列做自注意力
        x = x + attn_out  # 残差连接保留原信息
        cross_out = self.cross_attn(self.norm2(x), memory)  # 使用编码器记忆执行交叉注意力
        x = x + cross_out  # 残差连接融合上下文
        ffn_out = self.ffn(self.norm3(x))  # 通过前馈网络细化结果
        return x + ffn_out  # 残差输出最终表示
```

## Layer Norm
下例展示 LayerNorm 的手工实现，突出按特征维度标准化的思想。

```python
import torch  # 引入 PyTorch 支撑自动微分
import torch.nn as nn  # 导入 nn 模块定义模块接口
class SimpleLayerNorm(nn.Module):  # 定义简单的层归一化
    def __init__(self, features: int, eps: float = 1e-5):  # 初始化特征数量和稳定常数
        super().__init__()  # 初始化父类
        self.gamma = nn.Parameter(torch.ones(features))  # 可学习的缩放参数
        self.beta = nn.Parameter(torch.zeros(features))  # 可学习的偏移参数
        self.eps = eps  # 保存防止除零的常数
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 执行层归一化
        mean = x.mean(dim=-1, keepdim=True)  # 计算最后一维的均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算最后一维的方差
        normalized = (x - mean) / torch.sqrt(var + self.eps)  # 对输入做标准化
        return self.gamma * normalized + self.beta  # 应用缩放和平移得到结果
```

## Cross Attention
最后给出一个简化版的交叉注意力实现，用于解码器读取编码器输出。

```python
import torch  # 引入 PyTorch 获得矩阵运算能力
import torch.nn as nn  # 导入 nn 模块构建线性层
class SimpleCrossAttention(nn.Module):  # 定义交叉注意力层
    def __init__(self, d_model: int, num_heads: int):  # 接收输入维度和头数
        super().__init__()  # 初始化父类
        assert d_model % num_heads == 0  # 确保可以平均分配维度
        self.d_model = d_model  # 保存特征维度
        self.num_heads = num_heads  # 保存注意力头数
        self.head_dim = d_model // num_heads  # 计算单头维度
        self.q_proj = nn.Linear(d_model, d_model)  # 定义查询投影层
        self.k_proj = nn.Linear(d_model, d_model)  # 定义键投影层
        self.v_proj = nn.Linear(d_model, d_model)  # 定义值投影层
        self.o_proj = nn.Linear(d_model, d_model)  # 定义输出投影层
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:  # 执行交叉注意力
        batch_size, tgt_len, _ = query.shape  # 获取目标序列形状
        src_len = context.shape[1]  # 获取源序列长度
        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # 构建多头查询
        k = self.k_proj(context).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # 构建多头键
        v = self.v_proj(context).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # 构建多头值
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # 计算缩放注意力得分
        weights = torch.softmax(scores, dim=-1)  # 对源位置求权重
        context_out = torch.matmul(weights, v)  # 根据权重汇总值向量
        context_out = context_out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)  # 合并各头结果
        return self.o_proj(context_out)  # 线性变换输出最终表示
```
