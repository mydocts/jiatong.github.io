---
title: "AI Agent Memory Introduction"
date: 2026-01-30
draft: false
description: "9种从入门到高级的AI Agent内存优化技术，附代码与流程图解析"
categories:
  - 博客
tags:
  - AI Agent
  - 内存优化
  - LLM
slug: optimize-ai-agent-memory
katex: true
math: true
---
### 为什么要优化AI Agent的内存？

在对话式AI中，**内存管理**至关重要。这涉及到**历史上下文存储**、**工具调用**、**数据库检索**等多个组件。

本文将介绍**9种从入门到高级的内存优化技术**，每种技术都附有代码实现和学术风格的流程图。

<style>
main.max-w-4xl,
header.max-w-4xl {
  max-width: 100%;
}
.code-formula-container {
  display: flex;
  gap: 2em;
  align-items: flex-start;
}
.code-formula-container .code-side {
  flex: 1;
  min-width: 0;
  overflow-x: auto;
}
.code-formula-container .formula-side {
  flex: 1;
  min-width: 0;
}
.code-formula-container .highlight pre {
  margin: 0;
}
.formula-side img {
  max-width: 100%;
  height: auto;
}
</style>

---

### 策略1: Sequential Memory（顺序存储）

这是最基础的方法：将每条新消息添加到历史记录中，每次都将完整对话反馈给模型。

<details>
<summary>Sequential Memory 实现</summary>

<div class="code-formula-container">
<div class="code-side">

{{< highlight python "linenos=table,linenostart=1" >}}
class SequentialMemory(BaseMemoryStrategy):
    def __init__(self):
        """初始化空的历史记录列表"""
        self.history = []

    def add_message(self, user_input: str, ai_response: str):
        """将新的用户-AI交互添加到历史记录"""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": ai_response})

    def get_context(self, query: str) -> str:
        """获取完整的对话历史作为上下文"""
        return "\n".join([
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in self.history
        ])

    def clear(self):
        """清空对话历史"""
        self.history = []
        print("Sequential memory cleared.")
{{< /highlight >}}

</div>
<div class="formula-side">




![Sequential Memory Flow](/posts/sequential_memory_flow_1769744179925.png)

</div>
</div>

m

</details>

**流程解析：**

1. 用户发起对话 → Agent响应
2. 这一轮交互被保存为文本块
3. 下一轮时，Agent将**全部历史**与新查询合并
4. 整个文本块发送给LLM生成回复

**点评：**

- ✅ **优点**：实现简单，上下文完整
- ❌ **缺点**：Token成本随对话增长线性增加，很快会超出模型上下文窗口

---

### 策略2: Sliding Window（滑动窗口）

只保留最近N轮对话。当新消息到来时，最旧的消息被丢弃，窗口向前滑动。

<details>
<summary>Sliding Window Memory 实现</summary>

<div class="code-formula-container">
<div class="code-side">

{{< highlight python "linenos=table,linenostart=1" >}}
from collections import deque

class SlidingWindowMemory(BaseMemoryStrategy):
    def __init__(self, window_size: int = 4):
        """
        初始化固定大小的deque
        window_size: 保留的对话轮数
        """
        self.history = deque(maxlen=window_size)

    def add_message(self, user_input: str, ai_response: str):
        """添加新轮次，超出限制时自动丢弃最旧的"""
        self.history.append([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ])

    def get_context(self, query: str) -> str:
        """只返回窗口内的对话历史"""
        context_list = []
        for turn in self.history:
            for message in turn:
                context_list.append(
                    f"{message['role'].capitalize()}: {message['content']}"
                )
        return "\n".join(context_list)

    def clear(self):
        self.history.clear()
        print("Sliding window memory cleared.")
{{< /highlight >}}

</div>
<div class="formula-side">




![Sliding Window Memory Flow](/posts/sliding_window_flow_1769744202112.png)

</div>
</div>



</details>

**流程解析：**

1. 定义固定窗口大小，例如 `N = 2` 轮
2. 前两轮填满窗口
3. 第三轮到来时，第一轮被推出
4. 发送给LLM的上下文**只包含窗口内的内容**

**点评：**

- ✅ **优点**：Token成本恒定，可扩展
- ❌ **缺点**：可能丢失重要的早期上下文（如用户名、关键偏好）

---

### 策略3: Summarization（摘要压缩）

定期使用LLM生成对话摘要，而不是直接丢弃旧消息。这样既控制了Token又保留了核心信息。

<details>
<summary>Summarization Memory 实现</summary>

<div class="code-formula-container">
<div class="code-side">

{{< highlight python "linenos=table,linenostart=1" >}}
class SummarizationMemory(BaseMemoryStrategy):
    def __init__(self, summary_threshold: int = 4):
        """
        summary_threshold: 触发摘要的消息数量
        """
        self.running_summary = ""
        self.buffer = []
        self.summary_threshold = summary_threshold

    def add_message(self, user_input: str, ai_response: str):
        self.buffer.append({"role": "user", "content": user_input})
        self.buffer.append({"role": "assistant", "content": ai_response})

    if len(self.buffer) >= self.summary_threshold:
            self._consolidate_memory()

    def _consolidate_memory(self):
        """使用LLM合并摘要"""
        buffer_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in self.buffer
        ])

    summarization_prompt = (
            f"合并以下内容为一个简洁的摘要：\n\n"
            f"### 之前的摘要:\n{self.running_summary}\n\n"
            f"### 新对话:\n{buffer_text}"
        )

    self.running_summary = generate_text(
            "你是摘要专家。", summarization_prompt
        )
        self.buffer = []

    def get_context(self, query: str) -> str:
        buffer_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in self.buffer
        ])
        return f"### 历史摘要:\n{self.running_summary}\n\n### 近期消息:\n{buffer_text}"
{{< /highlight >}}

</div>
<div class="formula-side">


![Summarization Memory Flow](/posts/summarization_flow_1769744224695.png)

</div>
</div>


</details>

**流程解析：**

1. 近期消息存入**Buffer**
2. Buffer达到阈值时触发**摘要合并**
3. LLM将Buffer内容与旧摘要合并生成新摘要
4. 上下文 = 运行中的摘要 + 近期Buffer

**点评：**

- ✅ **优点**：大幅减少Token，保留对话主旨
- ❌ **缺点**：关键细节（如数字、专有名词）可能在摘要过程中丢失

---

### 策略4: Retrieval-Based Memory（RAG检索）

这是工业界最常用的策略。每条消息被转换为向量嵌入存入向量数据库，查询时通过语义相似度检索相关内容。

<details>
<summary>Retrieval Memory 实现</summary>

<div class="code-formula-container">
<div class="code-side">

{{< highlight python "linenos=table,linenostart=1" >}}
import numpy as np
import faiss

class RetrievalMemory(BaseMemoryStrategy):
    def __init__(self, k: int = 2, embedding_dim: int = 3584):
        self.k = k
        self.embedding_dim = embedding_dim
        self.documents = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)

    def add_message(self, user_input: str, ai_response: str):
        docs_to_add = [
            f"User said: {user_input}",
            f"AI responded: {ai_response}"
        ]
        for doc in docs_to_add:
            embedding = generate_embedding(doc)
            if embedding:
                self.documents.append(doc)
                vector = np.array([embedding], dtype='float32')
                self.index.add(vector)

    def get_context(self, query: str) -> str:
        if self.index.ntotal == 0:
            return "No information in memory yet."

    query_embedding = generate_embedding(query)
        query_vector = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_vector, self.k)

    retrieved_docs = [
            self.documents[i] for i in indices[0] if i != -1
        ]

    return "### 检索到的相关信息:\n" + "\n---\n".join(retrieved_docs)
{{< /highlight >}}

</div>
<div class="formula-side">


![Retrieval-Based Memory Flow](/posts/retrieval_rag_flow_1769744246686.png)

</div>
</div>


</details>

**流程解析：**

1. **存储路径**：消息 → 生成Embedding → 存入向量数据库
2. **查询路径**：用户查询 → 生成查询Embedding → 相似度搜索 → 检索Top-K文档
3. 检索到的文档注入上下文 → 发送给LLM

**点评：**

- ✅ **优点**：极低Token成本，支持海量历史，语义级检索
- ❌ **缺点**：实现复杂，依赖Embedding模型质量

---

### 策略5: Memory-Augmented Transformers（记忆增强）

结合滑动窗口（近期对话）和**事实提取**（长期关键信息）。类似于给学生一堆便签纸，重要信息随时可查。

<details>
<summary>Memory-Augmented Memory 实现</summary>

<div class="code-formula-container">
<div class="code-side">

{{< highlight python "linenos=table,linenostart=1" >}}
class MemoryAugmentedMemory(BaseMemoryStrategy):
    def __init__(self, window_size: int = 2):
        self.recent_memory = SlidingWindowMemory(window_size=window_size)
        self.memory_tokens = []

    def add_message(self, user_input: str, ai_response: str):
        self.recent_memory.add_message(user_input, ai_response)

    fact_extraction_prompt = (
            f"分析以下对话，是否包含需要长期记住的关键事实？\n"
            f"例如：用户偏好、关键决定、重要信息等。\n\n"
            f"对话:\nUser: {user_input}\nAI: {ai_response}\n\n"
            f"如果有，用一句话描述。否则回复'No important fact.'"
        )

    extracted_fact = generate_text("你是事实提取专家。", fact_extraction_prompt)

    if "no important fact" not in extracted_fact.lower():
            print(f"[Memory Token Created: '{extracted_fact}']")
            self.memory_tokens.append(extracted_fact)

    def get_context(self, query: str) -> str:
        recent_context = self.recent_memory.get_context(query)
        memory_token_context = "\n".join([
            f"- {token}" for token in self.memory_tokens
        ])

    return (
            f"### 关键记忆 (长期事实):\n{memory_token_context}\n\n"
            f"### 近期对话:\n{recent_context}"
        )
{{< /highlight >}}

</div>
<div class="formula-side">


![Memory-Augmented Flow](/posts/memory_augmented_flow_1769744264428.png)

</div>
</div>


</details>

**流程解析：**

1. 消息添加到**滑动窗口**（近期记忆）
2. LLM判断是否包含**关键事实**
3. 如果重要 → 添加到**Memory Tokens列表**
4. 上下文 = Memory Tokens + 近期对话

**点评：**

- ✅ **优点**：关键信息永久保留，即使滑动窗口移动也不丢失
- ❌ **缺点**：额外的LLM调用增加延迟和成本

---

### 策略6: Hierarchical Memory（分层记忆）

模仿人类记忆系统：**工作记忆**（最近几秒的内容）+ **长期记忆**（持久存储的重要信息）。

<details>
<summary>Hierarchical Memory 实现</summary>

<div class="code-formula-container">
<div class="code-side">

{{< highlight python "linenos=table,linenostart=1" >}}
class HierarchicalMemory(BaseMemoryStrategy):
    def __init__(self, window_size: int = 2, k: int = 2, embedding_dim: int = 3584):
        # 第一层：快速短期工作记忆
        self.working_memory = SlidingWindowMemory(window_size=window_size)
        # 第二层：持久长期记忆
        self.long_term_memory = RetrievalMemory(k=k, embedding_dim=embedding_dim)
        # 触发提升到长期记忆的关键词
        self.promotion_keywords = [
            "remember", "rule", "preference",
            "always", "never", "allergic"
        ]

    def add_message(self, user_input: str, ai_response: str):
        self.working_memory.add_message(user_input, ai_response)

    # 检查是否需要提升到长期记忆
        if any(keyword in user_input.lower()
               for keyword in self.promotion_keywords):
            print("[Promoting to long-term storage]")
            self.long_term_memory.add_message(user_input, ai_response)

    def get_context(self, query: str) -> str:
        working_context = self.working_memory.get_context(query)
        long_term_context = self.long_term_memory.get_context(query)

    return (
            f"### 长期记忆:\n{long_term_context}\n\n"
            f"### 工作记忆:\n{working_context}"
        )
{{< /highlight >}}

</div>
<div class="formula-side">


![Hierarchical Memory Flow](/posts/hierarchical_flow_1769744284503.png)

</div>
</div>


</details>

**流程解析：**

1. 消息添加到**工作记忆**
2. 检测关键词（如"remember", "always"）
3. 如果匹配 → **提升到长期记忆**（向量数据库）
4. 查询时：搜索长期记忆 + 获取工作记忆 → 合并上下文

**点评：**

- ✅ **优点**：兼顾响应速度和信息持久性
- ❌ **缺点**：需要设计合理的提升策略

---

### 策略7: Graph-Based Memory（知识图谱）

将信息表示为**节点**（实体）和**边**（关系）的知识图谱。支持复杂的关系推理。

<details>
<summary>Graph Memory 实现</summary>

<div class="code-formula-container">
<div class="code-side">

{{< highlight python "linenos=table,linenostart=1" >}}
import networkx as nx
import re

class GraphMemory(BaseMemoryStrategy):
    def __init__(self):
        self.graph = nx.DiGraph()

    def _extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        """使用LLM提取 (主体, 关系, 客体) 三元组"""
        extraction_prompt = (
            f"从以下文本中提取知识三元组。\n"
            f"格式: [('主体', '关系', '客体')]\n\n"
            f"文本:\"{text}\""
        )
        response = generate_text("你是知识图谱专家。", extraction_prompt)

    found_triples = re.findall(
            r"\([&#39;\"](.*?)['\"],\s*[&#39;\"](.*?)['\"],\s*[&#39;\"](.*?)['\"]\)",
            response
        )
        return found_triples

    def add_message(self, user_input: str, ai_response: str):
        full_text = f"User: {user_input}\nAI: {ai_response}"
        triples = self._extract_triples(full_text)
        for subject, relation, obj in triples:
            self.graph.add_edge(subject.strip(), obj.strip(),
                                relation=relation.strip())

    def get_context(self, query: str) -> str:
        if not self.graph.nodes:
            return "知识图谱为空。"

    # 实体链接
        query_entities = [
            word.capitalize() for word in query.replace('?','').split()
            if word.capitalize() in self.graph.nodes
        ]

    context_parts = []
        for entity in set(query_entities):
            for u, v, data in self.graph.out_edges(entity, data=True):
                context_parts.append(f"{u} --[{data['relation']}]--> {v}")
            for u, v, data in self.graph.in_edges(entity, data=True):
                context_parts.append(f"{u} --[{data['relation']}]--> {v}")

    return "### 知识图谱检索:\n" + "\n".join(set(context_parts))
{{< /highlight >}}

</div>
<div class="formula-side">


![Graph-Based Memory Flow](/posts/graph_memory_flow_1769744306375.png)

</div>
</div>


</details>

**流程解析：**

1. 消息 → LLM提取**三元组** (Subject, Relation, Object)
2. 三元组添加到**知识图谱**
3. 查询时：**实体链接** → 图遍历 → 检索相关事实

**点评：**

- ✅ **优点**：支持关系推理，构建专家知识库
- ❌ **缺点**：三元组提取不稳定，需要高质量的NLP

---

### 策略8: Compression & Consolidation（激进压缩）

将每条信息压缩为**最精简的事实陈述**，比摘要更激进。

<details>
<summary>Compression Memory 实现</summary>

<div class="code-formula-container">
<div class="code-side">

{{< highlight python "linenos=table,linenostart=1" >}}
class CompressionMemory(BaseMemoryStrategy):
    def __init__(self):
        self.compressed_facts = []

    def add_message(self, user_input: str, ai_response: str):
        text_to_compress = f"User: {user_input}\nAI: {ai_response}"

    compression_prompt = (
            f"你是数据压缩引擎。将以下文本压缩为一句核心事实。\n"
            f"去除所有废话。\n\n"
            f"文本:\"{text_to_compress}\""
        )

    compressed_fact = generate_text(
            "你是数据压缩专家。", compression_prompt
        )
        print(f"[Compressed: '{compressed_fact}']")
        self.compressed_facts.append(compressed_fact)

    def get_context(self, query: str) -> str:
        if not self.compressed_facts:
            return "没有压缩事实。"

    return "### 压缩事实记忆:\n- " + "\n- ".join(self.compressed_facts)

    def clear(self):
        self.compressed_facts = []
        print("Compression memory cleared.")
{{< /highlight >}}

</div>
<div class="formula-side">


![Compression Memory Flow](/posts/compression_flow_1769744326198.png)

</div>
</div>


</details>

**流程解析：**

1. 每轮对话 → 发送给**LLM压缩引擎**
2. 提取**单句核心事实**
3. 存入**压缩事实列表**
4. 上下文 = 所有压缩事实的列表

**点评：**

- ✅ **优点**：极致的Token效率
- ❌ **缺点**：同样依赖LLM调用，可能丢失细微差别

---

### 策略9: OS-Like Memory Management（操作系统式内存管理）

借鉴操作系统的**RAM + 硬盘**模型。活跃内容在RAM中快速访问，旧内容换出到硬盘，需要时再换入。

<details>
<summary>OS-Like Memory 实现</summary>

<div class="code-formula-container">
<div class="code-side">

{{< highlight python "linenos=table,linenostart=1" >}}
from collections import deque

class OSMemory(BaseMemoryStrategy):
    def __init__(self, ram_size: int = 2):
        self.ram_size = ram_size
        self.active_memory = deque()  # RAM
        self.passive_memory = {}      # Disk
        self.turn_count = 0

    def add_message(self, user_input: str, ai_response: str):
        turn_id = self.turn_count
        turn_data = f"User: {user_input}\nAI: {ai_response}"

    # RAM满了 → Page Out到Disk
        if len(self.active_memory) >= self.ram_size:
            lru_turn_id, lru_turn_data = self.active_memory.popleft()
            self.passive_memory[lru_turn_id] = lru_turn_data
            print(f"[Paging out Turn {lru_turn_id} to disk]")

    self.active_memory.append((turn_id, turn_data))
        self.turn_count += 1

    def get_context(self, query: str) -> str:
        active_context = "\n".join([
            data for _, data in self.active_memory
        ])

    # 模拟Page Fault
        paged_in_context = ""
        for turn_id, data in self.passive_memory.items():
            if any(word in data.lower()
                   for word in query.lower().split() if len(word) > 3):
                paged_in_context += f"\n(Paged in Turn {turn_id}): {data}"
                print(f"[Page fault! Paging in Turn {turn_id}]")

    return (
            f"### Active Memory (RAM):\n{active_context}\n\n"
            f"### Paged-In (Disk):\n{paged_in_context}"
        )
{{< /highlight >}}

</div>
<div class="formula-side">


![OS-Like Memory Flow](/posts/os_memory_flow_1769744346256.png)

</div>
</div>


</details>

**流程解析：**

1. 消息添加到**Active Memory (RAM)**
2. RAM满时 → **Page Out**最旧的到**Passive Memory (Disk)**
3. 查询时检查RAM → 如果没有：**Page Fault!**
4. 从Disk中**Page In**相关内容 → 合并上下文

**点评：**

- ✅ **优点**：概念清晰，理论上支持无限历史
- ❌ **缺点**：Page In的逻辑需要精心设计

---

### 如何选择策略？

| 场景             | 推荐策略                        |
| ---------------- | ------------------------------- |
| 简单短对话       | Sequential / Sliding Window     |
| 长创意对话       | Summarization                   |
| 需要精确长期召回 | Retrieval-Based (RAG)           |
| 可靠的个人助手   | Memory-Augmented / Hierarchical |
| 专家系统/知识库  | Graph-Based                     |
| 极致Token优化    | Compression                     |
| 大规模系统       | OS-Like + 混合策略              |

> 生产环境中最强大的Agent通常采用**混合策略**——例如分层系统中长期记忆同时使用向量数据库和知识图谱。
