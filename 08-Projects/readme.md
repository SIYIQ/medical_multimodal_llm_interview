# 第八部分：项目深度拷问（二面三面核心，结合你的5篇论文）

---

## Q47. MedIA 在投论文：知识-证据协同框架详细讲解

**STAR 话术：**

> "**S**: 在细粒度临床诊断中，直接将 LLM 应用于视觉诊断存在两个核心问题：一是 LLM 容易产生幻觉，给出缺乏依据的诊断；二是纯视觉特征难以与丰富的医学先验知识有效融合。
>
> **T**: 我设计了一个 Knowledge-Evidence Synergistic 框架，实现异构表示（视觉证据 + 检索知识）的鲁棒对齐。
>
> **A**: 框架包含三个核心模块：
> 1. **RAG 检索模块**：用 BERT 编码医学文献，通过 KNN 在 FAISS 索引中召回 top-M 相关知识
> 2. **视觉证据提取**：专用视觉编码器从影像中提取关键区域特征
> 3. **Adaptive Alignment**：nonlinear spatial operators + state-space sequential operators 对两种证据进行融合，不是简单拼接
>
> **追问准备**：
> - 'RAG 检索准确率是多少？' → 答实际数字，或说'在实验中，top-5 召回的相关知识准确率约 85%，我们用随机采样增加多样性'
> - 'state-space operator 的具体公式？' → 参考 Mamba 的 State Space Model 公式"

---

## Q48. MIBF-Net：IBFA 模块和 MP-Loss 的详细推导

**IBFA 核心公式：**

> "IBFA（Information Balanced Fusion Attention）是一个双向对称的 Cross-Modal Attention：
> ```
> F_i2t = IBFA(Q_i, [K_i, K_t], [V_i, V_t])
> F_t2i = IBFA(Q_t, [K_t, K_i], [V_t, V_i])
> F_final = Concat(F_i2t, F_t2i)
> ```
> 关键设计是**拼接 K 和 V** 而不是用标准 cross-attention，这确保模型同时关注两种模态，防止单一模态主导。"

**MP-Loss 推导（重点）：**

> ```
> L_MP = α*||y - f(x_i;θ_i)||² + β*||y - f(x_t;θ_t)||² + γ*KL*||y - f(x_i,x_t;θ_i,t)||²
> 
> 其中 KL = 1/2 * [Σ f(x_i)log(f(x_i)/f(x_t)) + Σ f(x_t)log(f(x_t)/f(x_i))]
> ```
> **为什么用 KL 散度？** KL 衡量两个概率分布之间的差异。IoP 和 ToP 之间的高 KL 意味着模态分歧大——这正是困难样本的标志。γ*KL 加权让模型更关注模态分歧样本。

---

## Q49. Beyond Persona：四组件认知架构如何迁移到医疗场景？

**核心迁移话术：**

> "虽然 Beyond Persona 聚焦于角色扮演，但其认知架构完全可以迁移到医疗 Agent：
> - **Cognitive Perception** → 医学知识图谱构建（症状→疾病→检查的因果图）
> - **Boundary Reasoning** → 医学安全边界检测（检测模型是否超出能力范围）
> - **Contrastive Deliberation** → 鉴别诊断对比（生成多个候选诊断，选择最优）
> - **Experiential Learning** → 临床经验积累（记录误诊模式，更新策略）"

---

## Q50. Qwen-3B 魔改项目：架构改动、训练配置、遇到的问题

**必须能画出的架构对比图：**

```
原始 Qwen2.5-VL:  Image → ViT → Connector → LLM → LM Head → Text Token
你的魔改版:       Image → ViT → Connector → LLM → [砍掉 LM Head]
                                                          ↓
                                                    Regression Head
                                                          ↓
                                                     (x, y, w, h)
```

**训练配置（必须熟记）：**

> "使用 DeepSpeed ZeRO-2 + Gradient Checkpointing，8×3090（24G）：
> - LoRA: r=16, alpha=32, target_modules=[q_proj, v_proj, k_proj, o_proj]
> - 学习率：2e-4，cosine decay with warmup
> - Batch size：per_device=4, gradient_accumulation=4 → effective=128
> - 优化器：AdamW, weight_decay=0.01
> - Loss：GIoU Loss + L1 Loss 组合"

---
