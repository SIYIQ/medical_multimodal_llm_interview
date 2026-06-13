# 医疗多模态大模型面试备战仓库 🏥🤖

本仓库基于《阿里巴巴-医疗多模态大模型算法工程师 面试逐题精答手册》整理，按照面试考察模块进行结构化拆分，方便系统性复习与面试冲刺。

> 目标岗位：阿里系 LLM / VLM / Agent 算法工程师（医疗方向）  
> 覆盖范围：大模型基础架构、多模态大模型、预训练与微调、RLHF/DPO/GRPO、Agent、RAG、模型评估、项目深挖、手撕代码、HR 面  
> 学习方式：按模块精读 → 公式推导 → 代码默写 → 项目话术串联

---

## 📋 面试流程总览

| 轮次 | 时长 | 主要考察内容 | 形式 |
|------|------|-------------|------|
| 一面 | 60-70min | 八股全覆盖（架构/预训练/微调/RLHF/RAG/Agent）+ 手撕1道 | 技术面 |
| 二面 | 60min | 项目深挖 + 多模态八股 + 场景题 | 技术面 |
| 三面 | 60min | 项目拷打 + RL深度追问 + 开放设计题 | 技术面 |
| HR面 | 60min | 软实力/团队协作/职业规划 | 行为面 |

---

## 🗺️ 学习路径

### 01 大模型基础架构（一面高频，必考）
- [x] [Transformer 完整架构与细节](./01-LLM-Foundation)
- [x] [Self-Attention / Multi-Head Attention 原理与手撕](./01-LLM-Foundation)
- [x] [位置编码：RoPE / ALiBi / M-RoPE](./01-LLM-Foundation)
- [x] [Encoder-only / Decoder-only / Encoder-Decoder 区别](./01-LLM-Foundation)
- [x] [Qwen 系列结构改动与 Qwen2 升级](./01-LLM-Foundation)
- [x] [ViT 架构与 Normalization 对比](./01-LLM-Foundation)
- [x] [MoE、DeepSpeed ZeRO、Flash Attention、显存优化](./01-LLM-Foundation)

### 02 多模态大模型 VLM（岗位最相关，重中之重）
- [x] [多模态对齐与融合挑战](./02-VLM)
- [x] [CLIP 对比学习](./02-VLM)
- [x] [LLaVA / MiniGPT-4 视觉-语言连接方案](./02-VLM)
- [x] [Q-Former vs MLP Connector 对比](./02-VLM)
- [x] [Qwen-VL / Qwen2.5-VL 训练流程与架构特点](./02-VLM)
- [x] [视觉指令微调与 VLM 幻觉问题](./02-VLM)

### 03 预训练与微调（一面高频）
- [x] [预训练目标函数：Next Token Prediction](./03-Training)
- [x] [SFT 完整流程与注意事项](./03-Training)
- [x] [LoRA 原理、公式推导与超参设置](./03-Training)
- [x] [Q-LoRA 与 LoRA 区别](./03-Training)
- [x] [P-tuning / Prefix-tuning / Adapter 对比](./03-Training)

### 04 RLHF / DPO / GRPO（阿里三面重点）
- [x] [RLHF 三步流程：SFT → RM → PPO](./04-Alignment)
- [x] [PPO vs GRPO 核心区别](./04-Alignment)
- [x] [GRPO 为什么不需要 Value Model](./04-Alignment)
- [x] [DPO 原理与损失函数推导](./04-Alignment)
- [x] [DPO vs GRPO 数据与指标差异](./04-Alignment)
- [x] [RL 训练诊断、Cold Start、TRPO/PPO/GRPO 演进](./04-Alignment)

### 05 AI Agent（二面重点，结合项目）
- [x] [AI Agent 核心组件：感知/规划/记忆/工具/执行](./05-Agent)
- [x] [ReAct / Reflexion / CoT / ToT 区别与应用](./05-Agent)
- [x] [Agent 长期一致性与记忆系统设计](./05-Agent)
- [x] [自我改进机制与 MCP 协议](./05-Agent)
- [x] [医疗诊断 Agent 开放设计题](./05-Agent)

### 06 RAG 检索增强（一面高频）
- [x] [RAG 完整架构：Indexing → Retrieval → Generation](./06-RAG)
- [x] [Chunk 切分策略与向量数据库选择](./06-RAG)
- [x] [检索知识与模型知识冲突处理](./06-RAG)
- [x] [高级 RAG：Self-RAG / CRAG / GraphRAG / HyDE](./06-RAG)

### 07 模型评估（三面场景题）
- [x] [分类任务效果不佳的系统优化方案](./07-Evaluation)
- [x] [优质数据标准与量化细则](./07-Evaluation)
- [x] [上线指标与效果测评体系设计](./07-Evaluation)

### 08 项目深度拷问（二面三面核心）
- [x] [MedIA：知识-证据协同框架](./08-Projects)
- [x] [MIBF-Net：IBFA 模块与 MP-Loss](./08-Projects)
- [x] [Beyond Persona：四组件认知架构迁移](./08-Projects)
- [x] [Qwen-3B 魔改：架构、配置、问题](./08-Projects)

### 09 手撕代码（一面必考）
- [x] [Self-Attention / Multi-Head Attention](./09-Coding)
- [x] [LoRA 层](./09-Coding)
- [x] [KL 散度 / RMSNorm / LayerNorm](./09-Coding)
- [x] [LeetCode 高频：无重复子串、反转链表、合并K链表、接雨水、LCS、LRU](./09-Coding)

### 10 HR 面
- [x] [项目团队与个人贡献](./10-HR)
- [x] [任务安排冲突处理](./10-HR)
- [x] [选阿里原因与职业规划](./10-HR)

---

## 🧰 附录

- [面试流程总览](./assets/interview-overview.md)
- [公式速查与 14 天冲刺日程表](./assets/appendix.md)

---

## 💡 使用建议

1. **一面前**：重点刷 01 / 02 / 03 / 06 / 09，确保八股流利、手撕代码熟练。
2. **二面前**：精读 02 / 05 / 08，准备项目 STAR 话术和医疗场景设计题。
3. **三面前**：死磕 04 / 07 / 08，能推导 DPO/GRPO 公式、讲清楚训练细节与评估指标。
4. **面试当天**：用 [公式速查表](./assets/appendix.md) 快速过一遍核心公式。

---

*本仓库为个人面试整理，内容源自真实面经与项目经验，持续迭代中。*
