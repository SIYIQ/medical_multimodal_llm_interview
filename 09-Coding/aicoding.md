# 第九部分：手撕代码（一面必考，必须熟练默写）

---

## 必刷题单（按优先级排序）

### 🔴 最高频（阿里/字节都考，必须30秒内开始写）

**1. 请用 Python 设计一个能检测并纠正医疗诊断幻觉的 AI Agent 系统代码框架**

一、 黄金五步法（AI 系统设计标准化流程）

拿到题目（比如：“请你设计一个医疗影像诊断的 Agent 系统”），千万不要上来就写代码，一定要按这 5 步走，边说边写注释：

> 第一步：需求澄清 (Clarification) - 耗时 2-3 分钟
>
> 输入输出：输入是什么（纯文本？还是图文多模态 CT片+症状描述）？输出是什么（单句结果？还是一份完整的长篇诊断报告）？
>
> 性能要求：需要实时响应（流式输出）吗？对准确率要求极高（医疗场景，0容忍幻觉）还是更看重召回率？

> 第二步：核心模块拆解 (Architecture Design) - 耗时 3-5 分钟
>
> 在代码里用注释写出你要设计的几个核心类（Class）：
>
> - Perception Module（感知/输入处理模块）
> - Memory / RAG Module（记忆与检索增强模块）
> - Reasoning Agent（核心推理大脑）
> - Tools / Action（外部工具库，如医学知识图谱API）
> - Evaluator / Reflection（反思与幻觉校验模块）

> 第三步：接口与类抽象 (Interface Definition) - 耗时 5-8 分钟
>
> 开始写 Python 类的框架（OOP）。不需要写具体的底层实现，写好 def 和参数类型，用 pass 代替，展现你的软件工程素养。

> 第四步：核心流转逻辑实现 (Core Pipeline) - 耗时 10 分钟
>
> 把上面定义的类串起来，写一个 run() 或 __call__() 方法，展现大模型的数据流是怎么走的（这步最关键）。

> 第五步：异常处理与边界优化 (Trade-offs & Edge Cases) - 耗时 3-5 分钟
>
> 主动告诉面试官：“这里可能会触发大模型的 Token 上限，我会做截断”、“这里 RAG 检索回来的知识可能有冲突，我会用一个 Rerank 模型重新打分”。

```python
import typing as t

# 1. 定义基础数据结构 (展示你的工程规范)
class MedicalRecord:
    def __init__(self, image_path: str, patient_desc: str):
        self.image_path = image_path
        self.patient_desc = patient_desc

class DiagnosisResult:
    def __init__(self, conclusion: str, confidence: float, evidence_list: t.List[str]):
        self.conclusion = conclusion
        self.confidence = confidence
        self.evidence_list = evidence_list

# 2. 外部工具模块抽象 (Tools)
class MedicalKnowledgeBase:
    def retrieve_guidelines(self, query: str, top_k: int = 3) -> t.List[str]:
        # 伪代码：调用向量数据库进行 RAG 检索
        pass

# 3. 核心 Agent 类
class MedicalDiagnosisAgent:
    def __init__(self, vlm_model, llm_model, retriever: MedicalKnowledgeBase):
        self.vlm = vlm_model  # 多模态大模型 (处理图片)
        self.llm = llm_model  # 纯文本大模型 (用于逻辑反思)
        self.retriever = retriever
        self.max_retries = 3  # 幻觉纠正的最大重试次数

    # --- 核心 Pipeline ---
    def generate_reliable_diagnosis(self, record: MedicalRecord) -> DiagnosisResult:

        # Step 1: 视觉多模态初步感知
        initial_diagnosis = self.vlm.generate(image=record.image_path, prompt=record.patient_desc)

        # Step 2: RAG 检索医学指南支持
        evidences = self.retriever.retrieve_guidelines(query=initial_diagnosis)

        # Step 3: 幻觉检测与反思循环 (Reflection) -> 极其加分的点！
        for attempt in range(self.max_retries):
            is_hallucination, feedback = self._detect_hallucination(initial_diagnosis, evidences)

            if not is_hallucination:
                return DiagnosisResult(conclusion=initial_diagnosis, confidence=0.95, evidence_list=evidences)

            # 如果存在幻觉，进行 Prompt 重写与自我纠正
            correction_prompt = f"原诊断为:{initial_diagnosis}。发现与医学指南冲突:{feedback}。请修正诊断。"
            initial_diagnosis = self.llm.generate(prompt=correction_prompt)

        # 兜底策略 (Fallback)
        return DiagnosisResult(conclusion="诊断存在冲突，请转交人工医生复核", confidence=0.1, evidence_list=[])

    # --- 内部私有方法 ---
    def _detect_hallucination(self, diagnosis: str, evidences: t.List[str]) -> t.Tuple[bool, str]:
        """
        基于自我一致性 (Self-Consistency) 或专用的判别模型来检测幻觉
        返回: (是否是幻觉, 具体冲突说明)
        """
        verification_prompt = f"请校验结论 '{diagnosis}' 是否符合医学证据 '{evidences}'。输出布尔值和理由。"
        response = self.llm.generate(prompt=verification_prompt)
        # 解析 response...
        pass
```

---
