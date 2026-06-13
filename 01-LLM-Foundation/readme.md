# 第一部分：大模型基础架构（一面高频，必考）

---

## Q1. 讲一讲 Transformer 的完整架构和细节

**回答框架（2分钟版）：**

> "Transformer 是一个完全基于注意力机制的序列到序列模型，由 Encoder 和 Decoder 两部分组成。原始论文《Attention Is All You Need》提出它用于机器翻译任务。
>
> **Encoder 部分**：由 N 个相同的层堆叠而成（BERT 中 N=12）。每一层包含两个子层：(1) Multi-Head Self-Attention，让序列中的每个位置都能 attend 到所有位置；(2) Position-wise Feed-Forward Network，对每个位置独立应用相同的 MLP。两个子层之间都有残差连接和 LayerNorm。
>
> **Decoder 部分**：同样由 N 层堆叠，但每一层有三个子层：(1) Masked Multi-Head Self-Attention，只让当前位置 attend 到之前的位置（ causal mask ）；(2) Cross-Attention，attend 到 Encoder 的输出；(3) FFN。最后经过 Linear + Softmax 输出概率分布。
>
> **关键设计**：位置编码（Positional Encoding）将位置信息注入模型，因为注意力机制本身不具备顺序感知能力。残差连接缓解梯度消失，LayerNorm 稳定训练。"

**追问1：为什么用 LayerNorm 而不是 BatchNorm？**

> "在 NLP 任务中，序列长度不固定，BatchNorm 对一个 batch 内的同一维度做归一化，但不同句子的同一位置可能没有语义关联。LayerNorm 对每个样本的所有特征做归一化，不受 batch size 和序列长度影响，更适合变长序列。"

**追问2：Pre-Norm vs Post-Norm？**

> "原始 Transformer 用 Post-Norm（x + Sublayer(Norm(x))），但深层模型训练不稳定。Pre-Norm（Norm(x + Sublayer(x))）将归一化放在子层之前，梯度路径更短，训练更稳定。现在的大模型（LLaMA、Qwen 等）普遍使用 Pre-Norm + RMSNorm。" 
> 解释：Post-Norm 梯度要穿过 L 个 LayerNorm上，LayerNorm 的导数不是 1，而是会缩放梯度，导致深层梯度爆炸/消失，且前向信号被反复扭曲。而且即使子层输出为 0，LN 也会扭曲信号
> Pre-Norm 残差连接提供了梯度高速公路，LayerNorm 只缩小子层的梯度。Pre-Norm：子层为 0 时，信号原封不动

---

## Q2. Self-Attention 的计算过程，为什么要除以 sqrt(d_k)？

**公式推导（必须能手写）：**

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

其中：
- Q, K, V 分别是查询、键、值矩阵，形状为 [seq_len, d_k]
- Q * K^T 得到注意力分数矩阵 [seq_len, seq_len]
- 除以 sqrt(d_k) 进行缩放
- softmax 归一化为概率分布
- 与 V 相乘得到输出 [seq_len, d_v]
```

**为什么要除以 sqrt(d_k)？**

> "当 d_k 很大时，Q * K^T 的点积值会变得非常大（方差随维度线性增长），导致 softmax 进入梯度极小的饱和区。除以 sqrt(d_k) 可以将点积的方差缩放到约 1，保持 softmax 的梯度流动，防止梯度消失。这是 scaled dot-product attention 的核心设计。"

**手撕代码（PyTorch，必须熟练默写）：**

```python
import torch
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, seq_len, d_k]
    mask: [batch, seq_len, seq_len] or None
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: [batch, seq_len, seq_len]
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

---

## Q3. Multi-Head Attention 的原理和计算过程

**核心思想：**

> "Multi-Head Attention 将 Q、K、V 投影到 h 个不同的子空间，在每个子空间独立计算注意力，然后将结果拼接再投影。这类似于 CNN 中的多个滤波器，不同 head 可以关注不同的特征模式（语法关系、指代关系、语义关联等）。"

**计算流程（能手写）：**

```
给定输入 X [batch, seq_len, d_model]:
1. 线性投影: Q_i = X * W_i^Q, K_i = X * W_i^K, V_i = X * W_i^V
   其中 i = 1..h, W_i 形状分别为 [d_model, d_k], [d_model, d_k], [d_model, d_v]
   通常 d_k = d_v = d_model / h

2. 每个 head 计算注意力: head_i = Attention(Q_i, K_i, V_i)

3. 拼接: Concat(head_1, ..., head_h) [batch, seq_len, h * d_v]

4. 最终线性投影: Output = Concat * W^O [batch, seq_len, d_model]
```

**手撕代码：**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. 线性投影并分头 [batch, h, seq_len, d_k]
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 拼接多头 [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. 最终投影
        return self.W_o(attn_output)
```

---

## Q4. 位置编码有哪些类型？各有什么优缺点？

| 类型 | 原理 | 优点 | 缺点 | 代表模型 |
|------|------|------|------|---------|
| **绝对正弦PE** | sin(pos/10000^(2i/d)) | 可处理任意长度，有理论保证 | 不是真正学到位置，相对位置信息弱 | 原始Transformer |
| **可学习绝对PE** | 直接学习位置嵌入向量 | 更灵活，模型自行决定 | 长度固定，外推性差 | BERT |
| **RoPE** | 通过旋转矩阵编码相对位置 | 天然支持相对位置，外推性好 | 实现稍复杂 | LLaMA, Qwen, GPT-NeoX |
| **ALiBi** | 在注意力分数上添加偏置 | 训练稳定，外推性极好 | 性能上限略低 | BLOOM, MPT |
| **M-RoPE** | 多模态RoPE，区分不同模态 | 适合多模态场景 | 实现复杂 | Qwen2.5-VL |

**RoPE 深度解释（阿里必考）：**

> "RoPE（Rotary Position Embedding）的核心思想是将位置信息编码为旋转矩阵，作用于 Q 和 K 向量。具体来说，对于位置 m 的向量 x，将其每两个维度组成一个复数，然后乘上 e^(i * m * theta_j)，其中 theta_j = 10000^(-2j/d)。
> 先做线性投影，再乘旋转矩阵。利用旋转矩阵和点积可交换的性质得到 不同维度用的是不同的旋转角
> 这样做的好处是：Q_m 和 K_n 的点积结果天然只依赖于 (m-n)，即**相对位置**，（m和n都是位置）而不是绝对位置。同时，RoPE 可以通过 NTK 插值或 YaRN 等方法进行长度外推，这是大模型长上下文的关键技术。"

---

## Q5. Encoder-only、Decoder-only、Encoder-Decoder 的代表模型和区别？

| 架构 | 代表模型 | 注意力类型 | 适用场景 | 预训练目标 |
|------|---------|-----------|---------|-----------|
| **Encoder-only** | BERT, ViT | 双向Self-Attention | 理解任务（分类、NER、检索） | Masked Language Model (随机 mask 掉一些词，让模型预测被 mask 的词。去噪自编码器)|
| **Decoder-only** | GPT系列, LLaMA, Qwen | Causal Masked Attention | 生成任务（对话、续写、代码） | 下一个token预测 |
| **Encoder-Decoder** | T5, BART | Encoder双向+Decoder因果 | 翻译、摘要 | Span Corruption eq2Seq mask的是一个句子|

**追问：为什么现在大模型主流是 Decoder-only？**

> "1. **生成能力更强**：自回归生成天然适合开放-ended文本生成。
> 2. **扩展性更好**：Decoder-only 架构在规模扩大时表现更稳定，涌现能力更强。
> 3. **训练效率更高**：统一用下一个token预测目标，不需要设计复杂的预训练任务。
> 4. **工程简单**：相比 Encoder-Decoder 少了 cross-attention，推理更容易优化（如 KV Cache）。
> 5. **上下文学习**：Causal attention 使得模型在预训练时就学会了从上下文中提取信息，in-context learning 能力更强。"

---

## Q6. Qwen 系列模型相比原始 Transformer 有哪些结构改动？Qwen2 又有哪些改进？

**Qwen（第一代）：**

> "Qwen 在原始 Transformer Decoder 基础上做了以下改动：
> 1. **RMSNorm 替代 LayerNorm**：
LayerNorm: 减均值 + 除标准差 + 乘权重 + 加偏置
`LayerNorm(x) = (x - mean) / sqrt(var + eps) * γ + β`
RMSNorm: 去掉 centering，只除 RMS + 乘权重
`RMSNorm(x) = x / sqrt(mean(x²) + eps) * γ`，去掉了mean和\beta。
> 2. **SwiGLU 激活函数**：替代 ReLU # ，SwiGLU(x) = Swish(xW) * xV 门：Swish(x) = x · sigmoid(βx)，值：xV 把 FFN 变成门控结构，两个线性投影一个当"门"、一个当"值"，表达能力更强，而且使用Swich实现更平滑的激活，值分支也不再经过额外的激活。
> 3. **RoPE 位置编码**：替代正弦位置编码，支持长度外推。
> 4. **分组查询注意力（GQA）**：在参数量大版本中，多个 query head 共享同一组 K/V head，减少推理时的 KV Cache 显存。

```
推理时 K/V 要缓存（KV Cache）
MHA 的 KV Cache = 2 × num_heads × seq × head_dim × batch
GQA 把 K/V head 减少，KV Cache 显存减少 num_heads/num_kv_heads 倍
牺牲少量效果，大幅降低推理成本

# MHA: Q/K/V 各有 num_heads 个头
Q: (batch, 32, seq, 128)   # 32 heads
K: (batch, 32, seq, 128)   # 32 heads  
V: (batch, 32, seq, 128)   # 32 heads

# GQA: Q 有 num_heads 个，K/V 只有 num_kv_heads 个（num_kv_heads < num_heads）
Q: (batch, 32, seq, 128)   # 32 heads
K: (batch, 8, seq, 128)    # 8 heads（每4个Q共享1个K）
V: (batch, 8, seq, 128)    # 8 heads
```

> 5. **采用 Slidding Window Attention + Full Attention 混合**：长文本场景兼顾效率和性能。"
```
# Full Attention: 每个词看所有词，O(n²)复杂度
# SWA: 每个词只看窗口内的词，O(n×w)复杂度

attn_mask[i, j] = 1 if |i - j| <= window_size else 0
```

**Qwen2 的改进：**

> "1. **GQA 全面应用**：所有尺寸模型都使用 GQA。
> 2. **双块注意力（Dual Chunk Attention）**：结合全局注意力和局部滑动窗口注意力。
```
# 把长序列切成 chunks
chunk_size = 2048

# 三种注意力模式同时存在：
# 1. Intra-chunk attention: chunk 内部互相看（局部）
# 2. Inter-chunk attention: chunk 之间稀疏看（跨chunk）  
# 3. Global attention: 某些位置全局可见（如<|im_end|>）

# 三种 mask 组合，通过一次性计算完成
纯 Full Attention：O(n²)，太长训不动
纯 Sliding Window：远处的信息传不过来
DCA 平衡了效率和长距离依赖
```
> 3. **更大规模的高质量预训练数据**：约 7T tokens，经过更严格的质量过滤。
> 4. **更优的指令微调数据配比**：增加了代码（HumanEval）、数学（GSM8K、MATH）、多语言数据的比例。
> 5. **训练范式改进**：Qwen1 用 RLHF（PPO）,Reward Model 打分 → PPO 优化 Policy,# 复杂、不稳定、需要维护 Reward Model。Qwen2 主要用 DPO（Direct Preference Optimization）直接用偏好数据（chosen rejected）优化 更简单、更稳定、效果不差 128K"

---

## Q7. 讲一讲 ViT（Vision Transformer）的架构和原理

**核心思想：**

> "ViT 将 Transformer 直接应用于图像分类任务。核心做法是将图像切分为固定大小的 patch（如 16x16），每个 patch 展平后通过线性投影得到 patch embedding，加上位置编码和 [CLS] token 后输入标准 Transformer Encoder。
>
> **具体流程**：
> 1. 输入图像 224x224x3，切分为 14x14=196 个 16x16 的 patch
> 2. 每个 patch 展平为 768 维向量（16*16*3=768）或者切分patch+展平=Conv2D
> 3. 通过线性投影（等价于一个卷积层）得到 patch embedding
> 4. 添加可学习的位置编码（逐个元素相加）和 [CLS] token（放在最前面，通过双向注意力实现看见后面，毕竟要是自回归的话，只能看见牵头的东西，而这这里前头啥都没有）
> 5. 输入 Transformer Encoder
> 6. [CLS] token 的输出接 MLP Head 做分类
>
> **ViT 的优势**：全局感受野，能建模长距离依赖；参数量大时性能超过 CNN。
> **劣势**：需要大量数据预训练，对局部特征不敏感（后续有 Swin Transformer 等改进）。"

---

## Q8. 几种主流 Normalization 的区别（LayerNorm / BatchNorm / RMSNorm / GroupNorm）

| 方法 | 归一化维度 | 公式 | 适用场景 |
|------|-----------|------|---------|
| **BatchNorm** | 对一个 batch 内同一通道的所有元素 | (x - mu_B) / sigma_B | CNN，固定 batch size |
| **LayerNorm** | 对一个样本的所有特征 | (x - mu) / sigma | NLP/RNN/Transformer，变长序列 |
| **RMSNorm** | 对一个样本的所有特征，只保留缩放 | x / RMS(x) * gamma | 大模型主流（LLaMA/Qwen），更快更稳定 |
| **GroupNorm** | 对一个样本的通道分组 | 类似LN但只对组内 | 小batch的CV任务 |

**RMSNorm 详解（面试加分项）：**

```python
def rms_norm(x, gamma, eps=1e-6):
    """
    x: [batch, seq_len, hidden_size]
    gamma: [hidden_size] 可学习参数
    """
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * gamma
```

> "RMSNorm 相比 LayerNorm 去除了 mean centering 操作，只保留 root mean square 缩放。这样做的原因是：
> 1. **更快**：少了一次求均值操作
> 2. **更稳定**：在深层 Transformer 中，mean centering 对训练稳定性贡献不大
> 3. **实验证明**：在 LLM 场景下，RMSNorm 的下游任务表现与 LayerNorm 相当甚至更优"

---

## Q9. 你了解 MoE（Mixture of Experts）吗？介绍一下架构

**核心思想：**

> "MoE 的核心思想是将模型中的 FFN 层替换为多个专家网络（Experts），通过一个门控网络（Gating Network）来决定每个 token 激活哪些专家。这样可以在不显著增加推理计算量的情况下大幅扩展模型参数量。
>
> **架构细节**：
> 1. **Experts**：N 个并行的 FFN（如 8 个或 64 个），每个 expert 是一个独立的 FFN
> 2. **Router/Gating Network**：一个线性层，输入是 hidden state，输出 N 个分数
> 3. **Top-K 路由**：对每个 token 选择分数最高的 K 个 expert（通常 K=1 或 K=2）
> 4. **负载均衡损失**：防止所有 token 都路由到少数几个 expert（负载不均衡）
>
> **代表性模型**：Mixtral 8x7B（8 个 expert，top-2 路由，总参数量约 47B，推理时只激活约 13B）
>
> **追问：负载均衡问题怎么解决？**
> 使用辅助负载均衡损失（Auxiliary Load Balancing Loss）：
> L_aux = alpha * N * sum(f_i * P_i)
> 其中 f_i 是路由到 expert i 的 token 比例，P_i 是门控网络分配给 expert i 的平均概率。这个损失鼓励均匀分配。"

---

## Q10. DeepSpeed 的 ZeRO 机制（ZeRO-1/2/3）分别做了什么优化？
1B 参数模型，fp16 混合精度训练，单卡显存占用：
参数:           2 GB   (fp16/bf16, 2 bytes × 1B)
梯度:           2 GB   (fp16/bf16, 2 bytes × 1B)
优化器状态:     12 GB  (Adam: fp32参数副本 + momentum + variance = 12 bytes × 1B)
激活值:         ~2-4 GB (取决于 batch、seq_len)

总计: ~18-20 GB 每张卡！8 张卡就是 8 份完整副本，大量冗余。
| Stage | 分片内容 | 显存节省 | 通信开销 |
|-------|---------|---------|---------|
| **ZeRO-0** | 无分片（DDP基线） | 0% | 低 |
| **ZeRO-1** | 只将优化器状态分八份，每张卡算完整梯度| 4x（Adam占用大量显存） | 低 |
| **ZeRO-2** | 优化器状态、梯度分片，每张卡算自己部分的梯度。使用reduce-scatter将梯度聚合到每张卡 | 8x | 中 |
| **ZeRO-3** | 优化器状态、梯度、模型参数分片 | 与数据并行度线性相关 | 高 |

**ZeRO-Offload（额外优化）：**

> "ZeRO-Offload 将优化器状态和计算卸载到 CPU/NVMe，进一步降低 GPU 显存占用。适合单卡/少卡训练大模型场景。"

**你的项目话术：**

> "我在魔改 Qwen-3B 时使用了 DeepSpeed ZeRO-2 + Gradient Checkpointing，在 8×3090（24G）上成功训练。ZeRO-2 将优化器状态和梯度分片到 8 张卡上，每张卡只需存 1/8 的优化器状态，配合 Gradient Checkpointing（用计算换显存），将显存从约 48G 降到了约 18G 每张卡。"

---

## Q11. Flash Attention 的原理和优势

**核心思想：**

> "Flash Attention 的核心洞察是：**减少 HBM（高带宽显存）和 SRAM（高速缓存）之间的数据搬运**。标准 Attention 需要将 Q、K、V、注意力矩阵全部存入 HBM，显存占用高且计算受限于内存带宽。
>
> **技术细节**：
> 1. **Tiling/分块计算**：将 Q、K、V 分成小块（tile），在 SRAM 中完成局部注意力计算
> 2. **Online Softmax**：避免一次性计算完整的注意力矩阵，增量计算 softmax
> 3. **Recomputation**：反向传播时不存储注意力矩阵，而是在需要时重新计算
>
> **优势**：
> - 显存：从 O(N^2) 降到 O(N)，可以处理更长的序列
> - 速度：减少 HBM 访问，实际训练速度提升 2-4x
> - IO-aware：考虑了 GPU 内存层次结构的实际特性"

---

## Q12. 显存不够一般怎么解决？列举所有手段

**回答框架（从工程到算法的完整方案）：**

> "1. **模型并行**：Tensor Parallelism（层内分片）、Pipeline Parallelism（层间分片）
> 2. **数据并行优化**：DeepSpeed ZeRO-1/2/3，分片优化器状态/梯度/参数
> 3. **CPU Offload**：ZeRO-Offload 将优化器状态卸载到 CPU 内存
> 4. **梯度检查点（Gradient Checkpointing）**：只保存关键层的激活值，其余重计算
> 5. **混合精度训练**：FP16/BF16 减少显存占用和计算时间
> 6. **量化训练**：QLoRA（4-bit 量化基模型 + LoRA 微调）
> 7. **高效微调**：LoRA/Adapter/Prompt Tuning，只训练少量参数
> 8. **减小 batch size + 梯度累积**：等效大 batch 训练
> 9. **清理不需要的缓存**：torch.cuda.empty_cache()"

---
