# 第九部分：手撕代码（一面必考，必须熟练默写）

---

## 必刷题单（按优先级排序）

### 🔴 最高频（阿里/字节都考，必须30秒内开始写）

**1. Self-Attention（已在上文给出完整代码）**

**2. LoRA 层（必须能手写完整 nn.Module）：**

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B初始化为0，保证训练开始时ΔW=0
    
    def forward(self, x, original_output):
        # x: [batch, seq_len, in_features]
        delta_w = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + delta_w
```

**3. KL 散度计算：**

```python
def kl_divergence(p, q, eps=1e-8):
    """KL(P||Q) = Σ P(x) * log(P(x)/Q(x))"""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return torch.sum(p * torch.log(p / q), dim=-1)
```

**4. 最长无重复字符子串（LeetCode 3）：**

```python
def lengthOfLongestSubstring(s: str) -> int:
    char_set = set()
    left = 0
    max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
```

**5. 反转链表（LeetCode 206）：**

```python
def reverseList(head):
    prev, curr = None, head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev
```

### 🟡 高频（需要熟练）

**6. 合并K个升序链表（LeetCode 23）**
**7. 接雨水（LeetCode 42）**
**8. 最长公共子序列（LeetCode 1143）**
**9. LRU 缓存（LeetCode 146）**
**10. RMSNorm / LayerNorm 手撕**

```python
def rms_norm(x, gamma, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * gamma

def layer_norm(x, gamma, beta, eps=1e-6):
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta
```

---
