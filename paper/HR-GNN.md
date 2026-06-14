困难样本是图像还是文本 还是都有 拼接是把batch_size整大了吗

具体讲解下三个算子的工作原理：

所以第一个算子不是“用知识节点方向的什么去乘”，而是：
1. 把样本和知识都归一化到球面上；
2. 使用正则化之后的知识节点减去图像节点作为矫正向量（从样本指向知识的测底线方向）
3. 使用门控更新：拼接图像节点和矫正向量先判断是否需要接受更新（使用MLP+Sigmoid作为门控），再乘上经过MLP映射的矫正向量
4. 图像节点的更新 = 门控 × MLP(校正向量)


第二个是把文本特征相对于图像特征拆成两个部分，验证分量：文本中与图像方向一致的部分（共线投影）；补充分量：文本中图像没有的新信息（正交残差）
把 h_txt 投影到 h_img 所在直线上，得到 verification_comp。这代表“文本在重复/验证图像已经知道的信息”。
用 h_txt - verification_comp 得到 supplementation_comp。这代表“文本提供的图像没有的新信息”，与图像方向正交。
dual_branch_gate 学习每个样本对中，验证信息和补充信息各应该占多少权重。
最终传递给图像节点的是加权后的 message。

第三个：找一个图像节点的邻居中“差异最大”的那些，把它们的差异向量传过来。
对于每个图像节点 i，先看它的邻居 j。
计算 h_j - h_i，也就是邻居相对于中心的差异向量。
||h_j - h_i|| 越大，说明这个邻居和中心越不一样。
用 softmax 给这些差异向量加权，差异大的邻居权重高。
权重乘“邻居 − 中心”的差分向量，然后求和得到聚合差异向量，再经过门控和 MLP 后用来更新中心节点。
图像节点的更新 = 门控 × MLP(校正向量) 
门控 gate_ss_img 控制调整幅度，MLP sample_update 控制怎么调整。

损失函数
分布同质化损失：计算mini-batch里面每个类别的方差，方差越小表示只是对同类样本的辐射更均匀
对齐知识损失：

原始图像/文本 → Encoder → img_feat (B,D), txt_feat (B,D)
                              ↓
                    Hard Case Queue 采样 H 个困难样本
                              ↓
                    拼接成 (B+H, D) 进入 HRC-GCN
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
   S←K 超球面校正       I←T 双分支分解           S←S 差异聚合
        ↓                     ↓                     ↓
   correction_vectors    verification/supplement   max_diff
        └─────────────────────┴─────────────────────┘
                              ↓
            fused_features → classifier → logits
                              ↓
            Loss = CE + align + knowledge_align
                + delta*correction_align
                + lambda_homo*homo
                + lambda_drift*drift
                + lambda_mi*orthogonality_MI
