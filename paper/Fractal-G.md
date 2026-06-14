原始图像/掩码
    │
    ▼
[utils.py] SegmentationDataset + 数据增强
    │
    ▼
[train.py] DataLoader ──> batch [B, 3, H, W]
    │
    ▼
[model.py] UNetFractalG.forward
    │
    ├── Encoder ──> x1, x2, x3, x4, x5
    │
    ├── FractalG_Module ──> feat_refined, split_prob
    │       │
    │       ├── FissionRouter ──> split_prob
    │       ├── assemble_graph ──> 异构图节点
    │       ├── AreaAwareGNN ──> 更新节点
    │       └── ImplicitRenderer ──> feat_refined
    │
    ├── Decoder ──> logits, logits_ds
    │
    ▼
[train.py] 损失计算 (CE + Dice + BCE + Boundary + Deep Supervision)
    │
    ▼
反向传播 ──> 更新 Router/GNN/Renderer/UNet 共享参数
    │
    ▼
[test.py] 评估 Dice + HD95


AreaAwareGNN
输入: node_feats [N, C], node_coords [N, 2], node_sizes [N, 1]
     │
     ▼
对每个节点 i:
    1. KNN: 找空间最近的 k 个邻居 j
    2. 投影: h_i = W * node_feats_i
    3. 注意力: e_ij = LeakyReLU(MLP([h_i, h_j, s_i, s_j]))
    4. 归一化: alpha_ij = softmax_j(e_ij)
    5. 聚合: message_i = sum_j(alpha_ij * h_j)
     │
     ▼
输出: node_feats + messages  [N, C]


从网格到图再回网格
4.1 路由决策：FissionRouter

路由函数在unet的第二层中，对于(W/4,H/4)的数据，计算裂变概率（split_prob = self.router(feat_deep) ）。大于0.5的宏节点分裂为微节点，小于0.5的部分保留为宏节点。route自己是个多层感知机

4.2 图构建：assemble_graph
根据 split_prob 把深层特征和浅层特征组装成异构图：

对每个样本：
  for y in range(H_d):
    for x in range(W_d):
      if split_prob[b,0,y,x] < 0.5:
         生成 1 个宏节点 (感受野 16×16，用 deep 特征)
      else:
         生成 16 个微节点 (感受野 1×1，用 shallow 投影特征)
输出三个 List：


node_feats_list:  List[[N_i, C_d]] 每个节点的特征向量，第i个节点有C_d维特征
node_coords_list: List[[N_i, 2]]   每个节点在原图/特征图的坐标，归一化之后的(y,x)
node_sizes_list:  List[[N_i, 1]] 每个节点对应的感受野面积，宏节点256，微节点1

4.3 图神经网络：AreaAwareGNN

对每个样本，GNN 拿到：
    node_feats:  [N, C]
    node_coords: [N, 2]
    node_sizes:  [N, 1]

KNN 图构建
    dist = torch.cdist(node_coords, node_coords)  # [N, N]
    _, knn_idx = torch.topk(dist, k=k, dim=-1, largest=False)  # [N, k]
    每个样本独立做空间 KNN 图构建 

特征投影
    h = self.feat_proj(node_feats)  # [N, C]
    先把节点特征投影一下，类似 GAT 里的线性变换。

面积感知注意力
    h_i = h.unsqueeze(1).expand(-1, k, -1)        # [N, k, C]
    s_i = node_sizes.unsqueeze(1).expand(-1, k, -1) # [N, k, 1]
    h_j = h[knn_idx]                              # [N, k, C]
    s_j = node_sizes[knn_idx]                     # [N, k, 1]

    att_input = torch.cat([h_i, h_j, s_i, s_j], dim=-1)  # [N, k, 2C+2]
    e = self.leaky_relu(self.att_mlp(att_input)).squeeze(-1)  # [N, k]
    alpha = F.softmax(e, dim=-1)                  # [N, k]

    [h_i, h_j, s_i, s_j]是注意力的输入，两个h是两个节点的投影特征，两个s是两个节点的感受野面积。所以面积项的作用是：显式编码节点尺度，让注意力网络在异构节点间做出更合理的加权。但如果不告诉模型“它们代表不同尺度的信息”，注意力可能默认给宏节点更高权重，因为它的特征幅度通常更大

    每个节点对每条边（或每个邻居）算一个注意力权重，然后在邻居上做 softmax得到权重，然后加权更新。

消息聚合
    messages = (alpha.unsqueeze(-1) * h_j).sum(dim=1)  # [N, C]
    out = node_feats + messages                          # 残差连接
    每个节点收到邻居的加权消息，然后和原特征相加。

4.4 隐式渲染：ImplicitRenderer

feat_refined = self.renderer(
    updated_nodes_list, node_coords_list, H_s, W_s
)
把 GNN 输出的“不规则离散节点”重新变回 CNN 能用的“规则网格特征图”。
传统插值（如双线性插值）是显式的：直接按距离加权已知点的值。

隐式渲染是学习的：对于网格上的每个查询位置，模型通过 MLP 学出一个从“邻域节点特征 + 相对位置”到“该位置特征”的映射。


1.首先生成2layer的归一化的格点坐标；
2.然后对于每个格点找k个最近（归一化的坐标距离）节点；
3.为MLP准备输入：计算每个格点相对于邻居节点的偏移
4.MLP解码：对每个 (查询点, 邻居) 对，将上面的偏移+邻居特征作为输入，MLP输出这个格点的输出初步特征
5.加权聚合，将上头跟每个邻居所得到的初步特征加权聚合，得到最终加权点的特征
6.reshape成规则特征图
# [B, C_shallow, H/4, W/4]
把更新后的节点通过 KNN + MLP + 距离加权，渲染回规则网格。

4.5 残差连接

return feat_shallow + feat_refined, split_prob





3. 为什么只改 down2 → up2 这一条？
因为这条 skip 有两个很合适的属性：

分辨率适中：H/4 × W/4，既保留了一定空间细节，又不会让图节点数量爆炸。
语义与细节平衡：x3 是浅层特征（边缘、纹理丰富），x5 是深层特征（语义、类别信息强）。
Fractal-G 的动机是：用深层语义指导浅层细节，所以在 H/4 这个“细节尚存、语义可及”的分辨率做 refine 最合适。

如果改 up1（H/8），空间细节太少；改 up3/up4（H/2 或 H），节点数会太多，GNN 计算开销大。

4. 是“替代”还是“增强”？
严格说是 enhanced substitution（增强式替代）：

feat_refined = FractalG(x5, x3)
内部最后一步是残差：feat_shallow + feat_refined_internal
所以 feat_refined 本身已经包含了 x3 的信息（通过残差），并加上了由深层特征引导的 refinement。

可以理解为：


标准 skip:     up2(x, x3)
Fractal-G skip: up2(x, x3 + delta(x5, x3))
其中 delta 是 Fractal-G 学到的一个自适应增强项。

5. 为什么不在所有 skip 上都加 Fractal-G？
主要考虑是计算效率与收益权衡：

Fractal-G 涉及图构建、KNN、GNN、隐式渲染，计算成本明显高于普通卷积。
只在最关键的分辨率（H/4）加一次，既能显著提升边界/细节，又不会让模型太重。
其他分辨率仍用标准 skip，保持 U-Net 的效率优势。
当然，从架构上讲，理论上可以在多条 skip 上加 Fractal-G，只是当前仓库选择了单点插入的性价比方案。