# Contrastive Loss (WeightedInfoNCE) 计算说明

## 公式

1. **特征归一化**：`f1 = L2norm(features1)`, `f2 = L2norm(features2)`  
2. **相似度矩阵**：`S = logit_scale * (f1 @ f2.T)`，形状 `(B, B)`  
3. **对每个 query i**：
   - 正样本：对角线 `S[i,i]`（匹配的 drone-sate 对）
   - 负样本：`S[i,j]`（j≠i）
   - `loss_i = (1-ε)*(-S[i,i] + logsumexp(S[i,:])) + ε*(-1/n*sum(S[i,:]) + logsumexp(S[i,:]))`
4. **ε (eps)**：由 `positive_weights` 决定，`ε = 1 - 1/(1+exp(-k*weight))`，用于半正样本加权  
5. **最终**：`loss = (loss_d2s + loss_s2d) / 2`

## NaN 可能原因

1. **特征含 NaN/Inf**：DPN 的 `x^p` 或 softP 残差导致数值爆炸  
2. **学习率过大**：lr=0.001 对 DPN+softP 可能偏高，建议 0.0001  
3. **logit_scale 过大**：`exp(logit_scale)` ≈ 14.3，若特征异常会放大  
4. **positive_weights 含 NaN**：数据集中 weight 异常
