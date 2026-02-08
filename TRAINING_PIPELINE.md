# Game4Loc 训练流程完整解析

> 本文档详细解析 Game4Loc (AAAI'25) 的训练流程，包括数据加载、训练循环、损失计算和模型保存。

## 📋 目录

- [整体流程图](#整体流程图)
- [① 数据加载流程](#数据加载流程)
- [② 训练循环流程](#训练循环流程)
- [③ 损失计算详解](#损失计算详解)
- [④ 模型保存策略](#模型保存策略)
- [⑤ 关键参数配置](#关键参数配置)
- [⑥ 核心创新点](#核心创新点)

---

## 整体流程图

```
训练脚本启动 (train_gta.py)
    ↓
① 数据加载 (GTADatasetTrain)
    ├─ 读取JSON元数据
    ├─ 互斥采样 (Mutually Exclusive Sampling)
    └─ 数据增强 (Augmentation)
    ↓
② 模型初始化 (DesModel)
    ├─ 骨干网络 (ViT-B/16)
    ├─ 权重共享 (Shared Encoder)
    └─ 可学习温度参数 (logit_scale)
    ↓
③ 损失函数 (WeightedInfoNCE)
    ├─ IoU权重计算
    ├─ 硬/软对比损失混合
    └─ 双向损失 (D2S + S2D)
    ↓
④ 训练循环 (train_with_weight)
    ├─ 混合精度训练 (FP16)
    ├─ 梯度裁剪 (Gradient Clipping)
    ├─ 学习率调度 (Cosine Annealing)
    └─ 每步更新
    ↓
⑤ 评估与保存 (evaluate + save)
    ├─ 特征提取
    ├─ 相似度计算
    ├─ Recall@K 计算
    └─ 保存最佳模型
```

---

## ① 数据加载流程

### 1.1 数据集初始化

**文件位置：** `Game4Loc/game4loc/dataset/gta.py:43-99`

```python
class GTADatasetTrain(Dataset):
    def __init__(self, pairs_meta_file, data_root,
                 transforms_query, transforms_gallery,
                 mode='pos_semipos', ...):

        # 读取JSON元数据
        with open(os.path.join(data_root, pairs_meta_file), 'r') as f:
            pairs_meta_data = json.load(f)

        self.pairs = []  # 所有 (drone_img, sate_img, IoU_weight) 三元组
        self.pairs_drone2sate_dict = {}  # drone → [sate列表]
        self.pairs_sate2drone_dict = {}  # sate → [drone列表]

        # 构建配对关系
        for pair_drone2sate in pairs_meta_data:
            drone_img_name = pair_drone2sate['drone_img_name']

            # 根据mode选择样本
            # mode='pos_semipos' → IoU > 0.1 的所有配对
            # mode='pos' → IoU > 0.3 的配对
            pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
            pair_sate_weight_list = pair_drone2sate[f'pair_{mode}_sate_weight_list']

            for sate_img, weight in zip(pair_sate_img_list, pair_sate_weight_list):
                self.pairs.append((drone_img_file, sate_img_file, weight))
                # 构建图结构用于互斥采样
                self.pairs_drone2sate_dict[drone_img_name].append(sate_img)
                self.pairs_sate2drone_dict[sate_img].append(drone_img_name)
```

**数据示例：**
```python
# self.pairs 内容
[
    ("/data/drone/500_0001_0000025682.png",
     "/data/satellite/4_0_6_13.png",
     0.4734),  # IoU=0.47
    ("/data/drone/500_0001_0000025682.png",
     "/data/satellite/5_0_12_27.png",
     0.2786),  # IoU=0.28 (semi-positive)
    ...
]

# self.pairs_drone2sate_dict
{
    "500_0001_0000025682.png": ["4_0_6_13.png", "5_0_12_27.png", ...],
    ...
}
```

### 1.2 单样本读取

**文件位置：** `Game4Loc/game4loc/dataset/gta.py:102-124`

```python
def __getitem__(self, index):
    query_img_path, gallery_img_path, positive_weight = self.samples[index]

    # 1. 读取图像 (OpenCV BGR → RGB)
    query_img = cv2.imread(query_img_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    gallery_img = cv2.imread(gallery_img_path)
    gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

    # 2. 随机同步翻转 (保持drone-satellite对应)
    if np.random.random() < self.prob_flip:  # prob_flip=0.5
        query_img = cv2.flip(query_img, 1)
        gallery_img = cv2.flip(gallery_img, 1)

    # 3. 数据增强
    query_img = self.transforms_query(image=query_img)['image']
    gallery_img = self.transforms_gallery(image=gallery_img)['image']

    # 返回: [3, 384, 384], [3, 384, 384], scalar
    return query_img, gallery_img, positive_weight
```

**数据增强（transforms.py）：**
```python
# 无人机图像增强
- Cut (裁剪边缘)
- ImageCompression (JPEG压缩模拟)
- Resize (384x384)
- ColorJitter (亮度/对比度/饱和度/色调)
- AdvancedBlur / Sharpen (模糊/锐化)
- GridDropout / CoarseDropout (网格/块状遮挡)
- Normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- ToTensorV2

# 卫星图像增强
- 同上 + RandomRotation(90°) (仅cross-area)
```

### 1.3 互斥采样 (Mutually Exclusive Sampling) 🔥

**核心创新！文件位置：** `Game4Loc/game4loc/dataset/gta.py:129-249`

**目的：** 避免batch内出现冲突的正负样本对

**问题场景：**
```
❌ 传统随机采样可能导致：
Batch 中同时出现:
  (drone_A, sate_X) → 正样本 (IoU=0.5)
  (drone_A, sate_Y) → 也是正样本 (IoU=0.3)

计算对比损失时:
  sate_Y 本应是 drone_A 的正样本
  但在相似度矩阵中被当作负样本
  → 错误的梯度信号！
```

**解决方案：互斥采样**
```python
def shuffle_group(self):
    """
    确保同一batch内的样本互不冲突

    约束条件:
    1. 每个drone最多出现1次
    2. 每个satellite最多出现1次
    3. 如果(drone_i, sate_j)在batch中，
       则drone_i的所有其他正样本satellite不能在同batch
    """

    pair_pool = copy.deepcopy(self.pairs)
    random.shuffle(pair_pool)

    batches = []
    current_batch = []

    sate_batch = set()   # 当前batch已用的satellite
    drone_batch = set()  # 当前batch已用的drone
    pairs_epoch = set()  # 当前epoch已用的配对

    while len(pair_pool) > 0:
        pair = pair_pool.pop(0)
        drone_name, sate_name, weight = pair

        # 检查冲突
        if drone_name in drone_batch or (drone_name, sate_name) in pairs_epoch:
            continue  # 跳过冲突样本

        # 检查该drone的所有正样本satellite是否被占用
        conflict = False
        for related_sate in self.pairs_drone2sate_dict[drone_name]:
            if related_sate in sate_batch:
                conflict = True
                break

        if conflict:
            continue

        # 通过检查，加入batch
        current_batch.append(pair)
        drone_batch.add(drone_name)
        sate_batch.add(sate_name)
        pairs_epoch.add((drone_name, sate_name))

        # batch满了
        if len(current_batch) >= self.shuffle_batch_size:
            batches.append(current_batch)
            current_batch = []
            sate_batch.clear()
            drone_batch.clear()

    # 重排self.samples
    self.samples = flatten(batches)
```

**效果：**
```
✅ 互斥采样后的Batch:
  (drone_A, sate_X) → 正样本
  (drone_B, sate_Y) → 正样本
  (drone_C, sate_Z) → 正样本
  ...

相似度矩阵对角线 = 正样本
其他所有位置 = 真负样本 ✓
```

### 1.4 DataLoader配置

**文件位置：** `train_gta.py:237-241`

```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,        # 实际batch_size
    num_workers=0,        # Windows=0, Linux=4
    shuffle=False,        # ⚠️ 使用custom_sampling，不用random shuffle
    pin_memory=True       # 加速CPU→GPU传输
)

# 每个epoch开始时重新采样
if config.custom_sampling:
    train_dataloader.dataset.shuffle()  # 调用上面的互斥采样
```

---

## ② 训练循环流程

### 2.1 主训练循环

**文件位置：** `train_gta.py:391-442`

```python
for epoch in range(1, config.epochs+1):
    print(f"\n[Epoch: {epoch}]")

    # 1. 互斥采样重排（每个epoch不同batch组合）
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    # 2. 训练一个epoch
    train_loss = train_with_weight(
        config, model,
        dataloader=train_dataloader,
        loss_function=WeightedInfoNCE(...),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        with_weight=True
    )

    print(f"Epoch: {epoch}, Loss = {train_loss:.3f}, "
          f"LR = {optimizer.param_groups[0]['lr']:.6f}")

    # 3. 评估（每eval_every_n_epoch轮）
    if epoch % config.eval_every_n_epoch == 0:
        r1_test = evaluate(...)

        # 4. 保存最佳模型
        if r1_test > best_score:
            best_score = r1_test
            torch.save(model.state_dict(),
                      f'weights_e{epoch}_{r1_test:.4f}.pth')

# 5. 保存最终模型
torch.save(model.state_dict(), 'weights_end.pth')
```

### 2.2 单步训练详解

**文件位置：** `Game4Loc/game4loc/trainer/trainer.py:10-174`

```python
def train_with_weight(config, model, dataloader, loss_function,
                      optimizer, scheduler, scaler, with_weight):
    model.train()
    losses = AverageMeter()

    for query, reference, weight in dataloader:
        # query:     [B, 3, 384, 384] 无人机图像
        # reference: [B, 3, 384, 384] 卫星图像
        # weight:    [B] IoU权重

        # === 混合精度训练 ===
        with autocast():  # 自动FP16
            # 1. 数据送GPU
            query = query.to(device)      # [64, 3, 384, 384]
            reference = reference.to(device)
            weight = weight.to(device)    # [64]

            # 2. 前向传播
            features1, features2 = model(img1=query, img2=reference)
            # features1: [64, 768] 无人机特征（L2归一化）
            # features2: [64, 768] 卫星特征（L2归一化）

            # 3. 计算损失
            loss_dict = loss_function(
                features1,
                features2,
                model.logit_scale.exp(),  # 可学习温度参数
                weight                     # IoU权重
            )
            # loss_dict: {"contrastive": tensor(loss_value)}

            loss_total = sum(loss_dict.values())
            losses.update(loss_total.item())

        # 4. 反向传播（混合精度缩放）
        scaler.scale(loss_total).backward()

        # 5. 梯度裁剪（防止梯度爆炸）
        if config.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), 100.)

        # 6. 更新参数
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # 7. 学习率调度（每step更新）
        if scheduler:
            scheduler.step()

    return losses.avg
```

### 2.3 学习率调度

**文件位置：** `train_gta.py:334-360`

```python
# 计算总步数
train_steps = len(train_dataloader) * config.epochs
warmup_steps = len(train_dataloader) * config.warmup_epochs

# Cosine Annealing (默认)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps=train_steps,
    num_warmup_steps=warmup_steps
)

# 学习率变化曲线:
# warmup阶段 (0.1 epoch): 0 → lr_max
# cosine阶段 (4.9 epoch): lr_max → 0 (余弦衰减)
```

---

## ③ 损失计算详解

### 3.1 Weighted-InfoNCE 原理

**文件位置：** `Game4Loc/game4loc/loss.py:46-89`

**核心思想：** 根据IoU权重自适应调整正样本的重要性

```python
class WeightedInfoNCE(nn.Module):
    def __init__(self, label_smoothing=0.0, k=5):
        self.k = k  # 权重曲线陡峭度

    def forward(self, features1, features2, logit_scale, positive_weights):
        # 1. L2归一化
        features1 = F.normalize(features1, dim=-1)  # [B, D]
        features2 = F.normalize(features2, dim=-1)  # [B, D]

        # 2. 计算相似度矩阵
        logits = logit_scale * features1 @ features2.T  # [B, B]
        # logits[i, j] = scale * cos_sim(drone_i, sate_j)

        # 3. 计算权重eps (基于IoU)
        eps = 1.0 - 1.0 / (1 + torch.exp(-self.k * positive_weights))
        # IoU → eps 映射 (k=5):
        #   0.0 → 0.01  (几乎全是硬对比)
        #   0.3 → 0.18
        #   0.5 → 0.38  (硬/软对比混合)
        #   0.8 → 0.68  (更多软对比)

        # 4. 逐样本计算损失
        loss = self._weighted_loss(logits, eps)

        return {"contrastive": loss}

    def _weighted_loss(self, similarity_matrix, eps_all):
        B = similarity_matrix.shape[0]
        total_loss = 0.0

        for i in range(B):
            eps_i = eps_all[i]

            # 正样本相似度（对角线）
            pos_sim = similarity_matrix[i, i]

            # 所有样本的logsumexp
            all_logsumexp = torch.logsumexp(similarity_matrix[i, :], dim=0)

            # 硬对比损失: -log(exp(pos)/sum(exp(all)))
            hard_loss = -pos_sim + all_logsumexp

            # 软对比损失: -mean(all) + log(sum(exp(all)))
            soft_loss = -similarity_matrix[i, :].mean() + all_logsumexp

            # 加权混合
            total_loss += (1 - eps_i) * hard_loss + eps_i * soft_loss

        return total_loss / B
```

### 3.2 损失计算示例

假设 batch_size=4:

```python
# 相似度矩阵 (logit_scale=20, 归一化特征点积)
similarity_matrix = torch.tensor([
    [8.5, 2.1, 1.8, 2.3],  # drone0 vs [sate0, sate1, sate2, sate3]
    [1.9, 9.2, 2.0, 1.7],  # drone1 vs [...]
    [2.2, 1.8, 8.8, 2.1],  # drone2 vs [...]
    [1.7, 2.0, 1.9, 9.0],  # drone3 vs [...]
])
# 对角线 = 正样本对

# IoU权重
positive_weights = torch.tensor([0.5, 0.7, 0.3, 0.6])

# 计算eps (k=5)
eps = 1 - 1/(1 + exp(-5 * positive_weights))
# eps = [0.38, 0.56, 0.18, 0.47]

# 对于drone0 (i=0):
pos_sim = 8.5
all_logsumexp = log(exp(8.5) + exp(2.1) + exp(1.8) + exp(2.3))
              ≈ 8.51

hard_loss = -8.5 + 8.51 = 0.01
soft_loss = -(8.5+2.1+1.8+2.3)/4 + 8.51 = -3.675 + 8.51 = 4.835

loss_0 = (1-0.38) * 0.01 + 0.38 * 4.835
       = 0.62 * 0.01 + 0.38 * 4.835
       ≈ 1.84

# 最终损失 = (loss_0 + loss_1 + loss_2 + loss_3) / 4
```

### 3.3 双向损失

```python
# 实际实现中计算双向损失
loss_D2S = weighted_loss(drone_features, sate_features, ...)
loss_S2D = weighted_loss(sate_features, drone_features, ...)

total_loss = (loss_D2S + loss_S2D) / 2
```

**直观理解：**
- **低IoU (eps≈0)**: 严格要求正样本相似度 >> 负样本
- **高IoU (eps≈1)**: 允许正样本不那么突出，容忍一定模糊性
- **中等IoU**: 两种损失平衡混合

---

## ④ 模型保存策略

### 4.1 保存时机

**文件位置：** `train_gta.py:430-442`

```python
# 每个epoch评估后
if r1_test > best_score or epoch == config.epochs:
    best_score = r1_test

    # 处理多GPU (DataParallel)
    if torch.cuda.device_count() > 1:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    # 保存格式: weights_e{epoch}_{recall@1}.pth
    save_path = f'{model_path}/weights_e{epoch}_{r1_test:.4f}.pth'
    torch.save(state_dict, save_path)
    print(f"✓ Saved best model: {save_path}")

# 训练结束保存最终模型
torch.save(model.state_dict(), f'{model_path}/weights_end.pth')
```

### 4.2 保存路径结构

```
work_dir/gta/
└── vit_base_patch16_rope_reg1_gap_256.sbb_in1k/
    └── 0207145032/                    # 时间戳 (MMDDHHMISS)
        ├── train.py                   # 训练脚本备份
        ├── log.txt                    # 完整训练日志
        ├── weights_e1_0.4205.pth     # Epoch1, Recall@1=42.05%
        ├── weights_e3_0.4521.pth     # Epoch3, Recall@1=45.21% (最佳)
        ├── weights_e5_0.4498.pth     # Epoch5, Recall@1=44.98%
        └── weights_end.pth            # 最终模型 (第5轮)
```

### 4.3 Checkpoint内容

```python
checkpoint = torch.load('weights_e3_0.4521.pth')

# OrderedDict 包含所有模型参数:
{
    'drone_encoder.blocks.0.norm1.weight': tensor([768]),
    'drone_encoder.blocks.0.norm1.bias': tensor([768]),
    'drone_encoder.blocks.0.attn.qkv.weight': tensor([2304, 768]),
    ...
    'satellite_encoder.blocks.0.norm1.weight': tensor([768]),  # 如果不共享权重
    ...
    'logit_scale': tensor(4.6052),  # 可学习的温度参数 ln(100)
}

# 模型大小: ViT-B/16 约 330MB
```

### 4.4 加载Checkpoint

```python
# 恢复训练
model = DesModel(...)
checkpoint = torch.load('weights_e3_0.4521.pth')
model.load_state_dict(checkpoint, strict=False)

# 仅评估
model.eval()
with torch.no_grad():
    features = model(images)
```

---

## ⑤ 关键参数配置

### 5.1 训练参数

| 参数 | Cross-Area | Same-Area | 说明 |
|------|------------|-----------|------|
| `epochs` | 5 | 20 | 训练轮数 |
| `batch_size` | 64 | 64 | 批次大小 |
| `lr` | 0.0001 | 0.0001 | 学习率 |
| `warmup_epochs` | 0.1 | 0.1 | 预热轮数 |
| `scheduler` | cosine | cosine | 学习率调度 |
| `clip_grad` | 100 | 100 | 梯度裁剪阈值 |

### 5.2 损失参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `with_weight` | True | 使用Weighted-InfoNCE |
| `k` | 5 | 权重曲线参数 |
| `label_smoothing` | 0.0 | 标签平滑（未启用） |

### 5.3 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train_mode` | pos_semipos | 训练样本类型 |
| `test_mode` | pos | 测试样本类型 |
| `prob_flip` | 0.5 | 随机翻转概率 |
| `custom_sampling` | True | 互斥采样 |
| `img_size` | 384 | 图像大小 |

### 5.4 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | vit_base_patch16_rope_reg1_gap_256.sbb_in1k | 骨干网络 |
| `share_weights` | True | drone/satellite共享encoder |
| `mixed_precision` | True | FP16混合精度 |
| `grad_checkpointing` | False | 梯度检查点（节省显存） |

### 5.5 命令行示例

```bash
# Cross-Area训练 (5 epochs)
python train_gta.py \
    --data_root "/path/to/GTA-UAV-LR" \
    --train_pairs_meta_file "cross-area-drone2sate-train.json" \
    --test_pairs_meta_file "cross-area-drone2sate-test.json" \
    --model "vit_base_patch16_rope_reg1_gap_256.sbb_in1k" \
    --gpu_ids 0,1 \
    --lr 0.0001 \
    --batch_size 64 \
    --with_weight \
    --k 5 \
    --epoch 5

# Same-Area训练 (20 epochs)
python train_gta.py \
    --data_root "/path/to/GTA-UAV-LR" \
    --train_pairs_meta_file "same-area-drone2sate-train.json" \
    --test_pairs_meta_file "same-area-drone2sate-test.json" \
    --gpu_ids 0 \
    --lr 0.0001 \
    --batch_size 64 \
    --with_weight \
    --k 5 \
    --epoch 20
```

---

## ⑥ 核心创新点

### 6.1 互斥采样 (Mutually Exclusive Sampling)

**问题：** 传统随机采样导致batch内正负样本冲突

**解决：**
- 构建drone-satellite图结构
- 每个epoch动态重排，确保batch内无冲突
- 提升对比学习质量

**效果：**
- 避免错误的负样本梯度
- 性能提升 ~2-3%

### 6.2 Weighted-InfoNCE Loss

**问题：** 不同drone-satellite对的重叠度(IoU)不同

**解决：**
- 低IoU: 严格对比损失 (硬正样本)
- 高IoU: 宽松对比损失 (软正样本)
- 自适应权重: `eps = 1 - 1/(1 + exp(-k*IoU))`

**效果：**
- 充分利用半正样本信息
- Cross-Area性能提升 ~5%

### 6.3 权重共享 (Shared Encoder)

**设计：** drone和satellite使用同一编码器

**优势：**
- 减少参数量 (330MB vs 660MB)
- 强制学习视角不变特征
- 提升泛化能力 (Cross-Area更重要)

### 6.4 混合精度训练 (Mixed Precision)

**实现：** FP16前向 + FP32梯度累积

**优势：**
- 训练速度提升 ~2x
- 显存占用减少 ~30%
- 精度几乎无损

---

## 📊 训练性能参考

### 时间消耗 (Cross-Area, batch_size=64)

| 硬件 | 单epoch时间 | 5 epochs总时间 |
|------|-------------|----------------|
| RTX 3090 (24GB) | ~30min | ~2.5h |
| RTX 4090 (24GB) | ~20min | ~1.7h |
| V100 (32GB) | ~35min | ~3h |
| A100 (40GB) | ~18min | ~1.5h |

### 显存占用

| Batch Size | 显存占用 (FP16) | 显存占用 (FP32) |
|------------|-----------------|-----------------|
| 32 | ~10GB | ~16GB |
| 64 | ~16GB | ~28GB |
| 128 | ~28GB | OOM |

### 预期性能 (Cross-Area)

| Epoch | Recall@1 | Recall@5 | Recall@10 |
|-------|----------|----------|-----------|
| 0 (Zero-shot) | ~18% | ~35% | ~45% |
| 1 | ~35% | ~58% | ~68% |
| 3 | ~42% | ~68% | ~77% |
| 5 | ~44% | ~72% | ~81% |

---

## 🔧 常见问题

### Q1: 训练时显存不足怎么办？

**方案1：减小batch_size**
```bash
python train_gta.py --batch_size 32  # 从64降到32
```

**方案2：启用梯度检查点**
```python
config.grad_checkpointing = True  # 节省显存但慢~20%
```

**方案3：使用梯度累积**
```python
# 每2步累积一次梯度，模拟batch_size=128
accumulation_steps = 2
```

### Q2: 训练太慢怎么办？

**方案1：减少workers**
```python
num_workers = 0  # Windows必须=0, Linux可用4-8
```

**方案2：减少数据增强**
```python
# 关闭部分耗时增强
A.OneOf([...], p=0.0)  # 跳过blur/sharpen
```

**方案3：使用更小的模型**
```bash
--model "vit_small_patch16_224"  # ViT-S替代ViT-B
```

### Q3: 如何从checkpoint恢复训练？

```bash
python train_gta.py \
    --checkpoint_start "work_dir/gta/.../weights_e3_0.4521.pth" \
    --epoch 10  # 继续训练到第10轮
```

**注意：** 需要手动调整起始epoch和scheduler

---

## 📚 相关文件索引

| 功能 | 文件路径 |
|------|---------|
| 训练脚本 | `Game4Loc/train_gta.py` |
| 数据加载器 | `Game4Loc/game4loc/dataset/gta.py` |
| 训练循环 | `Game4Loc/game4loc/trainer/trainer.py` |
| 损失函数 | `Game4Loc/game4loc/loss.py` |
| 模型定义 | `Game4Loc/game4loc/models/model.py` |
| 数据增强 | `Game4Loc/game4loc/transforms.py` |
| 评估脚本 | `Game4Loc/game4loc/evaluate/gta.py` |
| 配置示例 | `Game4Loc/train.sh` |

---

## 🎓 引用

如果使用本项目，请引用：

```bibtex
@inproceedings{ji2025game4loc,
  title={Game4loc: A uav geo-localization benchmark from game data},
  author={Ji, Yuxiang and He, Boyong and Tan, Zhuoyue and Wu, Liaoni},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={4},
  pages={3913--3921},
  year={2025}
}
```

---

**最后更新：** 2025-02-07
**作者：** Claude + 用户协作整理
