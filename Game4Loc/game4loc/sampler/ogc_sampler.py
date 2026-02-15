"""
Online Graph Creation (OGC) + Greedy Weighted Sampling (GWS)
参考 SAGE 论文，为 GTA-UAV 跨视角地理定位任务实现难样本挖掘采样策略。

核心思想 (OGC-Guided MES Shuffle):
1. 每个 epoch 开始时，用当前模型提取所有 satellite 特征
2. 构建亲和度图 (视觉相似度 / IOU×视觉相似度)
3. 贪心采样最难区分的 satellite 组成 clique
4. 按 clique 顺序重排所有 training pairs，传给 MES shuffle
   → MES 逻辑不变，但输入顺序使 hard negatives 自然聚集到同一 batch
   → 所有 pairs 都参与训练，覆盖率 100%
"""

import os
import time
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast


class _SatelliteFeatureDataset(Dataset):
    """轻量 Dataset：包装 satellite 图片路径 + val_transforms，用于干净的特征提取。"""

    def __init__(self, image_paths, transforms):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return img


class OGCSampler:
    """
    SAGE-style Online Graph Creation + Greedy Weighted Sampling.

    用法:
        ogc = OGCSampler(dataset, model, val_transforms, config)
        # 在 epoch 循环中替代 dataset.shuffle():
        ogc.reorder_samples(dataset)
    """

    def __init__(
        self,
        dataset,
        model,
        val_transforms,
        config,
        mode='visual',
        hard_ratio=0.5,
        top_k=100,
        feature_batch_size=128,
    ):
        """
        Args:
            dataset: GTADatasetTrain 实例
            model: DesModelGTA (可能被 DataParallel 包装)
            val_transforms: 无增强的 albumentations transforms
            config: Configuration dataclass (需含 device, batch_size, num_workers)
            mode: 'visual' (仅余弦相似度) | 'visual_iou' (余弦 × tile IOU)
            hard_ratio: OGC hard 样本占总样本的比例 (0~1)
            top_k: 稀疏图每节点保留的 top-k 邻居数
            feature_batch_size: 特征提取时的 batch size
        """
        self.model = model
        self.val_transforms = val_transforms
        self.device = config.device
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.mode = mode
        self.hard_ratio = hard_ratio
        self.top_k = top_k
        self.feature_batch_size = feature_batch_size

        # 预构建 satellite → pairs 的反向索引
        self._build_sate_index(dataset)

    # ------------------------------------------------------------------
    # 内部索引构建
    # ------------------------------------------------------------------

    def _build_sate_index(self, dataset):
        """构建 satellite basename → [(drone_path, sate_path, weight), ...] 映射"""
        self.sate_to_pairs = {}
        self.unique_sate_paths = {}  # basename → full_path

        for pair in dataset.pairs:
            drone_path, sate_path, weight = pair
            sate_name = os.path.basename(sate_path)
            self.sate_to_pairs.setdefault(sate_name, []).append(pair)
            if sate_name not in self.unique_sate_paths:
                self.unique_sate_paths[sate_name] = sate_path

        self.sate_names = list(self.unique_sate_paths.keys())
        self.sate_paths = [self.unique_sate_paths[n] for n in self.sate_names]
        self.sate_name_to_idx = {n: i for i, n in enumerate(self.sate_names)}

    # ------------------------------------------------------------------
    # Phase 1: 特征提取
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_satellite_features(self):
        """
        提取所有 unique satellite 的特征向量。

        Returns:
            features: (N_sate, D) L2 归一化的特征张量 (float32, CPU)
        """
        feat_dataset = _SatelliteFeatureDataset(self.sate_paths, self.val_transforms)
        feat_loader = DataLoader(
            feat_dataset,
            batch_size=self.feature_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.model.eval()
        all_feats = []

        for batch in feat_loader:
            with autocast(device_type='cuda'):
                batch = batch.to(self.device)
                feat = self.model(img1=batch)
                feat = F.normalize(feat, dim=-1)
            all_feats.append(feat.cpu().to(torch.float32))

        features = torch.cat(all_feats, dim=0)  # (N_sate, D)
        self.model.train()
        return features

    # ------------------------------------------------------------------
    # Phase 2: 构建亲和度图
    # ------------------------------------------------------------------

    def build_affinity_graph(self, features):
        """
        构建 satellite 之间的亲和度图。

        Args:
            features: (N_sate, D) 归一化特征

        Returns:
            W: (N_sate, N_sate) numpy 亲和度矩阵
        """
        # 余弦相似度矩阵
        sim_matrix = (features @ features.T).numpy()  # (N, N)
        np.fill_diagonal(sim_matrix, 0.0)  # 排除自身

        if self.mode == 'visual_iou':
            iou_matrix = self._compute_tile_iou_matrix()
            sim_matrix = sim_matrix * iou_matrix

        return sim_matrix

    def _compute_tile_iou_matrix(self):
        """
        从 satellite tile 坐标计算几何 IOU 矩阵。
        (第二阶段实现，当前返回全 1 矩阵作为占位)
        """
        n = len(self.sate_names)
        # TODO: 解析 {zoom}_{offset}_{tile_x}_{tile_y}.png 计算矩形 IOU
        print("  [WARNING] tile IOU not implemented yet, using identity (visual mode fallback)")
        return np.ones((n, n), dtype=np.float32)

    # ------------------------------------------------------------------
    # Phase 3: 贪心加权采样 (GWS)
    # ------------------------------------------------------------------

    def greedy_weighted_sampling(self, W):
        """
        Greedy Weighted Sampling: 贪心提取高亲和度 clique。

        每个 clique 大小 = batch_size，内含视觉上最难区分的 satellite 集合。

        Args:
            W: (N_sate, N_sate) 亲和度矩阵

        Returns:
            cliques: list of list[int], 每个 list 是 satellite 索引
        """
        n = W.shape[0]
        clique_size = self.batch_size
        available = np.ones(n, dtype=bool)
        cliques = []

        while available.sum() >= clique_size:
            # 找种子边：可用节点中权重最大的边
            W_masked = W.copy()
            W_masked[~available] = -np.inf
            W_masked[:, ~available] = -np.inf
            np.fill_diagonal(W_masked, -np.inf)

            # 找最大边
            flat_idx = np.argmax(W_masked)
            seed_i, seed_j = divmod(flat_idx, n)

            if W_masked[seed_i, seed_j] == -np.inf:
                break

            clique = [seed_i, seed_j]
            available[seed_i] = False
            available[seed_j] = False

            # 贪心扩展 clique
            while len(clique) < clique_size and available.any():
                # 向量化计算每个候选与 clique 的总亲和度
                clique_mask = np.zeros(n, dtype=bool)
                clique_mask[clique] = True
                scores = W[:, clique_mask].sum(axis=1)  # (N,)
                scores[~available] = -np.inf
                scores[clique_mask] = -np.inf

                best_c = np.argmax(scores)
                if scores[best_c] == -np.inf:
                    break

                clique.append(best_c)
                available[best_c] = False

            cliques.append(clique)

        return cliques

    # ------------------------------------------------------------------
    # Phase 4: 按 clique 顺序重排 pairs (OGC-Guided MES)
    # ------------------------------------------------------------------

    def _cliques_to_pair_order(self, cliques, dataset):
        """
        将 satellite cliques 转为 pair 排列顺序，供 MES shuffle 消费。

        前 hard_ratio 部分按 clique 分组排列 (视觉相似的 satellite 排在一起),
        剩余部分随机打乱。MES 顺序扫描时，同一 clique 的 pairs 自然进入同一 batch。

        Returns:
            reordered_pairs: list of (drone_path, sate_path, weight)
        """
        # 建立 sate_name → clique_idx 映射
        sate_to_clique = {}
        for cidx, clique in enumerate(cliques):
            for sate_idx in clique:
                sate_to_clique[self.sate_names[sate_idx]] = cidx

        # 按 clique 分组收集 pairs
        clique_groups = {}  # cidx → [pairs]
        non_clique_pairs = []

        for pair in dataset.pairs:
            sate_name = os.path.basename(pair[1])
            cidx = sate_to_clique.get(sate_name)
            if cidx is not None:
                clique_groups.setdefault(cidx, []).append(pair)
            else:
                non_clique_pairs.append(pair)

        # 按 clique 顺序排列，clique 内随机打乱
        clique_ordered_pairs = []
        for cidx in range(len(cliques)):
            group = clique_groups.get(cidx, [])
            random.shuffle(group)
            clique_ordered_pairs.extend(group)

        # hard_ratio 截断
        n_hard_target = int(len(dataset.pairs) * self.hard_ratio)
        hard_part = clique_ordered_pairs[:n_hard_target]

        # 剩余部分随机打乱
        remaining = clique_ordered_pairs[n_hard_target:] + non_clique_pairs
        random.shuffle(remaining)

        n_in_cliques = len(clique_ordered_pairs)
        n_total = len(dataset.pairs)
        print("  Pair reorder: {}/{} in cliques ({:.1%}), hard_target={}".format(
            n_in_cliques, n_total, n_in_cliques / n_total, len(hard_part)))

        return hard_part + remaining

    # ------------------------------------------------------------------
    # Phase 5: 主入口
    # ------------------------------------------------------------------

    def reorder_samples(self, dataset):
        """
        主入口，替代 dataset.shuffle()。

        OGC-Guided MES Shuffle:
        1. 提取 satellite 特征
        2. 构建亲和度图
        3. GWS 贪心采样 satellite cliques
        4. 按 clique 顺序重排 pairs (视觉相似的排在一起)
        5. 将重排后的 pairs 传给 dataset.shuffle(pair_order=...)
           MES 逻辑不变，但输入顺序使 hard negatives 自然聚集到同一 batch
        """
        print("\nOGC-Guided Shuffle (mode={}, hard_ratio={:.0%}):".format(self.mode, self.hard_ratio))
        t_total = time.time()

        # Step 1: 特征提取
        t0 = time.time()
        features = self.extract_satellite_features()
        print("  Feature extraction: {:.1f}s ({} unique satellites, dim={})".format(
            time.time() - t0, len(self.sate_names), features.shape[1]))

        # Step 2: 构建亲和度图
        t0 = time.time()
        W = self.build_affinity_graph(features)
        print("  Graph construction: {:.1f}s (mean affinity={:.4f})".format(
            time.time() - t0, W[W > 0].mean() if (W > 0).any() else 0))

        # Step 3: GWS
        t0 = time.time()
        cliques = self.greedy_weighted_sampling(W)
        print("  GWS: {:.1f}s ({} cliques, avg size={:.1f})".format(
            time.time() - t0, len(cliques),
            np.mean([len(c) for c in cliques]) if cliques else 0))

        # Step 4: 按 clique 顺序重排 pairs
        pair_order = self._cliques_to_pair_order(cliques, dataset)

        # Step 5: 调用 MES shuffle，传入重排后的 pairs 顺序
        dataset.shuffle(pair_order=pair_order)

        print("  OGC-Guided total: {:.1f}s".format(time.time() - t_total))
