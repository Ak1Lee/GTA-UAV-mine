"""
Online Graph Creation (OGC) + Greedy Weighted Sampling (GWS)
参考 SAGE 论文，为 GTA-UAV 跨视角地理定位任务实现难样本挖掘采样策略。

核心思想：
1. 每个 epoch 开始时，用当前模型提取所有 satellite 特征
2. 构建亲和度图 (视觉相似度 / IOU×视觉相似度)
3. 贪心采样最难区分的 satellite 组成 clique → 构造 hard negative batch
4. 映射 satellite clique → (drone, satellite, weight) training pairs (带 MES 约束)
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
    # Phase 4: 映射 satellite clique → training pairs (带 MES)
    # ------------------------------------------------------------------

    def _map_cliques_to_pairs(self, cliques, dataset):
        """
        将 satellite clique 映射为 (drone, satellite, weight) 训练 pairs。
        在每个 batch 内施加 MES 约束。

        Args:
            cliques: list of list[int] (satellite 索引)
            dataset: GTADatasetTrain

        Returns:
            hard_samples: list of (drone_path, sate_path, weight)
        """
        hard_samples = []

        for clique in cliques:
            batch_pairs = []
            used_drones = set()
            used_sates = set()

            for sate_idx in clique:
                sate_name = self.sate_names[sate_idx]
                candidate_pairs = list(self.sate_to_pairs.get(sate_name, []))
                random.shuffle(candidate_pairs)

                for pair in candidate_pairs:
                    drone_path, sate_path, weight = pair
                    drone_name = os.path.basename(drone_path)

                    # MES 检查: drone 不能已在 batch 的排除集中
                    if drone_name in used_drones:
                        continue

                    # MES 检查: satellite 不能已在 batch 的排除集中
                    sate_key = os.path.basename(sate_path)
                    if sate_key in used_sates:
                        continue

                    # 通过 MES 检查，加入 batch
                    batch_pairs.append(pair)

                    # 更新排除集 (同 shuffle() 逻辑)
                    drone2sates = dataset.pairs_drone2sate_dict.get(drone_name, [])
                    for s in drone2sates:
                        used_sates.add(s)
                    sate2drones = dataset.pairs_sate2drone_dict.get(sate_key, [])
                    for d in sate2drones:
                        used_drones.add(d)
                    break  # 已选中一个 pair，处理下一个 satellite

            hard_samples.extend(batch_pairs)

        return hard_samples

    # ------------------------------------------------------------------
    # Phase 5: 主入口
    # ------------------------------------------------------------------

    def reorder_samples(self, dataset):
        """
        主入口，替代 dataset.shuffle()。

        流程:
        1. 提取 satellite 特征
        2. 构建亲和度图
        3. GWS 贪心采样 clique
        4. 映射 clique → pairs (带 MES)
        5. 与 random MES 样本混合
        6. 写入 dataset.samples
        """
        print("\nOGC Sampling (mode={}, hard_ratio={:.0%}):".format(self.mode, self.hard_ratio))
        t_total = time.time()

        # Step 1: 特征提取
        t0 = time.time()
        features = self.extract_satellite_features()
        t_feat = time.time() - t0
        print("  Feature extraction: {:.1f}s ({} unique satellites, dim={})".format(
            t_feat, len(self.sate_names), features.shape[1]))

        # Step 2: 构建亲和度图
        t0 = time.time()
        W = self.build_affinity_graph(features)
        t_graph = time.time() - t0
        print("  Graph construction: {:.1f}s (mean affinity={:.4f})".format(
            t_graph, W[W > 0].mean() if (W > 0).any() else 0))

        # Step 3: GWS
        t0 = time.time()
        cliques = self.greedy_weighted_sampling(W)
        t_gws = time.time() - t0
        print("  GWS: {:.1f}s ({} cliques, avg size={:.1f})".format(
            t_gws, len(cliques),
            np.mean([len(c) for c in cliques]) if cliques else 0))

        # Step 4: 映射 satellite clique → training pairs
        hard_samples = self._map_cliques_to_pairs(cliques, dataset)

        # Step 5: 混合 hard + random
        n_target = len(dataset.pairs)
        n_hard = min(len(hard_samples), int(n_target * self.hard_ratio))
        hard_samples = hard_samples[:n_hard]

        # Random MES 部分: 复用 dataset.shuffle() 逻辑
        dataset.shuffle()
        random_samples = dataset.samples

        n_random = n_target - n_hard
        random_samples = random_samples[:n_random]

        # 合并: hard batches 在前 (特征最新鲜), random 在后
        dataset.samples = hard_samples + random_samples

        t_total = time.time() - t_total
        print("  Total: {:.1f}s - {} hard + {} random = {} samples".format(
            t_total, len(hard_samples), len(random_samples), len(dataset.samples)))
        if dataset.samples:
            print("  First: {} | Last: {}".format(
                os.path.basename(dataset.samples[0][1]),
                os.path.basename(dataset.samples[-1][1])))
