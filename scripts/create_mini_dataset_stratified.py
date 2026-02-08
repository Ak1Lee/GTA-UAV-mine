#!/usr/bin/env python3
"""
分层采样版本 - 保持地理分布
按地理位置网格均匀采样，避免某些区域完全没有样本
"""

import json
import random
import os
from collections import defaultdict

def create_stratified_mini(data_root, json_file, output_file, ratio, grid_size=10, seed=42):
    """
    按地理网格分层采样

    Args:
        grid_size: 将地图划分为 grid_size × grid_size 的网格
    """
    random.seed(seed)

    with open(os.path.join(data_root, json_file), 'r') as f:
        data = json.load(f)

    # 1. 按地理位置划分网格
    grid_dict = defaultdict(list)

    for item in data:
        x, y = item['drone_loc_x_y']

        # 计算网格索引（假设地图范围0-6400）
        grid_x = int(x // (6400 / grid_size))
        grid_y = int(y // (11200 / grid_size))
        grid_id = (grid_x, grid_y)

        grid_dict[grid_id].append(item)

    # 2. 每个网格按比例采样
    sampled_data = []
    for grid_id, items in grid_dict.items():
        num_sample = max(1, int(len(items) * ratio))  # 至少采样1个
        sampled = random.sample(items, num_sample)
        sampled_data.extend(sampled)

    print(f"  {json_file}:")
    print(f"    Original: {len(data):,} samples")
    print(f"    Grids: {len(grid_dict)} grids")
    print(f"    Sampled: {len(sampled_data):,} samples ({ratio*100:.1f}%)")
    print(f"    Samples per grid: avg {len(sampled_data)/len(grid_dict):.1f}")

    # 3. 保存
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

    print(f"    [OK] Saved: {output_file}\n")

    return len(sampled_data)


if __name__ == '__main__':
    DATA_ROOT = r"D:\Code\PythonProject\GTA-UAV\Game4Loc\game4loc\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu"

    ratios = [0.1, 0.05]

    for ratio in ratios:
        ratio_str = f"{int(ratio*100)}p"

        print(f"\n{'='*60}")
        print(f"Creating {ratio_str} Stratified Mini Dataset")
        print(f"{'='*60}\n")

        # 训练集
        create_stratified_mini(
            DATA_ROOT,
            "cross-area-drone2sate-train.json",
            f"stratified-mini-cross-area-drone2sate-train-{ratio_str}.json",
            ratio,
            grid_size=10
        )

        # 测试集
        create_stratified_mini(
            DATA_ROOT,
            "cross-area-drone2sate-test.json",
            f"stratified-mini-cross-area-drone2sate-test-{ratio_str}.json",
            ratio,
            grid_size=10
        )

        print(f"[OK] {ratio_str} Stratified Mini Created!")

    print("\n" + "="*60)
    print("Stratified datasets preserve geographic distribution!")
    print("="*60)
