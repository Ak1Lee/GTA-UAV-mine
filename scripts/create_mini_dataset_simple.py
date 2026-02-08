#!/usr/bin/env python3
"""
简单版迷你数据集创建脚本
只创建JSON索引，不复制图像
"""

import json
import random
import os

def create_mini_json(data_root, json_file, output_file, ratio, seed=42):
    """创建迷你JSON索引"""
    random.seed(seed)

    # 读取原始JSON
    with open(os.path.join(data_root, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机采样
    num_total = len(data)
    num_sample = max(1, int(num_total * ratio))
    data_mini = random.sample(data, num_sample)

    print(f"  {json_file}:")
    print(f"    Original: {num_total:,} samples")
    print(f"    Sampled:  {num_sample:,} samples ({ratio*100:.1f}%)")

    # 保存（带缩进格式化）
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_mini, f, indent=2, ensure_ascii=False)

    print(f"    [OK] Saved: {output_file}\n")

    return num_sample


if __name__ == '__main__':
    DATA_ROOT = r"D:\Code\PythonProject\GTA-UAV\Game4Loc\game4loc\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu"

    ratios = [0.1, 0.05, 0.01]

    for ratio in ratios:
        ratio_str = f"{int(ratio*100)}p"
        output_dir = f"data/GTA-UAV-Mini-{ratio_str}"

        print(f"\n{'='*60}")
        print(f"Creating {ratio_str} Mini Dataset ({ratio*100:.0f}%)")
        print(f"{'='*60}\n")

        # 训练集
        create_mini_json(
            DATA_ROOT,
            "cross-area-drone2sate-train.json",
            f"{output_dir}/mini-cross-area-drone2sate-train.json",
            ratio
        )

        # 测试集
        create_mini_json(
            DATA_ROOT,
            "cross-area-drone2sate-test.json",
            f"{output_dir}/mini-cross-area-drone2sate-test.json",
            ratio
        )

        print(f"[OK] {ratio_str} Mini Dataset Created!")
        print(f"Location: {output_dir}")
        print(f"\nUsage:")
        print(f"  python Game4Loc/train_gta.py \\")
        print(f"      --data_root \"{DATA_ROOT}\" \\")
        print(f"      --train_pairs_meta_file \"{output_dir}/mini-cross-area-drone2sate-train.json\" \\")
        print(f"      --test_pairs_meta_file \"{output_dir}/mini-cross-area-drone2sate-test.json\" \\")
        print(f"      --batch_size 64 --epoch 5\n")

    print("\n" + "="*60)
    print("All Mini Datasets Created Successfully!")
    print("="*60)
