#!/usr/bin/env python3
"""
创建GTA-UAV迷你数据集用于快速实验

功能:
- 随机采样训练集和测试集
- 可选复制图像文件或仅创建JSON
- 保持数据结构一致性

使用:
    python scripts/create_mini_dataset.py --ratio 0.1 --copy-images
"""

import json
import random
import os
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path


def create_mini_dataset(
    data_root,
    train_json,
    test_json,
    output_root,
    sample_ratio=0.1,
    copy_images=True,
    seed=42
):
    """
    创建迷你数据集

    Args:
        data_root: 原始数据根目录
        train_json: 训练JSON文件名
        test_json: 测试JSON文件名
        output_root: 输出目录
        sample_ratio: 采样比例 (0.1 = 10%)
        copy_images: 是否复制图像文件
        seed: 随机种子
    """
    random.seed(seed)

    os.makedirs(output_root, exist_ok=True)

    # 1. 处理训练集
    print(f"\n{'='*60}")
    print("Processing Training Set...")
    print(f"{'='*60}")

    train_json_path = os.path.join(data_root, train_json)
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"Training JSON not found: {train_json_path}")

    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 随机采样
    num_train = len(train_data)
    num_sample_train = max(1, int(num_train * sample_ratio))
    train_data_mini = random.sample(train_data, num_sample_train)

    print(f"  Original: {num_train:,} samples")
    print(f"  Sampled:  {num_sample_train:,} samples ({sample_ratio*100:.1f}%)")

    # 保存训练JSON
    output_train_json = os.path.join(output_root, f"mini-{train_json}")
    with open(output_train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data_mini, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved: {output_train_json}")

    # 2. 处理测试集
    print(f"\n{'='*60}")
    print("Processing Test Set...")
    print(f"{'='*60}")

    test_json_path = os.path.join(data_root, test_json)
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Test JSON not found: {test_json_path}")

    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    num_test = len(test_data)
    num_sample_test = max(1, int(num_test * sample_ratio))
    test_data_mini = random.sample(test_data, num_sample_test)

    print(f"  Original: {num_test:,} samples")
    print(f"  Sampled:  {num_sample_test:,} samples ({sample_ratio*100:.1f}%)")

    # 保存测试JSON
    output_test_json = os.path.join(output_root, f"mini-{test_json}")
    with open(output_test_json, 'w', encoding='utf-8') as f:
        json.dump(test_data_mini, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved: {output_test_json}")

    # 3. 复制图像文件（可选）
    if copy_images:
        print(f"\n{'='*60}")
        print("Copying Images...")
        print(f"{'='*60}")

        # 收集所有需要的图像
        drone_imgs = set()
        sate_imgs = set()

        for item in train_data_mini + test_data_mini:
            drone_imgs.add(item['drone_img_name'])

            # 收集所有satellite图像（包括pos和semipos）
            if 'pair_pos_semipos_sate_img_list' in item:
                for sate_img in item['pair_pos_semipos_sate_img_list']:
                    sate_imgs.add(sate_img)
            elif 'pair_pos_sate_img_list' in item:
                for sate_img in item['pair_pos_sate_img_list']:
                    sate_imgs.add(sate_img)

        print(f"  Unique drone images: {len(drone_imgs):,}")
        print(f"  Unique satellite images: {len(sate_imgs):,}")

        # 复制drone图像
        drone_src_dir = os.path.join(data_root, "drone/images")
        drone_dst_dir = os.path.join(output_root, "drone/images")
        os.makedirs(drone_dst_dir, exist_ok=True)

        print(f"\n  Copying drone images from {drone_src_dir}...")
        copied_drone = 0
        missing_drone = 0
        for img_name in tqdm(drone_imgs, desc="  Drone"):
            src = os.path.join(drone_src_dir, img_name)
            dst = os.path.join(drone_dst_dir, img_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_drone += 1
            else:
                missing_drone += 1
                print(f"  [!] Missing: {src}")

        print(f"  [OK] Copied {copied_drone:,} drone images")
        if missing_drone > 0:
            print(f"  [!]  Missing {missing_drone:,} drone images")

        # 复制satellite图像
        sate_src_dir = os.path.join(data_root, "satellite")
        sate_dst_dir = os.path.join(output_root, "satellite")
        os.makedirs(sate_dst_dir, exist_ok=True)

        print(f"\n  Copying satellite images from {sate_src_dir}...")
        copied_sate = 0
        missing_sate = 0
        for img_name in tqdm(sate_imgs, desc="  Satellite"):
            src = os.path.join(sate_src_dir, img_name)
            dst = os.path.join(sate_dst_dir, img_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_sate += 1
            else:
                missing_sate += 1
                print(f"  [!] Missing: {src}")

        print(f"  [OK] Copied {copied_sate:,} satellite images")
        if missing_sate > 0:
            print(f"  [!]  Missing {missing_sate:,} satellite images")

        # 估算磁盘占用
        if copied_drone > 0 and copied_sate > 0:
            # 假设平均每张图像约50KB (LR版本)
            estimated_size_mb = (copied_drone + copied_sate) * 50 / 1024
            print(f"\n  📊 Estimated disk usage: ~{estimated_size_mb:.1f} MB")

    print(f"\n{'='*60}")
    print("✅ Mini Dataset Created Successfully!")
    print(f"{'='*60}")
    print(f"📁 Location: {output_root}")
    print(f"📄 Train JSON: mini-{train_json}")
    print(f"📄 Test JSON:  mini-{test_json}")
    print(f"\n💡 Usage:")
    print(f"  python train_gta.py \\")
    print(f"      --data_root \"{output_root}\" \\")
    print(f"      --train_pairs_meta_file \"mini-{train_json}\" \\")
    print(f"      --test_pairs_meta_file \"mini-{test_json}\" \\")
    print(f"      --batch_size 64 --epoch 5")


def main():
    parser = argparse.ArgumentParser(description="Create mini dataset for GTA-UAV")

    parser.add_argument(
        '--data_root',
        type=str,
        help='Path to original GTA-UAV dataset',
        default=None
    )

    parser.add_argument(
        '--output_root',
        type=str,
        help='Output directory for mini dataset',
        default=None
    )

    parser.add_argument(
        '--train_json',
        type=str,
        help='Training JSON filename',
        default='cross-area-drone2sate-train.json'
    )

    parser.add_argument(
        '--test_json',
        type=str,
        help='Test JSON filename',
        default='cross-area-drone2sate-test.json'
    )

    parser.add_argument(
        '--ratio',
        type=float,
        help='Sampling ratio (0.0-1.0)',
        default=0.1
    )

    parser.add_argument(
        '--no-copy-images',
        action='store_true',
        help='Only create JSON files without copying images'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed',
        default=42
    )

    args = parser.parse_args()

    # 默认路径配置
    if args.data_root is None:
        # 尝试自动检测
        possible_paths = [
            "D:/BaiduNetdiskDownload/GTA-UAV-LR/GTA-UAV-LR-baidu",
            "./data/GTA-UAV-LR",
            "../data/GTA-UAV-LR",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.data_root = path
                print(f"📁 Auto-detected data_root: {path}")
                break

        if args.data_root is None:
            print("[X] Error: --data_root not specified and auto-detection failed")
            print("   Please specify the path to GTA-UAV dataset:")
            print("   python scripts/create_mini_dataset.py --data_root /path/to/GTA-UAV-LR")
            return

    if args.output_root is None:
        ratio_str = f"{int(args.ratio*100)}p"  # 0.1 -> 10p
        args.output_root = f"./data/GTA-UAV-Mini-{ratio_str}"
        print(f"📁 Output directory: {args.output_root}")

    # 验证输入路径存在
    if not os.path.exists(args.data_root):
        print(f"[X] Error: data_root does not exist: {args.data_root}")
        return

    # 执行创建
    try:
        create_mini_dataset(
            data_root=args.data_root,
            train_json=args.train_json,
            test_json=args.test_json,
            output_root=args.output_root,
            sample_ratio=args.ratio,
            copy_images=not args.no_copy_images,
            seed=args.seed
        )
    except Exception as e:
        print(f"\n[X] Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
