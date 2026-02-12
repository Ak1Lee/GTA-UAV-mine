#!/usr/bin/env python
"""
单步诊断 NaN：跑一个 batch，逐阶段检查 features/loss 是否含 NaN/Inf.
用法: 在 Game4Loc 目录下
  python scripts/debug_nan.py --model eva_gta --batch_size 8
"""
# 解决 Windows 下 OpenMP 重复加载 (libiomp5md.dll already initialized)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game4loc.dataset.gta import GTADatasetTrain, get_transforms
from torch.utils.data import DataLoader


def check_tensor(name, t, desc=""):
    if t is None:
        print(f"  {name}: None")
        return True
    if isinstance(t, dict):
        for k, v in t.items():
            check_tensor(f"{name}.{k}", v, desc)
        return True
    ok = torch.isfinite(t).all().item()
    info = f"  {name}: shape={tuple(t.shape)}"
    if t.numel() > 0:
        info += f" min={t.min().item():.4f} max={t.max().item():.4f} mean={t.float().mean().item():.4f}"
    if not ok:
        nan = torch.isnan(t).sum().item()
        inf = torch.isinf(t).sum().item()
        info += f"  [NaN: {nan}, Inf: {inf}]"
    print(info)
    return ok


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="eva_gta")
    p.add_argument("--data_root", default=None,
                   help="数据根目录，默认尝试 game4loc/dataset/... 或 /root/autodl-tmp/dataset/...")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--no_mixed_precision", action="store_true", help="关闭 AMP 排查")
    p.add_argument("--fake_data", action="store_true", help="用随机张量代替真实数据，快速定位")
    p.add_argument("--trace", action="store_true", help="逐 block 检查，定位 NaN 出现位置")
    args = p.parse_args()

    print("[1/5] 构建模型...")
    # 构建模型
    if args.model == "eva_gta":
        from game4loc.models.model_gta_vit import DesModelGTA
        model = DesModelGTA(model_name="eva_gta", pretrained=True, img_size=384,
                           share_weights=True, global_pool="avg", dpn_layers=4,
                           freeze_backbone=True)
    else:
        from game4loc.models.model import DesModel
        model = DesModel(model_name=args.model, pretrained=True, img_size=384)

    model.train()  # 与训练时一致，便于复现 NaN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print("      模型已加载")

    # 数据
    if args.fake_data:
        print("[2/5] 使用随机数据 (--fake_data)")
        torch.manual_seed(42)
        query = torch.randn(args.batch_size, 3, 384, 384, device=device) * 0.5 + 0.5
        reference = torch.randn(args.batch_size, 3, 384, 384, device=device) * 0.5 + 0.5
        weight = torch.ones(args.batch_size, device=device) * 0.5
    else:
        print("[2/5] 加载数据集...")
        data_root = args.data_root or "game4loc/dataset/GTA-UAV-LR/GTA-UAV-LR-baidu"
        if not os.path.exists(data_root):
            data_root = "/root/autodl-tmp/dataset/GTA-UAV-LR/GTA-UAV-LR-baidu"
        if not os.path.exists(data_root):
            print(f"错误: 数据目录不存在，请用 --data_root 指定 或加 --fake_data")
            sys.exit(1)
        val_t, sat_t, drone_t = get_transforms((384, 384), sat_rot=True)
        ds = GTADatasetTrain(data_root=data_root, pairs_meta_file="cross-area-drone2sate-train.json",
                              transforms_query=drone_t, transforms_gallery=sat_t, group_len=2,
                              prob_flip=0.5, shuffle_batch_size=args.batch_size, mode="pos_semipos", train_ratio=0.01)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        query, reference, weight = next(iter(dl))
        query = query.to(device)
        reference = reference.to(device)
        weight = weight.to(device)

    print("[3/5] 检查输入...")
    print("=" * 50)
    check_tensor("weight", weight)
    check_tensor("query", query)
    check_tensor("reference", reference)

    # 前向 (与训练相同，可选关闭 AMP)
    print("[4/5] 模型前向 (逐阶段检查 NaN)...")
    use_amp = not args.no_mixed_precision and device == "cuda"
    if args.trace and args.model == "eva_gta":
        eva = model.model
        ctx = torch.amp.autocast("cuda") if use_amp else torch.no_grad()
        with ctx:
            x = eva.patch_embed(query)
        check_tensor("1_patch_embed", x)
        if not torch.isfinite(x).all():
            print(">>> NaN 出现在 patch_embed")
        else:
            x, rope = eva._base_ref._pos_embed(x)
            check_tensor("2_pos_embed", x)
            for i, blk in enumerate(eva.blocks):
                with ctx:
                    x = blk(x, None, rope)
                ok = torch.isfinite(x).all().item()
                print(f"  3_block_{i}: {'OK' if ok else 'NaN!'}")
                if not ok:
                    print(f">>> NaN 出现在 block {i} (blocks 8-11 含 DPN)")
                    break
            if torch.isfinite(x).all():
                with ctx:
                    x = eva.norm(x)
                check_tensor("4_norm", x)
                with ctx:
                    x = eva.softp(x)
                check_tensor("5_softp", x)
                with ctx:
                    x = eva.forward_head(x)
                check_tensor("6_head", x)
        features1 = x
        with torch.amp.autocast("cuda") if use_amp else torch.no_grad():
            _, features2 = model(img1=query, img2=reference)
    else:
        if use_amp:
            with torch.amp.autocast("cuda"):
                features1, features2 = model(img1=query, img2=reference)
        else:
            features1, features2 = model(img1=query, img2=reference)
    check_tensor("features1", features1)
    check_tensor("features2", features2)

    logit_scale = model.logit_scale.exp()
    check_tensor("logit_scale", logit_scale)

    # Loss
    print("[5/5] 计算 Loss...")
    from game4loc.loss import WeightedInfoNCE
    loss_fn = WeightedInfoNCE(label_smoothing=0.1, k=5, device=device)
    f1 = features1.detach().clone()
    f2 = features2.detach().clone()
    ls = logit_scale.detach().clone()
    w = weight.detach().clone()

    # 手动算 similarity 看哪一步出问题
    import torch.nn.functional as F
    f1n = F.normalize(f1, dim=-1)
    f2n = F.normalize(f2, dim=-1)
    check_tensor("features1_normalized", f1n)
    check_tensor("features2_normalized", f2n)
    sim = ls * (f1n @ f2n.T)
    check_tensor("similarity_matrix", sim)

    ls_val = model.logit_scale.exp()
    out = loss_fn(features1, features2, ls_val, weight)
    check_tensor("loss", out.get("contrastive"))

    print("\n--- 诊断结束 ---")
    if not torch.isfinite(out.get("contrastive", torch.tensor(0.0))).all():
        print(">>> 发现 NaN/Inf，请根据上面标记的 [NaN/Inf] 定位问题模块")


if __name__ == "__main__":
    main()
