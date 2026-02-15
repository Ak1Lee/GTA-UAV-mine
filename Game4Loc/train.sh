#!/bin/sh

# Cross-area setting with weighted-InfoNCE k=5
python train_gta.py --data_root <The directory of the GTA-UAV dataset> --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64

# Cross-area setting with standard InfoNCE
python train_gta.py --data_root <The directory of the GTA-UAV dataset> --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64


# Same-area setting with weighted-InfoNCE k=5
python train_gta.py --data_root <The directory of the GTA-UAV dataset> --train_pairs_meta_file "same-area-drone2sate-train.json" --test_pairs_meta_file "same-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 20 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64

# Same-area setting with standard InfoNCE
python train_gta.py --data_root "D:\BaiduNetdiskDownload\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "same-area-drone2sate-train.json" --test_pairs_meta_file "same-area-drone2sate-test.json" --gpu_ids 0 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 4
python train_gta.py --data_root <The directory of the GTA-UAV dataset> --train_pairs_meta_file "same-area-drone2sate-train.json" --test_pairs_meta_file "same-area-drone2sate-test.json" --gpu_ids 0 --epoch 20 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64



# Or try the performance on the UAV-VisLoc after running preparing script (../scripts/prepare_dataset/visloc.py)
python train_visloc.py --data_root <The directory of the UAV-VisLoc dataset> --train_pairs_meta_file "same-area-drone2sate-train.json" --test_pairs_meta_file "same-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 20 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64

# 小样本
python train_gta.py --data_root "game4loc\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "mini-cross-area-drone2sate-train.json" --test_pairs_meta_file "mini-cross-area-drone2sate-test.json" --gpu_ids 0 --epoch 10 --model "vit_medium_patch16_rope_reg1_gap_256.sbb_in1k" --lr 0.0001 --batch_size 8
Recall@1: 39.8048 - Recall@5: 64.7505 - Recall@10: 72.7766 - Recall@top1: 92.4078 - AP: 51.3155 - SDM@1: 0.7071 - SDM@3: 0.6384 - SDM@5: 0.5978 - Dis@1: 684.6771 - Dis@3: 901.5898 - Dis@5: 1015.6826
ssh -p 40213 root@connect.westc.gpuhub.com


# Cross-area baseline setting with weighted-InfoNCE k=5
python train_gta.py --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 32
------------------------------[Evaluate]------------------------------
Extract Features and Compute Scores:
Processing each query: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9056/9056 [00:19<00:00, 454.19it/s]
Recall@1: 61.2853 - Recall@5: 85.3578 - Recall@10: 89.2005 - Recall@top1: 96.9302 - AP: 71.6573 - SDM@1: 0.8215 - SDM@3: 0.7559 - SDM@5: 0.7088 - Dis@1: 333.0190 - Dis@3: 527.0131 - Dis@5: 662.3857
Recall@1: 62.3785 - Recall@5: 85.6338 - Recall@10: 89.6643 - Recall@top1: 96.8860 - AP: 72.3459 - SDM@1: 0.8234 - SDM@3: 0.7608 - SDM@5: 0.7146 - Dis@1: 338.8207 - Dis@3: 515.9145 - Dis@5: 650.5140
很奇怪啊 不同的epoch会影响结果？

# need to do Cross-area baseline setting with weighted-InfoNCE k=5 with medium model
python train_gta.py --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 5 --model 'vit_medium_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 32

# DINOv2 (timm): base / large / giant, reduce batch_size if OOM
# python train_gta.py --data_root <dataset_dir> --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 5 --model 'vit_base_patch14_dinov2.lvd142m' --lr 0.0001 --batch_size 16
# python train_gta.py ... --model 'vit_large_patch14_dinov2.lvd142e' --batch_size 8
# train dinov2:
python train_gta.py --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 5 --model 'vit_base_patch14_dinov2.lvd142m' --lr 0.0001 --batch_size 32------------------------------[Evaluate]------------------------------
Extract Features and Compute Scores:
Processing each query: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9056/9056 [00:20<00:00, 446.80it/s]
Recall@1: 46.0799 - Recall@5: 75.0994 - Recall@10: 81.6696 - Recall@top1: 94.8101 - AP: 58.1938 - SDM@1: 0.7330 - SDM@3: 0.6659 - SDM@5: 0.6252 - Dis@1: 533.3091 - Dis@3: 758.4565 - Dis@5: 889.2196

# 20 epoch ViT
# need to do Cross-area baseline setting with weighted-InfoNCE k=5 with medium model
python train_gta.py --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 20 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 32
Recall@1: 53.9863 - Recall@5: 79.6489 - Recall@10: 85.2915 - Recall@top1: 96.3560 - AP: 65.0409 - SDM@1: 0.7859 - SDM@3: 0.7221 - SDM@5: 0.6757 - Dis@1: 419.6235 - Dis@3: 620.2992 - Dis@5: 766.6094


# 5 epoch dinov2 gap
python train_gta.py --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 5 --model 'vit_base_patch14_dinov2.lvd142m' --lr 0.0001 --batch_size 32 --global_pool avg
Recall@1: 44.0040 - Recall@5: 73.6749 - Recall@10: 80.9077 - Recall@top1: 95.0640 - AP: 56.5931 - SDM@1: 0.7215 - SDM@3: 0.6571 - SDM@5: 0.6184 - Dis@1: 592.7422 - Dis@3: 799.0147 - Dis@5: 913.0639
# 5 epoch dinov2 gem
python train_gta.py --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 5 --model 'vit_base_patch14_dinov2.lvd142m' --lr 0.0001 --batch_size 32 --global_pool gem
Recall@1: 22.6148 - Recall@5: 47.9240 - Recall@10: 57.6303 - Recall@top1: 87.2571 - AP: 33.8336 - SDM@1: 0.5512 - SDM@3: 0.5144 - SDM@5: 0.4921 - Dis@1: 1056.5646 - Dis@3: 1194.5820 - Dis@5: 1267.7858


# 20 epoch eva_gta 后4层block添加dpn 尾层添加softp
python train_gta.py --model eva_gta --data_root "game4loc\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json"  --test_pairs_meta_file "cross-area-drone2sate-test.json"  --gpu_ids 0 --with_weight --k 5 --epochs 20 --lr 0.0001 --batch_size 32
python train_gta.py --model eva_gta --data_root "game4loc\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json"  --test_pairs_meta_file "cross-area-drone2sate-test.json"  --gpu_ids 0 --with_weight --k 5 --epochs 20 --lr 0.0001 --batch_size 32
python train_gta.py --model eva_gta --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json"  --test_pairs_meta_file "cross-area-drone2sate-test.json"  --gpu_ids 0 --with_weight --k 5 --epochs 20 --lr 0.0001 --batch_size 64
------------------------------[Evaluate]------------------------------
Extract Features and Compute Scores:
Processing each query: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9056/9056 [00:20<00:00, 446.27it/s]
Recall@1: 18.1206 - Recall@5: 40.3048 - Recall@10: 50.6625 - Recall@top1: 82.3322 - AP: 28.4332 - SDM@1: 0.5151 - SDM@3: 0.4847 - SDM@5: 0.4654 - Dis@1: 1119.5372 - Dis@3: 1256.7233 - Dis@5: 1323.3381

python scripts/debug_nan.py --model eva_gta --data_root "game4loc\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --batch_size 4

# 5 epoch eva_gta 0dpn 尾层添加softp 全量微调5 epoch
python train_gta.py --model eva_gta  --dpn_layers 0  --no_freeze_backbone  --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu"   --train_pairs_meta_file "cross-area-drone2sate-train.json"  --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epochs 5 --lr_backbone 0.00008 --lr_extra 0.0005 --batch_size 32
Recall@1: 60.2584 - Recall@5: 83.7898 - Recall@10: 88.3944 - Recall@top1: 96.5216 - AP: 70.4298 - SDM@1: 0.8113 - SDM@3: 0.7502 - SDM@5: 0.7039 - Dis@1: 367.1190 - Dis@3: 543.8847 - Dis@5: 685.1971

python train_gta.py --model eva_gta  --dpn_layers 0  --no_freeze_backbone  --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu"   --train_pairs_meta_file "cross-area-drone2sate-train.json"  --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epochs 5 --lr_backbone 0.0001 --lr_extra 0.0005 --batch_size 32
Recall@1: 58.8339 - Recall@5: 83.1935 - Recall@10: 87.9086 - Recall@top1: 96.4554 - AP: 69.1889 - SDM@1: 0.8046 - SDM@3: 0.7403 - SDM@5: 0.6944 - Dis@1: 397.7988 - Dis@3: 592.1241 - Dis@5: 734.1847
python train_gta.py --model eva_gta  --dpn_layers 0  --no_freeze_backbone  --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu"   --train_pairs_meta_file "cross-area-drone2sate-train.json"  --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epochs 5 --lr_backbone 0.0002 --lr_extra 0.0006 --batch_size 32 --global_pool gem
Processing each query: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9056/9056 [00:20<00:00, 448.86it/s]
Recall@1: 51.0490 - Recall@5: 77.9483 - Recall@10: 83.7898 - Recall@top1: 94.7438 - AP: 62.3341 - SDM@1: 0.7709 - SDM@3: 0.7017 - SDM@5: 0.6561 - Dis@1: 466.2387 - Dis@3: 682.2548 - Dis@5: 816.4153
# gem 效果相当差啊

python train_gta.py \
  --model eva_gta \
  --dpn_layers 0 \
  --no_freeze_backbone \
  --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" \
  --train_pairs_meta_file "cross-area-drone2sate-train.json" \
  --test_pairs_meta_file "cross-area-drone2sate-test.json" \
  --gpu_ids 0 \
  --with_weight \
  --k 5 \
  --epochs 15 \
  --lr_backbone 0.00001 \
  --lr_extra 0.0005 \
  --batch_size 32

  # Baseline (MES only)
python train_gta.py --model eva_gta --epochs 5

# OGC visual, 50% hard
python train_gta.py --model eva_gta --use_ogc --ogc_mode visual --ogc_hard_ratio 0.5 --epochs 5

# OGC 延迟启动 (前2个epoch用MES热身)
python train_gta.py --model eva_gta --use_ogc --ogc_start_epoch 3 --epochs 5

# 先测试一下 小样本
python train_gta.py --data_root "game4loc\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "mini-cross-area-drone2sate-train.json" --test_pairs_meta_file "mini-cross-area-drone2sate-test.json" --gpu_ids 0 --epoch 2 --model "vit_medium_patch16_rope_reg1_gap_256.sbb_in1k" --lr 0.0001 --batch_size 8 --use_ogc --ogc_mode visual --ogc_hard_ratio 0.5
# 能跑 Recall@1: 40.2386 - Recall@5: 66.3774 - Recall@10: 74.4035 - Recall@top1: 91.2148 - AP: 51.6939 - SDM@1: 0.7264 - SDM@3: 0.6576 - SDM@5: 0.6163 - Dis@1: 610.9821 - Dis@3: 851.1047 - Dis@5: 979.4390
# 跑一下baseline+OGC
python train_gta.py --data_root "\root\autodl-tmp\dataset\GTA-UAV-LR\GTA-UAV-LR-baidu" --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0 --with_weight --k 5 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 32 --use_ogc --ogc_mode visual --ogc_hard_ratio 0.5