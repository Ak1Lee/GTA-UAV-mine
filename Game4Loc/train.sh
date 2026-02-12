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