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