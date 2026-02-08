# 在 AutoDL 上跑 GTA-UAV（不同步大文件）

目标：**代码用 Git 同步，数据集在 AutoDL 上只放一份**，后续改代码只需 `git pull`，不用再传数据。

---

## 1. 保证本地仓库不包含数据集

- 数据集不要放在仓库目录里，或放在已被 `.gitignore` 忽略的路径（如 `data/`、`GTA-UAV-LR` 等）。
- 你当前用 `--data_root "D:\BaiduNetdiskDownload\GTA-UAV-LR\..."` 是**仓库外**的路径，这样就不会被 Git 提交，是对的。
- 若曾误把数据目录加进过 Git，可执行（慎用，会从历史里删大文件）：
  ```bash
  git rm -r --cached data/   # 或你的数据集目录名
  git commit -m "stop tracking data dir"
  ```

---

## 2. 第一次在 AutoDL 上部署

1. **同步代码（不含数据）**
   - 把项目推到 GitHub/Gitee（只会上传代码，数据集已被 ignore）。
   - 在 AutoDL 实例里：
     ```bash
     cd /root/autodl-tmp   # 或你常用的工作目录
     git clone https://github.com/你的用户名/GTA-UAV.git
     cd GTA-UAV/Game4Loc
     ```

2. **数据集只传一次**
   - 用 AutoDL 的「文件」、网盘、或 `scp` 把数据集上传到实例的**固定目录**，例如：
     - `/root/autodl-tmp/GTA-UAV-LR`  
     或  
     - `/root/autodl-fs/GTA-UAV-LR`（持久盘，关机不丢）
   - 不要放在 `GTA-UAV` 仓库目录下，这样以后 `git pull` 不会动到数据。

3. **安装环境并跑训练**
   ```bash
   pip install -r requirements.txt   # 若有
   # 或按你当前环境安装依赖
   python train_gta.py --data_root "/root/autodl-tmp/GTA-UAV-LR" \
     --train_pairs_meta_file "same-area-drone2sate-train.json" \
     --test_pairs_meta_file "same-area-drone2sate-test.json" \
     --gpu_ids 0 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' \
     --lr 0.0001 --batch_size 2
   ```
   把 `--data_root` 换成你实际放数据的路径。

---

## 3. 后续只同步代码（不改数据）

- **本机**：改完代码后
  ```bash
  git add .
  git commit -m "xxx"
  git push
  ```
- **AutoDL**：只拉代码，数据不动
  ```bash
  cd /root/autodl-tmp/GTA-UAV
  git pull
  ```
  然后照常运行 `train_gta.py`，`--data_root` 仍指向你第一次放数据的目录。

这样就不会每次同步都传大文件，数据只在 AutoDL 上保留一份即可。

---

## 4. 可选：用环境变量区分本机 / AutoDL 数据路径

若不想每次改 `--data_root`，可以：

- 在 AutoDL 上写一个启动脚本（例如 `run_autodl.sh`），**不要提交数据集路径**，用环境变量或固定 AutoDL 路径：

  ```bash
  # run_autodl.sh（仅 AutoDL 使用，可加入 .gitignore 或写死 AutoDL 路径）
  export DATA_ROOT=/root/autodl-tmp/GTA-UAV-LR
  python train_gta.py --data_root "$DATA_ROOT" \
    --train_pairs_meta_file "same-area-drone2sate-train.json" \
    --test_pairs_meta_file "same-area-drone2sate-test.json" \
    --gpu_ids 0 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' \
    --lr 0.0001 --batch_size 2
  ```

- 本机继续用本机路径传参即可，例如：
  `--data_root "D:\BaiduNetdiskDownload\GTA-UAV-LR\GTA-UAV-LR-baidu"`。

---

## 小结

| 内容       | 是否用 Git 同步 | 说明 |
|------------|------------------|------|
| 代码       | 是               | `git push` / `git pull` |
| 数据集     | 否               | 本机放本地，AutoDL 上单独传一次并固定路径 |
| 后续改动   | 只同步代码       | 在 AutoDL 上 `git pull` 即可，无需再传数据 |
