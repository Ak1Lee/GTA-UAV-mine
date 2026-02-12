#!/usr/bin/env python
"""
下载 vit_base_patch16_rope_reg1_gap_256 预训练权重到默认路径，供 eva_gta 使用.
来源: HuggingFace timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k
"""
import os
import sys

# 默认保存路径，与 model_gta_vit.build_eva_gta 的 checkpoint_path 一致
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAME4LOC_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_SAVE_DIR = os.path.join(GAME4LOC_ROOT, "pretrained", "vit_base_patch16_rope_reg1_gap_256")
DEFAULT_SAVE_PATH = os.path.join(DEFAULT_SAVE_DIR, "pytorch_model.bin")

# HuggingFace 直链 (timm 官方)
HF_URL = "https://huggingface.co/timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/resolve/main/pytorch_model.bin"


def download_with_urllib(url: str, save_path: str) -> bool:
    """使用 urllib 下载."""
    try:
        from urllib.request import urlopen, Request
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as r:
            total = int(r.headers.get("content-length", 0))
            size = 0
            with open(save_path, "wb") as f:
                chunk = 8192
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    size += len(buf)
                    if total and size % (10 * 1024 * 1024) < chunk:
                        pct = 100 * size / total if total else 0
                        print(f"\r  {pct:.1f}% ({size / 1024 / 1024:.1f} MB)", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"urllib 下载失败: {e}")
        return False


def download_with_requests(url: str, save_path: str) -> bool:
    """使用 requests 下载 (支持断点续传)."""
    try:
        import requests
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        size = 0
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                size += len(chunk)
                if total and size % (10 * 1024 * 1024) < 8192:
                    pct = 100 * size / total if total else 0
                    print(f"\r  {pct:.1f}% ({size / 1024 / 1024:.1f} MB)", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"requests 下载失败: {e}")
        return False


def main():
    save_path = os.path.abspath(
        os.environ.get("EVA_CHECKPOINT_PATH", DEFAULT_SAVE_PATH)
    )
    save_dir = os.path.dirname(save_path)

    if os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"权重已存在: {save_path} ({size_mb:.1f} MB)")
        print("如需重新下载，请先删除该文件")
        return 0

    print(f"下载 vit_base_patch16_rope_reg1_gap_256 预训练权重")
    print(f"  URL: {HF_URL}")
    print(f"  保存: {save_path}")
    os.makedirs(save_dir, exist_ok=True)

    # 优先用 huggingface_hub，其次 requests，最后 urllib
    ok = False
    try:
        from huggingface_hub import hf_hub_download
        print("使用 huggingface_hub 下载...")
        hf_hub_download(
            repo_id="timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
            filename="pytorch_model.bin",
            local_dir=save_dir,
            local_dir_use_symlinks=False,
        )
        ok = os.path.exists(save_path)
    except ImportError:
        pass
    except Exception as e:
        print(f"huggingface_hub: {e}")

    if not ok:
        try:
            import requests
            print("使用 requests 下载...")
            ok = download_with_requests(HF_URL, save_path)
        except ImportError:
            pass

    if not ok:
        print("使用 urllib 下载...")
        ok = download_with_urllib(HF_URL, save_path)

    if ok and os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"下载完成: {save_path} ({size_mb:.1f} MB)")
        return 0
    else:
        print("下载失败，请检查网络或手动下载:")
        print(f"  {HF_URL}")
        print(f"  保存到: {save_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
