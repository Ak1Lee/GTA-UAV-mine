import timm
import torch

# 测试 Medium 版本
model_name = ' vit_medium_patch16_rope_reg1_gap_256.sbb_in1k'

model = timm.create_model(model_name, pretrained=True, num_classes=0)
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# 测试推理
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    out = model(x)
print(f"输出维度: {out.shape}")  # 应该是 (1, 256)