"""
GTA-UAV 模型: 在 Eva ViT 最后 4 个 block 插入 DPN, 输出使用 softP, 可冻结 backbone 训练.
- DPN: AvgPool -> Linear -> ReLU -> Linear -> Sigmoid -> Point-wise Power
- softP: L2 Norm -> MLP (Linear-ReLU-Linear) -> Sigmoid -> Residual Connection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .eva import (
    Eva, EvaBlock,
    Mlp, GluMlp, SwiGLU,
    LayerNorm, DropPath,
    PatchEmbed, PatchDropout, RotaryEmbeddingCat,
    use_fused_attn, apply_rot_embed_cat, apply_keep_indices_nlc,
    resample_abs_pos_embed, trunc_normal_, to_2tuple,
)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Callable, Optional, Tuple


# -----------------------------------------------------------------------------
# DPN: Dynamic Power Normalization
# AvgPool -> Linear -> ReLU -> Linear -> Sigmoid -> Point-wise Power
# -----------------------------------------------------------------------------
class DPNModule(nn.Module):
    """Learnable Dynamic Power Normalization.
    Flow: AvgPool(seq) -> Linear -> ReLU -> Linear -> Sigmoid -> per-channel power.
    Power p in (0.5, 2) for stable training; x_out = sign(x) * |x|^p.
    """
    def __init__(self, dim: int, hidden_ratio: float = 0.25):
        super().__init__()
        hidden = max(32, int(dim * hidden_ratio))
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.ReLU(inplace=True)
        # sigmoid output maps to power in [0.5, 2]: p = 0.5 + 1.5 * sigmoid
        self.p_min, self.p_max = 0.5, 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        pooled = x.mean(dim=1)  # (B, D)
        h = self.fc1(pooled)
        h = self.act(h)
        p = self.fc2(h)
        p = torch.sigmoid(p) * (self.p_max - self.p_min) + self.p_min  # (B, D)
        # point-wise power: sign(x) * |x|^p
        p = p.unsqueeze(1)  # (B, 1, D)
        abs_x = x.abs().clamp(min=1e-6)
        out = torch.sign(x) * torch.pow(abs_x, p)
        return out


# -----------------------------------------------------------------------------
# softP
# -----------------------------------------------------------------------------
class SoftPModule(nn.Module):
    """
    SAGE Soft Probing (SoftP) Module
    Paper Ref: Eq. (1), (2), (3)
    Logic: L2 Norm (Scalar) -> MLP (Scalar Output) -> Reweight (1 + beta) * x
    """
    def __init__(self, dim: int, hidden_dim: int = 64, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        
        # 论文 cite: 269 "compact predictor (a two-layer MLP)"
        # 输入是 1 (scalar L2 norm)，输出是 1 (scalar weight)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        
        # 1. 计算每个 patch 的 L2 范数 (Scalar)
        # cite: 269 "s_i = ||X_i||_2"
        # 加上 clamp 防止 0 梯度
        norms = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-6)  # (B, N, 1)
        
        # 2. 预测加权系数 beta
        # cite: 270 "beta_i = alpha * sigmoid(phi(s_i))"
        # cite: 272 "0 <= beta_i <= alpha"
        beta = self.alpha * self.mlp(norms)  # (B, N, 1)
        
        # 3. 残差加权 (Rescaling)
        # cite: 274 "X_bar = (1 + beta_i) * X_i"
        # 这是一个广播乘法，不改变特征方向，只放大模长
        out = x * (1 + beta)
        
        return out


# -----------------------------------------------------------------------------
# EvaBlockWithDPN: 在 attn 与 mlp 之间插入 DPN
# -----------------------------------------------------------------------------
class EvaBlockWithDPN(nn.Module):
    """EvaBlock with DPN inserted between attention and MLP residuals."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 1,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        attn_head_dim: Optional[int] = None,
    ):
        super().__init__()
        from .eva import EvaAttention
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens, attn_drop=attn_drop, proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.dpn = DPNModule(dim, hidden_dim=64)

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                self.mlp = SwiGLU(in_features=dim, hidden_features=hidden_features,
                                  norm_layer=norm_layer if scale_mlp else None, drop=proj_drop)
            else:
                self.mlp = GluMlp(in_features=dim, hidden_features=hidden_features * 2,
                                 norm_layer=norm_layer if scale_mlp else None, act_layer=nn.SiLU,
                                 gate_last=False, drop=proj_drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=hidden_features,
                          act_layer=act_layer, norm_layer=norm_layer if scale_mlp else None, drop=proj_drop)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, rgb, d=None, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            rgb = rgb + self.drop_path1(self.attn(self.norm1(rgb), rope=rope, attn_mask=attn_mask))
        else:
            rgb = rgb + self.drop_path1(self.gamma_1 * self.attn(self.norm1(rgb), rope=rope, attn_mask=attn_mask))
        rgb = self.dpn(rgb)  # DPN
        if self.gamma_2 is None:
            rgb = rgb + self.drop_path2(self.mlp(self.norm2(rgb)))
        else:
            rgb = rgb + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(rgb)))
        return rgb


# -----------------------------------------------------------------------------
# EvaGTA: Eva + 最后 4 层 DPN + softP + freeze_backbone
# -----------------------------------------------------------------------------
def build_eva_gta(
    pretrained: bool = True,
    checkpoint_path: str = './pretrained/vit_base_patch16_rope_reg1_gap_256/pytorch_model.bin',
    in_chans: int = 3,
    img_size: int = 384,
    global_pool: str = 'avg',
    dpn_layers: int = 4,
    **kwargs,
):
    """构建 EvaGTA 模型. 最后 dpn_layers 个 block 使用 DPN, 输出前应用 softP."""
    from .eva import vit_base_patch16_rope_reg1_gap_256
    base = vit_base_patch16_rope_reg1_gap_256(pretrained=False, in_chans=in_chans, img_size=img_size, global_pool=global_pool)
    model = EvaGTA(base, dpn_layers=dpn_layers)
    if pretrained and checkpoint_path:
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            # 兼容 state_dict 嵌套 (model/model_ema/state_dict)
            ckpt = ckpt.get('model_ema', ckpt)
            ckpt = ckpt.get('model', ckpt)
            ckpt = ckpt.get('state_dict', ckpt)
            model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            print(f"Warning: could not load checkpoint from {checkpoint_path}: {e}")
    return model


def _get_drop_path(blk) -> float:
    """从 DropPath 或 Identity 获取 drop path rate."""
    dp = getattr(blk, 'drop_path1', None)
    if dp is None:
        return 0.
    return getattr(dp, 'drop_prob', getattr(dp, 'p', 0.))


class EvaGTA(nn.Module):
    """Eva ViT with DPN in last N blocks, softP before pooling, and freeze_backbone support."""
    def __init__(self, base: Eva, dpn_layers: int = 4):
        super().__init__()
        self._base_ref = base  # 保留引用用于 _pos_embed
        self.dpn_layers = dpn_layers
        depth = len(base.blocks)
        assert dpn_layers <= depth, "dpn_layers must be <= depth"
        # 复制 base 的所有组件，保证 state_dict 兼容
        self.patch_embed = base.patch_embed
        self.cls_token = base.cls_token
        self.reg_token = base.reg_token
        self.pos_embed = base.pos_embed
        self.pos_drop = base.pos_drop
        self.patch_drop = base.patch_drop
        self.rope = base.rope
        self.num_prefix_tokens = base.num_prefix_tokens
        self.embed_dim = base.embed_dim
        self.num_heads = base.blocks[0].attn.num_heads
        self.global_pool = base.global_pool
        self.fc_norm = base.fc_norm
        self.head_drop = base.head_drop
        self.head = base.head
        self.gem = getattr(base, 'gem', None)
        self.grad_checkpointing = False

        # 替换最后 dpn_layers 个 block
        start_replace = depth - dpn_layers
        new_blocks = list(base.blocks)
        for i in range(start_replace, depth):
            blk = base.blocks[i]
            drop_path_val = _get_drop_path(blk)
            init_val = blk.gamma_1[0].item() if blk.gamma_1 is not None else None
            # 与 base block 的 attn 结构一致 (qkv_fused 时 bias 在 q_bias/v_bias)
            attn = blk.attn
            qkv_fused = attn.qkv is not None
            qkv_bias = attn.q_bias is not None if qkv_fused else (attn.q_proj.bias is not None if attn.q_proj is not None else True)
            new_blk = EvaBlockWithDPN(
                dim=base.embed_dim,
                num_heads=self.num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=4.,
                swiglu_mlp=isinstance(blk.mlp, (GluMlp, SwiGLU)),
                scale_mlp='norm.weight' in blk.mlp.state_dict(),
                scale_attn_inner=False,
                num_prefix_tokens=base.num_prefix_tokens,
                proj_drop=0.,
                attn_drop=0.,
                drop_path=drop_path_val,
                init_values=init_val,
                norm_layer=type(blk.norm1),
            )
            new_blk.norm1.load_state_dict(blk.norm1.state_dict())
            new_blk.attn.load_state_dict(blk.attn.state_dict())
            new_blk.norm2.load_state_dict(blk.norm2.state_dict())
            new_blk.mlp.load_state_dict(blk.mlp.state_dict())
            if blk.gamma_1 is not None:
                new_blk.gamma_1.data.copy_(blk.gamma_1.data)
            if blk.gamma_2 is not None:
                new_blk.gamma_2.data.copy_(blk.gamma_2.data)
            new_blocks[i] = new_blk
        self.blocks = nn.ModuleList(new_blocks)
        self.norm = base.norm
        self.softp = SoftPModule(base.embed_dim, hidden_ratio=0.5)

    def freeze_backbone(self):
        """冻结主干，仅训练 DPN 与 softP."""
        for n, p in self.named_parameters():
            if 'dpn' in n or 'softp' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def forward_features(self, rgb, d=None, intermediate=False):
        from torch.utils.checkpoint import checkpoint
        rgb = self.patch_embed(rgb)
        rgb, rot_pos_embed = self._base_ref._pos_embed(rgb)
        x_intermediate = []
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                rgb = checkpoint(blk, rgb, d, rope=rot_pos_embed, use_reentrant=False)
            else:
                rgb = blk(rgb, d, rope=rot_pos_embed)
            if intermediate:
                x_intermediate.append(rgb)
        x = self.norm(rgb)
        x = self.softp(x)
        if intermediate:
            return x, x_intermediate
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool == 'max':
            x = x[:, self.num_prefix_tokens:].max(dim=1)[0]
        elif self.global_pool == 'gem':
            x = self.gem(x[:, self.num_prefix_tokens:])
        elif self.global_pool == 'cls':
            x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, rgb, d=None):
        x = self.forward_features(rgb, d)
        x = self.forward_head(x)
        return x

    def get_config(self):
        return dict(
            input_size=(3, 384, 384),
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )


# -----------------------------------------------------------------------------
# DesModelGTA: 与 DesModel 接口兼容的 GTA 模型
# -----------------------------------------------------------------------------
class DesModelGTA(nn.Module):
    """兼容 DesModel 的 GTA 模型 (DPN + softP, 支持 freeze_backbone)."""
    def __init__(
        self,
        model_name: str = 'eva_gta',
        pretrained: bool = True,
        img_size: int = 384,
        share_weights: bool = True,
        global_pool: str = 'avg',
        dpn_layers: int = 4,
        checkpoint_path: str = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        self.global_pool = global_pool
        ckpt = checkpoint_path or './pretrained/vit_base_patch16_rope_reg1_gap_256/pytorch_model.bin'
        if share_weights:
            self.model = build_eva_gta(
                pretrained=pretrained,
                checkpoint_path=ckpt,
                img_size=img_size,
                global_pool=global_pool or 'avg',
                dpn_layers=dpn_layers,
            )
        else:
            self.model1 = build_eva_gta(pretrained=pretrained, checkpoint_path=ckpt, img_size=img_size,
                                        global_pool=global_pool or 'avg', dpn_layers=dpn_layers)
            self.model2 = build_eva_gta(pretrained=pretrained, checkpoint_path=ckpt, img_size=img_size,
                                        global_pool=global_pool or 'avg', dpn_layers=dpn_layers)
        if freeze_backbone:
            if share_weights:
                self.model.freeze_backbone()
            else:
                self.model1.freeze_backbone()
                self.model2.freeze_backbone()

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self):
        m = self.model if self.share_weights else self.model1
        return m.get_config()

    def set_grad_checkpointing(self, enable: bool = True):
        if self.share_weights:
            self.model.set_grad_checkpointing(enable)
        else:
            self.model1.set_grad_checkpointing(enable)
            self.model2.set_grad_checkpointing(enable)

    def _get_features(self, backbone, img):
        return backbone(img)

    def forward(self, img1=None, img2=None):
        if self.share_weights:
            if img1 is not None and img2 is not None:
                f1 = self._get_features(self.model, img1)
                f2 = self._get_features(self.model, img2)
                return f1, f2
            return self._get_features(self.model, img1 if img1 is not None else img2)
        if img1 is not None and img2 is not None:
            f1 = self._get_features(self.model1, img1)
            f2 = self._get_features(self.model2, img2)
            return f1, f2
        return self._get_features(self.model1 if img1 is not None else self.model2,
                                   img1 if img1 is not None else img2)
