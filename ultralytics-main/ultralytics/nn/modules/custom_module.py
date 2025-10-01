import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

__all__ = (
    "MyTransformerBlock",
    "ATFT",
    "Add",
    "Add2",
)

# loweeminh
class MyTransformerBlock(nn.Module):
    def __init__(self, d_model, h, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, dropout=attn_pdrop, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x

# loweeminh
class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_h, output_w):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w

    def forward(self, x):
        B, C, H, W = x.shape

        if H > self.output_h or W > self.output_w:
            stride_h = H // self.output_h
            stride_w = W // self.output_w
            kernel_h = H - (self.output_h - 1) * stride_h
            kernel_w = W - (self.output_w - 1) * stride_w

            return F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
        else:
            return F.interpolate(x, size=(self.output_h, self.output_w), mode='bilinear', align_corners=False)

# loweeminh
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, N, C)
        scale = self.fc(x.transpose(1, 2)).unsqueeze(1)  # (B, 1, C)
        return x * scale
    
# loweeminh
class ATFT(nn.Module):
    def __init__(self, d_model, h=8, block_exp=4, n_layer=2,
                 grid_h=8, grid_w=8, embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2):
        super().__init__()
        self.d_model = d_model
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.avgpool = AdaptiveAvgPool2d(grid_h, grid_w)

        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * grid_h * grid_w, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # SE block for channel-wise attention
        self.se = SEBlock(d_model)

        # Transformer blocks
        self.trans_blocks = nn.Sequential(
            *[MyTransformerBlock(d_model, h, block_exp, attn_pdrop, resid_pdrop) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(embd_pdrop)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        event, rgb = x
        B, C, H, W = event.shape

        event_pooled = self.avgpool(event).flatten(2).transpose(1, 2)  # (B, N, C)
        rgb_pooled = self.avgpool(rgb).flatten(2).transpose(1, 2)      # (B, N, C)

        # Fusion via concatenation
        fused_tokens = torch.cat([event_pooled, rgb_pooled], dim=1)  # (B, 2*N, C)

        # Add positional encoding and apply dropout
        fused_tokens = self.drop(fused_tokens + self.pos_emb)

        # Apply SE block
        fused_tokens = self.se(fused_tokens)

        # Transformer encoder
        x = self.trans_blocks(fused_tokens)
        x = self.ln_f(x)

        # Reshape and interpolate back to original resolution
        x = x.view(B, 2, self.grid_h, self.grid_w, self.d_model).permute(0, 1, 4, 2, 3)  # (B, 2, C, H', W')

        event_out = F.interpolate(x[:, 0], size=(H, W), mode='bilinear', align_corners=False)
        rgb_out = F.interpolate(x[:, 1], size=(H, W), mode='bilinear', align_corners=False)

        return event_out, rgb_out

# loweeminh
class Add(nn.Module):
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])

# loweeminh
class Add2(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return torch.add(x[0], x[1][self.index])