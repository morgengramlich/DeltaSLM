from torch import nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .utils import factorized_dynamic_linear


class DynamicSwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, W_up_base, W_down_base, factors_up, factors_down, layer_idx, context_state=None):
        R_up, C_up, depth_up = factors_up
        R_dn, C_dn, depth_dn = factors_down

        up_proj = factorized_dynamic_linear(x, W_up_base, R_up, C_up, depth_up[:, layer_idx], layer_idx, context_state)
        gate, up = up_proj.chunk(2, dim=-1)
        x = F.silu(gate) * up
        return factorized_dynamic_linear(x, W_down_base, R_dn, C_dn, depth_dn[:, layer_idx], layer_idx, context_state)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.RMSNorm(d_model)
        self.ffn = DynamicSwiGLU()
        self.norm_ff = nn.RMSNorm(d_model)

    def forward(self, x, base_attn, base_ffn, factors_attn, factors_up, factors_down, layer_idx, context_state=None, attn_mask=None):
        x = x + self.dropout(
            self.self_attn(
                self.norm1(x), base_attn, factors_attn, layer_idx, context_state, is_causal=True, attn_mask=attn_mask
            )
        )
        x = x + self.dropout(
            self.ffn(
                self.norm_ff(x), base_ffn[0], base_ffn[1], factors_up, factors_down, layer_idx, context_state
            )
        )
        return x
