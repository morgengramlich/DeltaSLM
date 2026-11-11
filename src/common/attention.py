from torch import nn
import torch.nn.functional as F
from .utils import factorized_dynamic_linear


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer using dynamically generated weights."""
    def __init__(self, d_model, n_heads, qk_norm=True):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.d_k, eps=1e-6, elementwise_affine=False)
            self.k_norm = nn.RMSNorm(self.d_k, eps=1e-6, elementwise_affine=False)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, base_attn, factors_attn, layer_idx,
                context_state=None, is_causal=False, attn_mask=None):
        B, S, E = x.shape
        W_q, W_k, W_v, W_o = base_attn

        R_q, C_q, q_layer_depth = factors_attn[0]
        q_proj = factorized_dynamic_linear(x, W_q, R_q, C_q, q_layer_depth[:, layer_idx], layer_idx, context_state)
        q = q_proj.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        R_k, C_k, k_layer_depth = factors_attn[1]
        k_proj = factorized_dynamic_linear(x, W_k, R_k, C_k, k_layer_depth[:, layer_idx], layer_idx, context_state)
        k = k_proj.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        R_v, C_v, v_layer_depth = factors_attn[2]
        v_proj = factorized_dynamic_linear(x, W_v, R_v, C_v, v_layer_depth[:, layer_idx], layer_idx, context_state)
        v = v_proj.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal and attn_mask is None,
            dropout_p=0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, E)
        R_o, C_o, o_layer_depth = factors_attn[3]
        return factorized_dynamic_linear(out, W_o, R_o, C_o, o_layer_depth[:, layer_idx], layer_idx, context_state)
