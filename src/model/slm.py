import torch
from torch import nn
import torch.nn.functional as F
from .modulated.loop import LoopBlock


class DeltaSlm(nn.Module):
    def __init__(self, embeddings, n_heads, d_ff, num_layers,
                 expansion_order, state_dim, dropout=0.1):
        super().__init__()
        vocab_size, d_model = embeddings.shape
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.expansion_order = expansion_order

        self.token_emb = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.dropout = nn.Dropout(dropout)

        self.in_block = LoopBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, num_layers=num_layers[0],
            expansion_order=expansion_order, state_dim=state_dim,
            max_mat_cycles=2, max_rc_cycles=24, max_depth_cycles=2,
        )
        self.internal_block = LoopBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, num_layers=num_layers[1],
            expansion_order=expansion_order, state_dim=state_dim,
            max_mat_cycles=2, max_rc_cycles=24, max_depth_cycles=4,
        )
        self.out_block = LoopBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, num_layers=num_layers[2],
            expansion_order=expansion_order, state_dim=state_dim,
            max_mat_cycles=2, max_rc_cycles=24, max_depth_cycles=2,
        )

        self.norm_f = nn.RMSNorm(d_model)

    def forward(self, idx, attn_mask=None):
        # B, T = idx.shape

        x = self.dropout(self.token_emb(idx))
        x = self.in_block(x, attn_mask=attn_mask)
        x = self.internal_block(x, attn_mask=attn_mask)
        x = self.out_block(x, attn_mask=attn_mask)

        x = self.norm_f(x)
        return x @ self.token_emb.weight.T

    def get_param_groups(self, base_lr, coeff_lr, freq_lr, phase_lr):
        groups_in = self.in_block.get_param_groups(base_lr, coeff_lr, freq_lr, phase_lr)
        groups_mid = self.internal_block.get_param_groups(base_lr, coeff_lr, freq_lr, phase_lr)
        groups_out = self.out_block.get_param_groups(base_lr, coeff_lr, freq_lr, phase_lr)

        merged_groups = {}
        for group_list in [groups_in, groups_mid, groups_out]:
            for g in group_list:
                name = g["name"]
                if name not in merged_groups:
                    merged_groups[name] = {
                        "params": [],
                        "lr": g["lr"],
                        "weight_decay": g["weight_decay"],
                        "name": name
                    }
                merged_groups[name]["params"].extend(g["params"])

        merged_groups["base"]["params"].extend(list(self.norm_f.parameters()))
        if self.token_emb.weight.requires_grad:
            merged_groups["base"]["params"].extend(list(self.token_emb.parameters()))

        return list(merged_groups.values())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, max_seq_len=1024, temperature=1.0, top_k=None):
        """Autoregressive generation loop for testing out the model."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_seq_len:]

            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
