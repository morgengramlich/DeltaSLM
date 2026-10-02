import torch
from torch import nn
import torch.nn.functional as F
from .generators import LatentAttentionDeltaGenerator, LinearDeltaGenerator
from .basic.decoder import LatentDecoderBlock


class DeltaSlm(nn.Module):
    def __init__(self, embeddings, n_heads, d_c, d_ff, num_layers, max_seq_len,
                 expansion_order, dropout=0.1):
        super().__init__()
        vocab_size, d_model = embeddings.shape
        self.d_model = d_model
        self.d_c = d_c
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.dropout = nn.Dropout(dropout)

        self.W_dq = nn.Parameter(torch.empty(d_c, d_model))
        self.W_dkv = nn.Parameter(torch.empty(d_c, d_model))
        self.W_uq = nn.Parameter(torch.empty(d_model, d_c))
        self.W_uk = nn.Parameter(torch.empty(d_model, d_c))
        self.W_uv = nn.Parameter(torch.empty(d_model, d_c))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        self.W_up = nn.Parameter(torch.empty(2 * d_ff, d_model))
        self.W_down = nn.Parameter(torch.empty(d_model, d_ff))
        self._init_weights()

        self.attn_gen = LatentAttentionDeltaGenerator(d_model, d_c, num_layers, expansion_order)
        self.ffn_up_gen = LinearDeltaGenerator(d_model, 2 * d_ff, num_layers, expansion_order)
        self.ffn_down_gen = LinearDeltaGenerator(d_ff, d_model, num_layers, expansion_order)

        self.decoder = LatentDecoderBlock(d_model=d_model, d_c=d_c, n_heads=n_heads, dropout=dropout)
        self.norm_f = nn.RMSNorm(d_model)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_dq)
        nn.init.xavier_uniform_(self.W_dkv)
        nn.init.xavier_uniform_(self.W_uq)
        nn.init.xavier_uniform_(self.W_uk)
        nn.init.xavier_uniform_(self.W_uv)
        nn.init.xavier_uniform_(self.W_o)
        nn.init.xavier_uniform_(self.W_up)
        nn.init.xavier_uniform_(self.W_down)

    def _base_weights(self):
        return (self.W_dq, self.W_dkv, self.W_uq, self.W_uk, self.W_uv, self.W_o,
                self.W_up, self.W_down)

    def forward(self, idx, attn_mask=None):
        B, T = idx.shape
        assert T <= self.max_seq_len

        x = self.dropout(self.token_emb(idx))

        base_attn = (self.W_dq, self.W_dkv, self.W_uq, self.W_uk, self.W_uv, self.W_o)
        base_ffn = (self.W_up, self.W_down)

        factors_attn = self.attn_gen.generate_harmonics()
        factors_up = self.ffn_up_gen.generate_harmonics()
        factors_down = self.ffn_down_gen.generate_harmonics()

        for n in range(self.num_layers):
            x = self.decoder(
                x,
                base_attn,
                base_ffn,
                factors_attn,
                factors_up,
                factors_down,
                layer_idx=n,
                attn_mask=attn_mask,
            )

        x = self.norm_f(x)
        return self.token_emb.compute_logits(x)

    def get_param_groups(self, base_lr, coeff_lr, freq_lr, phase_lr, scale_lr):
        base_params = [self.W_dq, self.W_dkv, self.W_uq, self.W_uk, self.W_uv, self.W_o, self.W_up, self.W_down]
        base_params += list(self.decoder.parameters())
        base_params += list(self.norm_f.parameters())

        coeff_params = []
        freq_params = []
        phase_params = []

        def split_surface(module):
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if "coefficients" in name or "amplitudes" in name:
                    coeff_params.append(p)
                elif "freqs" in name:
                    freq_params.append(p)
                elif "phases" in name:
                    phase_params.append(p)
                else:
                    raise ValueError(f"Unknown surface parameter '{name}' in {module.__class__.__name__}")

        split_surface(self.attn_gen)
        split_surface(self.ffn_up_gen)
        split_surface(self.ffn_down_gen)

        base_ids = {id(p) for p in base_params}
        coeff_ids = {id(p) for p in coeff_params}
        freq_ids = {id(p) for p in freq_params}
        phase_ids = {id(p) for p in phase_params}
        all_ids = (base_ids | coeff_ids | freq_ids | phase_ids )

        for name, p in self.named_parameters():
            if p.requires_grad and id(p) not in all_ids:
                raise ValueError(f"Unassigned parameter: {name}")

        return [
            { "params": base_params, "lr": base_lr, "weight_decay": 0.1, "name": "base" },
            { "params": coeff_params, "lr": coeff_lr, "weight_decay": 0.01, "name": "coeff" },
            { "params": freq_params, "lr": freq_lr, "weight_decay": 0.0, "name": "freq" },
            { "params": phase_params, "lr": phase_lr, "weight_decay": 0.0, "name": "phase" },
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]

            B, T = idx_cond.shape

            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
