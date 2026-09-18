import math
import torch
from torch import nn
from .context import ContextTracker
from .decoder import DecoderBlock
from ..generators import AttentionDeltaGenerator, LinearDeltaGenerator
from ..basis import TriangleSurface3D, TriangleSurface4D


class LoopBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, max_seq_len,
                 expansion_order, state_dim, max_mat_cycles=1, max_rc_cycles=5, max_depth_cycles=1,
                 dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.expansion_order = expansion_order

        self.ctx_tracker = ContextTracker(d_model, state_dim)
        self.ctx_comp = nn.Sequential(
            nn.Linear(state_dim, expansion_order, bias=False),
            nn.RMSNorm(expansion_order),
        )

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.W_up = nn.Parameter(torch.empty(2 * d_ff, d_model))
        self.W_down = nn.Parameter(torch.empty(d_model, d_ff))

        self.attn_gen = AttentionDeltaGenerator(
            d_model, num_layers, expansion_order,
            max_mat_cycles=max_mat_cycles, max_rc_cycles=max_rc_cycles, max_depth_cycles=max_depth_cycles,
            delta_fn=TriangleSurface4D
        )
        self.ffn_up_gen = LinearDeltaGenerator(
            d_model, 2 * d_ff, num_layers, expansion_order,
            max_rc_cycles=max_rc_cycles, max_depth_cycles=max_depth_cycles,
            delta_fn=TriangleSurface3D
        )
        self.ffn_down_gen = LinearDeltaGenerator(
            d_ff, d_model, num_layers, expansion_order,
            max_rc_cycles=max_rc_cycles, max_depth_cycles=max_depth_cycles,
            delta_fn=TriangleSurface3D
        )

        self.decoder = DecoderBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)

        self._init_weights()

    def _init_amplitudes(self, module, alpha, n_in, n_out, expansion_order):
        sigma_a = math.sqrt((27.0 * alpha) / (expansion_order * (n_in + n_out)))
        nn.init.normal_(module.amplitudes, mean=0.0, std=sigma_a)

    def _init_weights(self):
        """Initialize base transformer weights."""
        alpha = 0.1
        gain = math.sqrt(1.0 - alpha)

        nn.init.xavier_uniform_(self.W_q, gain=gain)
        nn.init.xavier_uniform_(self.W_k, gain=gain)
        nn.init.xavier_uniform_(self.W_v, gain=gain)
        nn.init.xavier_uniform_(self.W_o, gain=gain)
        nn.init.xavier_uniform_(self.W_up, gain=gain)
        nn.init.xavier_uniform_(self.W_down, gain=gain)

        self._init_amplitudes(self.attn_gen.diff_gen, alpha=alpha, n_in=self.d_model, n_out=self.d_model, expansion_order=self.expansion_order)
        self._init_amplitudes(self.ffn_up_gen.diff_gen, alpha=alpha, n_in=self.d_model, n_out=2 * self.d_ff, expansion_order=self.expansion_order)
        self._init_amplitudes(self.ffn_down_gen.diff_gen, alpha=alpha, n_in=self.d_ff, n_out=self.d_model, expansion_order=self.expansion_order)

    def _base_weights(self):
        return (self.W_q, self.W_k, self.W_v, self.W_o, self.W_up, self.W_down)

    def forward(self, x, attn_mask=None):
        B, T, E = x.shape
        assert T <= self.max_seq_len

        state = self.ctx_tracker(x)
        state = self.ctx_comp(state)

        base_attn = (self.W_q, self.W_k, self.W_v, self.W_o)
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
                n,
                state,
                attn_mask=attn_mask
            )

        return x

    def get_param_groups(self, base_lr, coeff_lr, freq_lr, phase_lr):
        base_params = [self.W_q, self.W_k, self.W_v, self.W_o, self.W_up, self.W_down]
        base_params += list(self.ctx_tracker.parameters())
        base_params += list(self.ctx_comp.parameters())
        base_params += list(self.decoder.parameters())

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
        all_ids = (base_ids | coeff_ids | freq_ids | phase_ids)

        for name, p in self.named_parameters():
            if p.requires_grad and id(p) not in all_ids:
                raise ValueError(f"Unassigned parameter: {name}")

        return [
            { "params": base_params, "lr": base_lr, "weight_decay": 0.1, "name": "base" },
            { "params": coeff_params, "lr": coeff_lr, "weight_decay": 0.01, "name": "coeff" },
            { "params": freq_params, "lr": freq_lr, "weight_decay": 0.0, "name": "freq" },
            { "params": phase_params, "lr": phase_lr, "weight_decay": 0.0, "name": "phase" },
        ]
