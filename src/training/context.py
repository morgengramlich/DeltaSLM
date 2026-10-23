import torch
from torch import nn
from fla.ops.gla import chunk_gla


class ContextTracker(nn.Module):
    def __init__(self, d_model, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.proj_v = nn.Linear(d_model, state_dim, bias=False)
        self.proj_i = nn.Linear(d_model, state_dim)
        self.proj_f = nn.Linear(d_model, state_dim)
        self.norm = nn.RMSNorm(state_dim)

    def _run_triton_gla(self, iv, f_gate):
        B, T, E = iv.shape
        D = 16
        assert E % D == 0, "state_dim must be divisible by 16"
        H = E // D

        iv_reshaped = iv.view(B, T, H, D).transpose(1, 2)
        f_gate_reshaped = f_gate.view(B, T, H, D).transpose(1, 2)

        q = torch.ones_like(iv_reshaped)
        k = torch.ones_like(iv_reshaped)
        gk = torch.log(f_gate_reshaped.clamp(min=1e-6))

        S, _ = chunk_gla(q, k, iv_reshaped, gk)
        return S.transpose(1, 2).reshape(B, T, E)

    def forward(self, x):
        v = self.proj_v(x)
        i_gate = torch.sigmoid(self.proj_i(x))
        f_gate = torch.sigmoid(self.proj_f(x))
        iv = i_gate * v

        S = self._run_triton_gla(iv, f_gate)
        return self.norm(S)
