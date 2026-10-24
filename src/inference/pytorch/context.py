import torch
from torch import nn


class ContextTracker(nn.Module):
    def __init__(self, d_model, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.proj_v = nn.Linear(d_model, state_dim, bias=False)
        self.proj_i = nn.Linear(d_model, state_dim)
        self.proj_f = nn.Linear(d_model, state_dim)
        self.norm = nn.RMSNorm(state_dim)

    def forward(self, x):
        B, T, _ = x.shape
        E = self.state_dim
        D = 16
        assert E % D == 0, "state_dim must be divisible by 16"
        H = E // D

        v = self.proj_v(x)
        i_gate = torch.sigmoid(self.proj_i(x))
        f_gate = torch.sigmoid(self.proj_f(x))
        iv = i_gate * v

        iv_reshaped = iv.view(B, T, H, D)
        f_gate_reshaped = f_gate.view(B, T, H, D)
        state = torch.zeros(B, H, D, D, device=x.device)
        outputs = []

        for t in range(T):
            v_t = iv_reshaped[:, t, :, :]
            f_t = f_gate_reshaped[:, t, :, :]
            state = state * f_t.unsqueeze(-1) + v_t.unsqueeze(-2)
            o_t = state.sum(dim=-2)
            outputs.append(o_t)

        S = torch.stack(outputs, dim=1)
        S = S.view(B, T, E)

        return self.norm(S)
