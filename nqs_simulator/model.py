"""自回归 Transformer 波函数。

自已验证的 paper_reproduction/model.py 抽出（arXiv:2606.08707v1）。
比特 0 是态矢量索引的最高位。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True, bias=True)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, z, mask):
        v = self.norm1(z)
        z = z + self.attn(v, v, v, attn_mask=mask, need_weights=False)[0]
        return z + self.fc2(F.gelu(self.fc1(self.norm2(z))))


class Statebank(nn.Module):
    """归一化复自回归波函数；比特 0 为最高位。"""

    def __init__(self, n_qubits, d_model=128, n_layers=2, n_heads=4, d_ff=256, dropout=0.0):
        super().__init__()
        if dropout != 0:
            raise ValueError("This implementation uses zero dropout")
        self.n_qubits = n_qubits
        self.config = dict(n_qubits=n_qubits, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.tokens = nn.Embedding(3, d_model)
        self.positions = nn.Parameter(torch.empty(n_qubits, d_model))
        nn.init.normal_(self.positions, std=0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output = nn.Linear(d_model, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def initialize_plus(self):
        """精确表示 |+>^N：所有条件概率为 1/2，相位为零。"""
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def initialize_zero(self, logit_scale: float = 2.5):
        """用有限输出偏置近似 |0>^N，使输出头仍可训练。

        偏置过大时 softmax 饱和，后续 H 门几乎学不动。
        """
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor([logit_scale, -logit_scale, 0.0, 0.0],
                                                dtype=self.output.bias.dtype))

    def conditionals(self, bits):
        """返回形状 [B,N,2] 的对数概率和未缠绕相位增量。"""
        bits = bits.long()
        b, n = bits.shape
        if n != self.n_qubits:
            raise ValueError("Wrong bit width")
        shifted = torch.cat((torch.full((b, 1), 2, device=bits.device, dtype=torch.long), bits[:, :-1]), dim=1)
        z = self.tokens(shifted) + self.positions[None]
        mask = torch.ones(n, n, device=bits.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            z = block(z, mask)
        out = self.output(self.norm(z))
        return F.log_softmax(out[..., :2], dim=-1), out[..., 2:]

    def log_amplitude_phase(self, bits):
        logp, phases = self.conditionals(bits)
        selected = bits.long().unsqueeze(-1)
        return 0.5 * logp.gather(-1, selected).squeeze(-1).sum(-1), phases.gather(-1, selected).squeeze(-1).sum(-1)

    def log_psi(self, bits):
        a, phase = self.log_amplitude_phase(bits)
        return torch.complex(a, phase)

    def amplitude(self, bits):
        return torch.exp(self.log_psi(bits))

    @torch.no_grad()
    def sample(self, n_samples):
        device = self.positions.device
        bits = torch.zeros(n_samples, self.n_qubits, dtype=torch.long, device=device)
        for i in range(self.n_qubits):
            logp, _ = self.conditionals(bits)
            bits[:, i] = torch.multinomial(logp[:, i].exp(), 1).squeeze(-1)
        return bits


def expected_parameter_count(n_qubits: int) -> int:
    return 266116 + 128 * n_qubits


def bits_from_indices(indices, n_qubits):
    shifts = torch.arange(n_qubits - 1, -1, -1, device=indices.device)
    return ((indices[:, None] >> shifts) & 1).long()


def indices_from_bits(bits):
    weights = 2 ** torch.arange(bits.shape[-1] - 1, -1, -1, device=bits.device)
    return (bits.long() * weights).sum(-1)


def bits_to_strings(bits) -> list[str]:
    rows = bits.detach().to("cpu").long().tolist()
    return ["".join(str(v) for v in row) for row in rows]


def strings_to_bits(bitstrings, n_qubits, device=None):
    bits = torch.zeros(len(bitstrings), n_qubits, dtype=torch.long, device=device)
    for i, text in enumerate(bitstrings):
        text = str(text).strip()
        if len(text) != n_qubits or any(ch not in "01" for ch in text):
            raise ValueError(f"Bitstring {text!r} is not length-{n_qubits} binary")
        for j, ch in enumerate(text):
            bits[i, j] = int(ch)
    return bits
