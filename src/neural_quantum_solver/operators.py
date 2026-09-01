from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch

from .hilbert import SpinHalfHilbert


@dataclass(frozen=True)
class PauliTerm:
    coefficient: complex
    operators: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        sites = [site for site, _ in self.operators]
        if len(sites) != len(set(sites)):
            raise ValueError("a Pauli term cannot act twice on the same site")
        if any(op not in {"I", "X", "Y", "Z"} for _, op in self.operators):
            raise ValueError("supported Pauli labels are I, X, Y, and Z")


@dataclass(frozen=True)
class Connections:
    states: torch.Tensor
    matrix_elements: torch.Tensor
    sample_indices: torch.Tensor


@dataclass(frozen=True)
class RealConnections:
    """将 Hamiltonian 矩阵元拆成两个实数数组保存的连接结构。"""

    states: torch.Tensor
    matrix_elements_real: torch.Tensor
    matrix_elements_imag: torch.Tensor
    sample_indices: torch.Tensor


class PauliHamiltonian:
    """带显式系数的 Pauli 字符串之和。"""

    def __init__(self, num_sites: int, terms: Iterable[PauliTerm]) -> None:
        self.num_sites = num_sites
        self.terms = tuple(terms)
        for term in self.terms:
            if any(site < 0 or site >= num_sites for site, _ in term.operators):
                raise ValueError("Pauli term site is outside the Hilbert space")

    def connections(
        self, configurations: torch.Tensor, *, dtype: torch.dtype = torch.complex128
    ) -> Connections:
        """返回每个输入基矢对应的打包 H[row, column] 连接。"""
        if configurations.ndim != 2 or configurations.shape[1] != self.num_sites:
            raise ValueError("configurations must have shape [batch, num_sites]")
        device = configurations.device
        states_out, elements_out, samples_out = [], [], []
        batch = configurations.shape[0]
        sample_ids = torch.arange(batch, device=device, dtype=torch.int64)
        for term in self.terms:
            states = configurations.clone()
            values = torch.full((batch,), complex(term.coefficient), device=device, dtype=dtype)
            for site, operator in term.operators:
                spin = configurations[:, site].to(dtype)
                if operator == "Z":
                    values = values * spin
                elif operator == "X":
                    states[:, site] = -states[:, site]
                elif operator == "Y":
                    # 行指标约定：<z|Y|-z> = -i*z。
                    values = values * (-1j * spin)
                    states[:, site] = -states[:, site]
            states_out.append(states)
            elements_out.append(values)
            samples_out.append(sample_ids)
        if not states_out:
            return Connections(
                configurations[:0],
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=torch.int64),
            )
        return Connections(
            torch.cat(states_out), torch.cat(elements_out), torch.cat(samples_out)
        )

    def real_connections(
        self,
        configurations: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> RealConnections:
        """返回不使用复数加速器张量或内核的打包连接。"""
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("real connections require torch.float32 or torch.float64")
        if configurations.ndim != 2 or configurations.shape[1] != self.num_sites:
            raise ValueError("configurations must have shape [batch, num_sites]")
        device = configurations.device
        states_out, real_out, imag_out, samples_out = [], [], [], []
        batch = configurations.shape[0]
        sample_ids = torch.arange(batch, device=device, dtype=torch.int64)
        for term in self.terms:
            states = configurations.clone()
            coefficient = complex(term.coefficient)
            real = torch.full(
                (batch,), coefficient.real, device=device, dtype=dtype
            )
            imag = torch.full(
                (batch,), coefficient.imag, device=device, dtype=dtype
            )
            for site, operator in term.operators:
                spin = configurations[:, site].to(dtype)
                if operator == "Z":
                    real = real * spin
                    imag = imag * spin
                elif operator == "X":
                    states[:, site] = -states[:, site]
                elif operator == "Y":
                    # 使用恒等式：(a+i*b)*(-i*z) = b*z - i*a*z。
                    real, imag = imag * spin, -real * spin
                    states[:, site] = -states[:, site]
            states_out.append(states)
            real_out.append(real)
            imag_out.append(imag)
            samples_out.append(sample_ids)
        if not states_out:
            empty = torch.empty(0, device=device, dtype=dtype)
            return RealConnections(
                configurations[:0],
                empty,
                empty.clone(),
                torch.empty(0, device=device, dtype=torch.int64),
            )
        return RealConnections(
            torch.cat(states_out),
            torch.cat(real_out),
            torch.cat(imag_out),
            torch.cat(samples_out),
        )

    def dense_matrix(
        self,
        hilbert: SpinHalfHilbert,
        *,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if hilbert.num_sites != self.num_sites:
            raise ValueError("Hamiltonian and Hilbert-space sizes differ")
        states = hilbert.all_states(device=device)
        matrix = torch.zeros((hilbert.size, hilbert.size), dtype=dtype, device=device)
        conn = self.connections(states, dtype=dtype)
        columns = hilbert.state_to_index(conn.states)
        matrix.index_put_((conn.sample_indices, columns), conn.matrix_elements, accumulate=True)
        return matrix
