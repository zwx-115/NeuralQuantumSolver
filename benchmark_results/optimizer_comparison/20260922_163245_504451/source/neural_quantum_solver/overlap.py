from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class OverlapResult:
    loss: torch.Tensor
    overlap: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    log_abs_A: torch.Tensor
    arg_A: torch.Tensor
    log_abs_B: torch.Tensor
    arg_B: torch.Tensor
    log_norm_phi: torch.Tensor
    log_norm_chi: torch.Tensor
    ratio_quantiles_A: torch.Tensor
    ratio_quantiles_B: torch.Tensor
    cancellation_A: torch.Tensor
    cancellation_B: torch.Tensor
    finite_flag: bool


def _diagnostics(phi, chi, r_a, r_b, A, B, loss) -> OverlapResult:
    norm_phi = torch.sum(torch.abs(phi) ** 2)
    norm_chi = torch.sum(torch.abs(chi) ** 2)
    terms_a = torch.abs(chi) ** 2 / norm_chi * r_a
    terms_b = torch.abs(phi) ** 2 / norm_phi * r_b
    tiny = torch.finfo(phi.real.dtype).tiny
    cancellation_a = torch.sum(torch.abs(terms_a)) / torch.clamp(torch.abs(A), min=tiny)
    cancellation_b = torch.sum(torch.abs(terms_b)) / torch.clamp(torch.abs(B), min=tiny)
    quantiles = torch.tensor(
        [0.0, 0.5, 0.9, 0.99, 1.0], device=phi.device, dtype=phi.real.dtype
    )
    qa = torch.quantile(torch.log(torch.abs(r_a)), quantiles)
    qb = torch.quantile(torch.log(torch.abs(r_b)), quantiles)
    values = torch.cat((
        torch.stack([loss.to(phi.dtype), A, B]), r_a.reshape(-1), r_b.reshape(-1)
    ))
    return OverlapResult(
        loss, A * B, A, B, torch.log(torch.abs(A)), torch.angle(A),
        torch.log(torch.abs(B)), torch.angle(B), torch.log(norm_phi),
        torch.log(norm_chi), qa, qb, cancellation_a, cancellation_b,
        bool(torch.all(torch.isfinite(values))),
    )


def legacy_ratio_loss(phi: torch.Tensor, chi: torch.Tensor, unitary: torch.Tensor) -> OverlapResult:
    """Exact-sum form of the two legacy Born-distribution expectations."""
    transformed_phi = unitary @ phi
    transformed_chi = unitary.mH @ chi
    norm_phi = torch.sum(torch.abs(phi) ** 2)
    norm_chi = torch.sum(torch.abs(chi) ** 2)
    r_a = transformed_phi / chi
    r_b = transformed_chi / phi
    A = torch.sum(torch.abs(chi) ** 2 / norm_chi * r_a)
    B = torch.sum(torch.abs(phi) ** 2 / norm_phi * r_b)
    return _diagnostics(phi, chi, r_a, r_b, A, B, 1 - (A * B).real)


def direct_overlap_loss(phi: torch.Tensor, chi: torch.Tensor, unitary: torch.Tensor) -> OverlapResult:
    transformed_phi = unitary @ phi
    transformed_chi = unitary.mH @ chi
    norm_phi = torch.vdot(phi, phi).real
    norm_chi = torch.vdot(chi, chi).real
    inner = torch.vdot(chi, transformed_phi)
    fidelity = torch.abs(inner) ** 2 / (norm_phi * norm_chi)
    A, B = inner / norm_chi, inner.conj() / norm_phi
    return _diagnostics(
        phi, chi, transformed_phi / chi, transformed_chi / phi,
        A, B, 1 - fidelity.real,
    )


def normalized_ratio_loss(phi: torch.Tensor, chi: torch.Tensor, unitary: torch.Tensor) -> OverlapResult:
    return legacy_ratio_loss(
        phi / torch.linalg.vector_norm(phi), chi / torch.linalg.vector_norm(chi), unitary
    )


def gauge_fixed_ratio_loss(phi: torch.Tensor, chi: torch.Tensor, unitary: torch.Tensor) -> OverlapResult:
    phi_n = phi / torch.linalg.vector_norm(phi)
    chi_n = chi / torch.linalg.vector_norm(chi)
    phi_n = phi_n * torch.exp(-1j * torch.angle(phi_n[torch.argmax(torch.abs(phi_n))]))
    chi_n = chi_n * torch.exp(-1j * torch.angle(chi_n[torch.argmax(torch.abs(chi_n))]))
    return legacy_ratio_loss(phi_n, chi_n, unitary)


def log_domain_ratio_loss(phi: torch.Tensor, chi: torch.Tensor, unitary: torch.Tensor) -> OverlapResult:
    """Small-system stabilized path using normalized dense matrix products."""
    phi_n = phi / torch.linalg.vector_norm(phi)
    chi_n = chi / torch.linalg.vector_norm(chi)
    return direct_overlap_loss(phi_n, chi_n, unitary)
