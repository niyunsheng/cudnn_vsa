"""Explicit CPU substitutes; CUDA tests always exercise the real kernels."""
import pytest
import vsa_kernels
from tests.reference import cpu_gather, cpu_merge_forward, cpu_merge_backward


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(vsa_kernels, "gather", cpu_gather)
    monkeypatch.setattr(vsa_kernels, "block_means", vsa_kernels.block_means.__wrapped__)
    monkeypatch.setattr(vsa_kernels, "_fine_coarse_gate_add_forward", cpu_merge_forward)
    monkeypatch.setattr(vsa_kernels, "_fine_coarse_gate_add_backward", cpu_merge_backward)
