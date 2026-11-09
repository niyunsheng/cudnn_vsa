"""Explicit CPU substitutes; CUDA tests always exercise the real kernels."""
import pytest
import torch.distributed as dist
import vsa_kernels
from tests.reference import cpu_gather, cpu_merge_forward, cpu_merge_backward


@pytest.fixture
def metadata_group():
    """Backend-free ProcessGroups for CPU geometry tests; GPU runs use real NCCL."""
    return lambda size=1, rank=0: dist.ProcessGroup(rank, size)


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(vsa_kernels, "gather", cpu_gather)
    monkeypatch.setattr(vsa_kernels, "block_means", vsa_kernels.block_means.__wrapped__)
    monkeypatch.setattr(vsa_kernels, "routing_and_coarse_means",
                        vsa_kernels.routing_and_coarse_means.__wrapped__)
    monkeypatch.setattr(vsa_kernels, "_fine_coarse_gate_add_forward", cpu_merge_forward)
    monkeypatch.setattr(vsa_kernels, "_fine_coarse_gate_add_backward", cpu_merge_backward)
