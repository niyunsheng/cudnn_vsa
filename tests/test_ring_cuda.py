"""CUDA checks for the online softmax state used by ring attention."""
import pytest
import torch
import triton

from vsa_ring import (
    _merge_partial, _partition_indices, _partition_indices_reference, _window_block_sizes,
)


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_partial_merge_with_empty_steps():
    torch.manual_seed(61)
    n, heads, dim = 129, 2, 128
    output = torch.zeros(n, heads, dim, device="cuda", dtype=torch.float32)
    lse = torch.full((1, heads, n), -torch.inf, device="cuda")
    next_lse = torch.empty_like(lse)
    parts, normalizers = [], []
    for step in range(8):
        part = torch.randn(1, heads, n, dim, device="cuda", dtype=torch.bfloat16)
        part_lse = torch.randn(1, heads, n, device="cuda") * 10
        part_lse[:, :, step::8] = -torch.inf
        part_lse[:, :, 0] = -torch.inf  # A padding query has no keys at any step.
        part.masked_fill_(~torch.isfinite(part_lse).unsqueeze(-1), 0)
        parts.append(part.double())
        normalizers.append(part_lse.double())
        _merge_partial[(triton.cdiv(output.numel(), 1024),)](
            output, lse, next_lse, part, part_lse, n, heads, dim,
            part.stride(1), part.stride(2), part.stride(3), 1024)
        lse, next_lse = next_lse, lse
    target_lse = torch.stack(normalizers).logsumexp(0)
    weights = [(x - target_lse).exp().nan_to_num() for x in normalizers]
    target = sum(w.unsqueeze(-1) * x for w, x in zip(weights, parts))
    target = target.squeeze(0).transpose(0, 1)
    torch.testing.assert_close(output.double(), target, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(lse.double(), target_lse, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("heads,blocks,keys", [(1, 1, 3), (2, 5, 40), (3, 7, 53),
                                              (2, 128, 160), (2, 128, 1024)])
def test_compiled_peer_metadata_alignment(heads, blocks, keys):
    # Odd local block counts must not offset peer bases by 4 or 8 bytes.
    index = torch.rand(1, heads, blocks, 8 * blocks, device="cuda").argsort(-1)[..., :keys]
    index = index.sort(-1).values.to(torch.int32).contiguous()
    counts = torch.randint(0, keys + 1, (1, heads, blocks), device="cuda", dtype=torch.int32)
    actual = _partition_indices(index, counts, 8, blocks)
    expected = _partition_indices_reference(index, counts, 8, blocks)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
        assert all(peer.data_ptr() % 16 == 0 and peer.is_contiguous() for peer in a)


@pytest.mark.parametrize("ring_size,peer_blocks", [(8, 1385), (1, 11073)])
def test_long_peer_metadata_with_mixed_counts(ring_size, peer_blocks):
    # Exercise empty, sparse, scan-boundary and full rows without constructing
    # thousands of long rows just to check compaction and zeroed output tails.
    keys = ring_size * peer_blocks
    counts = torch.tensor([0, 1, 1136, 2047, 2048, 2049, 4097, keys],
                          device="cuda", dtype=torch.int32).view(1, 1, -1)
    index = torch.arange(keys, device="cuda", dtype=torch.int32)
    index = index.expand(1, 1, counts.numel(), keys).contiguous()
    actual = _partition_indices(index, counts, ring_size, peer_blocks)
    expected = _partition_indices_reference(index, counts, ring_size, peer_blocks)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
        assert all(peer.data_ptr() % 16 == 0 and peer.is_contiguous() for peer in a)


@pytest.mark.parametrize("chunk_size", [2, 4])
@pytest.mark.parametrize("rank", [0, 3, 7])
@pytest.mark.parametrize("peer_blocks", [5, 1385])
def test_coalesced_metadata_preserves_owner_mapping(chunk_size, rank, peer_blocks):
    keys, ring = 8 * peer_blocks, 8
    lengths = [0, 1, min(1136, keys), min(2049, keys), keys]
    index = torch.arange(keys, device="cuda", dtype=torch.int32)
    index = index.expand(1, 1, len(lengths), keys).contiguous()
    counts = torch.tensor(lengths, device="cuda", dtype=torch.int32).view(1, 1, -1)
    indices, numbers = _partition_indices(index, counts, ring, peer_blocks, chunk_size, rank)
    peer_sizes = torch.arange(keys, device="cuda", dtype=torch.int32).view(ring, peer_blocks) % 65
    window_sizes = _window_block_sizes(peer_sizes, chunk_size, rank)
    for tensor in (indices, numbers, window_sizes):
        assert all(peer.data_ptr() % 16 == 0 and peer.is_contiguous() for peer in tensor)
    indices, numbers = indices.cpu(), numbers.cpu()
    for step in range(ring // chunk_size):
        owners = [(rank - step*chunk_size - i) % ring for i in range(chunk_size)]
        assert window_sizes[step].tolist() == [
            value for owner in owners for value in peer_sizes[owner].tolist()]
        for row, length in enumerate(lengths):
            expected = [owners.index(block % ring) * peer_blocks + block // ring
                        for block in range(length) if block % ring in owners]
            assert numbers[step, 0, 0, row].item() == len(expected)
            assert indices[step, 0, 0, row, :len(expected)].tolist() == expected
            assert not indices[step, 0, 0, row, len(expected):].any()
