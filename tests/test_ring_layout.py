"""THD ownership and global routing contracts, without distributed execution."""
import pytest
import torch

from vsa_core import route
from vsa_layout import Segment, build_ring_vsa_layout, build_vsa_layout
from vsa_kernels import partition_indices
from vsa_core import ring_route


PACK = (
    (Segment("video", (5, 9, 11), .7), Segment("text", 67)),
    (Segment("text", 3), Segment("video", (1, 3, 5))),
    (Segment("video", (4, 8, 12), .5), Segment("video", (5, 4, 7), .9),
     Segment("audio", 17), Segment("text", 0)),
)


@pytest.mark.parametrize("size", [1, 2, 8, 64])
@pytest.mark.parametrize("dense_query_full", [False, True])
@pytest.mark.parametrize("input_order", ["raster", "block"])
def test_thd_partition(size, dense_query_full, input_order):
    full = build_vsa_layout(PACK, head_num=2, dense_query_full=dense_query_full)
    owners, seen = [], []
    for rank in range(size):
        local = build_ring_vsa_layout(PACK, head_num=2, ring_size=size, ring_rank=rank,
                                     dense_query_full=dense_query_full, input_order=input_order)
        ids = local.block_ids[local.block_ids >= 0]
        assert ids.tolist() == list(range(rank, full.blocks, size))
        if input_order == "raster":
            assert local.input_rows.tolist() == sorted(local.input_rows.tolist())
            assert not local.slots_are_input
        else:
            assert local.slot_row[local.slot_row >= 0].tolist() == list(range(local.tokens))
            assert local.slots_are_input == (local.tokens == local.padded_tokens)
        assert local.padded_tokens == ((full.blocks + size - 1) // size) * 64
        valid = local.slot_row >= 0
        original_slots = torch.full_like(local.slot_row, -1)
        original_slots[valid] = local.input_rows[local.slot_row[valid]]
        assert torch.equal(original_slots.view(-1, 64)[:len(ids)],
                           full.slot_row.view(-1, 64)[ids])
        assert torch.equal(local.slot_row[local.row_slot], torch.arange(local.tokens))
        assert torch.equal(local.block_sizes[:len(ids)], full.block_sizes[ids])
        assert all(row.data_ptr() % 16 == 0 for row in local.peer_block_sizes)
        assert torch.equal(local.routing_counts[:, :, :len(ids)], full.routing_counts[:, :, ids])
        assert not local.routing_counts[:, :, len(ids):].any()
        for g in full.groups:
            amount = int(((ids >= g.start) & (ids < g.end)).sum())
            assert g.blocks // size <= amount <= (g.blocks + size - 1) // size
        owners.extend(ids.tolist())
        seen.extend(local.input_rows.tolist())
    assert sorted(owners) == list(range(full.blocks))
    assert sorted(seen) == list(range(full.tokens))


@pytest.mark.parametrize("dense_query_full", [False, True])
def test_global_topk_and_peer_lists(dense_query_full):
    torch.manual_seed(38)
    heads, size = 2, 8
    full = build_vsa_layout(PACK, head_num=heads, dense_query_full=dense_query_full)
    q, k = (torch.randn(full.blocks, heads, 128) for _ in range(2))
    selected = route(q[full.coarse_blocks], k[full.coarse_blocks], full.groups,
                     full.routing_indices, dense_query_full=dense_query_full)
    for rank in range(size):
        local = build_ring_vsa_layout(PACK, head_num=heads, ring_size=size, ring_rank=rank,
                                     dense_query_full=dense_query_full)
        ids = local.block_ids[local.block_ids >= 0]
        query = torch.zeros(local.blocks, heads, 128)
        query[:len(ids)] = q[ids]
        actual = ring_route(query, k, local)
        assert torch.equal(actual[:, :, :len(ids)], selected[:, :, ids])
        indices, counts = partition_indices(
            actual, local.routing_counts, size, local.blocks)
        assert all(x.data_ptr() % 16 == 0 and x.is_contiguous()
                   for table in (indices, counts) for x in table)
        assert torch.equal(counts.sum(0), local.routing_counts)
        for head in range(heads):
            for row in range(local.blocks):
                gathered = []
                for peer in range(size):
                    n = counts[peer, 0, head, row].item()
                    gathered.extend((indices[peer, 0, head, row, :n] * size + peer).tolist())
                n = local.routing_counts[0, head, row].item()
                assert sorted(gathered) == sorted(actual[0, head, row, :n].tolist())


@pytest.mark.parametrize("input_order", ["raster", "block"])
def test_64k_h2_ring8(input_order):
    samples = [[Segment("video", (16, 60, 64), .9), Segment("text", 4096)]]
    for rank in range(8):
        local = build_ring_vsa_layout(samples, head_num=2, ring_size=8, ring_rank=rank,
                                     input_order=input_order)
        assert local.global_tokens == 65536
        assert local.tokens == 8192
        assert local.blocks == 128
        assert local.coarse_max_q == 128
        assert local.coarse_max_kv == 1024
        assert torch.all(local.routing_counts == 160)
        assert local.slots_are_input == (input_order == "block")


@pytest.mark.parametrize("size,rank", [(0, 0), (8, 8), (8, -1), (True, 0)])
def test_invalid_group_shape(size, rank):
    with pytest.raises(ValueError):
        build_ring_vsa_layout(PACK, head_num=2, ring_size=size, ring_rank=rank)


@pytest.mark.parametrize("order", ["slots", "", None, 1])
def test_invalid_input_order(order):
    with pytest.raises(ValueError, match="input_order"):
        build_ring_vsa_layout(PACK, head_num=2, ring_size=8, ring_rank=0, input_order=order)
