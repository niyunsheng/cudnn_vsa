"""THD ownership and global routing contracts, without distributed execution."""
from dataclasses import replace

import pytest
import torch

from vsa_core import route
from vsa_layout import VSALayout, Segment, build_vsa_layout
from vsa_kernels import partition_indices
from vsa_core import ring_route
from vsa import to_block_order, to_raster_order, vsa_attention


PACK = (
    (Segment("video", (5, 9, 11), .7), Segment("text", 67)),
    (Segment("text", 3), Segment("video", (1, 3, 5))),
    (Segment("video", (4, 8, 12), .5), Segment("video", (5, 4, 7), .9),
     Segment("audio", 17), Segment("text", 0)),
)


@pytest.mark.parametrize("size", [1, 2, 8])
@pytest.mark.parametrize("dense_query_full", [False, True])
def test_thd_partition(size, dense_query_full):
    full = build_vsa_layout(PACK, head_num=2, dense_query_full=dense_query_full)
    owners, seen = [], []
    for rank in range(size):
        local = build_vsa_layout(PACK, head_num=2, ring_size=size, ring_rank=rank,
                                     dense_query_full=dense_query_full, input_order="block")
        ids = local.ring.block_ids[local.ring.block_ids >= 0]
        assert ids.tolist() == list(range(rank, full.blocks, size))
        assert local.input_order == "block"
        assert local.tokens > 0
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
        assert all(row.data_ptr() % 16 == 0 for row in local.ring.peer_block_sizes)
        assert torch.equal(local.routing_counts[:, :, :len(ids)], full.routing_counts[:, :, ids])
        assert not local.routing_counts[:, :, len(ids):].any()
        for g in full.groups:
            amount = int(((ids >= g.start) & (ids < g.end)).sum())
            assert g.blocks // size <= amount <= (g.blocks + size - 1) // size
        owners.extend(ids.tolist())
        seen.extend(local.input_rows.tolist())
    assert sorted(owners) == list(range(full.blocks))
    assert sorted(seen) == list(range(full.tokens))


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("dense", [False, True])
def test_ring_rejects_fewer_global_blocks_than_ranks(rank, dense):
    # Every rank rejects, including those that would own a valid block.
    samples = [[Segment("video", (1, 1, 1), None if dense else .5)] for _ in range(3)]
    with pytest.raises(ValueError, match="global_blocks=3, ring_size=4"):
        build_vsa_layout(samples, head_num=2, input_order="block",
                         ring_size=4, ring_rank=rank, device="cuda")


@pytest.mark.parametrize("blocks", [4, 5, 7])
def test_ring_accepts_small_samples_and_uneven_block_counts(blocks):
    # Each sample has one block (< CP), but the complete pack has enough.
    samples = [[Segment("video", (1, 1, 1), .5)] for _ in range(blocks)]
    for rank in range(4):
        layout = build_vsa_layout(samples, head_num=2, input_order="block",
                                  ring_size=4, ring_rank=rank)
        owned = list(range(rank, blocks, 4))
        assert layout.tokens == len(owned) > 0
        assert layout.ring.block_ids[layout.ring.block_ids >= 0].tolist() == owned
        assert layout.padded_tokens == ((blocks + 3) // 4) * 64


@pytest.mark.parametrize("ring", [False, True])
def test_attention_rejects_empty_tokens_before_cuda_or_communication(ring):
    kwargs = dict(input_order="block", ring_size=1, ring_rank=0) if ring else {}
    layout = build_vsa_layout([[Segment("video", (1, 1, 1))]], head_num=2, **kwargs)
    x = torch.empty(0, 2, 128)
    with pytest.raises(ValueError, match="positive dimensions"):
        vsa_attention(x, x, x, x, layout, ring_cp_group=object() if ring else None)


@pytest.mark.parametrize("dense_query_full", [False, True])
def test_global_topk_and_peer_lists(dense_query_full):
    torch.manual_seed(38)
    heads, size = 2, 8
    full = build_vsa_layout(PACK, head_num=heads, dense_query_full=dense_query_full)
    q, k = (torch.randn(full.blocks, heads, 128) for _ in range(2))
    selected = route(q[full.coarse_blocks], k[full.coarse_blocks], full.groups,
                     full.routing_indices, dense_query_full=dense_query_full)
    for rank in range(size):
        local = build_vsa_layout(PACK, head_num=heads, ring_size=size, ring_rank=rank,
                                     dense_query_full=dense_query_full, input_order="block")
        ids = local.ring.block_ids[local.ring.block_ids >= 0]
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


def test_64k_h2_ring8():
    samples = [[Segment("video", (16, 60, 64), .9), Segment("text", 4096)]]
    for rank in range(8):
        local = build_vsa_layout(samples, head_num=2, ring_size=8, ring_rank=rank, input_order="block")
        assert local.ring.global_tokens == 65536
        assert local.tokens == 8192
        assert local.blocks == 128
        assert local.coarse_max_q == 128
        assert local.coarse_max_kv == 1024
        assert torch.all(local.routing_counts == 160)
        assert local.slots_are_input


@pytest.mark.parametrize("size,rank", [(0, 0), (8, 8), (8, -1), (True, 0), (1.5, 0), (8, 0.5)])
def test_invalid_group_shape(size, rank):
    with pytest.raises(ValueError):
        build_vsa_layout(PACK, head_num=2, ring_size=size, ring_rank=rank, input_order="block")


@pytest.mark.parametrize("order", ["raster", "slots", "", None, 1])
def test_invalid_input_order(order):
    with pytest.raises(ValueError, match="input_order"):
        build_vsa_layout(PACK, head_num=2, ring_size=8, ring_rank=0, input_order=order)


def test_size_one_ring_is_still_compact_block_order():
    ring = build_vsa_layout(PACK, head_num=2, ring_size=1, ring_rank=0, input_order="block")
    single = build_vsa_layout(PACK, head_num=2, input_order="block")
    assert type(ring) is type(single) is VSALayout
    assert ring.ring is not None and single.ring is None
    for name in ("input_rows", "slot_row", "row_slot", "block_sizes", "routing_indices",
                 "routing_counts", "coarse_blocks", "coarse_kv_blocks", "coarse_cu_q",
                 "coarse_cu_kv", "row_coarse", "coarse_rows"):
        torch.testing.assert_close(getattr(ring, name), getattr(single, name), rtol=0, atol=0)
    assert single.coarse_cu_q is single.coarse_cu_kv
    assert single.coarse_blocks is single.coarse_kv_blocks
    with pytest.raises(ValueError, match='input_order must be "block"'):
        build_vsa_layout(PACK, head_num=2, ring_size=1, ring_rank=0, input_order="raster")
    with pytest.raises(ValueError, match='input_order must be "block"'):
        replace(ring, input_order="raster")


@pytest.mark.parametrize("convert", [to_block_order, to_raster_order])
@pytest.mark.parametrize("size", [1, 2])
def test_boundary_reorder_requires_full_layout(convert, size):
    shard = build_vsa_layout(PACK, head_num=2, ring_size=size, ring_rank=0, input_order="block")
    with pytest.raises(ValueError, match="full VSALayout"):
        convert(torch.empty(shard.tokens, 32), vsa_layout=shard)


@pytest.mark.parametrize("kwargs", [{"ring_size": 2}, {"ring_rank": 0}])
def test_ring_parameters_must_be_paired(kwargs):
    with pytest.raises(ValueError, match="provided together"):
        build_vsa_layout(PACK, head_num=2, input_order="block", **kwargs)


@pytest.mark.parametrize("size", [1, 2])
def test_ring_requires_explicit_block_order(size):
    with pytest.raises(ValueError, match='input_order must be "block"'):
        build_vsa_layout(PACK, head_num=2, ring_size=size, ring_rank=0)


def test_omitted_ring_parameters_preserve_full_layout():
    raster = build_vsa_layout(PACK, head_num=2)
    block = build_vsa_layout(PACK, head_num=2, input_order="block", ring_size=None, ring_rank=None)
    assert isinstance(raster, VSALayout) and raster.input_order == "raster"
    assert isinstance(block, VSALayout) and block.input_order == "block"
    assert raster.ring is block.ring is None
    assert raster.tokens == block.tokens
    assert raster.input_rows.tolist() == list(range(raster.tokens))


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("sharded", [False, True])
def test_attention_rejects_layout_group_mismatch(size, sharded):
    kwargs = dict(ring_size=size, ring_rank=0) if sharded else {}
    layout = build_vsa_layout(PACK, head_num=2, input_order="block", **kwargs)
    x = torch.empty(layout.tokens, 2, 128)
    with pytest.raises(ValueError, match="explicit ring_cp_group"):
        vsa_attention(x, x, x, x, layout, ring_cp_group=None if sharded else object())


@pytest.mark.parametrize("rank,size", [(1, 2), (0, 4)])
def test_ring_metadata_must_match_process_group(monkeypatch, rank, size):
    import vsa_core as core

    layout = build_vsa_layout(PACK, head_num=2, input_order="block", ring_size=2, ring_rank=0)
    monkeypatch.setattr(core.dist, "get_world_size", lambda group: size)
    monkeypatch.setattr(core.dist, "get_rank", lambda group: rank)
    with pytest.raises(ValueError, match="rank/size differ"):
        core._ring_runtime(layout, object(), torch.device("cpu"), False, None)


def test_coarse_metadata_preserves_sample_isolation_and_global_collectives():
    samples = [
        [Segment("video", (4, 4, 8), .5)],  # Global blocks 0, 1: coarse.
        [Segment("video", (1, 1, 128))],   # Global blocks 2, 3: fine only.
        [Segment("video", (4, 4, 8), .5)],  # Global blocks 4, 5: coarse.
    ]
    for rank in range(4):
        layout = build_vsa_layout(samples, head_num=2, input_order="block",
                                  ring_size=4, ring_rank=rank)
        # All ranks must join pooled collectives, even ranks 2/3 with no coarse Q.
        assert not layout.dense_only
        assert layout.sample_lengths == (128, 128, 128)
        if rank < 2:
            assert layout.coarse_blocks.tolist() == [0, 1]
            assert layout.coarse_kv_blocks.tolist() == [0, 1, 4, 5]
            assert layout.coarse_cu_q.tolist() == [0, 1, 2]
            assert layout.coarse_cu_kv.tolist() == [0, 2, 4]
            assert (layout.coarse_max_q, layout.coarse_max_kv) == (1, 2)
        else:
            assert layout.coarse_blocks.numel() == layout.coarse_kv_blocks.numel() == 0
            assert layout.coarse_cu_q.tolist() == layout.coarse_cu_kv.tolist() == [0]
            assert (layout.coarse_max_q, layout.coarse_max_kv) == (0, 0)
            assert not layout.row_coarse.any()  # All valid rows use the zero sentinel.
