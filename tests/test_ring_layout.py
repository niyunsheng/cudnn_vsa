"""THD ownership and global routing contracts, without distributed execution."""
from dataclasses import FrozenInstanceError, replace
import random

import pytest
import torch
import torch.distributed as dist

from vsa_core import route
from vsa_layout import VSALayout, SampleBlocks, Segment, _plan_ring_blocks, build_vsa_layout
from vsa_kernels import partition_indices
from vsa import raster_to_block_wo_padding, block_wo_padding_to_raster, vsa_attention


PACK = (
    (Segment("video", (5, 9, 11), .7), Segment("text", 67)),
    (Segment("text", 3), Segment("video", (1, 3, 5))),
    (Segment("video", (4, 8, 12), .5), Segment("video", (5, 4, 7), .9),
     Segment("audio", 17), Segment("text", 0)),
)


@pytest.mark.parametrize("size", [1, 2, 8])
@pytest.mark.parametrize("dense_query_full", [False, True])
@pytest.mark.parametrize("partition", ["cyclic", "contiguous"])
def test_thd_partition(size, dense_query_full, partition, metadata_group):
    full = build_vsa_layout(PACK, head_num=2, dense_query_full=dense_query_full)
    owners, seen = [], []
    for rank in range(size):
        local = build_vsa_layout(PACK, head_num=2, ring_cp_group=metadata_group(size, rank),
                                     dense_query_full=dense_query_full, input_order="block",
                                     ring_partition=partition)
        assert local.ring.partition == partition
        ids = local.ring.block_ids[local.ring.block_ids >= 0]
        assert ids.tolist() == sorted(ids.tolist())
        mapping = local.ring.global_to_gather
        assert mapping[ids].tolist() == list(range(rank * local.blocks,
                                                  rank * local.blocks + len(ids)))
        if rank == 0:
            expected_mapping = mapping.clone()
        else:
            assert torch.equal(mapping, expected_mapping)
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
            part = ids[(ids >= g.start) & (ids < g.end)].tolist()
            stride = size if partition == "cyclic" else 1
            assert not part or part == list(range(part[0], part[-1] + 1, stride))
        owners.extend(ids.tolist())
        seen.extend(local.input_rows.tolist())
    assert sorted(owners) == list(range(full.blocks))
    assert sorted(seen) == list(range(full.tokens))


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16, 72, 128])
@pytest.mark.parametrize("partition", ["cyclic", "contiguous"])
def test_planner_coverage_capacity_and_sample_boundaries(size, partition):
    rng = random.Random(71 + size)
    for _ in range(80):
        total = rng.randint(size, size + 192)
        cuts = sorted({0, total, *[rng.randrange(total + 1) for _ in range(30)]})
        groups = [SampleBlocks(a, b, b-a, ()) for a, b in zip(cuts, cuts[1:])]
        owned = _plan_ring_blocks(groups, size, partition)
        assert sorted(b for ids in owned for b in ids) == list(range(total))
        assert all(0 < len(ids) <= (total + size - 1) // size for ids in owned)
        for ids in owned:
            assert ids == sorted(ids)
            for g in groups:
                part = [b for b in ids if g.start <= b < g.end]
                stride = size if partition == "cyclic" else 1
                assert not part or part == list(range(part[0], part[-1] + 1, stride))
        for g in groups:
            counts = [sum(g.start <= b < g.end for b in ids) for ids in owned]
            cumulative = [sum(b < g.end for b in ids) for ids in owned]
            assert max(counts) - min(counts) <= 1
            assert max(cumulative) - min(cumulative) <= 1
        assert _plan_ring_blocks(groups, size, partition) == owned


def test_next_sample_rotates_past_extra_blocks_and_keeps_start_without_remainder():
    groups = [SampleBlocks(a, b, b-a, ()) for a, b in [(0, 5), (5, 10), (10, 14)]]
    # A's extra block goes to rank 0, B's to rank 1. C starts on rank 2
    # and divides evenly, so a fourth sample still starts on rank 2.
    assert _plan_ring_blocks(groups, 4, "contiguous") == [
        [0, 1, 9, 12], [2, 5, 6, 13], [3, 7, 10], [4, 8, 11]]
    groups.append(SampleBlocks(14, 15, 1, ()))
    assert _plan_ring_blocks(groups, 4, "contiguous") == [
        [0, 1, 9, 12], [2, 5, 6, 13], [3, 7, 10, 14], [4, 8, 11]]


@pytest.mark.parametrize("partition,blocks,tokens", [
    ("contiguous", [[0, 1, 6], [2, 3, -1], [4, 5, -1]], [91, 96, 47]),
    ("cyclic", [[0, 3, 6], [1, 4, -1], [2, 5, -1]], [59, 96, 79]),
])
def test_partial_blocks_balance_block_counts_and_restore_rank_major_means(
        partition, blocks, tokens, metadata_group):
    samples = [[Segment("video", (4, 6, 8), .5), Segment("text", 20)],
               [Segment("video", (1, 3, 5)), Segment("text", 7)]]
    layouts = [build_vsa_layout(samples, head_num=2, input_order="block",
                                ring_cp_group=metadata_group(3, rank),
                                ring_partition=partition) for rank in range(3)]
    assert [x.ring.block_ids.tolist() for x in layouts] == blocks
    assert [x.tokens for x in layouts] == tokens
    # Synthetic means mark global IDs; the gather permutation discards dummy rows.
    gathered = torch.cat([x.ring.block_ids for x in layouts])
    for x in layouts:
        assert gathered[x.ring.global_to_gather].tolist() == list(range(7))


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("dense", [False, True])
def test_ring_rejects_fewer_global_blocks_than_ranks(rank, dense, metadata_group):
    # Every rank rejects, including those that would own a valid block.
    samples = [[Segment("video", (1, 1, 1), None if dense else .5)] for _ in range(3)]
    with pytest.raises(ValueError, match="global_blocks=3, ring_size=4"):
        build_vsa_layout(samples, head_num=2, input_order="block",
                         ring_cp_group=metadata_group(4, rank), device="cuda")


@pytest.mark.parametrize("blocks", [4, 5, 7])
def test_ring_accepts_small_samples_and_uneven_block_counts(blocks, metadata_group):
    # Each sample has one block (< CP), but the complete pack has enough.
    samples = [[Segment("video", (1, 1, 1), .5)] for _ in range(blocks)]
    for rank in range(4):
        layout = build_vsa_layout(samples, head_num=2, input_order="block",
                                  ring_cp_group=metadata_group(4, rank))
        owned = list(range(rank, blocks, 4))
        assert layout.tokens == len(owned) > 0
        assert layout.ring.block_ids[layout.ring.block_ids >= 0].tolist() == owned
        assert layout.padded_tokens == ((blocks + 3) // 4) * 64


@pytest.mark.parametrize("ring", [False, True])
def test_attention_rejects_empty_tokens_before_cuda_or_communication(ring, metadata_group):
    kwargs = dict(input_order="block", ring_cp_group=metadata_group(1, 0)) if ring else {}
    layout = build_vsa_layout([[Segment("video", (1, 1, 1))]], head_num=2, **kwargs)
    x = torch.empty(0, 2, 128)
    with pytest.raises(ValueError, match="positive dimensions"):
        vsa_attention(x, x, x, x, layout)


@pytest.mark.parametrize("dense_query_full", [False, True])
@pytest.mark.parametrize("partition", ["cyclic", "contiguous"])
def test_global_topk_and_peer_lists(dense_query_full, partition, metadata_group):
    torch.manual_seed(38)
    heads, size = 2, 8
    full = build_vsa_layout(PACK, head_num=heads, dense_query_full=dense_query_full)
    q, k = (torch.randn(full.blocks, heads, 128) for _ in range(2))
    selected = route(q[full.coarse_blocks], k[full.coarse_blocks], full)
    for rank in range(size):
        local = build_vsa_layout(PACK, head_num=heads, ring_cp_group=metadata_group(size, rank),
                                     dense_query_full=dense_query_full, input_order="block",
                                     ring_partition=partition)
        ids = local.ring.block_ids[local.ring.block_ids >= 0]
        query = torch.zeros(local.blocks, heads, 128)
        query[:len(ids)] = q[ids]
        actual = route(query, k, local)
        assert torch.equal(actual[:, :, :len(ids)], selected[:, :, ids])
        indices, counts = partition_indices(
            actual, local.routing_counts, local.ring.global_to_gather, size, local.blocks)
        assert all(x.data_ptr() % 16 == 0 and x.is_contiguous()
                   for table in (indices, counts) for x in table)
        assert torch.equal(counts.sum(0), local.routing_counts)
        inverse = torch.full((size * local.blocks,), -1, dtype=torch.int64)
        inverse[local.ring.global_to_gather] = torch.arange(full.blocks)
        for head in range(heads):
            for row in range(local.blocks):
                gathered = []
                for peer in range(size):
                    n = counts[peer, 0, head, row].item()
                    positions = indices[peer, 0, head, row, :n].long() + peer * local.blocks
                    gathered.extend(inverse[positions].tolist())
                n = local.routing_counts[0, head, row].item()
                assert sorted(gathered) == sorted(actual[0, head, row, :n].tolist())


def test_64k_h2_ring8(metadata_group):
    samples = [[Segment("video", (16, 60, 64), .9), Segment("text", 4096)]]
    for rank in range(8):
        local = build_vsa_layout(samples, head_num=2, ring_cp_group=metadata_group(8, rank), input_order="block")
        assert local.ring.global_tokens == 65536
        assert local.tokens == 8192
        assert local.blocks == 128
        assert local.coarse_max_q == 128
        assert local.coarse_max_kv == 1024
        assert torch.all(local.routing_counts == 160)
        assert local.slots_are_input


@pytest.mark.parametrize("group", [object(), False, 1, dist.GroupMember.NON_GROUP_MEMBER])
def test_invalid_or_nonmember_group(group):
    with pytest.raises(ValueError, match="ProcessGroup containing this rank"):
        build_vsa_layout(PACK, head_num=2, ring_cp_group=group, input_order="block")


@pytest.mark.parametrize("order", ["raster", "slots", "", None, 1])
def test_invalid_input_order(order, metadata_group):
    with pytest.raises(ValueError, match="input_order"):
        build_vsa_layout(PACK, head_num=2, ring_cp_group=metadata_group(8, 0), input_order=order)


def test_size_one_ring_is_still_compact_block_order(metadata_group):
    ring = build_vsa_layout(PACK, head_num=2, ring_cp_group=metadata_group(1, 0), input_order="block")
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
        build_vsa_layout(PACK, head_num=2, ring_cp_group=metadata_group(1, 0), input_order="raster")
    with pytest.raises(ValueError, match='input_order must be "block"'):
        replace(ring, input_order="raster")


@pytest.mark.parametrize("size", [1, 2])
def test_inverse_boundary_reorder_requires_full_layout(size, metadata_group):
    shard = build_vsa_layout(PACK, head_num=2, ring_cp_group=metadata_group(size, 0), input_order="block")
    with pytest.raises(ValueError, match="full VSALayout"):
        block_wo_padding_to_raster(torch.empty(shard.tokens, 32), vsa_layout=shard)


def test_ring_boundary_requires_global_raster_rows(metadata_group):
    shard = build_vsa_layout(PACK, head_num=2, ring_cp_group=metadata_group(2, 0), input_order="block")
    with pytest.raises(ValueError, match="shape"):
        raster_to_block_wo_padding(torch.empty(shard.tokens, 32), vsa_layout=shard)
    with pytest.raises(ValueError, match="CUDA"):
        raster_to_block_wo_padding(torch.empty(shard.ring.global_tokens, 32), vsa_layout=shard)


@pytest.mark.parametrize("size", [1, 2])
def test_ring_requires_explicit_block_order(size, metadata_group):
    with pytest.raises(ValueError, match='input_order must be "block"'):
        build_vsa_layout(PACK, head_num=2, ring_cp_group=metadata_group(size, 0))


def test_omitted_ring_parameters_preserve_full_layout():
    raster = build_vsa_layout(PACK, head_num=2)
    block = build_vsa_layout(PACK, head_num=2, input_order="block", ring_cp_group=None)
    assert isinstance(raster, VSALayout) and raster.input_order == "raster"
    assert isinstance(block, VSALayout) and block.input_order == "block"
    assert raster.ring is block.ring is None
    assert raster.tokens == block.tokens
    assert raster.input_rows.tolist() == list(range(raster.tokens))


def test_default_partition_is_global_cyclic(metadata_group):
    full = build_vsa_layout(PACK, head_num=2)
    for rank in range(3):
        layout = build_vsa_layout(PACK, head_num=2, input_order="block",
                                  ring_cp_group=metadata_group(3, rank))
        assert layout.ring.partition == "cyclic"
        assert layout.ring.block_ids[layout.ring.block_ids >= 0].tolist() == list(
            range(rank, full.blocks, 3))


@pytest.mark.parametrize("partition", ["", "round_robin", None, 3, False])
@pytest.mark.parametrize("ring", [False, True])
def test_invalid_partition_rejected(partition, ring, metadata_group):
    kwargs = dict(input_order="block", ring_cp_group=metadata_group(3, 0)) if ring else {}
    with pytest.raises(ValueError, match="ring_partition"):
        build_vsa_layout(PACK, head_num=2, ring_partition=partition, **kwargs)


def test_microbatch_layouts_borrow_one_group(metadata_group, monkeypatch):
    group = metadata_group(4, 1)
    def unexpected(*args, **kwargs):
        pytest.fail("Layout construction must not create or destroy process groups")
    monkeypatch.setattr(dist, "new_group", unexpected)
    monkeypatch.setattr(dist, "destroy_process_group", unexpected)
    first = build_vsa_layout(PACK, head_num=2, input_order="block", ring_cp_group=group)
    second = build_vsa_layout([[Segment("video", (4, 8, 8), .5)]], head_num=2,
                              input_order="block", ring_cp_group=group)
    assert first.tokens != second.tokens
    assert first.ring.group is second.ring.group is group
    assert (first.ring.size, first.ring.rank) == (4, 1)
    with pytest.raises(FrozenInstanceError):
        first.ring.group = metadata_group(4, 1)
    del first
    assert second.ring.group is group
    assert group.size() == 4 and group.rank() == 1


def test_same_shape_groups_keep_distinct_runtimes(metadata_group, monkeypatch):
    import vsa_core as core

    groups = [metadata_group(2, 0), metadata_group(2, 0)]
    peers = {groups[0]: [0, 2], groups[1]: [1, 3]}
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: peers[group])
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    layouts = [build_vsa_layout(PACK, head_num=2, input_order="block", ring_cp_group=group,
                                ring_cp_batch_p2p=False, ring_cp_bwd_bucket_size=None)
               for group in groups]
    runtimes = [core._ring_runtime(layout, torch.device("cpu")) for layout in layouts]
    for layout, runtime, group in zip(layouts, runtimes, groups):
        assert runtime.group is layout.ring.group is group
        assert runtime.peers == tuple(peers[group])
        assert core._ring_runtime(layout, torch.device("cpu")) is runtime
    assert runtimes[0] is not runtimes[1]


@pytest.mark.parametrize("batch", [None, False, True])
@pytest.mark.parametrize("bound", [False, True])
def test_runtime_resolves_shared_transport(batch, bound, monkeypatch):
    from types import SimpleNamespace
    import vsa_core as core

    group = SimpleNamespace(bound_device_id=torch.device("cuda:0") if bound else None)
    ring = SimpleNamespace(group=group, batch_p2p=batch, transport="ring",
                           bwd_bucket_size=None, size=2, rank=0, _runtime={})
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [1, 3])
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    layout = SimpleNamespace(ring=ring)
    runtime = core._ring_runtime(layout, torch.device("cpu"))
    assert runtime.group is group
    assert runtime.batch_p2p == (bound if batch is None else batch)
    assert core._ring_runtime(layout, torch.device("cpu")) is runtime


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("rank", range(4))
def test_ring_exchange_shares_group_and_reverses_peers(batch, rank, metadata_group, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import vsa_core as core

    group = metadata_group(4, rank)
    peers = (1, 3, 5, 7)
    transfers, waits, batches = [], [], []
    def operation(kind):
        def run(tensor, peer, actual_group):
            assert actual_group is group
            transfers.append((kind, tensor, peer))
            return SimpleNamespace(wait=lambda: waits.append(peer))
        return run
    def batched(ops):
        batches.append(len(ops))
        return [op.op(op.tensor, op.peer, op.group) for op in ops]
    monkeypatch.setattr(core.dist, "isend", operation("send"))
    monkeypatch.setattr(core.dist, "irecv", operation("receive"))
    monkeypatch.setattr(core.dist, "P2POp", lambda op, tensor, peer, group:
                        SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group))
    monkeypatch.setattr(core.dist, "batch_isend_irecv", batched)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    runtime = SimpleNamespace(group=group, peers=peers, rank=rank, size=4,
                              batch_p2p=batch, stream=SimpleNamespace(wait_stream=lambda stream: None))
    pairs = [(object(), object()), (object(), object())]
    for reverse in (False, True):
        transfers.clear()
        core._ring_exchange(runtime, pairs, reverse=reverse)
        previous, following = peers[(rank-1) % 4], peers[(rank+1) % 4]
        sender, receiver = (previous, following) if reverse else (following, previous)
        assert [event[1:] for event in transfers if event[0] == "send"] == [
            (send, sender) for send, _ in pairs]
        assert [event[1:] for event in transfers if event[0] == "receive"] == [
            (receive, receiver) for _, receive in pairs]
    assert len(waits) == 8
    assert batches == ([4, 4] if batch else [])


def test_full_layout_needs_no_distributed_state(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Full layouts must not read distributed state")
    for name in ("get_world_size", "get_rank", "get_process_group_ranks", "new_group"):
        monkeypatch.setattr(dist, name, unexpected)
    assert build_vsa_layout(PACK, head_num=2).ring is None


@pytest.mark.parametrize("batch,bucket,transport", [
    (None, "auto", "ring"), (False, None, "window"), (True, 64, "ring")])
def test_transport_options_are_bound_to_layout(batch, bucket, transport, metadata_group):
    layout = build_vsa_layout(PACK, head_num=2, input_order="block", ring_cp_group=metadata_group(),
                              ring_cp_batch_p2p=batch, ring_cp_bwd_bucket_size=bucket,
                              ring_cp_transport=transport)
    assert layout.ring.batch_p2p is batch
    assert layout.ring.bwd_bucket_size == bucket
    assert layout.ring.transport == transport
    with pytest.raises(FrozenInstanceError):
        layout.ring.transport = "window"


def test_default_transport_is_the_step_ring(metadata_group):
    assert build_vsa_layout(PACK, head_num=2, input_order="block",
                            ring_cp_group=metadata_group()).ring.transport == "ring"


@pytest.mark.parametrize("size", [1, 2, 3, 8])
@pytest.mark.parametrize("partition", ["cyclic", "contiguous"])
def test_window_block_sizes_follow_gather_order(size, partition, metadata_group):
    """The all-gathered KV window is indexed exactly like the pooled-mean gather."""
    full = build_vsa_layout(PACK, head_num=2)
    for rank in range(size):
        local = build_vsa_layout(PACK, head_num=2, input_order="block",
                                 ring_partition=partition,
                                 ring_cp_group=metadata_group(size, rank), ring_cp_transport="window")
        ring, capacity = local.ring, local.blocks
        assert capacity == -(-full.blocks // size)
        assert ring.window_block_sizes.shape == (size * capacity,)
        assert ring.window_block_sizes.dtype == torch.int32
        slots = ring.global_to_gather
        # Every real block keeps its valid length; the remaining slots are dummies.
        assert torch.equal(ring.window_block_sizes[slots], full.block_sizes)
        dummy = torch.ones(size * capacity, dtype=torch.bool)
        dummy[slots] = False
        assert int(dummy.sum()) == size * capacity - full.blocks
        assert not ring.window_block_sizes[dummy].any()
        # This rank's own shard occupies the window blocks of its gather slot.
        own = ring.window_block_sizes[rank * capacity:(rank + 1) * capacity]
        assert torch.equal(own, local.block_sizes)


@pytest.mark.parametrize("options,match", [
    ({"ring_cp_batch_p2p": 1}, "ring_cp_batch_p2p"),
    ({"ring_cp_batch_p2p": "False"}, "ring_cp_batch_p2p"),
    *[({"ring_cp_bwd_bucket_size": value}, "ring_cp_bwd_bucket_size")
      for value in [0, -1, True, False, 1.5, "backend"]],
    ({"ring_cp_transport": "p2p"}, "ring_cp_transport"),
    ({"ring_cp_transport": None}, "ring_cp_transport"),
])
def test_invalid_transport_options_fail_at_construction(options, match, metadata_group):
    with pytest.raises(ValueError, match=match):
        build_vsa_layout(PACK, head_num=2, input_order="block", ring_cp_group=metadata_group(), **options)


@pytest.mark.parametrize("options", [
    {"ring_cp_batch_p2p": False}, {"ring_cp_batch_p2p": True},
    {"ring_cp_bwd_bucket_size": None}, {"ring_cp_bwd_bucket_size": 64},
    {"ring_partition": "contiguous"},
    {"ring_cp_transport": "window"},
])
def test_transport_options_require_a_bound_group(options):
    with pytest.raises(ValueError, match="require ring_cp_group"):
        build_vsa_layout(PACK, head_num=2, **options)


def test_coarse_metadata_preserves_sample_isolation_and_global_collectives(metadata_group):
    samples = [
        [Segment("video", (4, 4, 8), .5)],  # Global blocks 0, 1: coarse.
        [Segment("video", (1, 1, 128))],   # Global blocks 2, 3: fine only.
        [Segment("video", (4, 4, 8), .5)],  # Global blocks 4, 5: coarse.
    ]
    for rank in range(4):
        layout = build_vsa_layout(samples, head_num=2, input_order="block",
                                  ring_cp_group=metadata_group(4, rank))
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
