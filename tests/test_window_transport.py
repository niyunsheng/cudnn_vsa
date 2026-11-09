"""Window transport: all-gathered KV, one block-sparse call, FP32 gradient reduce-scatter.

R ranks are emulated in one process with test substitutes for the collectives
and for the cuDNN BSA entry points. The reference is the same masked attention
over the whole pack in canonical block order.
"""
from types import SimpleNamespace

import pytest
import torch

import vsa_core as core
import vsa_kernels
from vsa_layout import Segment, build_vsa_layout

pytestmark = pytest.mark.usefixtures("cpu_kernels")

# Seven blocks: uneven samples, partial blocks, and dummy slots for R > 1.
PACK = ((Segment("video", (4, 8, 8)), Segment("text", 7)),
        (Segment("text", 3), Segment("video", (1, 3, 5))))
HEADS, DIM = 2, 16


def masked_attention(q, k, v, indices, counts, sizes, scale):
    """Test substitute for cuDNN BSA whose KV block count may exceed the Q block count."""
    heads, n_q, capacity = indices.shape[1:]
    n_kv = k.shape[0] // 64
    valid = (torch.arange(capacity) < counts[0, ..., None]).int()
    blocks = torch.zeros(heads, n_q, n_kv, dtype=torch.int32)
    blocks.scatter_add_(-1, indices[0].long(), valid)  # ignored tail entries may repeat id 0
    mask = (blocks > 0).repeat_interleave(64, -2).repeat_interleave(64, -1)
    mask &= (torch.arange(k.shape[0]) % 64 < sizes.repeat_interleave(64))[None, None, :]
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    scores = scores.masked_fill(~mask, -torch.inf)
    scores = torch.where(mask.any(-1, keepdim=True), scores, 0)
    probs = scores.softmax(-1).masked_fill(~mask, 0)
    return torch.einsum("hqk,khd->qhd", probs, v)


def _fake_forward(q, k, v, indices, counts, sizes, scale, *, allow_empty=False):
    assert allow_empty, "window rows of dummy blocks have zero selected blocks"
    assert k.shape[0] == v.shape[0] == sizes.numel() * 64
    out = masked_attention(q, k, v, indices, counts, sizes, scale)
    return {"o_tensor": out.transpose(0, 1).unsqueeze(0),
            "lse_tensor": torch.zeros(1, q.shape[1], q.shape[0])}


def _fake_backward(grad, q, k, v, out, lse, indices, counts, sizes, scale, *, bucket_size=None):
    # Autograd runs custom backward functions with grad mode disabled.
    with torch.enable_grad():
        leaves = [x.detach().clone().requires_grad_() for x in (q, k, v)]
        masked_attention(*leaves, indices, counts, sizes, scale).backward(grad)
    return tuple(x.grad for x in leaves)


@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("partition", ["cyclic", "contiguous"])
def test_window_matches_global_block_sparse_attention(monkeypatch, size, partition, metadata_group):
    torch.manual_seed(size)
    full = build_vsa_layout(PACK, head_num=HEADS, input_order="block")
    scale = DIM ** -0.5
    q, k, v, dy = (torch.randn(full.tokens, HEADS, DIM) for _ in range(4))
    # Reference: the whole pack in canonical block order through the same masked kernel.
    leaves = [x.clone().requires_grad_() for x in (q, k, v)]
    slots = [vsa_kernels.block_wo_padding_to_block_w_padding(x, full.slot_row, full.row_slot) for x in leaves]
    dy_slots = vsa_kernels.block_wo_padding_to_block_w_padding(dy, full.slot_row, full.row_slot)
    expected = masked_attention(*slots, full.routing_indices, full.routing_counts, full.block_sizes, scale)
    expected.backward(dy_slots)
    expected_grads = [x.grad for x in leaves]

    layouts = [build_vsa_layout(PACK, head_num=HEADS, input_order="block", ring_cp_transport="window",
                                ring_partition=partition,
                                ring_cp_group=metadata_group(size, rank)) for rank in range(size)]
    capacity = layouts[0].blocks
    local_inputs, shards = [], []
    for local in layouts:
        xs = [x[local.input_rows].clone().requires_grad_() for x in (q, k, v)]
        local_inputs.append(xs)
        shards.append([vsa_kernels.block_wo_padding_to_block_w_padding(x, local.slot_row, local.row_slot).detach()
                       for x in xs[1:]])
    gathered, partials = [], []

    def all_gather(window, x, *, group):
        # K and V are gathered separately; identify the operand by its values.
        which = next(i for i, shard in enumerate(shards[0]) if torch.equal(shard, x)
                     or any(torch.equal(peer[i], x) for peer in shards))
        gathered.append((x.dtype, x.shape, which))
        window.copy_(torch.cat([peer[which] for peer in shards]))

    def reduce_scatter(local, partial, *, group):
        assert partial.dtype == torch.float32 and partial.is_contiguous()
        partials.append(partial.clone())
        local.zero_()  # The real sum is checked below across all emulated ranks.

    monkeypatch.setattr(core.dist, "all_gather_into_tensor", all_gather)
    monkeypatch.setattr(core.dist, "reduce_scatter_tensor", reduce_scatter)
    monkeypatch.setattr(core, "bsa_forward", _fake_forward)
    monkeypatch.setattr(core, "bsa_backward", _fake_backward)
    for rank, (local, xs) in enumerate(zip(layouts, local_inputs)):
        runtime = SimpleNamespace(size=size, rank=rank, group=object(), transport="window",
                                  bwd_bucket_size=None)
        q_s, k_s, v_s = (vsa_kernels.block_wo_padding_to_block_w_padding(x, local.slot_row, local.row_slot) for x in xs)
        out = core._WindowFine.apply(q_s, k_s, v_s, local.routing_indices, local.routing_counts,
                                     local, runtime, scale)
        owned = local.ring.block_ids.clamp_min(0)
        rows = local.slot_row >= 0
        expected_rows = expected.view(full.blocks, 64, HEADS, DIM)[owned].reshape(-1, HEADS, DIM)[rows]
        torch.testing.assert_close(out[rows], expected_rows)
        dy_local = vsa_kernels.block_wo_padding_to_block_w_padding(dy[local.input_rows], local.slot_row, local.row_slot)
        dq = torch.autograd.grad(out, xs[0], dy_local)[0]
        torch.testing.assert_close(dq, expected_grads[0][local.input_rows])
    expected_gathers = [(q.dtype, (capacity * 64, HEADS, DIM), which)
                        for _ in range(2 * size if size > 1 else 0) for which in (0, 1)]
    assert gathered == expected_gathers
    if size == 1:
        return
    # Summing every rank's FP32 window partial reproduces dK/dV in gather order.
    reduced = torch.stack(partials).sum(0).view(size, capacity, 64, 2, HEADS, DIM)
    for rank, local in enumerate(layouts):
        owned = local.ring.block_ids.clamp_min(0)
        rows = local.slot_row >= 0
        for which, expected_grad in enumerate(expected_grads[1:]):
            got = reduced[rank, :, :, which].reshape(-1, HEADS, DIM)[rows]
            want = vsa_kernels.block_wo_padding_to_block_w_padding(expected_grad, full.slot_row, full.row_slot)
            want = want.view(full.blocks, 64, HEADS, DIM)[owned].reshape(-1, HEADS, DIM)[rows]
            torch.testing.assert_close(got, want)
