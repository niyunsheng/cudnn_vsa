"""VSA tensor operations: slot conversion, pooling, gated merge and backward."""
import math

import torch
import triton
import triton.language as tl

from vsa_layout import BLOCK


@triton.jit(do_not_specialize=["TOKENS"])
def _reorder_tokens_kernel(X, OUT, ROWS, TOKENS,
                           WIDTH: tl.constexpr, INVERSE: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = offsets < TOKENS * WIDTH
    row, col = offsets // WIDTH, offsets % WIDTH
    original = tl.load(ROWS + row, valid, other=0)
    mapped = original * WIDTH + col
    if INVERSE:
        values = tl.load(X + offsets, valid, other=0)
        # ROWS is a full permutation: every destination is written once.
        tl.store(OUT + mapped, values, valid)
    else:
        values = tl.load(X + mapped, valid, other=0)
        tl.store(OUT + offsets, values, valid)


def _reorder_tokens(x, input_rows, inverse):
    x = x.contiguous()
    out = torch.empty_like(x)
    width = math.prod(x.shape[1:])
    count = x.numel()
    if count:
        _reorder_tokens_kernel[(triton.cdiv(count, 1024),)](
            x, out, input_rows, input_rows.numel(), width, inverse, 1024)
    return out


class _TokenReorder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, input_rows, inverse):
        ctx.save_for_backward(input_rows)
        ctx.inverse = inverse
        return _reorder_tokens(x, input_rows, inverse)

    @staticmethod
    def backward(ctx, grad):
        input_rows, = ctx.saved_tensors
        return reorder_tokens(grad, input_rows, inverse=not ctx.inverse), None, None


def reorder_tokens(x, input_rows, *, inverse=False):
    """Permute one token tensor in one launch, with the inverse adjoint."""
    return _TokenReorder.apply(x, input_rows, inverse)


@triton.jit(do_not_specialize=["N"])
def _gather(X, MAP, OUT, N, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    # Batch-dependent lengths reuse the same kernel for a fixed row width.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = offsets // WIDTH, offsets % WIDTH
    source = tl.load(MAP + rows, mask=offsets < N, other=-1)
    values = tl.load(X + source * WIDTH + cols, mask=(offsets < N) & (source >= 0), other=0)
    tl.store(OUT + offsets, values, mask=offsets < N)


def gather(x, row_map):
    x = x.contiguous()
    out = torch.empty((row_map.numel(), *x.shape[1:]), device=x.device, dtype=x.dtype)
    if out.numel():
        _gather[(triton.cdiv(out.numel(), 1024),)](
            x, row_map, out, N=out.numel(), WIDTH=math.prod(x.shape[1:]), BLOCK=1024)
    return out


class SlotGather(torch.autograd.Function):
    """Slot permutation/padding with a first-order gather backward."""

    @staticmethod
    def forward(ctx, x, row_map, inverse_map):
        ctx.save_for_backward(inverse_map)
        return gather(x, row_map)

    @staticmethod
    def backward(ctx, grad):
        inverse_map, = ctx.saved_tensors
        return gather(grad, inverse_map), None, None


def to_slots(x, slot_row, row_slot):
    """Gather original rows into padded slots; row_slot is the backward map."""
    return SlotGather.apply(x, slot_row, row_slot)


def from_slots(x, slot_row, row_slot):
    """Restore original rows; slot_row is the backward map."""
    return SlotGather.apply(x, row_slot, slot_row)


@triton.jit(do_not_specialize=["N"])
def _gather_qkv(Q, K, V, MAP, OUT, N, WIDTH: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    rows, cols = offsets // WIDTH, offsets % WIDTH
    source = tl.load(MAP + rows, mask=offsets < N, other=-1)
    valid = (offsets < N) & (source >= 0)
    address = source * WIDTH + cols
    q = tl.load(Q + address, valid, other=0)
    k = tl.load(K + address, valid, other=0)
    v = tl.load(V + address, valid, other=0)
    tl.store(OUT + offsets, q, offsets < N)
    tl.store(OUT + N + offsets, k, offsets < N)
    tl.store(OUT + 2 * N + offsets, v, offsets < N)


def gather_qkv(q, k, v, row_map):
    q, k, v = (x.contiguous() for x in (q, k, v))
    out = torch.empty((3, row_map.numel(), *q.shape[1:]), device=q.device, dtype=q.dtype)
    count = out[0].numel()
    if count:
        _gather_qkv[(triton.cdiv(count, 1024),)](
            q, k, v, row_map, out, count, WIDTH=math.prod(q.shape[1:]), TILE=1024)
    return tuple(out.unbind(0))


class _QKVSlotGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, slot_row, row_slot):
        ctx.save_for_backward(row_slot)
        return gather_qkv(q, k, v, slot_row)

    @staticmethod
    def backward(ctx, dq, dk, dv):
        row_slot, = ctx.saved_tensors
        return (*gather_qkv(dq, dk, dv, row_slot), None, None)


def to_slots_qkv(q, k, v, slot_row, row_slot):
    """Pad compact Q/K/V together, sharing the map and launch in both directions."""
    return _QKVSlotGather.apply(q, k, v, slot_row, row_slot)


@torch.compile(fullgraph=True)
def block_means(q, k, v, block_sizes, blocks=None):
    """Return differentiable FP32 Q/K/V means over zero-padded block slots.

    Select ``blocks`` before reducing, or pool every block when omitted.
    Valid lengths exclude padding; zero-filled empty blocks produce zero means.
    """
    sizes = block_sizes if blocks is None else block_sizes[blocks]
    denominator = sizes.clamp_min(1)[:, None, None]
    means = []
    for x in (q, k, v):
        slots = x.view(block_sizes.numel(), BLOCK, *x.shape[1:])
        if blocks is not None:
            slots = slots[blocks]
        means.append(slots.sum(1, dtype=torch.float32) / denominator)
    return tuple(means)


@torch.compile(fullgraph=True)
def routing_and_coarse_means(q, k, v, block_sizes, coarse_blocks):
    """Return detached FP32 routing Q/K and differentiable coarse Q/K/V.

    Coarse means use the input dtype (BF16 in VSA). Include all regions of
    VSA samples and exclude entire fine-only samples.
    """
    q_mean, k_mean, v_mean = block_means(q, k, v, block_sizes, coarse_blocks)
    return (q_mean.detach(), k_mean.detach(),
            q_mean.to(q.dtype), k_mean.to(k.dtype), v_mean.to(v.dtype))


@triton.jit(do_not_specialize=["N", "COARSE_N"])
def _fine_coarse_gate_add_kernel(FINE, GATE, COARSE, MAP, OUT, N, COARSE_N,
             WIDTH: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    rows, cols = offsets // WIDTH, offsets % WIDTH
    block = tl.load(MAP + rows, mask=offsets < N, other=COARSE_N)
    active = (offsets < N) & (block < COARSE_N)
    fine = tl.load(FINE + offsets, mask=offsets < N, other=0).to(tl.float32)
    gate = tl.load(GATE + offsets, mask=active, other=0).to(tl.float32)
    coarse = tl.load(COARSE + block * WIDTH + cols, mask=active, other=0).to(tl.float32)
    tl.store(OUT + offsets, fine + gate * coarse, mask=offsets < N)


@triton.jit(do_not_specialize=["N", "COARSE_N"])
def _fine_coarse_gate_add_dgate(GRAD, COARSE, MAP, DGATE, N, COARSE_N,
                   WIDTH: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    rows, cols = offsets // WIDTH, offsets % WIDTH
    block = tl.load(MAP + rows, mask=offsets < N, other=COARSE_N)
    grad = tl.load(GRAD + offsets, mask=offsets < N, other=0).to(tl.float32)
    coarse = tl.load(COARSE + block * WIDTH + cols,
                     mask=(offsets < N) & (block < COARSE_N), other=0).to(tl.float32)
    tl.store(DGATE + offsets, grad * coarse, mask=offsets < N)


@triton.jit
def _fine_coarse_gate_add_dcoarse(GRAD, GATE, ROWS, DCOARSE, WIDTH: tl.constexpr,
                     COLS: tl.constexpr):
    block = tl.program_id(0)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    rows = tl.load(ROWS + block * 64 + tl.arange(0, 64))
    offsets = rows[:, None] * WIDTH + cols[None, :]
    valid = (rows[:, None] >= 0) & (cols[None, :] < WIDTH)
    grad = tl.load(GRAD + offsets, mask=valid, other=0).to(tl.float32)
    gate = tl.load(GATE + offsets, mask=valid, other=0).to(tl.float32)
    result = tl.sum(grad * gate, axis=0)
    tl.store(DCOARSE + block * WIDTH + cols, result, mask=cols < WIDTH)


def _fine_coarse_gate_add_forward(fine, gate, coarse, row_coarse):
    out = torch.empty_like(fine)
    _fine_coarse_gate_add_kernel[(triton.cdiv(fine.numel(), 1024),)](
        fine, gate, coarse, row_coarse, out, fine.numel(), coarse.shape[0],
        fine.shape[1] * fine.shape[2], 1024, enable_fp_fusion=False)
    return out


def _fine_coarse_gate_add_backward(grad, gate, coarse, row_coarse, coarse_rows):
    grad = grad.contiguous()
    dgate, dcoarse = torch.empty_like(gate), torch.empty_like(coarse)
    width = grad.shape[1] * grad.shape[2]
    _fine_coarse_gate_add_dgate[(triton.cdiv(grad.numel(), 1024),)](
        grad, coarse, row_coarse, dgate, grad.numel(), coarse.shape[0], width, 1024)
    _fine_coarse_gate_add_dcoarse[(coarse.shape[0], triton.cdiv(width, 128))](
        grad, gate, coarse_rows, dcoarse, width, 128, num_warps=4)
    return grad, dgate, dcoarse


class _FineCoarseGateAdd(torch.autograd.Function):
    """Fine + gated coarse with a block-reduction adjoint instead of scatter."""

    @staticmethod
    def forward(ctx, fine, gate, coarse, row_coarse, coarse_rows):
        fine, gate, coarse = (x.contiguous() for x in (fine, gate, coarse))
        ctx.save_for_backward(gate, coarse)
        ctx.row_coarse, ctx.coarse_rows = row_coarse, coarse_rows
        return _fine_coarse_gate_add_forward(fine, gate, coarse, row_coarse)

    @staticmethod
    def backward(ctx, grad):
        gate, coarse = ctx.saved_tensors
        df, dg, dc = _fine_coarse_gate_add_backward(
            grad, gate, coarse, ctx.row_coarse, ctx.coarse_rows)
        return df, dg, dc, None, None


def fine_coarse_gate_add(fine, gate, coarse, row_coarse, coarse_rows):
    """Return fine + gate * coarse[row_coarse], with fine-only rows unchanged.

    BF16 inputs are promoted inside Triton; accumulate each block's gradient
    in FP32 and cast once on store, without a materialized FP32 coarse copy.
    """
    return _FineCoarseGateAdd.apply(fine, gate, coarse, row_coarse, coarse_rows)


def _align_peers(tensor):
    """Align every peer's int32 metadata base to the CuTe 16-byte contract."""
    flat = tensor.flatten(1)
    padding = -flat.shape[1] % 4
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))[:, :flat.shape[1]]
    return flat.view(tensor.shape)


def _partition_indices_reference(index, counts, ring_size: int, peer_blocks: int):
    """Fixed-capacity peer lists; zero counts are valid and never fabricated."""
    positions = torch.arange(index.shape[-1], device=index.device)
    valid = positions < counts[..., None]
    owners = torch.arange(ring_size, device=index.device)[:, None, None, None, None]
    belongs = valid.unsqueeze(0) & ((index % ring_size).unsqueeze(0) == owners)
    local = torch.where(belongs, (index // ring_size).unsqueeze(0), peer_blocks)
    compact = local.sort(-1).values[..., :min(peer_blocks, index.shape[-1])]
    compact = torch.where(compact < peer_blocks, compact, 0).to(torch.int32)
    return (_align_peers(compact.contiguous()),
            _align_peers(belongs.sum(-1, dtype=torch.int32).contiguous()))


@triton.jit
def _partition_kernel(INDEX, COUNTS, OUT, OUT_COUNTS, KEYS: tl.constexpr,
                      CAP: tl.constexpr, RING: tl.constexpr, STRIDE: tl.constexpr,
                      COUNT_STRIDE: tl.constexpr, TILE: tl.constexpr):
    row, peer = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, TILE)
    count = tl.load(COUNTS + row)
    base = OUT + peer * STRIDE + row * CAP
    # Preserve source order while translating IDs into a peer/window's slots.
    # Tail and active stores target disjoint addresses.
    if KEYS <= TILE:
        index = tl.load(INDEX + row * KEYS + lane, lane < count, other=0)
        owner = index % RING
        local = index // RING
        selected = (lane < count) & (owner == peer)
        destination = tl.cumsum(selected.to(tl.int32)) - 1
        total = tl.sum(selected.to(tl.int32))
        tl.store(base + lane, 0, (lane >= total) & (lane < CAP))
        tl.store(base + destination, local, selected)
    else:
        # A few full-attention rows can make KEYS much larger than the usual
        # sparse count. Bound scan state and skip each row's unused capacity.
        total = tl.full((), 0, tl.int32)
        for start in range(tl.cdiv(count, TILE)):
            position = start * TILE + lane
            index = tl.load(INDEX + row * KEYS + position, position < count, other=0)
            owner = index % RING
            local = index // RING
            selected = (position < count) & (owner == peer)
            destination = total + tl.cumsum(selected.to(tl.int32)) - 1
            tl.store(base + destination, local, selected)
            total += tl.sum(selected.to(tl.int32))
        for start in range(total // TILE, tl.cdiv(CAP, TILE)):
            position = start * TILE + lane
            tl.store(base + position, 0, (position >= total) & (position < CAP))
    tl.store(OUT_COUNTS + peer * COUNT_STRIDE + row, total)


def partition_indices(index, counts, ring_size, peer_blocks):
    if not index.is_cuda:
        return _partition_indices_reference(index, counts, ring_size, peer_blocks)
    cap = min(peer_blocks, index.shape[-1])
    steps = ring_size
    rows = counts.numel()
    stride, count_stride = triton.cdiv(rows * cap, 4) * 4, triton.cdiv(rows, 4) * 4
    indices = torch.empty((steps, stride), device=index.device, dtype=torch.int32)
    numbers = torch.empty((steps, count_stride), device=index.device, dtype=torch.int32)
    _partition_kernel[(rows, steps)](
        index, counts, indices, numbers, index.shape[-1], cap, ring_size, stride,
        count_stride, min(2048, triton.next_power_of_2(index.shape[-1])),
        num_warps=4)
    return (indices[:, :rows * cap].view(steps, *index.shape[:-1], cap),
            numbers[:, :rows].view(steps, *counts.shape))


@triton.jit
def _accumulate_gradients(DQ, DK, DV, DQ_ACC, KV_ACC, INCOMING,
                          N: tl.constexpr, NKV: tl.constexpr, ADD_INCOMING: tl.constexpr,
                          TILE: tl.constexpr):
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = offset < N
    dq = tl.load(DQ + offset, valid, other=0).to(tl.float32)
    valid_kv = offset < NKV
    dk = tl.load(DK + offset, valid_kv, other=0).to(tl.float32)
    dv = tl.load(DV + offset, valid_kv, other=0).to(tl.float32)
    dq += tl.load(DQ_ACC + offset, valid, other=0)
    if ADD_INCOMING:
        dk += tl.load(INCOMING + offset, valid_kv, other=0)
        dv += tl.load(INCOMING + NKV + offset, valid_kv, other=0)
    tl.store(DQ_ACC + offset, dq, valid)
    tl.store(KV_ACC + offset, dk, valid_kv)
    tl.store(KV_ACC + NKV + offset, dv, valid_kv)


@triton.jit
def _merge_partial(O, L, NEXT_L, PART, PART_L, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   S_H: tl.constexpr, S_T: tl.constexpr, S_D: tl.constexpr,
                   TILE: tl.constexpr):
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = offset < N * H * D
    token, head, col = offset // (H * D), offset // D % H, offset % D
    l_offset = head * N + token
    old_l = tl.load(L + l_offset, valid, other=-float("inf"))
    part_l = tl.load(PART_L + l_offset, valid, other=-float("inf"))
    maximum = tl.maximum(old_l, part_l)
    maximum = tl.where(maximum == -float("inf"), 0., maximum)
    old_weight, part_weight = tl.exp(old_l - maximum), tl.exp(part_l - maximum)
    denominator = old_weight + part_weight
    inverse = tl.where(denominator > 0, 1. / denominator, 0.)
    old_o = tl.load(O + offset, valid, other=0.)
    part_o = tl.load(PART + head * S_H + token * S_T + col * S_D, valid, other=0.).to(tl.float32)
    merged = tl.where(old_weight > 0, old_o * old_weight, 0.)
    merged += tl.where(part_weight > 0, part_o * part_weight, 0.)
    tl.store(O + offset, merged * inverse, valid)
    # Separate LSE buffers avoid a cross-warp read/write race: every D lane
    # reads the row's old LSE, while only one lane writes its new value.
    tl.store(NEXT_L + l_offset, maximum + tl.log(denominator), valid & (col == 0))
