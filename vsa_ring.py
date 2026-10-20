"""Ring VSA on packed THD blocks, independent of external Ulysses transport."""
from dataclasses import dataclass
from math import gcd

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from cudnn import BSA
from flash_attn.cute.interface import flash_attn_varlen_func

import vsa_kernels
from vsa import _validate, fp32_routing
from vsa_layout import BLOCK, RingVSALayout


@dataclass
class _RingRuntime:
    group: object
    rank: int
    size: int
    stream: object
    batch_p2p: bool
    bwd_bucket_size: int | None
    fwd_use_clc: bool | None
    fwd_kv_splits: int
    chunk_size: int
    peers: tuple
    chunk_block_sizes: object


def _runtime(layout, group, device, batch_p2p, bwd_bucket_size, fwd_use_clc, fwd_kv_splits, chunk_size):
    if batch_p2p is None:
        # Eager NCCL groups serialize individual P2P operations on their
        # parent communicator. Lazy groups can use independent pair comms.
        batch_p2p = group.bound_device_id is not None
    if bwd_bucket_size == "auto":
        # A bounded tuning entry, not a claim about unmeasured geometries.
        # Splitting dense-KV work helps H2/B128 single-sample backward on
        # SM100. Packed samples already split that work and retain defaults.
        bwd_bucket_size = (
            64 if layout.head_num == 2 and layout.blocks == 128
            and len(layout.groups) == 1
            and torch.cuda.get_device_capability(device) == (10, 0) else None)
    if bwd_bucket_size is not None and (
        isinstance(bwd_bucket_size, bool) or not isinstance(bwd_bucket_size, int) or bwd_bucket_size < 1
    ):
        raise ValueError('ring_cp_bwd_bucket_size must be "auto", a positive integer or None')
    if fwd_use_clc == "auto":
        fwd_use_clc = None
    if fwd_use_clc is not None and not isinstance(fwd_use_clc, bool):
        raise ValueError('ring_cp_fwd_use_clc must be "auto", a bool or None')
    if isinstance(fwd_kv_splits, bool) or not isinstance(fwd_kv_splits, int) or not 1 <= fwd_kv_splits <= 256:
        raise ValueError("ring_cp_fwd_kv_splits must be an integer in [1, 256]")
    if (isinstance(chunk_size, bool) or not isinstance(chunk_size, int)
            or chunk_size not in (1, 2, 4) or layout.ring_size % chunk_size):
        raise ValueError("ring_cp_chunk_size must be 1, 2 or 4 and divide the ring size")
    key = (group, device, batch_p2p, bwd_bucket_size, fwd_use_clc, fwd_kv_splits, chunk_size)
    if key not in layout._runtime:
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        if (size, rank) != (layout.ring_size, layout.ring_rank):
            raise ValueError("Ring layout rank/size differ from the supplied process group")
        ranks = dist.get_process_group_ranks(group)
        chunk_sizes = (None if chunk_size == 1 else
                       _window_block_sizes(layout.peer_block_sizes, chunk_size, rank))
        layout._runtime[key] = _RingRuntime(
            group, rank, size, torch.cuda.Stream(device=device), batch_p2p, bwd_bucket_size,
            fwd_use_clc, fwd_kv_splits, chunk_size, tuple(ranks), chunk_sizes)
    return layout._runtime[key]


def _window_block_sizes(peer_sizes, chunk_size, rank):
    """Keep every concatenated window's base aligned for cuDNN metadata."""
    size, blocks = peer_sizes.shape
    count = chunk_size * blocks
    storage = peer_sizes.new_empty((size // chunk_size, triton.cdiv(count, 4) * 4))
    windows = storage[:, :count]
    for step in range(size // chunk_size):
        windows[step].copy_(torch.cat([
            peer_sizes[(rank - step * chunk_size - index) % size]
            for index in range(chunk_size)]))
    return windows


def _exchange(runtime, pairs, *, hop=1, reverse=False):
    """Enqueue matched transfers on the communication stream.

    Callers keep buffers live until the compute stream joins this stream.
    Waiting Work objects here establishes CUDA stream dependencies.
    """
    compute = torch.cuda.current_stream()
    runtime.stream.wait_stream(compute)
    direction = -hop if reverse else hop
    next_peer = runtime.peers[(runtime.rank + direction) % runtime.size]
    previous_peer = runtime.peers[(runtime.rank - direction) % runtime.size]
    with torch.cuda.stream(runtime.stream):
        requests = []
        for send, receive in pairs:
            send_op = dist.P2POp(dist.isend, send, next_peer, runtime.group)
            receive_op = dist.P2POp(dist.irecv, receive, previous_peer, runtime.group)
            parity = runtime.rank // gcd(runtime.size, hop) % 2
            ordered = (send_op, receive_op) if parity == 0 else (receive_op, send_op)
            requests.extend(ordered)
        if runtime.batch_p2p:
            requests = dist.batch_isend_irecv(requests)
        else:
            # Lazy groups use pair communicators. Alternating the posting
            # order avoids a circular wait during communicator initialization.
            requests = [op.op(op.tensor, op.peer, op.group) for op in requests]
        for work in requests:
            work.wait()


def _kv_window(k, v, runtime):
    """Coalesce adjacent owner shards, preserving whole physical blocks."""
    current = torch.stack((k, v))
    if runtime.chunk_size == 1:
        return current
    tokens = k.shape[0]
    window = current.new_empty((2, runtime.chunk_size * tokens, *k.shape[1:]))
    window[:, :tokens].copy_(current)
    for index in range(1, runtime.chunk_size):
        incoming = torch.empty_like(current)
        _exchange(runtime, [(current, incoming)])
        torch.cuda.current_stream().wait_stream(runtime.stream)
        window[:, index*tokens:(index+1)*tokens].copy_(incoming)
        current = incoming
    return window


@torch.compile(fullgraph=True)
def _pool(q, k, v, sizes):
    shape = (sizes.numel(), BLOCK, *q.shape[1:])
    denominator = sizes.clamp_min(1)[:, None, None]
    return tuple(x.view(shape).sum(1, dtype=torch.float32) / denominator for x in (q, k, v))


class _GatherMeans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, runtime):
        ctx.runtime = runtime
        if runtime.size == 1:
            return means
        gathered = means.new_empty((runtime.size * means.shape[0], *means.shape[1:]))
        dist.all_gather_into_tensor(gathered, means.contiguous(), group=runtime.group)
        return gathered

    @staticmethod
    def backward(ctx, grad):
        runtime = ctx.runtime
        if runtime.size == 1:
            return grad, None
        local = grad.new_empty((grad.shape[0] // runtime.size, *grad.shape[1:]))
        dist.reduce_scatter_tensor(local, grad.contiguous(), group=runtime.group)
        return local, None


@torch.no_grad()
def _route(q, k, layout):
    """Local queries select the same global per-region top-k as CP1."""
    index = layout.routing_indices.clone()
    for group, begin, end in layout.local_groups:
        if group.dense_only or group.kept == group.blocks:
            continue
        query = q[begin:end].transpose(0, 1)
        cursor = group.dense_blocks_num
        for start, stop, keep in group.video_spans:
            if keep < stop - start:
                with fp32_routing(q.device.type):
                    scores = query @ k[start:stop].permute(1, 2, 0)
                selected = scores.topk(keep, dim=-1, sorted=False).indices
                index[0, :, begin:end, cursor:cursor + keep] = selected.sort(-1).values + start
            cursor += keep
        if layout.dense_query_full and group.dense_blocks_num:
            dense_end = min(end, max(begin, (
                group.start + group.dense_blocks_num - layout.ring_rank
                + layout.ring_size - 1) // layout.ring_size))
            index[:, :, begin:dense_end] = layout.routing_indices[:, :, begin:dense_end]
    return index


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
                      COUNT_STRIDE: tl.constexpr, TILE: tl.constexpr,
                      CHUNK: tl.constexpr, RANK: tl.constexpr, PEER_BLOCKS: tl.constexpr):
    row, peer = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, TILE)
    count = tl.load(COUNTS + row)
    base = OUT + peer * STRIDE + row * CAP
    # Preserve source order while translating IDs into a peer/window's slots.
    # Tail and active stores target disjoint addresses.
    if KEYS <= TILE:
        index = tl.load(INDEX + row * KEYS + lane, lane < count, other=0)
        owner = index % RING if CHUNK == 1 else ((RANK - index % RING + RING) % RING) // CHUNK
        local = index // RING
        if CHUNK > 1:
            local += ((RANK - index % RING + RING) % CHUNK) * PEER_BLOCKS
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
            owner = index % RING if CHUNK == 1 else ((RANK - index % RING + RING) % RING) // CHUNK
            local = index // RING
            if CHUNK > 1:
                local += ((RANK - index % RING + RING) % CHUNK) * PEER_BLOCKS
            selected = (position < count) & (owner == peer)
            destination = total + tl.cumsum(selected.to(tl.int32)) - 1
            tl.store(base + destination, local, selected)
            total += tl.sum(selected.to(tl.int32))
        for start in range(total // TILE, tl.cdiv(CAP, TILE)):
            position = start * TILE + lane
            tl.store(base + position, 0, (position >= total) & (position < CAP))
    tl.store(OUT_COUNTS + peer * COUNT_STRIDE + row, total)


def _partition_indices(index, counts, ring_size, peer_blocks, chunk_size=1, rank=0):
    if not index.is_cuda:
        if chunk_size != 1:
            raise ValueError("Coalesced peer partitioning requires CUDA")
        return _partition_indices_reference(index, counts, ring_size, peer_blocks)
    cap = min(chunk_size * peer_blocks, index.shape[-1])
    steps = ring_size // chunk_size
    rows = counts.numel()
    stride, count_stride = triton.cdiv(rows * cap, 4) * 4, triton.cdiv(rows, 4) * 4
    indices = torch.empty((steps, stride), device=index.device, dtype=torch.int32)
    numbers = torch.empty((steps, count_stride), device=index.device, dtype=torch.int32)
    _partition_kernel[(rows, steps)](
        index, counts, indices, numbers, index.shape[-1], cap, ring_size, stride,
        count_stride, min(2048, triton.next_power_of_2(index.shape[-1])),
        chunk_size, rank, peer_blocks, num_warps=4)
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


class _RingFine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, counts, layout, runtime, scale):
        kv = _kv_window(k, v, runtime)
        steps = runtime.size // runtime.chunk_size
        buffers = [kv, torch.empty_like(kv)] if steps > 1 else [kv]
        out = torch.zeros_like(q, dtype=torch.float32)
        lse = torch.full((1, q.shape[1], q.shape[0]), -torch.inf,
                         dtype=torch.float32, device=q.device)
        next_lse = torch.empty_like(lse)
        qh = q.transpose(0, 1).unsqueeze(0)
        for step in range(steps):
            owner = (runtime.rank - step) % runtime.size if runtime.chunk_size == 1 else step
            sizes = layout.peer_block_sizes[owner] if runtime.chunk_size == 1 else runtime.chunk_block_sizes[step]
            current = buffers[step % len(buffers)]
            if step + 1 < steps:
                _exchange(runtime, [(current, buffers[(step + 1) % 2])], hop=runtime.chunk_size)
            kh, vh = (x.transpose(0, 1).unsqueeze(0) for x in current.unbind(0))
            result = BSA.block_sparse_attention_forward(
                qh, kh, vh, indices[owner], None, sizes,
                counts[owner], sparse_block_size=BLOCK, allow_empty_block_nums=True,
                softmax_scale=scale, layout="bhsd", use_clc=runtime.fwd_use_clc,
                kv_splits=runtime.fwd_kv_splits)
            part = result["o_tensor"]
            _merge_partial[(triton.cdiv(out.numel(), 1024),)](
                out, lse, next_lse, part, result["lse_tensor"], q.shape[0], q.shape[1], q.shape[2],
                part.stride(1), part.stride(2), part.stride(3), 1024)
            lse, next_lse = next_lse, lse
            if step + 1 < steps:
                torch.cuda.current_stream().wait_stream(runtime.stream)
        out = out.to(q.dtype)
        # Start backward at the final forward window and traverse the ring
        # in reverse. Its last window then belongs to this rank, eliminating
        # both backward's window construction and the final gradient hop.
        ctx.save_for_backward(q, current, out, lse, indices, counts)
        ctx.layout, ctx.runtime, ctx.scale = layout, runtime, scale
        return out

    @staticmethod
    def backward(ctx, grad):
        q, saved_kv, out, lse, indices, counts = ctx.saved_tensors
        runtime, layout = ctx.runtime, ctx.layout
        steps = runtime.size // runtime.chunk_size
        # Ping-pong receives must not mutate saved tensors: retained-graph
        # backward can revisit this same window.
        kv = saved_kv.clone() if steps > 1 else saved_kv
        buffers = [kv, torch.empty_like(kv)] if steps > 1 else [kv]
        dq = torch.zeros_like(q, dtype=torch.float32)
        accumulated = None
        incoming = torch.empty_like(kv, dtype=torch.float32)
        grad = grad.contiguous()
        for step in range(steps):
            forward_step = steps - 1 - step
            owner = ((runtime.rank - forward_step) % runtime.size
                     if runtime.chunk_size == 1 else forward_step)
            sizes = (layout.peer_block_sizes[owner] if runtime.chunk_size == 1
                     else runtime.chunk_block_sizes[forward_step])
            current = buffers[step % len(buffers)]
            pairs = []
            if step + 1 < steps:
                pairs.append((current, buffers[(step + 1) % 2]))
            if step:
                pairs.append((accumulated, incoming))
            if pairs:
                _exchange(runtime, pairs, hop=runtime.chunk_size, reverse=True)
            result = BSA.block_sparse_attention_backward(
                grad.unsqueeze(0), q.unsqueeze(0), current[0].unsqueeze(0),
                current[1].unsqueeze(0), out.unsqueeze(0), lse, indices[owner], None,
                sizes, counts[owner], sparse_block_size=BLOCK,
                softmax_scale=ctx.scale, layout="bshd",
                bucket_size_blocks=runtime.bwd_bucket_size)
            if pairs:
                torch.cuda.current_stream().wait_stream(runtime.stream)
            accumulated = torch.empty_like(incoming)
            partials = [result[name].contiguous() for name in ("dq_tensor", "dk_tensor", "dv_tensor")]
            _accumulate_gradients[(triton.cdiv(current[0].numel(), 1024),)](
                *partials, dq, accumulated, incoming, q.numel(), current[0].numel(), bool(step), 1024)
        if runtime.chunk_size > 1:
            # Each strided cycle accumulated only its Q ranks. Adjacent cycles
            # return the remaining components to the original KV owners.
            window_grads = accumulated.view(2, runtime.chunk_size, *q.shape)
            own = window_grads[:, 0].contiguous()
            for index in range(1, runtime.chunk_size):
                send = window_grads[:, index].contiguous()
                receive = torch.empty_like(send)
                _exchange(runtime, [(send, receive)], hop=index, reverse=True)
                torch.cuda.current_stream().wait_stream(runtime.stream)
                own.add_(receive)
            accumulated = own
        return (dq.to(q.dtype), accumulated[0].to(q.dtype), accumulated[1].to(q.dtype),
                None, None, None, None, None)


def ring_vsa_attention(q, k, v, gate, layout, group, *, scale=None, batch_p2p=None,
                       bwd_bucket_size="auto", fwd_use_clc="auto", fwd_kv_splits=1,
                       chunk_size=1):
    """Complete VSA for one sequence shard; the explicit group contains its heads."""
    if not isinstance(layout, RingVSALayout):
        raise ValueError("ring_cp_group requires build_ring_vsa_layout metadata")
    _validate(q, k, v, gate, allow_empty=True)
    if q.shape[:2] != (layout.tokens, layout.head_num):
        raise ValueError("Ring inputs must match the layout's local tokens and heads")
    if q.device != layout.block_sizes.device:
        raise ValueError("Ring layout and activations must be on the same device")
    if not q.is_cuda or q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError("cuDNN BSA ring training requires CUDA BF16 and head_dim=128")
    runtime = _runtime(layout, group, q.device, batch_p2p, bwd_bucket_size,
                       fwd_use_clc, fwd_kv_splits, chunk_size)
    scale = q.shape[-1] ** -.5 if scale is None else scale
    if layout.slots_are_input:
        qs, ks, vs = q, k, v
    elif layout.input_order == "block":
        qs, ks, vs = vsa_kernels.to_slots_qkv(q, k, v, layout.slot_row, layout.row_slot)
    else:
        qs, ks, vs = (vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot) for x in (q, k, v))
    indices = layout.routing_indices
    means = None
    if not layout.dense_only:
        qm, km, vm = _pool(qs, ks, vs, layout.block_sizes)
        gathered = _GatherMeans.apply(torch.stack((km, vm), dim=1), runtime)
        means = gathered.index_select(0, layout.global_to_gather)
        indices = _route(qm.detach(), means[:, 0].detach(), layout)
    peer_indices, peer_counts = _partition_indices(
        indices, layout.routing_counts, runtime.size, layout.blocks,
        runtime.chunk_size, runtime.rank)
    fine_slots = _RingFine.apply(qs, ks, vs, peer_indices, peer_counts, layout, runtime, scale)
    fine = fine_slots if layout.slots_are_input else vsa_kernels.from_slots(
        fine_slots, layout.slot_row, layout.row_slot)
    if means is None:
        return fine
    if layout.coarse_blocks.numel() == 0:
        # Every peer participates in the gather's adjoint, even without queries.
        # A fine-only local sample has identically zero gate derivatives.
        return fine + (means.sum() + gate.masked_fill(torch.ones_like(gate, dtype=torch.bool), 0).sum()) * 0
    cq = qm.index_select(0, layout.coarse_blocks).to(q.dtype)
    ckv = means.index_select(0, layout.coarse_kv_blocks).to(q.dtype)
    coarse, _ = flash_attn_varlen_func(
        cq, ckv[:, 0].contiguous(), ckv[:, 1].contiguous(),
        cu_seqlens_q=layout.coarse_cu_q, cu_seqlens_k=layout.coarse_cu_kv,
        max_seqlen_q=layout.coarse_max_q, max_seqlen_k=layout.coarse_max_kv,
        softmax_scale=scale, causal=False)
    return vsa_kernels.fine_coarse_gate_add(
        fine, gate, coarse, layout.row_coarse, layout.coarse_rows)
