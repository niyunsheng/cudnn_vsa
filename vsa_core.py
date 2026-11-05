"""Shared routing/backends and single-device or ring attention execution."""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
from cudnn import BSA
from flash_attn.cute.interface import flash_attn_varlen_func

import vsa_kernels
from vsa_layout import BLOCK


def bsa_forward(q, k, v, indices, counts, sizes, scale, *, allow_empty=False):
    """Use BHSD views of THD storage to avoid frontend Q/K/V copies."""
    qh, kh, vh = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
    return BSA.block_sparse_attention_forward(
        qh, kh, vh, indices, None, sizes, counts, sparse_block_size=BLOCK,
        softmax_scale=scale, layout="bhsd", allow_empty_block_nums=allow_empty)


def bsa_backward(grad, q, k, v, out, lse, indices, counts, sizes, scale, *, bucket_size=None):
    """Backward consumes final O/LSE; ring callers supply the global softmax."""
    grads = BSA.block_sparse_attention_backward(
        grad.contiguous().unsqueeze(0), q.unsqueeze(0), k.unsqueeze(0),
        v.unsqueeze(0), out.unsqueeze(0), lse, indices, None, sizes, counts,
        softmax_scale=scale, sparse_block_size=BLOCK, layout="bshd",
        bucket_size_blocks=bucket_size)
    return tuple(grads[name].squeeze(0) for name in ("dq_tensor", "dk_tensor", "dv_tensor"))


class BlockSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, counts, sizes, scale):
        result = bsa_forward(q, k, v, indices, counts, sizes, scale)
        out = result["o_tensor"].squeeze(0).transpose(0, 1).contiguous()
        ctx.save_for_backward(q, k, v, out, result["lse_tensor"], indices, counts)
        ctx.block_sizes, ctx.scale = sizes, scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse, indices, counts = ctx.saved_tensors
        grads = bsa_backward(grad_out, q, k, v, out, lse, indices, counts,
                             ctx.block_sizes, ctx.scale)
        return (*grads, None, None, None, None)


@contextmanager
def fp32_routing(device_type):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _route_group(q, k, group, index, *, key_offset=0):
    """Fill one sample's query rows using its global per-region KV budgets."""
    query = q.transpose(0, 1)
    cursor = group.dense_blocks_num
    for start, end, keep in group.video_spans:
        if keep < end - start:
            with fp32_routing(q.device.type):
                key = k[start+key_offset:end+key_offset].permute(1, 2, 0)
                scores = query @ key
            selected = scores.topk(keep, dim=-1, sorted=False).indices.to(torch.int32)
            index[..., cursor:cursor+keep] = selected.sort(-1).values + start
        cursor += keep


@torch.no_grad()
def route(q, k, layout):
    """Shared router; layout prepares pooled-mean offsets and sample-local rows."""
    if not layout.routing_groups:
        return layout.routing_indices
    index = layout.routing_indices.clone()
    for g, begin, end, offset in layout.routing_groups:
        _route_group(q[begin+offset:end+offset], k, g,
                     index[:, :, begin:end], key_offset=offset)
    return index


def coarse_attention(q, k, v, cu_seqlens, max_seqlen, scale=None, *,
                     cu_seqlens_kv=None, max_seqlen_kv=None):
    """BF16 FA4 varlen; ring Q and KV can have different per-sample lengths."""
    scale = q.shape[-1]**-0.5 if scale is None else scale
    out, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens if cu_seqlens_kv is None else cu_seqlens_kv,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen if max_seqlen_kv is None else max_seqlen_kv,
        softmax_scale=scale, causal=False)
    return out


def attention_slots(q, k, v, layout, *, scale=None):
    """Internal cuDNN BSA fine and FA4 coarse attention.

    Returns fine in slot order and compact coarse (None for entirely dense packs).
    No projection, CP communication, row conversion or training-framework imports.
    """
    scale = q.shape[-1]**-0.5 if scale is None else scale
    indices = layout.routing_indices
    if not layout.dense_only:
        q_route, k_route, q_coarse, k_coarse, v_coarse = vsa_kernels.routing_and_coarse_means(
            q, k, v, layout.block_sizes, layout.coarse_blocks)
        indices = route(q_route, k_route, layout)
    sizes = None if layout.all_blocks_full else layout.block_sizes
    fine = BlockSparseAttention.apply(q, k, v, indices, layout.routing_counts, sizes, scale)
    # Enqueue the long fine kernel first so GPU execution
    # covers coarse's host setup. Both run on the current stream; no sync here.
    coarse = None
    if not layout.dense_only:
        coarse = coarse_attention(q_coarse, k_coarse, v_coarse, layout.coarse_cu_q,
                                  layout.coarse_max_q, scale=scale)
    return fine, coarse


@dataclass
class _RingRuntime:
    group: object
    bwd_group: object
    rank: int
    size: int
    stream: object
    batch_p2p: bool
    bwd_batch_p2p: bool
    bwd_bucket_size: int | None
    peers: tuple


def _ring_runtime(layout, device):
    ring = layout.ring
    group, batch_p2p, bwd_bucket_size = ring.group, ring.batch_p2p, ring.bwd_bucket_size
    bwd_batch_p2p = batch_p2p
    if batch_p2p is None:
        # Eager NCCL groups serialize individual P2P operations on their
        # parent communicator. Lazy groups can use independent pair comms.
        batch_p2p = group.bound_device_id is not None
        bwd_batch_p2p = ring.bwd_group.bound_device_id is not None
    if bwd_bucket_size == "auto":
        # A bounded tuning entry, not a claim about unmeasured geometries.
        # Splitting dense-KV work helps H2/B128 single-sample backward on
        # SM100. Packed samples already split that work and retain defaults.
        bwd_bucket_size = (
            64 if layout.head_num == 2 and layout.blocks == 128
            and len(layout.groups) == 1
            and torch.cuda.get_device_capability(device) == (10, 0) else None)
    key = (device, batch_p2p, bwd_batch_p2p, bwd_bucket_size)
    if key not in ring._runtime:
        ranks = dist.get_process_group_ranks(group)
        ring._runtime[key] = _RingRuntime(
            group, ring.bwd_group, ring.rank, ring.size, torch.cuda.Stream(device=device),
            batch_p2p, bwd_batch_p2p,
            bwd_bucket_size, tuple(ranks))
    return ring._runtime[key]


def _ring_exchange(runtime, pairs, *, reverse=False):
    """Enqueue matched transfers on the communication stream.

    Callers keep buffers live until the compute stream joins this stream.
    Waiting Work objects here establishes CUDA stream dependencies.
    """
    compute = torch.cuda.current_stream()
    runtime.stream.wait_stream(compute)
    direction = -1 if reverse else 1
    group = runtime.bwd_group if reverse else runtime.group
    batch_p2p = runtime.bwd_batch_p2p if reverse else runtime.batch_p2p
    next_peer = runtime.peers[(runtime.rank + direction) % runtime.size]
    previous_peer = runtime.peers[(runtime.rank - direction) % runtime.size]
    with torch.cuda.stream(runtime.stream):
        requests = []
        for send, receive in pairs:
            send_op = dist.P2POp(dist.isend, send, next_peer, group)
            receive_op = dist.P2POp(dist.irecv, receive, previous_peer, group)
            parity = runtime.rank % 2
            ordered = (send_op, receive_op) if parity == 0 else (receive_op, send_op)
            requests.extend(ordered)
        if batch_p2p:
            requests = dist.batch_isend_irecv(requests)
        else:
            # Lazy groups use pair communicators. Alternating the posting
            # order avoids a circular wait during communicator initialization.
            requests = [op.op(op.tensor, op.peer, op.group) for op in requests]
        for work in requests:
            work.wait()


class _GatherMeans(torch.autograd.Function):
    """Gather FP32 K / BF16 V, returning FP32 pooled gradients after SUM."""

    @staticmethod
    def forward(ctx, k, v, runtime):
        ctx.runtime = runtime
        # Cast inside this boundary so reduced dV reaches FP32 pooling directly.
        v = v.to(torch.bfloat16)
        if runtime.size == 1:
            return k, v
        # Preserve each field's bits in one collective, without dtype promotion.
        # D=128 keeps both reinterpreted fields naturally aligned.
        key_bytes = k.shape[-1] * k.element_size()
        packed = torch.cat((k.contiguous().view(torch.uint8), v.contiguous().view(torch.uint8)), dim=-1)
        gathered = packed.new_empty((runtime.size * packed.shape[0], *packed.shape[1:]))
        dist.all_gather_into_tensor(gathered, packed, group=runtime.group)
        return (gathered[..., :key_bytes].view(torch.float32),
                gathered[..., key_bytes:].view(torch.bfloat16))

    @staticmethod
    def backward(ctx, dk, dv):
        runtime = ctx.runtime
        if runtime.size == 1:
            return dk.float(), dv.float(), None
        # Unused outputs receive materialized zero gradients on every rank.
        grad = torch.stack((dk.float(), dv.float()), dim=1)
        local = grad.new_empty((grad.shape[0] // runtime.size, *grad.shape[1:]))
        dist.reduce_scatter_tensor(local, grad, group=runtime.bwd_group)
        return local[:, 0], local[:, 1], None


class _RingFine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, counts, layout, runtime, scale):
        kv = torch.stack((k, v))
        steps = runtime.size
        buffers = [kv, torch.empty_like(kv)] if steps > 1 else [kv]
        out = torch.zeros_like(q, dtype=torch.float32)
        lse = torch.full((1, q.shape[1], q.shape[0]), -torch.inf,
                         dtype=torch.float32, device=q.device)
        next_lse = torch.empty_like(lse)
        for step in range(steps):
            owner = (runtime.rank - step) % runtime.size
            sizes = layout.ring.peer_block_sizes[owner]
            current = buffers[step % len(buffers)]
            if step + 1 < steps:
                _ring_exchange(runtime, [(current, buffers[(step + 1) % 2])])
            result = bsa_forward(q, current[0], current[1], indices[owner],
                                          counts[owner], sizes, scale, allow_empty=True)
            part = result["o_tensor"]
            vsa_kernels._merge_partial[(triton.cdiv(out.numel(), 1024),)](
                out, lse, next_lse, part, result["lse_tensor"], q.shape[0], q.shape[1], q.shape[2],
                part.stride(1), part.stride(2), part.stride(3), 1024)
            lse, next_lse = next_lse, lse
            if step + 1 < steps:
                torch.cuda.current_stream().wait_stream(runtime.stream)
        out = out.to(q.dtype)
        # Start backward at the final forward shard and traverse the ring
        # in reverse. Its last shard then belongs to this rank, eliminating
        # the final gradient-return hop.
        ctx.save_for_backward(q, current, out, lse, indices, counts)
        ctx.layout, ctx.runtime, ctx.scale = layout, runtime, scale
        return out

    @staticmethod
    def backward(ctx, grad):
        q, saved_kv, out, lse, indices, counts = ctx.saved_tensors
        runtime, layout = ctx.runtime, ctx.layout
        steps = runtime.size
        # Ping-pong receives must not mutate saved tensors: retained-graph
        # backward can revisit this same shard.
        kv = saved_kv.clone() if steps > 1 else saved_kv
        buffers = [kv, torch.empty_like(kv)] if steps > 1 else [kv]
        dq = torch.zeros_like(q, dtype=torch.float32)
        accumulated = None
        incoming = torch.empty_like(kv, dtype=torch.float32)
        grad = grad.contiguous()
        for step in range(steps):
            forward_step = steps - 1 - step
            owner = (runtime.rank - forward_step) % runtime.size
            sizes = layout.ring.peer_block_sizes[owner]
            current = buffers[step % len(buffers)]
            pairs = []
            if step + 1 < steps:
                pairs.append((current, buffers[(step + 1) % 2]))
            if step:
                pairs.append((accumulated, incoming))
            if pairs:
                _ring_exchange(runtime, pairs, reverse=True)
            grads = bsa_backward(
                grad, q, current[0], current[1], out, lse, indices[owner], counts[owner],
                sizes, ctx.scale, bucket_size=runtime.bwd_bucket_size)
            if pairs:
                torch.cuda.current_stream().wait_stream(runtime.stream)
            accumulated = torch.empty_like(incoming)
            partials = [x.contiguous() for x in grads]
            vsa_kernels._accumulate_gradients[(triton.cdiv(current[0].numel(), 1024),)](
                *partials, dq, accumulated, incoming, q.numel(), current[0].numel(), bool(step), 1024)
        return (dq.to(q.dtype), accumulated[0].to(q.dtype), accumulated[1].to(q.dtype),
                None, None, None, None, None)


def ring_attention_slots(q, k, v, layout, *, scale):
    """Return local fine slots and compact coarse output; row handling is caller-owned."""
    runtime = _ring_runtime(layout, q.device)
    indices = layout.routing_indices
    if not layout.dense_only:
        qm, km, vm = vsa_kernels.block_means(q, k, v, layout.block_sizes)
        km, vm = _GatherMeans.apply(km, vm, runtime)
        km = km.index_select(0, layout.ring.global_to_gather)
        vm = vm.index_select(0, layout.ring.global_to_gather)
        indices = route(qm.detach(), km.detach(), layout)
    peer_indices, peer_counts = vsa_kernels.partition_indices(
        indices, layout.routing_counts, layout.ring.global_to_gather, runtime.size, layout.blocks)
    fine = _RingFine.apply(q, k, v, peer_indices, peer_counts, layout, runtime, scale)
    if layout.dense_only:
        return fine, None
    if layout.coarse_blocks.numel() == 0:
        # Peers without coarse queries must still run the gather's reduce-scatter
        # adjoint. Keep the zero dependency in autograd; never skip the collective.
        return fine + (km.sum() + vm.float().sum()) * 0, None
    cq = qm.index_select(0, layout.coarse_blocks).to(q.dtype)
    ck = km.index_select(0, layout.coarse_kv_blocks).to(q.dtype)
    cv = vm.index_select(0, layout.coarse_kv_blocks)
    coarse = coarse_attention(
        cq, ck, cv,
        layout.coarse_cu_q, layout.coarse_max_q, scale,
        cu_seqlens_kv=layout.coarse_cu_kv, max_seqlen_kv=layout.coarse_max_kv)
    return fine, coarse
