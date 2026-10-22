"""Ring transport and autograd scheduling over padded VSA slots."""
from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton

import vsa_core
import vsa_kernels


@dataclass
class _RingRuntime:
    group: object
    rank: int
    size: int
    stream: object
    batch_p2p: bool
    bwd_bucket_size: int | None
    peers: tuple


def _runtime(layout, group, device, batch_p2p, bwd_bucket_size):
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
    key = (group, device, batch_p2p, bwd_bucket_size)
    if key not in layout._runtime:
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        if (size, rank) != (layout.ring_size, layout.ring_rank):
            raise ValueError("Ring layout rank/size differ from the supplied process group")
        ranks = dist.get_process_group_ranks(group)
        layout._runtime[key] = _RingRuntime(
            group, rank, size, torch.cuda.Stream(device=device), batch_p2p,
            bwd_bucket_size, tuple(ranks))
    return layout._runtime[key]


def _exchange(runtime, pairs, *, reverse=False):
    """Enqueue matched transfers on the communication stream.

    Callers keep buffers live until the compute stream joins this stream.
    Waiting Work objects here establishes CUDA stream dependencies.
    """
    compute = torch.cuda.current_stream()
    runtime.stream.wait_stream(compute)
    direction = -1 if reverse else 1
    next_peer = runtime.peers[(runtime.rank + direction) % runtime.size]
    previous_peer = runtime.peers[(runtime.rank - direction) % runtime.size]
    with torch.cuda.stream(runtime.stream):
        requests = []
        for send, receive in pairs:
            send_op = dist.P2POp(dist.isend, send, next_peer, runtime.group)
            receive_op = dist.P2POp(dist.irecv, receive, previous_peer, runtime.group)
            parity = runtime.rank % 2
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
            sizes = layout.peer_block_sizes[owner]
            current = buffers[step % len(buffers)]
            if step + 1 < steps:
                _exchange(runtime, [(current, buffers[(step + 1) % 2])])
            result = vsa_core.bsa_forward(q, current[0], current[1], indices[owner],
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
            sizes = layout.peer_block_sizes[owner]
            current = buffers[step % len(buffers)]
            pairs = []
            if step + 1 < steps:
                pairs.append((current, buffers[(step + 1) % 2]))
            if step:
                pairs.append((accumulated, incoming))
            if pairs:
                _exchange(runtime, pairs, reverse=True)
            grads = vsa_core.bsa_backward(
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


def attention_slots(q, k, v, layout, group, *, scale, batch_p2p=None, bwd_bucket_size="auto"):
    """Return local fine slots and compact coarse output; row handling is caller-owned."""
    runtime = _runtime(layout, group, q.device, batch_p2p, bwd_bucket_size)
    indices = layout.routing_indices
    means = None
    if not layout.dense_only:
        qm, km, vm = vsa_kernels.ring_block_means(q, k, v, layout.block_sizes)
        gathered = _GatherMeans.apply(torch.stack((km, vm), dim=1), runtime)
        means = gathered.index_select(0, layout.global_to_gather)
        indices = vsa_core.ring_route(qm.detach(), means[:, 0].detach(), layout)
    peer_indices, peer_counts = vsa_kernels.partition_indices(
        indices, layout.routing_counts, runtime.size, layout.blocks)
    fine = _RingFine.apply(q, k, v, peer_indices, peer_counts, layout, runtime, scale)
    if means is None:
        return fine, None
    if layout.coarse_blocks.numel() == 0:
        # Peers without coarse queries must still run the gather's reduce-scatter
        # adjoint. Keep the zero dependency in autograd; never skip the collective.
        return fine + means.sum() * 0, None
    cq = qm.index_select(0, layout.coarse_blocks).to(q.dtype)
    ckv = means.index_select(0, layout.coarse_kv_blocks).to(q.dtype)
    coarse = vsa_core.coarse_attention(
        cq, ckv[:, 0].contiguous(), ckv[:, 1].contiguous(),
        layout.coarse_cu_q, layout.coarse_max_q, scale,
        cu_seqlens_kv=layout.coarse_cu_kv, max_seqlen_kv=layout.coarse_max_kv)
    return fine, coarse
