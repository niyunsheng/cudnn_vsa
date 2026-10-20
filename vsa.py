"""Packed multimodal VSA with cuDNN block-sparse fine and FA4 varlen coarse."""
from contextlib import contextmanager

import torch
from cudnn import BSA
from flash_attn.cute.interface import flash_attn_varlen_func

import vsa_kernels


def _validate(q, k, v, gate):
    if q is None or k is None or v is None:
        raise ValueError("Q, K and V are required separate tensors")
    if gate is None:
        raise ValueError("VSA requires a gate tensor; use FA4 full attention without a gate")
    if q.ndim != 3 or any(n <= 0 for n in q.shape):
        raise ValueError("Q/K/V must have shape [tokens, heads, head_dim] with positive dimensions")
    for x in (q, k, v, gate):
        if x.shape != q.shape or x.dtype != q.dtype or x.device != q.device:
            raise ValueError("Q/K/V/gate must have identical shape, dtype and device")
        if not x.is_contiguous():
            raise ValueError("Q/K/V/gate must be contiguous; materialize projections before calling attention")


def vsa_attention(q, k, v, gate, vsa_layout, *, scale=None):
    """[T,H,D] Q/K/V/gate -> O in the caller's original row order.

    Pass the current batch's prepared layout on every call. Q's head count must
    match layout.head_num; routing templates are ready at layout construction.
    Gate is required. Samples with only dense segments use fine only.
    Segment sparsities are resolved into top-k counts when building the layout;
    every call uses these counts and selects blocks from the current Q/K.
    layout.dense_query_full controls whether dense queries bypass top-k.
    Inputs follow layout.input_order: raster or compact block-major, without
    padding. Output uses the same order; Q/K/V/gate must share that order.
    """
    _validate(q, k, v, gate)
    layout = vsa_layout
    if q.shape[0] != layout.tokens:
        raise ValueError("Expected [layout.tokens, heads, head_dim] inputs")
    if q.shape[1] != layout.head_num:
        raise ValueError("Q/K/V head count must match VSALayout.head_num")
    if layout.block_sizes.device != q.device:
        raise ValueError("VSALayout and activations must be on the same device")
    if not q.is_cuda or q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError("cuDNN BSA training requires CUDA BF16 and head_dim=128")
    if layout.slots_are_input:
        q_slots, k_slots, v_slots = q, k, v
    elif layout.input_order == "block":
        q_slots, k_slots, v_slots = vsa_kernels.to_slots_qkv(
            q, k, v, layout.slot_row, layout.row_slot)
    else:
        q_slots, k_slots, v_slots = (
            vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot) for x in (q, k, v))
    fine, coarse = _attention_slots(q_slots, k_slots, v_slots, layout, scale=scale)
    out = fine if layout.slots_are_input else vsa_kernels.from_slots(
        fine, layout.slot_row, layout.row_slot)
    if coarse is not None:
        out = vsa_kernels.fine_coarse_gate_add(
            out, gate, coarse, layout.row_coarse, layout.coarse_rows)
    return out.contiguous()


class BlockSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, counts, sizes, scale):
        # NOTE: Use BHSD views to avoid the SM100 blk64 frontend's BSHD
        # transpose(...).contiguous() copies of Q/K/V. The kernel accepts
        # the existing THD storage through strides; D remains contiguous.
        # https://github.com/NVIDIA/cudnn-frontend/blob/v1.31.0/python/cudnn/block_sparse_attention/_interface.py#L1429
        qh, kh, vh = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
        result = BSA.block_sparse_attention_forward(
            qh, kh, vh, indices,
            None, sizes, counts, sparse_block_size=64, softmax_scale=scale,
            layout="bhsd",
        )
        # Keep O in the same THD storage order as dO/Q/K/V for BSA backward.
        out = result["o_tensor"].squeeze(0).transpose(0, 1).contiguous()
        ctx.save_for_backward(q, k, v, out, result["lse_tensor"], indices, counts)
        ctx.block_sizes = sizes
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse, indices, counts = ctx.saved_tensors
        grads = BSA.block_sparse_attention_backward(
            grad_out.contiguous().unsqueeze(0), q.unsqueeze(0), k.unsqueeze(0),
            v.unsqueeze(0), out.unsqueeze(0), lse, indices, None, ctx.block_sizes, counts,
            softmax_scale=ctx.scale, sparse_block_size=64, layout="bshd",
        )
        return (*(grads[name].squeeze(0) for name in ("dq_tensor", "dk_tensor", "dv_tensor")),
                None, None, None, None)


@contextmanager
def fp32_routing(device_type):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
def route(q, k, groups, template, *, dense_query_full):
    """Route by KV budgets, preserve dense KV and optionally exempt dense queries.

    q/k are compact FP32 [N,H,D] means; return BSA block indices.
    Per-query counts are static layout metadata and do not depend on routing.
    """
    if not any(g.kept < g.blocks for g in groups):
        return template
    index = template.clone()
    offset = 0
    for g in groups:
        if g.dense_only:
            continue
        if g.kept < g.blocks:
            query = q[offset:offset+g.blocks].transpose(0, 1)
            cursor = g.dense_blocks_num
            for start, end, keep in g.video_spans:
                if keep < end-start:
                    with fp32_routing(q.device.type):
                        key = k[offset+start-g.start:offset+end-g.start].permute(1, 2, 0)
                        scores = query @ key
                    selected = scores.topk(keep, dim=-1, sorted=False).indices.to(torch.int32)
                    index[0, :, g.start:g.end, cursor:cursor+keep] = selected.sort(-1).values + start
                cursor += keep
            if dense_query_full and g.dense_blocks_num:
                # Top-k also writes dense query rows; restore their full KV lists.
                dense_end = g.start + g.dense_blocks_num
                index[:, :, g.start:dense_end] = template[:, :, g.start:dense_end]
        offset += g.blocks
    return index


def coarse_attention(q, k, v, cu_seqlens, max_seqlen, scale=None):
    """One BF16 FA4 varlen call over the compact video-sample blocks."""
    scale = q.shape[-1]**-0.5 if scale is None else scale
    out, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=scale, causal=False)
    return out


def _attention_slots(q, k, v, layout, *, scale=None):
    """Internal cuDNN BSA fine and FA4 coarse attention.

    Returns fine in slot order and compact coarse (None for entirely dense packs).
    No projection, CP communication, row conversion or training-framework imports.
    """
    scale = q.shape[-1]**-0.5 if scale is None else scale
    indices = layout.routing_indices
    if not layout.dense_only:
        q_route, k_route, q_coarse, k_coarse, v_coarse = vsa_kernels.block_means(
            q, k, v, layout.block_sizes, layout.coarse_blocks)
        indices = route(q_route, k_route, layout.groups, indices,
            dense_query_full=layout.dense_query_full)
    sizes = None if layout.all_blocks_full else layout.block_sizes
    fine = BlockSparseAttention.apply(q, k, v, indices, layout.routing_counts, sizes, scale)
    # Enqueue the long fine kernel first so GPU execution
    # covers coarse's host setup. Both run on the current stream; no sync here.
    coarse = None
    if not layout.dense_only:
        coarse = coarse_attention(q_coarse, k_coarse, v_coarse, layout.coarse_cu_seqlens,
                                  max(layout.coarse_lengths), scale=scale)
    return fine, coarse
