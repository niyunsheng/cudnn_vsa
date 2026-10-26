"""Shared routing and backend calls, plus single-device slot attention."""
from contextlib import contextmanager

import torch
from cudnn import BSA
from flash_attn.cute.interface import flash_attn_varlen_func

import vsa_kernels
from vsa_layout import BLOCK


def bsa_forward(q, k, v, indices, counts, sizes, scale):
    """Use BHSD views of THD storage to avoid frontend Q/K/V copies."""
    qh, kh, vh = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
    return BSA.block_sparse_attention_forward(
        qh, kh, vh, indices, None, sizes, counts, sparse_block_size=BLOCK,
        softmax_scale=scale, layout="bhsd")


def bsa_backward(grad, q, k, v, out, lse, indices, counts, sizes, scale):
    """Backward uses the saved fine-attention output and log-sum-exp."""
    grads = BSA.block_sparse_attention_backward(
        grad.contiguous().unsqueeze(0), q.unsqueeze(0), k.unsqueeze(0),
        v.unsqueeze(0), out.unsqueeze(0), lse, indices, None, sizes, counts,
        softmax_scale=scale, sparse_block_size=BLOCK, layout="bshd")
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


def _route_group(q, k, group, index, template, *, key_offset=0, dense_rows=0):
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
    if dense_rows:
        index[:, :, :dense_rows] = template[:, :, :dense_rows]


@torch.no_grad()
def route(q, k, groups, template, *, dense_query_full):
    """Route Q/K compact means of samples that include coarse attention."""
    if not any(g.kept < g.blocks for g in groups):
        return template
    index = template.clone()
    offset = 0
    for g in groups:
        if g.dense_only:
            continue
        if g.kept < g.blocks:
            _route_group(q[offset:offset+g.blocks], k, g,
                         index[:, :, g.start:g.end], template[:, :, g.start:g.end],
                         key_offset=offset-g.start,
                         dense_rows=g.dense_blocks_num if dense_query_full else 0)
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


def attention_slots(q, k, v, layout, *, scale=None):
    """Internal cuDNN BSA fine and FA4 coarse attention.

    Returns fine in slot order and compact coarse (None for entirely dense packs).
    No projection, row conversion or training-framework imports.
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
