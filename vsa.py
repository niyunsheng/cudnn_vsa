"""Public packed VSA entry: validate, pad, attend, restore and merge."""
import torch

import vsa_core
import vsa_kernels
from vsa_layout import VSALayout


def _reorder_tokens(tensors, layout, inverse):
    if not isinstance(layout, VSALayout) or layout.input_order != "block":
        raise ValueError('Token reordering requires a full VSALayout with input_order="block"')
    if not tensors:
        raise ValueError("At least one tensor is required")
    for x in tensors:
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[0] != layout.tokens:
            raise ValueError("Each tensor must have shape [layout.tokens, ...]")
        if any(size == 0 for size in x.shape[1:]) or not x.is_contiguous():
            raise ValueError("Token-reorder tensors must be contiguous with nonempty row dimensions")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                           torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool):
            raise ValueError("Unsupported token-reorder dtype")
        if not x.is_cuda or x.device != layout.input_rows.device:
            raise ValueError("Token-reorder tensors and layout must be on the same CUDA device")
    return vsa_kernels.reorder_tokens(tensors, layout.input_rows, inverse=inverse)


def to_block_order(*tensors, vsa_layout):
    """Fuse raster-to-compact-block reordering without padding; return a tuple.

    Accept contiguous CUDA [T, ...] tensors with different row shapes/dtypes,
    such as hidden states, Q/K/V/gate and integer positions. Use a full block
    layout and call once before the VSA stack, before any sequence sharding.
    Every tensor retains its shape and dtype; gradients invert the permutation.
    Already block-ordered inputs do not need this conversion.
    """
    return _reorder_tokens(tensors, vsa_layout, inverse=False)


def to_raster_order(*tensors, vsa_layout):
    """Restore compact block tensors to raster order, without padding.

    Return a tuple, including for one tensor. Use the same full block layout
    as to_block_order, after the VSA stack if the caller needs raster outputs.
    """
    return _reorder_tokens(tensors, vsa_layout, inverse=True)


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
    The caller reorders tokens and positions once before the entire VSA stack.
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
    fine, coarse = vsa_core.attention_slots(q_slots, k_slots, v_slots, layout, scale=scale)
    out = fine if layout.slots_are_input else vsa_kernels.from_slots(
        fine, layout.slot_row, layout.row_slot)
    if coarse is not None:
        out = vsa_kernels.fine_coarse_gate_add(
            out, gate, coarse, layout.row_coarse, layout.coarse_rows)
    return out.contiguous()
