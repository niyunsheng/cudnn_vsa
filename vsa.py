"""Public packed VSA entry: validate, pad, attend, restore and merge."""
import torch

import vsa_core
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
