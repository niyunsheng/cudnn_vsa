"""Public packed VSA entry: validate, pad, attend, restore and merge."""
import torch

import vsa_core
import vsa_kernels
from vsa_layout import RingVSALayout, VSALayout


def _reorder_tokens(x, layout, inverse):
    if not isinstance(layout, VSALayout) or layout.input_order != "block":
        raise ValueError('Token reordering requires a full VSALayout with input_order="block"')
    if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[0] != layout.tokens:
        raise ValueError("Expected one tensor with shape [layout.tokens, ...]")
    if any(size == 0 for size in x.shape[1:]) or not x.is_contiguous():
        raise ValueError("Token-reorder input must be contiguous with nonempty row dimensions")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                       torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool):
        raise ValueError("Unsupported token-reorder dtype")
    if not x.is_cuda or x.device != layout.input_rows.device:
        raise ValueError("Token-reorder input and layout must be on the same CUDA device")
    return vsa_kernels.reorder_tokens(x, layout.input_rows, inverse=inverse)


def to_block_order(x, *, vsa_layout):
    """Reorder one raster tensor into compact blocks, without padding.

    Accept and return one contiguous CUDA [T, ...] tensor, preserving its shape
    and dtype. Use a full block layout and call once at the stack's input
    boundary. Gradients apply the inverse permutation.
    Already block-ordered inputs do not need this conversion.
    """
    return _reorder_tokens(x, vsa_layout, inverse=False)


def to_raster_order(x, *, vsa_layout):
    """Restore one compact block tensor to raster order, without padding.

    Return one tensor with the same shape and dtype. Use the same full block
    layout as to_block_order at the stack's output boundary if needed.
    """
    return _reorder_tokens(x, vsa_layout, inverse=True)


def _validate(q, k, v, gate, *, allow_empty=False):
    if q is None or k is None or v is None:
        raise ValueError("Q, K and V are required separate tensors")
    if gate is None:
        raise ValueError("VSA requires a gate tensor; use FA4 full attention without a gate")
    if q.ndim != 3 or any(n <= 0 for n in q.shape[1:]) or (
        q.shape[0] == 0 and not allow_empty
    ):
        raise ValueError("Q/K/V must have shape [tokens, heads, head_dim] with positive dimensions")
    for x in (q, k, v, gate):
        if x.shape != q.shape or x.dtype != q.dtype or x.device != q.device:
            raise ValueError("Q/K/V/gate must have identical shape, dtype and device")
        if not x.is_contiguous():
            raise ValueError("Q/K/V/gate must be contiguous; materialize projections before calling attention")


def vsa_attention(q, k, v, gate, vsa_layout, *, scale=None, ring_cp_group=None,
                  ring_cp_batch_p2p=None, ring_cp_bwd_bucket_size="auto"):
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

    An explicit ring_cp_group requires a RingVSALayout with compact local
    block inputs, including size-one groups. Pass group-local rank/size to
    build_vsa_layout with input_order="block"; the caller owns whole-block
    sharding and any external Ulysses transformations. ring_cp_batch_p2p=None chooses batched
    P2P for device-bound groups and individual P2P for lazy groups.
    ring_cp_bwd_bucket_size="auto" selects the measured SM100/H2/local-128-
    block/single-sample setting; None uses the backend default, and a positive
    integer overrides it.
    """
    layout = vsa_layout
    is_ring = ring_cp_group is not None
    if is_ring != isinstance(layout, RingVSALayout):
        raise ValueError("RingVSALayout requires an explicit ring_cp_group, and vice versa")
    _validate(q, k, v, gate, allow_empty=is_ring)
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
    if is_ring:
        scale = q.shape[-1]**-0.5 if scale is None else scale
        fine, coarse = vsa_core.ring_attention_slots(
            q_slots, k_slots, v_slots, layout, ring_cp_group, scale=scale,
            batch_p2p=ring_cp_batch_p2p, bwd_bucket_size=ring_cp_bwd_bucket_size)
    else:
        fine, coarse = vsa_core.attention_slots(q_slots, k_slots, v_slots, layout, scale=scale)
    out = fine if layout.slots_are_input else vsa_kernels.from_slots(
        fine, layout.slot_row, layout.row_slot)
    if coarse is not None:
        out = vsa_kernels.fine_coarse_gate_add(
            out, gate, coarse, layout.row_coarse, layout.coarse_rows)
    elif is_ring and not layout.dense_only:
        # Fine-only peers still need a zero gate gradient, including NaN gates.
        out = out + gate.masked_fill(torch.ones_like(gate, dtype=torch.bool), 0).sum() * 0
    return out.contiguous()
