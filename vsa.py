"""Public packed VSA entry: validate, pad, attend, restore and merge."""
import torch

import vsa_core
import vsa_kernels
from vsa_layout import RingVSALayout


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
    """CUDA BF16 [T,H,128] Q/K/V/gate -> output in the supplied token order.

    Reuse the current geometry's layout across layers. Non-ring calls accept
    original FHW/raster or compact block inputs as declared by input_order.
    An explicit ring group requires build_ring_vsa_layout and compact local
    block rows, even for a size-one group. Inputs never contain padding;
    this entry owns slot padding. A one-time FHW-to-block reorder before the
    VSA stack and whole-block dataset sharding belong to the caller.
    The caller owns groups and any external Ulysses transformations.

    batch_p2p=None selects batched P2P for device-bound groups and individual
    P2P for lazy groups. bwd_bucket_size="auto" uses the measured SM100/H2/
    local-128-block/single-sample choice; None uses the backend default, and a
    positive integer overrides it. Non-ring calls retain the single-device path.
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
    scale = q.shape[-1]**-0.5 if scale is None else scale
    if layout.slots_are_input:
        slots = q, k, v
    elif layout.input_order == "block":
        slots = vsa_kernels.to_slots_qkv(q, k, v, layout.slot_row, layout.row_slot)
    else:
        slots = tuple(vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot) for x in (q, k, v))
    if is_ring:
        from vsa_ring import attention_slots
        fine, coarse = attention_slots(*slots, layout, ring_cp_group, scale=scale,
                                       batch_p2p=ring_cp_batch_p2p,
                                       bwd_bucket_size=ring_cp_bwd_bucket_size)
    else:
        fine, coarse = vsa_core.attention_slots(*slots, layout, scale=scale)
    out = fine if layout.slots_are_input else vsa_kernels.from_slots(
        fine, layout.slot_row, layout.row_slot)
    if coarse is not None:
        out = vsa_kernels.fine_coarse_gate_add(
            out, gate, coarse, layout.row_coarse, layout.coarse_rows)
    elif not layout.dense_only:
        # A ring peer can have only fine queries in a globally mixed pack.
        # Its gate gradient is zero even for NaN/Inf gates on ignored rows.
        out = out + gate.masked_fill(torch.ones_like(gate, dtype=torch.bool), 0).sum() * 0
    return out.contiguous()
