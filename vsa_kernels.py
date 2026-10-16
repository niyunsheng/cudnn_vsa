"""VSA tensor operations: slot conversion, pooling, gated merge and backward."""
import math

import torch
import triton
import triton.language as tl

from vsa_layout import BLOCK


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


@torch.compile(fullgraph=True)
def block_means(q, k, v, block_sizes, coarse_blocks):
    """Return detached FP32 routing Q/K and differentiable coarse Q/K/V.

    Coarse means use the input dtype (BF16 in VSA). Include all regions of
    VSA samples and exclude entire fine-only samples.
    """
    q_mean, k_mean, v_mean = (
        x.view(block_sizes.numel(), BLOCK, *x.shape[1:])[coarse_blocks].sum(1, dtype=torch.float32)
        / block_sizes[coarse_blocks, None, None] for x in (q, k, v)
    )
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
