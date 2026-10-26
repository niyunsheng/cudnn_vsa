"""Compact block order preserves raster attention, padding and gradients."""
import pytest
import torch

import vsa_kernels
from vsa import to_block_order, to_raster_order, vsa_attention
from vsa_layout import Segment, build_vsa_layout
from tests.reference import cpu_attention, reference_pack


CASES = [
    ((Segment("video", (4, 4, 8), .5), Segment("text", 64)),),
    ((Segment("video", (5, 9, 11), .7), Segment("text", 67)),
     (Segment("video", (1, 3, 5)), Segment("audio", 3))),
    ((Segment("text", 3), Segment("video", (5, 4, 7), .5),
      Segment("video", (1, 7, 9), .8), Segment("audio", 0)),),
]


@pytest.mark.parametrize("samples", CASES)
def test_compact_mapping(samples, cpu_kernels):
    raster = build_vsa_layout(samples, head_num=2)
    block = build_vsa_layout(samples, head_num=2, input_order="block")
    rows = block.input_rows
    assert raster.input_order == "raster" and block.input_order == "block"
    assert raster.input_rows.tolist() == list(range(raster.tokens))
    assert not raster.slots_are_input  # Even full FHW blocks need a permutation.
    assert rows.sort().values.tolist() == list(range(raster.tokens))
    assert block.slot_row[block.slot_row >= 0].tolist() == list(range(block.tokens))
    assert block.sample_lengths == raster.sample_lengths
    assert block.slots_are_input == (block.tokens == block.padded_tokens)
    x = torch.randn(raster.tokens, 2, 8, requires_grad=True)
    reordered = x[rows].detach().requires_grad_()
    first = vsa_kernels.to_slots(x, raster.slot_row, raster.row_slot)
    second = vsa_kernels.to_slots(reordered, block.slot_row, block.row_slot)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not second[block.slot_row < 0].count_nonzero()
    dy = torch.randn_like(first)
    actual, = torch.autograd.grad(second, (reordered,), dy)
    expected, = torch.autograd.grad(first, (x,), dy)
    torch.testing.assert_close(actual, expected[rows], rtol=0, atol=0)


@pytest.mark.parametrize("order", ["slots", "", None, 1])
def test_invalid_order(order):
    with pytest.raises(ValueError, match="input_order"):
        build_vsa_layout(CASES[0], head_num=2, input_order=order)


@pytest.mark.parametrize("samples", CASES)
@pytest.mark.parametrize("dense_query_full", [False, True])
def test_native_reference(samples, dense_query_full, cpu_kernels):
    raster = build_vsa_layout(samples, head_num=2, dense_query_full=dense_query_full)
    block = build_vsa_layout(samples, head_num=2, dense_query_full=dense_query_full,
                             input_order="block")
    torch.manual_seed(612)
    xs = [torch.randn(raster.tokens, 2, 8, requires_grad=True) for _ in range(4)]
    bx = [x.detach()[block.input_rows].requires_grad_() for x in xs]
    expected = reference_pack(samples, raster.sample_slices,
                              dense_query_full=dense_query_full)(*xs)
    actual = cpu_attention(*bx, vsa_layout=block)
    dy = torch.randn_like(expected)
    want = torch.autograd.grad(expected, xs, dy)
    got = torch.autograd.grad(actual, bx, dy[block.input_rows])
    for a, b in zip((actual, *got), (expected, *want)):
        torch.testing.assert_close(a, b[block.input_rows], atol=3e-6, rtol=3e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("samples", CASES)
@pytest.mark.parametrize("dense_query_full", [False, True])
def test_cuda_orders_and_graph(samples, dense_query_full):
    from benchmark.utils import capture_graph, similarity

    raster = build_vsa_layout(samples, head_num=2, dense_query_full=dense_query_full, device="cuda")
    block = build_vsa_layout(samples, head_num=2, dense_query_full=dense_query_full,
                             input_order="block", device="cuda")
    torch.manual_seed(612)
    xs = [torch.randn(raster.tokens, 2, 128, device="cuda", dtype=torch.bfloat16,
                      requires_grad=True) for _ in range(4)]
    dy = torch.randn_like(xs[0])
    expected = reference_pack(samples, raster.sample_slices,
                              dense_query_full=dense_query_full)(*xs)
    want = torch.autograd.grad(expected, xs, dy)
    for layout in (raster, block):
        inputs = [x.detach().clone().requires_grad_() for x in xs]
        def fn(*values):
            ordered = (to_block_order(*values, vsa_layout=layout)
                       if layout.input_order == "block" else values)
            out = vsa_attention(*ordered, vsa_layout=layout)
            return (to_raster_order(out, vsa_layout=layout)[0]
                    if layout.input_order == "block" else out)
        actual = fn(*inputs)
        got = torch.autograd.grad(actual, inputs, dy)
        for a, b in zip((actual, *got), (expected, *want)):
            metrics = similarity(a, b)
            assert min(metrics["cosine"], metrics["tensor_similarity"]) >= .99999, metrics
        # Capture owns fresh leaves; reference autograd nodes live on the
        # default stream rather than the graph's warmup/capture stream.
        graph_inputs = [x.detach().clone().requires_grad_() for x in inputs]
        capture_graph(fn, graph_inputs, dy, 3, 1e-5)


@pytest.mark.parametrize("convert", [to_block_order, to_raster_order])
@pytest.mark.parametrize("invalid", ["raster", "empty", "length", "scalar", "stride", "dtype", "cpu"])
def test_reorder_contract(convert, invalid):
    layout = build_vsa_layout(CASES[0], head_num=2,
                              input_order="raster" if invalid == "raster" else "block")
    tensor = torch.empty(layout.tokens, 2)
    match = "CUDA"
    if invalid == "raster":
        match = "input_order"
    elif invalid == "empty":
        with pytest.raises(ValueError, match="At least one"):
            convert(vsa_layout=layout)
        return
    elif invalid == "length":
        tensor, match = tensor[:-1], "shape"
    elif invalid == "scalar":
        tensor, match = torch.tensor(1.), "shape"
    elif invalid == "stride":
        tensor, match = torch.empty(2, layout.tokens).T, "contiguous"
    elif invalid == "dtype":
        tensor, match = tensor.to(torch.complex64), "dtype"
    with pytest.raises(ValueError, match=match):
        convert(tensor, vsa_layout=layout)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("samples", CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_fused_reorder_round_trip_and_adjoint(samples, dtype):
    layout = build_vsa_layout(samples, head_num=2, input_order="block", device="cuda")
    x = torch.randn(layout.tokens, 2, 7, device="cuda", dtype=dtype, requires_grad=True)
    hidden = torch.randn(layout.tokens, 1025, device="cuda", requires_grad=True)
    positions = torch.arange(layout.tokens * 3, device="cuda", dtype=torch.int64).view(-1, 3) + 2**60
    mask = torch.arange(layout.tokens, device="cuda") % 2 == 0
    tensors = (x, hidden, positions, mask)
    ordered = to_block_order(*tensors, vsa_layout=layout)
    restored = to_raster_order(*ordered, vsa_layout=layout)
    for original, block, raster in zip(tensors, ordered, restored):
        assert block.shape == original.shape and block.dtype == original.dtype
        torch.testing.assert_close(block, original[layout.input_rows], rtol=0, atol=0)
        torch.testing.assert_close(raster, original, rtol=0, atol=0)
    # One unused floating output and integer outputs have no incoming gradient.
    dy = torch.randn(7, 2, layout.tokens, device="cuda", dtype=dtype).permute(2, 1, 0)
    got = torch.autograd.grad(ordered[0], (x, hidden), dy, retain_graph=True, allow_unused=True)
    expected, = torch.autograd.grad(x[layout.input_rows], (x,), dy)
    torch.testing.assert_close(got[0], expected, rtol=0, atol=0)
    assert got[1] is None
    dh = torch.randn_like(hidden)
    dx, dhidden = torch.autograd.grad(restored[:2], (x, hidden), (dy, dh))
    torch.testing.assert_close(dx, dy, rtol=0, atol=0)
    torch.testing.assert_close(dhidden, dh, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("samples", CASES)
@pytest.mark.parametrize("heads", [2, 4])
def test_fused_padding_and_adjoint(samples, heads):
    layout = build_vsa_layout(samples, head_num=heads, input_order="block", device="cuda")
    xs = [torch.randn(layout.tokens, heads, 128, device="cuda", dtype=torch.bfloat16,
                      requires_grad=True) for _ in range(3)]
    actual = vsa_kernels.to_slots_qkv(*xs, layout.slot_row, layout.row_slot)
    # A final constant row implements padding independently in native PyTorch.
    expected = tuple(torch.cat((x, torch.zeros_like(x[:1])))[layout.slot_row] for x in xs)
    dy = [torch.randn(heads, layout.padded_tokens, 128, device="cuda",
                      dtype=torch.bfloat16).transpose(0, 1) for _ in xs]
    got = torch.autograd.grad(actual, xs, dy)
    want = torch.autograd.grad(expected, xs, dy)
    for a, b in zip((*actual, *got), (*expected, *want)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
