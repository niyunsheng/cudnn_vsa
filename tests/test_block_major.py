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
        rows = layout.input_rows
        inputs = [x.detach()[rows].requires_grad_() for x in xs]
        def fn(*values):
            return vsa_attention(*values, vsa_layout=layout)
        actual = fn(*inputs)
        got = torch.autograd.grad(actual, inputs, dy[rows])
        for a, b in zip((actual, *got), (expected, *want)):
            metrics = similarity(a, b[rows])
            assert min(metrics["cosine"], metrics["tensor_similarity"]) >= .99999, metrics
        # Capture owns fresh leaves; reference autograd nodes live on the
        # default stream rather than the graph's warmup/capture stream.
        graph_inputs = [x.detach().clone().requires_grad_() for x in inputs]
        capture_graph(fn, graph_inputs, dy[rows], 3, 1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("samples", CASES)
def test_boundary_reorder_and_graph(samples):
    from benchmark.utils import capture_graph, similarity

    layout = build_vsa_layout(samples, head_num=2, input_order="block", device="cuda")
    torch.manual_seed(612)
    x = torch.randn(layout.tokens, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4 * 2 * 128, 32, device="cuda", dtype=torch.bfloat16) / 32**.5

    def project(tokens):
        projected = torch.nn.functional.linear(tokens, weight).view(layout.tokens, 4, 2, 128)
        return tuple(t.contiguous() for t in projected.unbind(1))

    def fn(tokens):
        tokens = to_block_order(tokens, vsa_layout=layout)
        out = vsa_attention(*project(tokens), vsa_layout=layout)
        return to_raster_order(out, vsa_layout=layout)

    expected = reference_pack(samples, layout.sample_slices)(*project(x))
    actual = fn(x)
    dy = torch.randn_like(expected)
    want, = torch.autograd.grad(expected, (x,), dy)
    got, = torch.autograd.grad(actual, (x,), dy)
    for a, b in ((actual, expected), (got, want)):
        metrics = similarity(a, b)
        assert min(metrics["cosine"], metrics["tensor_similarity"]) >= .99999, metrics
    capture_graph(fn, [x.detach().clone().requires_grad_()], dy, 3, 1e-5)


@pytest.mark.parametrize("convert", [to_block_order, to_raster_order])
@pytest.mark.parametrize("invalid", ["raster", "tuple", "length", "scalar", "empty_rows",
                                     "stride", "dtype", "cpu"])
def test_reorder_contract(convert, invalid):
    layout = build_vsa_layout(CASES[0], head_num=2,
                              input_order="raster" if invalid == "raster" else "block")
    tensor = torch.empty(layout.tokens, 2)
    match = "CUDA"
    if invalid == "raster":
        match = "input_order"
    elif invalid == "tuple":
        tensor, match = (tensor,), "one tensor"
    elif invalid == "length":
        tensor, match = tensor[:-1], "shape"
    elif invalid == "scalar":
        tensor, match = torch.tensor(1.), "shape"
    elif invalid == "empty_rows":
        tensor, match = torch.empty(layout.tokens, 0), "nonempty"
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
@pytest.mark.parametrize("row_shape", [(), (2, 7), (1025,)])
def test_fused_reorder_round_trip_and_adjoint(samples, dtype, row_shape):
    layout = build_vsa_layout(samples, head_num=2, input_order="block", device="cuda")
    x = torch.randn(layout.tokens, *row_shape, device="cuda", dtype=dtype, requires_grad=True)
    ordered = to_block_order(x, vsa_layout=layout)
    restored = to_raster_order(ordered, vsa_layout=layout)
    assert isinstance(ordered, torch.Tensor) and isinstance(restored, torch.Tensor)
    assert ordered.shape == x.shape and ordered.dtype == x.dtype
    torch.testing.assert_close(ordered, x[layout.input_rows], rtol=0, atol=0)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)
    dy = torch.randn(*x.shape, 2, device="cuda", dtype=dtype)[..., 0]
    assert not dy.is_contiguous()
    got, = torch.autograd.grad(ordered, (x,), dy, retain_graph=True)
    expected, = torch.autograd.grad(x[layout.input_rows], (x,), dy)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    dx, = torch.autograd.grad(restored, (x,), dy)
    torch.testing.assert_close(dx, dy, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("samples", CASES)
@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_reorder_discrete_tensor(samples, dtype):
    layout = build_vsa_layout(samples, head_num=2, input_order="block", device="cuda")
    if dtype == torch.int64:
        x = torch.arange(layout.tokens * 3, device="cuda").view(-1, 3) + 2**60
    else:
        x = torch.arange(layout.tokens, device="cuda") % 2 == 0
    ordered = to_block_order(x, vsa_layout=layout)
    assert ordered.shape == x.shape and ordered.dtype == x.dtype
    torch.testing.assert_close(ordered, x[layout.input_rows], rtol=0, atol=0)
    torch.testing.assert_close(to_raster_order(ordered, vsa_layout=layout), x, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", [1, 3, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("row_shape", [(), (2, 7), (1025,)])
def test_ring_boundary_gather_and_scatter_adjoint(size, dtype, row_shape, metadata_group):
    samples = CASES[1]
    total = sum(s.tokens for sample in samples for s in sample)
    x = torch.randn(total, *row_shape, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(*x.shape, 2, device="cuda", dtype=dtype)[..., 0]
    combined = torch.zeros_like(x)
    covered = []
    for rank in range(size):
        layout = build_vsa_layout(samples, head_num=2, input_order="block", device="cuda",
                                  ring_cp_group=metadata_group(size, rank))
        rows = layout.input_rows
        actual = to_block_order(x, vsa_layout=layout)
        expected = x[rows]
        assert actual.is_contiguous() and actual.shape == (layout.tokens, *row_shape)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Preserve a noncontiguous gradient to exercise the scatter's row strides.
        grad = torch.stack((dy[rows], dy[rows]), dim=-1)[..., 0]
        got, = torch.autograd.grad(actual, x, grad)
        want, = torch.autograd.grad(expected, x, grad)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        combined += got
        covered.extend(rows.cpu().tolist())
    assert sorted(covered) == list(range(total))
    torch.testing.assert_close(combined, dy, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", [1, 2])
def test_ring_boundary_large_offsets(size, metadata_group):
    # Cover both flattened output offsets and source offsets beyond int32.
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("Large-offset regression requires 8 GiB free CUDA memory")
    samples = ((Segment("text", 128),),)
    layout = build_vsa_layout(samples, head_num=1, input_order="block", device="cuda",
                              ring_cp_group=metadata_group(size, size - 1))
    x = torch.arange(128, device="cuda", dtype=torch.uint8)[:, None]
    x = x.expand(128, 2**24 + 1).contiguous()
    actual = to_block_order(x, vsa_layout=layout)
    assert torch.equal(actual, x[layout.input_rows])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ring_boundary_positions_masks_and_graph(metadata_group, monkeypatch):
    from benchmark.utils import capture_graph
    import torch.distributed as dist

    layout = build_vsa_layout(CASES[1], head_num=2, input_order="block", device="cuda",
                              ring_cp_group=metadata_group(3, 1))
    def unexpected(*args, **kwargs):
        pytest.fail("Boundary conversion must not communicate or create groups")
    for name in ("all_gather_into_tensor", "all_reduce", "reduce_scatter_tensor", "new_group"):
        monkeypatch.setattr(dist, name, unexpected)
    total = layout.ring.global_tokens
    positions = torch.arange(total * 3, device="cuda").view(total, 3) + 2**60
    mask = torch.arange(total, device="cuda") % 3 == 0
    for field in (positions, mask):
        torch.testing.assert_close(to_block_order(field, vsa_layout=layout),
                                   field[layout.input_rows], rtol=0, atol=0)
    x = torch.randn(total, 32, device="cuda", requires_grad=True)
    fn = lambda value: to_block_order(value, vsa_layout=layout)
    dy = torch.randn(layout.tokens, 32, device="cuda")
    capture_graph(fn, [x], dy, 3, 1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ring_boundary_second_derivative(metadata_group):
    layout = build_vsa_layout(CASES[1], head_num=2, input_order="block", device="cuda",
                              ring_cp_group=metadata_group(3, 2))
    x = torch.randn(layout.ring.global_tokens, device="cuda", dtype=torch.float64,
                    requires_grad=True)
    y = to_block_order(x, vsa_layout=layout)
    dx, = torch.autograd.grad(y.square().sum(), x, create_graph=True)
    ddx, = torch.autograd.grad(dx.sum(), x)
    expected = torch.zeros_like(x)
    expected[layout.input_rows] = 2
    torch.testing.assert_close(ddx, expected, rtol=0, atol=0)


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
