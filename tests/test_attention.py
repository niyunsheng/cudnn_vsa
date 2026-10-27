"""Routing, packed attention, precision and backend call contracts."""
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import vsa_core as core
import vsa_kernels
from vsa import vsa_attention
from vsa_core import route
from vsa_layout import Segment, build_vsa_layout

from tests import reference
from tests.reference import cpu_attention, dense_attention, modality_slices, pool, routing_mask, sample_input

pytestmark = pytest.mark.usefixtures("cpu_kernels")


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("used", ["both", "k", "v", "zero"])
def test_pooled_transport_preserves_fp32_gradients(monkeypatch, size, used):
    runtime = SimpleNamespace(size=size, group=object())
    wire_dtypes, reductions = [], []

    def gather(out, value, *, group):
        assert group is runtime.group
        wire_dtypes.append(value.dtype)
        assert value.element_size() * value.numel() == 2 * 1 * 2 * 6
        out.copy_(torch.cat([value] * size))

    def reduce_scatter(out, grad, *, group):
        assert group is runtime.group and grad.dtype == torch.float32
        assert grad.is_contiguous()
        reductions.append(grad.shape)
        # A remote rank contributes 1/256 of the local derivative. The sum
        # 1 + 1/256 must survive until FP32 pooling, without BF16 rounding.
        contribution = grad[:out.shape[0]]
        out.copy_(contribution + contribution / 256)

    monkeypatch.setattr(core.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(core.dist, "reduce_scatter_tensor", reduce_scatter)
    k, v = [torch.tensor([1.001, -0.003, 3.14159, 0.0]).view(2, 1, 2).requires_grad_()
            for _ in range(2)]
    kg, vg = core._GatherMeans.apply(k, v, runtime)
    assert kg.dtype == torch.float32 and vg.dtype == torch.bfloat16
    torch.testing.assert_close(kg, torch.cat([k] * size), atol=0, rtol=0)
    torch.testing.assert_close(vg, torch.cat([v.bfloat16()] * size), atol=0, rtol=0)
    loss = {"both": lambda: kg.sum() + vg.float().sum(),
            "k": lambda: kg.sum(), "v": lambda: vg.float().sum(),
            "zero": lambda: (kg.sum() + vg.float().sum()) * 0}[used]()
    dk, dv = torch.autograd.grad(loss, (k, v))
    for name, grad in (("k", dk), ("v", dv)):
        expected = (1 if size == 1 else 1 + 1/256) if used in ("both", name) else 0
        assert grad.dtype == torch.float32
        torch.testing.assert_close(grad, torch.full_like(grad, expected), atol=0, rtol=0)
    assert wire_dtypes == ([] if size == 1 else [torch.uint8])
    assert len(reductions) == (0 if size == 1 else 1)

SAMPLES = (
    (Segment("video", (2, 3, 5), None), Segment("text", 7)),
    (Segment("video", (5, 4, 8), 0.5), Segment("text", 65)),
    (Segment("video", (1, 9, 8)), Segment("text", 3)),
    (Segment("video", (2, 3, 5), 0.5), Segment("text", 9)),
)


def test_core_accepts_current_layout_and_heads():
    first = build_vsa_layout([[Segment("video", (2, 4, 8), .9), Segment("text", 3)],
                              [Segment("video", (1, 2, 3)), Segment("text", 0)]], head_num=2)
    second = build_vsa_layout([[Segment("video", (1, 3, 5)), Segment("text", 7)]], head_num=4)
    assert first.dense_query_full is False and second.dense_query_full is False
    # The attention entry stays the same; each layout fixes its own head count.
    for layout, dim in ((first, 8), (second, 16), (first, 16), (first, 8)):
        heads = layout.head_num
        xs = [torch.randn(layout.tokens, heads, dim, requires_grad=True) for _ in range(4)]
        out = cpu_attention(*xs, vsa_layout=layout)
        assert out.shape == xs[0].shape and out.is_contiguous()
        grads = torch.autograd.grad(out.sum(), xs, allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert first.routing_indices.shape[1] == 2
    assert second.routing_indices.shape[1] == 4


@pytest.mark.parametrize("head_num", [1, 2, 4])
@pytest.mark.parametrize("dense_query_full", [True, False])
def test_routing_templates_are_ready_at_construction_and_remain_immutable(head_num, dense_query_full):
    layout = build_vsa_layout([
        [Segment("video", (4, 4, 16), .5), Segment("text", 3)],
        [Segment("video", (1, 2, 3))],
    ], head_num=head_num, dense_query_full=dense_query_full)
    metadata = layout.routing_indices, layout.routing_counts
    for template in metadata:
        assert template.is_contiguous() and template.shape[1] == head_num
        assert template.dtype == torch.int32
        torch.testing.assert_close(template, template[:, :1].expand_as(template), atol=0, rtol=0)
    saved = [x.clone() for x in metadata]
    q, k = [torch.randn(layout.coarse_blocks.numel(), head_num, 8) for _ in range(2)]
    index = core.route(q, k, layout.groups, layout.routing_indices,
        dense_query_full=layout.dense_query_full)
    assert index.data_ptr() != metadata[0].data_ptr()
    for actual, expected in zip(metadata, saved):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_noncontiguous_and_mismatched_inputs_fail_before_backend():
    layout = build_vsa_layout([[Segment("video", (2, 2, 2))]], head_num=2)
    x = torch.randn(8, 2, 128)
    noncontiguous = torch.randn(8, 128, 2).transpose(1, 2)
    with pytest.raises(ValueError, match="contiguous"):
        vsa_attention(noncontiguous, x, x, x, layout)
    with pytest.raises(ValueError, match="identical shape"):
        vsa_attention(x, x.double(), x, x, layout)
    with pytest.raises(ValueError, match="layout.tokens"):
        vsa_attention(x[:7], x[:7], x[:7], x[:7], layout)
    wrong_heads = torch.randn(8, 4, 128)
    with pytest.raises(ValueError, match="VSALayout.head_num"):
        vsa_attention(wrong_heads, wrong_heads, wrong_heads, wrong_heads, layout)


@pytest.mark.parametrize("sparsity", [None, .5])
def test_missing_gate_fails_before_backend(sparsity):
    layout = build_vsa_layout([[Segment("video", (4, 4, 8), sparsity)]], head_num=2)
    x = torch.randn(layout.tokens, 2, 128)
    with pytest.raises(ValueError, match="VSA requires a gate tensor"):
        vsa_attention(x, x, x, None, layout)


@pytest.mark.parametrize("text_tokens", [0, 3])
def test_sample_sparsities_are_prepared_once_and_preserved_across_calls(text_tokens):
    from tests.reference import reference_pack

    grids, texts = [(4, 4, 16), (4, 4, 16), (1, 2, 3)], [text_tokens, 2, 4]
    rates = ([.75, .5, None], [.5, .75, None], [0, .5, None], [.75, .5, None])
    layouts = [build_vsa_layout([[Segment("video", grid, rate), Segment("text", n)]
                                for grid, n, rate in zip(grids, texts, sample_rates)], head_num=2)
               for sample_rates in rates]
    torch.manual_seed(287)
    xs = [torch.randn(layouts[0].tokens, 2, 8, requires_grad=True) for _ in range(4)]
    templates = [(layout.routing_indices, layout.routing_counts) for layout in layouts]
    # Leave several forward graphs alive across layout switches to catch cache/route mutation.
    outputs = [cpu_attention(*xs, layout) for layout in layouts]
    for layout, out, sample_rates, (indices, counts) in zip(layouts, outputs, rates, templates):
        refs = [x.detach().clone().requires_grad_() for x in xs]
        expected = reference_pack(layout.samples, layout.sample_slices)(*refs)
        dy = torch.randn_like(out)
        got = torch.autograd.grad(out, xs, dy, retain_graph=True)
        want = torch.autograd.grad(expected, refs, dy)
        for a, b in zip((out, *got), (expected, *want)):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        for group, rate in zip(layout.groups[:2], sample_rates):
            assert group.video_spans[0][2] == int(4*(1-rate))
        assert layout.groups[2].video_spans == ()  # None selects fine only, regardless of geometry.
        assert layout.routing_indices is indices and layout.routing_counts is counts
    torch.testing.assert_close(outputs[0], outputs[-1], atol=0, rtol=0)
    assert layouts[0].routing_indices is not layouts[1].routing_indices


def test_exempt_routing_and_all_modality_coarse():
    layout = build_vsa_layout([(Segment("text", 7), Segment("audio", 5), Segment("video", (4, 4, 8), 0.5))],
                              head_num=2, dense_query_full=True)
    torch.manual_seed(42)
    q, k, v = [torch.randn(layout.tokens, 2, 8, requires_grad=True) for _ in range(3)]
    gate = torch.randn_like(q).requires_grad_()
    slots = [vsa_kernels.to_slots(t, layout.slot_row, layout.row_slot) for t in (q, k, v)]
    means = [pool(t, layout) for t in slots]
    scores = means[0] @ means[1].transpose(-2, -1) / 8**0.5
    q_route, k_route, _, _, _ = vsa_kernels.routing_and_coarse_means(*slots, layout.block_sizes, layout.coarse_blocks)
    indices = route(q_route, k_route, layout.groups, layout.routing_indices,
        dense_query_full=layout.dense_query_full)
    mask = routing_mask(indices, layout.routing_counts)[0]
    assert mask[:, :2].all() and mask[:, :, :2].all()
    assert (mask[:, 2:, 2:].sum(-1) == 1).all()
    fine = cpu_attention(q, k, v, torch.zeros_like(gate), layout)
    out = cpu_attention(q, k, v, gate, layout)
    coarse = (scores.softmax(-1) @ means[2]).permute(1, 0, 2)
    coarse_tokens = coarse[layout.row_slot // 64]
    torch.testing.assert_close(out, torch.addcmul(fine, gate, coarse_tokens))
    dy = torch.randn_like(out)
    out.backward(dy)
    torch.testing.assert_close(gate.grad, dy * coarse_tokens)
    for _, rows in modality_slices(layout):
        assert gate.grad[rows].abs().sum() > 0


def test_dense_limit_is_single_joint_softmax():
    layout = build_vsa_layout([(Segment("video", (3, 4, 5), 0), Segment("text", 11), Segment("audio", 3))], head_num=2)
    torch.manual_seed(17)
    leaves = [torch.randn(layout.tokens, 2, 8, requires_grad=True) for _ in range(3)]
    copies = [x.detach().clone().requires_grad_() for x in leaves]
    got = cpu_attention(*leaves, torch.zeros_like(leaves[0]), layout)
    q, k, v = [x.transpose(0, 1) for x in copies]
    expected = ((q @ k.transpose(-2, -1) / 8**0.5).softmax(-1) @ v).transpose(0, 1)
    dy = torch.randn_like(got)
    got_grads = torch.autograd.grad(got, leaves, dy)
    ref_grads = torch.autograd.grad(expected, copies, dy)
    for a, b in zip((got, *got_grads), (expected, *ref_grads)):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_gpu_backend_never_silently_falls_back():
    layout = build_vsa_layout([(Segment("video", (4, 4, 4)),)], head_num=2)
    x = torch.randn(64, 2, 128)
    with pytest.raises(ValueError, match="CUDA BF16"):
        vsa_attention(x, x, x, x, layout)


@pytest.mark.parametrize("scale", [None, .2])
def test_coarse_uses_one_fa4_varlen_call(monkeypatch, scale):
    import vsa_core as core

    layout = build_vsa_layout([
        [Segment("video", (4, 4, 8), .5), Segment("text", 3)],
        [Segment("video", (1, 2, 3))],
        [Segment("video", (4, 4, 4), .5)],
    ], head_num=2)
    xs = [torch.randn(4, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
    calls = []
    def fake_fa4(q, k, v, **kwargs):
        calls.append(kwargs)
        assert all(a is b for a, b in zip((q, k, v), xs))
        return v, None
    monkeypatch.setattr(core, "flash_attn_varlen_func", fake_fa4)
    assert core.coarse_attention(*xs, layout.coarse_cu_seqlens, max(layout.coarse_lengths), scale) is xs[2]
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["cu_seqlens_q"] is kwargs["cu_seqlens_k"] is layout.coarse_cu_seqlens
    assert kwargs["cu_seqlens_q"].tolist() == [0, 3, 4]
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 3
    assert kwargs["softmax_scale"] == (128**-0.5 if scale is None else scale)
    assert kwargs["causal"] is False


def test_coarse_preserves_asymmetric_ring_boundaries(monkeypatch):
    cu_q = torch.tensor([0, 1, 3], dtype=torch.int32)
    cu_kv = torch.tensor([0, 4, 7], dtype=torch.int32)
    q = torch.randn(3, 2, 128, dtype=torch.bfloat16)
    k, v = (torch.randn(7, 2, 128, dtype=torch.bfloat16) for _ in range(2))
    def fake_fa4(query, key, value, **kwargs):
        assert query is q and key is k and value is v
        assert kwargs["cu_seqlens_q"] is cu_q
        assert kwargs["cu_seqlens_k"] is cu_kv
        assert (kwargs["max_seqlen_q"], kwargs["max_seqlen_k"]) == (2, 4)
        assert kwargs["softmax_scale"] == .125
        return query, None
    monkeypatch.setattr(core, "flash_attn_varlen_func", fake_fa4)
    assert core.coarse_attention(q, k, v, cu_q, 2, .125,
                                 cu_seqlens_kv=cu_kv, max_seqlen_kv=4) is q


@pytest.mark.parametrize("dense_query_full", [True, False])
def test_zero_sparsity_does_not_override_other_regions(dense_query_full):
    layout = build_vsa_layout([(Segment("video", (8, 8, 8), 0.9), Segment("text", 7),
                           Segment("video", (4, 4, 8), 0))], head_num=2, dense_query_full=dense_query_full)
    assert layout.groups[0].video_spans == ((1, 9, 1), (9, 11, 2))
    assert not layout.dense_only  # Numeric zero retains VSA tiling and coarse.
    q, k = [torch.randn(layout.blocks, 2, 8) for _ in range(2)]
    index = route(q, k, layout.groups, layout.routing_indices,
        dense_query_full=dense_query_full)
    counts = layout.routing_counts
    mask = routing_mask(index, counts)[0]
    assert mask[:, :, 0].all() and mask[:, :, 9:11].all()
    routed_start = 1 if dense_query_full else 0
    assert mask[:, routed_start:, 1:9].sum(-1).eq(1).all()
    assert counts[0, :, routed_start:].eq(4).all()


@pytest.mark.parametrize("dense_query_full", [True, False])
def test_segment_sparsity_controls_kv_columns_including_cross_video_attention(dense_query_full):
    layout = build_vsa_layout([
        [Segment("video", (4, 4, 16), .25), Segment("text", 3),
         Segment("video", (4, 4, 16), .75), Segment("audio", 2)],
        [Segment("video", (4, 4, 16), .5)],
    ], head_num=2, dense_query_full=dense_query_full)
    torch.manual_seed(310)
    q, k = [torch.randn(layout.coarse_blocks.numel(), 2, 8) for _ in range(2)]
    index = route(q, k, layout.groups, layout.routing_indices,
        dense_query_full=dense_query_full)
    counts = layout.routing_counts
    mask = routing_mask(index, counts)[0]
    assert index.shape[-1] == (10 if dense_query_full else 6)
    # Both video query regions choose three blocks from video0 and one from
    # video1. Crossing a segment boundary does not make video attention dense.
    assert mask[:, 2:10, 2:6].sum(-1).eq(3).all()
    assert mask[:, 2:10, 6:10].sum(-1).eq(1).all()
    assert mask[:, :10, :2].all()
    if dense_query_full:
        assert mask[:, :2, :10].all()
        assert counts[0, :, :2].eq(10).all()
    else:
        assert mask[:, :2, 2:6].sum(-1).eq(3).all()
        assert mask[:, :2, 6:10].sum(-1).eq(1).all()
        assert counts[0, :, :10].eq(6).all()
    assert mask[:, 10:14, 10:14].sum(-1).eq(2).all()
    assert not mask[:, :10, 10:].any() and not mask[:, 10:, :10].any()
    # Also check the selected IDs against scores, separately for each KV span.
    routed_start = 2 if dense_query_full else 0
    for q0, q1, k0, k1, keep in ((routed_start, 10, 2, 6, 3), (routed_start, 10, 6, 10, 1),
                                (10, 14, 10, 14, 2)):
        scores = q[q0:q1].transpose(0, 1) @ k[k0:k1].permute(1, 2, 0)
        expected = torch.zeros_like(scores, dtype=torch.bool)
        expected.scatter_(-1, scores.topk(keep, dim=-1).indices, True)
        torch.testing.assert_close(mask[:, q0:q1, k0:k1], expected)


@pytest.mark.parametrize("frames,sizes", [(1, [64, 17, 64, 1]), (2, [64, 64, 34, 64, 1])])
def test_none_sparsity_only_pads_and_uses_full_fine(monkeypatch, frames, sizes):
    import importlib
    core = importlib.import_module("vsa_core")
    layout = build_vsa_layout([(Segment("video", (frames, 9, 9), sparsity=None), Segment("text", 65))], head_num=2)
    assert layout.dense_only and layout.groups[0].video_spans == ()
    assert layout.block_sizes.tolist() == sizes
    assert layout.padded_tokens == 64 * len(sizes)
    # Padding may move row addresses, but valid rows retain their exact order.
    torch.testing.assert_close(layout.slot_row[layout.slot_row >= 0], torch.arange(layout.tokens))
    def forbidden(*args, **kwargs):
        raise AssertionError("An all-None sample must skip pooling, routing and coarse")
    monkeypatch.setattr(vsa_kernels, "block_means", forbidden)
    monkeypatch.setattr(core, "route", forbidden)
    monkeypatch.setattr(reference, "dense_coarse", forbidden)
    leaves = [torch.randn(layout.tokens, 2, 8, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    slots = vsa_kernels.to_slots(leaves[0], layout.slot_row, layout.row_slot)
    torch.testing.assert_close(slots[layout.slot_row >= 0], leaves[0], rtol=0, atol=0)
    assert not slots[layout.slot_row < 0].count_nonzero()
    copies = [x.detach().clone().requires_grad_() for x in leaves[:3]]
    out = cpu_attention(*leaves, layout)
    q, k, v = (x.transpose(0, 1) for x in copies)
    expected = ((q @ k.transpose(-1, -2) / 8**0.5).softmax(-1) @ v).transpose(0, 1)
    dy = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, leaves, dy, allow_unused=True)
    expected_grads = torch.autograd.grad(expected, copies, dy)
    assert actual_grads[-1] is None  # No gate graph for a fine-only sample.
    for a, b in zip((out, *actual_grads[:3]), (expected, *expected_grads)):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("sparsity,keep", [(0, 4), (.5, 2)])
def test_numeric_sparsity_enables_image_vsa(sparsity, keep):
    from tests.reference import reference_sample

    sample = [Segment("video", (1, 8, 8), sparsity), Segment("text", 7)]
    layout = build_vsa_layout([sample], head_num=2)
    assert not layout.dense_only
    assert layout.groups[0].video_spans == ((1, 5, keep),)
    assert layout.block_sizes.tolist() == [7, 16, 16, 16, 16]
    assert layout.coarse_lengths == (5,)
    torch.manual_seed(725)
    xs = [torch.randn(layout.tokens, 2, 8, requires_grad=True) for _ in range(4)]
    refs = [x.detach().clone().requires_grad_() for x in xs]
    actual = cpu_attention(*xs, layout)
    expected = reference_sample(sample, *refs)
    dy = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, xs, dy)
    ref_grads = torch.autograd.grad(expected, refs, dy)
    for a, b in zip((actual, *grads), (expected, *ref_grads)):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert grads[-1].abs().sum() > 0


@pytest.mark.parametrize("full", (False, True))
def test_bsa_forward_backward_contract(monkeypatch, full):
    from types import SimpleNamespace

    import vsa_core as core

    sizes = None if full else torch.tensor([63], dtype=torch.int32)
    index = torch.zeros((1, 2, 1, 1), dtype=torch.int32)
    counts = torch.ones((1, 2, 1), dtype=torch.int32)
    xs = [torch.randn(64, 2, 8, requires_grad=True) for _ in range(3)]
    def forward(q, k, v, indices, _, block_sizes, nums, **kwargs):
        assert block_sizes is sizes
        assert indices is index and nums is counts
        assert kwargs["layout"] == "bhsd"
        assert q.shape == (1, 2, 64, 8)
        return {"o_tensor": v.clone(), "lse_tensor": q.new_zeros((1, 2, 64))}
    def backward(dy, q, k, v, out, lse, indices, _, block_sizes, nums, **kwargs):
        assert block_sizes is sizes
        assert indices is index and nums is counts
        assert kwargs["layout"] == "bshd"
        assert q.shape == (1, 64, 2, 8)
        return {"dq_tensor": torch.zeros_like(q), "dk_tensor": torch.zeros_like(k), "dv_tensor": dy}
    monkeypatch.setattr(core, "BSA", SimpleNamespace(
        block_sparse_attention_forward=forward, block_sparse_attention_backward=backward))
    out = core.BlockSparseAttention.apply(*xs, index, counts, sizes, 8**-0.5)
    assert out.is_contiguous()
    torch.testing.assert_close(out, xs[2], atol=0, rtol=0)
    grads = torch.autograd.grad(out.sum(), xs)
    for grad in grads[:2]:
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=0, rtol=0)
    torch.testing.assert_close(grads[2], torch.ones_like(xs[2]), atol=0, rtol=0)


@pytest.mark.parametrize("reference_sparsity", [None, .5])
@pytest.mark.parametrize("dense_query_full", [True, False])
def test_joint_reference_regions_and_arbitrary_segment_order(reference_sparsity, dense_query_full):
    """Packed multi-region samples match isolated token math and stay isolated."""
    from tests.reference import reference_pack

    samples = [
        [Segment("text", 3), Segment("video", (1, 3, 5), reference_sparsity), Segment("audio", 5),
         Segment("video", (5, 4, 8), .75), Segment("text", 0), Segment("video", (2, 3, 5), .5)],
        [Segment("video", (1, 3, 5)), Segment("text", 7), Segment("video", (1, 2, 3)), Segment("audio", 0)],
        [Segment("video", (2, 2, 3), .5)],
    ]
    layout = build_vsa_layout(samples, head_num=2, dense_query_full=dense_query_full)
    assert layout.sample_lengths == (213, 28, 12)
    assert tuple(g.dense_only for g in layout.groups) == (False, True, False)
    assert [len(g.video_spans) for g in layout.groups] == [2 if reference_sparsity is None else 3, 0, 1]
    assert [len(rows) for rows in layout.sample_slices] == [6, 4, 1]
    assert [name for name, _ in modality_slices(layout)] == [
        f"sample_{i}/{s.kind}" for i, sample in enumerate(samples) for s in sample]
    # Dense regions retain raster order even inside a mixed VSA sample.
    for sample, slices in zip(samples, layout.sample_slices):
        for segment, rows in zip(sample, slices):
            if segment.sparsity is None:
                kept = layout.slot_row[(layout.slot_row >= rows.start) & (layout.slot_row < rows.stop)]
                torch.testing.assert_close(kept, torch.arange(rows.start, rows.stop))
    torch.manual_seed(901)
    xs = [torch.randn(layout.tokens, 2, 8, requires_grad=True) for _ in range(4)]
    refs = [x.detach().clone().requires_grad_() for x in xs]
    actual = cpu_attention(*xs, layout)
    expected = reference_pack(samples, layout.sample_slices, dense_query_full=dense_query_full)(*refs)
    dy = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, xs, dy, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, refs, dy)
    for a, b in zip((actual, *grads), (expected, *expected_grads)):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert not sample_input(grads[3], layout.sample_slices[1]).count_nonzero()
    assert grads[3][layout.sample_slices[0][1]].abs().sum() > 0  # Joint reference image has coarse.
    isolated_grads = torch.autograd.grad(sample_input(actual, layout.sample_slices[0]).sum(), xs)
    for grad in isolated_grads:
        for rows in layout.sample_slices[1:]:
            assert not sample_input(grad, rows).count_nonzero()


def make_case(samples=SAMPLES, dtype=torch.float64, *, dense_query_full=False):
    layout = build_vsa_layout(samples, head_num=2, dense_query_full=dense_query_full)
    fn = partial(cpu_attention, vsa_layout=layout)
    torch.manual_seed(11)
    xs = [torch.randn(layout.tokens, 2, 8, dtype=dtype, requires_grad=True) for _ in range(4)]
    return layout, fn, xs


def test_packed_calls_and_metadata_exclude_whole_dense_samples(monkeypatch):
    layout, fn, xs = make_case(dtype=torch.bfloat16)
    calls = []
    fine, coarse = reference.dense_fine, reference.dense_coarse
    def counted_fine(*args):
        calls.append("fine")
        return fine(*args)
    def counted_coarse(q, k, v, cu_seqlens, max_seqlen, scale=None):
        calls.append("coarse")
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        assert q.shape == k.shape == v.shape == (9, 2, 8)
        assert cu_seqlens.tolist() == [0, 6, 9] and max_seqlen == 6
        return coarse(q, k, v, cu_seqlens, max_seqlen, scale)
    monkeypatch.setattr(reference, "dense_fine", counted_fine)
    monkeypatch.setattr(reference, "dense_coarse", counted_coarse)
    out = fn(*xs)
    assert calls == ["fine", "coarse"]
    slots = tuple(vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot)
                  for x in xs[:3])
    q_route, k_route, q_coarse, k_coarse, v_coarse = vsa_kernels.routing_and_coarse_means(
        *slots, layout.block_sizes, layout.coarse_blocks)
    for routing, coarse in ((q_route, q_coarse), (k_route, k_coarse)):
        assert routing.dtype == torch.float32 and not routing.requires_grad
        torch.testing.assert_close(coarse, routing.to(torch.bfloat16), atol=0, rtol=0)
    for coarse in (q_coarse, k_coarse, v_coarse):
        assert coarse.dtype == torch.bfloat16 and coarse.requires_grad
    index = core.route(q_route, k_route, layout.groups, layout.routing_indices,
        dense_query_full=layout.dense_query_full)
    mask = routing_mask(index, layout.routing_counts)[0]
    assert index.shape[-1] == 4 < layout.blocks  # Kept KV capacity, including dense KV.
    for i, g in enumerate(layout.groups):
        assert not mask[:, g.start:g.end, :g.start].any()
        assert not mask[:, g.start:g.end, g.end:].any()
        rows = layout.sample_slices[i]
        row_ids = sample_input(layout.row_coarse, rows)
        if g.dense_only:
            assert torch.all(row_ids == layout.coarse_blocks.numel())
        else:
            assert torch.all(row_ids < layout.coarse_blocks.numel())  # Includes text rows.
    grads = torch.autograd.grad(out.float().sum(), xs)
    for i in (0, 2):
        assert not sample_input(grads[-1], layout.sample_slices[i]).count_nonzero()
    for i in (1, 3):
        assert grads[-1][layout.sample_slices[i][1]].abs().sum() > 0


@pytest.mark.parametrize("dense_query_full", [True, False])
def test_all_dense_pack_never_pools_routes_or_calls_coarse(monkeypatch, dense_query_full):
    layout, fn, xs = make_case(samples=(SAMPLES[0], SAMPLES[2]), dense_query_full=dense_query_full)
    def forbidden(*args, **kwargs):
        raise AssertionError("All-None samples must have only fine attention")
    monkeypatch.setattr(vsa_kernels, "block_means", forbidden)
    monkeypatch.setattr(core, "route", forbidden)
    monkeypatch.setattr(reference, "dense_coarse", forbidden)
    monkeypatch.setattr(vsa_kernels, "_fine_coarse_gate_add_forward", forbidden)
    out = fn(*xs)
    assert torch.autograd.grad(out.sum(), xs[-1], allow_unused=True)[0] is None
    expected = []
    for rows in layout.sample_slices:
        q, k, v = (sample_input(x, rows).transpose(0, 1) for x in xs[:3])
        expected.append(((q @ k.transpose(-1, -2)/8**0.5).softmax(-1) @ v).transpose(0, 1))
    expected = torch.cat(expected)
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)


def test_fp32_routing_survives_autocast_and_templates_are_immutable():
    layout = build_vsa_layout([SAMPLES[1]], head_num=2)
    xs = [torch.randn(layout.tokens, 2, 8) for _ in range(3)]
    slots = tuple(vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot)
                  for x in xs)
    q_route, k_route, _, _, _ = vsa_kernels.routing_and_coarse_means(*slots, layout.block_sizes, layout.coarse_blocks)
    template = layout.routing_indices
    saved = template.clone()
    expected = core.route(q_route, k_route, layout.groups, template,
        dense_query_full=layout.dense_query_full)
    original = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = core.route(q_route, k_route, layout.groups, template,
                dense_query_full=layout.dense_query_full)
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(template, saved, atol=0, rtol=0)


def test_independent_token_math_reference():
    from tests.reference import reference_pack
    layout, fn, xs = make_case(dtype=torch.float32)
    copies = [x.detach().clone().requires_grad_() for x in xs]
    actual = fn(*xs)
    expected = reference_pack(SAMPLES, layout.sample_slices)(*copies)
    dy = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, xs, dy)
    expected_grads = torch.autograd.grad(expected, copies, dy)
    for a, b in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_mixed_pack_ignores_nonfinite_dense_sample_gate():
    layout, fn, xs = make_case()
    original = fn(*xs)
    changed = [x.detach().clone().requires_grad_() for x in xs]
    with torch.no_grad():
        for i in (0, 2):
            for rows in layout.sample_slices[i]:
                changed[3][rows] = float("nan")
    out = fn(*changed)
    torch.testing.assert_close(out, original, atol=0, rtol=0)
    grads = torch.autograd.grad(out.sum(), changed)
    assert all(torch.isfinite(g).all() for g in grads)
    for i in (0, 2):
        assert not sample_input(grads[-1], layout.sample_slices[i]).count_nonzero()


@pytest.mark.parametrize("heads", [2, 4])
def test_packed_vsa_matches_independent_samples_and_isolates_gradients(heads):
    samples = ((Segment("video", (2, 4, 8), 0.5), Segment("text", 3)),
               (Segment("video", (1, 2, 3)), Segment("text", 4)),
               (Segment("video", (1, 1, 5)), Segment("text", 1)))
    layout = build_vsa_layout(samples, head_num=heads)
    fn = partial(cpu_attention, vsa_layout=layout)
    assert tuple(g.dense_only for g in layout.groups) == (False, True, True)
    assert layout.sample_slices == ((slice(0, 64), slice(64, 67)),
                                   (slice(67, 73), slice(73, 77)),
                                   (slice(77, 82), slice(82, 83)))

    torch.manual_seed(42)
    original = [torch.randn(83, heads, 8, dtype=torch.float64) for _ in range(4)]
    xs = [x.clone().requires_grad_() for x in original]
    got = fn(*xs)
    reference_inputs = [x.clone().requires_grad_() for x in original]
    first_layout = build_vsa_layout([samples[0]], head_num=heads)
    expected = torch.cat([
        cpu_attention(*(x[:67] for x in reference_inputs), first_layout),
        dense_attention(*(x[67:77] for x in reference_inputs[:3])),
        dense_attention(*(x[77:] for x in reference_inputs[:3]))])
    dy = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(got, xs, dy)
    expected_grads = torch.autograd.grad(expected, reference_inputs, dy)
    for a, b in zip((got, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    assert got.is_contiguous()

    changed = [x.detach().clone() for x in xs]
    for x in changed:
        for rows in layout.sample_slices[1]:
            x[rows] += 7
    modified = fn(*changed)
    for i in (0, 2):
        torch.testing.assert_close(sample_input(modified, layout.sample_slices[i]),
                                   sample_input(got, layout.sample_slices[i]), rtol=0, atol=0)

    out = fn(*xs)
    grads = torch.autograd.grad(sample_input(out, layout.sample_slices[0]).sum(), xs)
    for grad in grads:
        for i in (1, 2):
            assert torch.count_nonzero(sample_input(grad, layout.sample_slices[i])) == 0
    assert grads[-1][layout.sample_slices[0][1]].abs().sum() > 0  # Text coarse/gate is active.
