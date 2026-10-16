"""Benchmark input definition, FA4 boundaries and useful-FLOP accounting."""
import pytest
import torch
from benchmark import compare
from benchmark import inputs as benchmark_inputs
from benchmark.inputs import HEAD_COUNTS, HEAD_DIM, describe, make_geometry, make_inputs
from benchmark.utils import summarize
from vsa_layout import Segment, build_vsa_layout
from tests.reference import dense_attention

pytestmark = pytest.mark.usefixtures("cpu_kernels")

SAMPLES = ((Segment("video", (2, 4, 8), 0.5), Segment("text", 3)),
           (Segment("video", (1, 2, 3)), Segment("text", 4)),
           (Segment("video", (1, 1, 5)), Segment("text", 1)))


def dense_varlen(q, k, v, *, cu_seqlens, max_seqlen, total_tokens):
    boundaries = cu_seqlens.tolist()
    assert q.shape[0] == total_tokens == boundaries[-1]
    assert cu_seqlens.dtype == torch.int32
    assert max_seqlen == max(b-a for a, b in zip(boundaries, boundaries[1:]))
    return torch.cat([dense_attention(q[a:b], k[a:b], v[a:b])
                      for a, b in zip(boundaries, boundaries[1:])])


def test_workload_definition():
    lengths = (113648, 5648, 5648)
    samples, info = make_geometry(), describe()
    assert HEAD_DIM == 128 and HEAD_COUNTS == (2, 4)
    assert tuple(info["sample_lengths"]) == lengths
    assert [video.size[0] for video, _ in samples] == [31] + [1]*(len(lengths)-1)
    assert [text.tokens for _, text in samples] == [2048]*len(lengths)
    assert [video.sparsity for video, _ in samples] == [.9, None, None]
    assert info["token_order"] == "sample_major"
    assert [(s["sample"], s["kind"]) for s in info["segments"]] == [
        (i, kind) for i in range(len(lengths)) for kind in ("video", "text")]
    assert info["cu_seqlens"] == [sum(lengths[:i]) for i in range(len(lengths)+1)]
    assert info["segments"][-1]["stop"] == sum(lengths)


def test_generator_produces_reproducible_contiguous_leaves(monkeypatch):
    monkeypatch.setattr(benchmark_inputs, "RESOLUTION", (16, 16))
    xs, dy = make_inputs(2, device="cpu")
    repeated, repeated_dy = make_inputs(2, device="cpu")
    assert len(xs) == 4 and not dy.requires_grad
    assert all(x.is_leaf and x.requires_grad for x in xs)
    for a, b in zip((*xs, dy), (*repeated, repeated_dy)):
        assert a.is_contiguous() and a.shape == (6177, 2, 128)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_full_pack_uses_one_varlen_call_and_isolates_samples(monkeypatch):
    calls = []
    def counted(q, k, v, **kw):
        calls.append((q.shape, kw["cu_seqlens"].tolist(), kw["max_seqlen"]))
        return dense_varlen(q, k, v, **kw)
    monkeypatch.setattr(compare, "_fa4_full_attention", counted)
    fn = compare.prepare_fa4_full((67, 10, 6), device="cpu")
    xs = [torch.randn(83, 2, 8, requires_grad=True) for _ in range(3)]
    got = fn(*xs)
    expected = torch.cat([dense_attention(*(x[a:b] for x in xs)) for a, b in ((0, 67), (67, 77), (77, 83))])
    torch.testing.assert_close(got, expected)
    assert calls == [(torch.Size([83, 2, 8]), [0, 67, 77, 83], 67)]
    grads = torch.autograd.grad(got[:67].sum(), xs)
    assert all(torch.count_nonzero(g[67:]) == 0 for g in grads)
    assert fn.keywords["cu_seqlens"].tolist() == [0, 67, 77, 83]


def test_invalid_full_lengths():
    for lengths in ((), (0,), (-1, 3)):
        with pytest.raises(ValueError, match="positive"):
            compare.prepare_fa4_full(lengths, device="cpu")
    fn = compare.prepare_fa4_full((3, 5), device="cpu")
    x = torch.randn(7, 2, 8)
    with pytest.raises(ValueError, match="sample boundaries"):
        fn(x, x, x)


def test_summary_uses_independent_medians_and_dense_peak():
    samples = [{"forward": 1., "backward": 7., "total": 8.},
               {"forward": 2., "backward": 2., "total": 4.},
               {"forward": 3., "backward": 4., "total": 7.}]
    result = summarize(samples, {"forward": 2e12, "backward": 4e12, "total": 6e12}, 2500)
    assert result["forward"] == {"median_ms": 2., "tflops": 1000., "mfu_percent": 40.}
    assert result["backward"] == {"median_ms": 4., "tflops": 1000., "mfu_percent": 40.}
    assert result["total"]["median_ms"] == 7.  # Not the sum of phase medians.
    assert result["total"]["mfu_percent"] == pytest.approx(100 * 6e12 / 7e9 / 2500)


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("dense_query_full", [True, False])
def test_mfu_counts_valid_isolated_pairs_and_no_image_coarse(heads, dense_query_full):
    from benchmark.utils import packed_geometry_and_flops
    layout = build_vsa_layout(SAMPLES, head_num=heads, dense_query_full=dense_query_full)
    xs = [torch.randn(83, heads, 8) for _ in range(4)]
    geometry, flops = packed_geometry_and_flops(layout, xs, 8)
    # Two equally sized 32-token video blocks: keep one plus all three text
    # keys per video query. Text follows the layout policy. Images are 10,6.
    pairs_per_head = 64 * (32 + 3) + 3 * (67 if dense_query_full else 35) + 10**2 + 6**2
    assert flops["valid_pairs_all_heads"] == heads * pairs_per_head
    assert flops["fine_forward"] == 4 * 8 * heads * pairs_per_head
    assert flops["coarse_forward"] == 4 * heads * 8 * 3**2
    assert flops["routing_forward"] == 2 * heads * 8 * 3 * 2  # All Q blocks × video KV blocks.
    assert flops["forward"] == flops["fine_forward"] + flops["coarse_forward"]
    assert [g["coarse_blocks"] for g in geometry] == [3, 0, 0]
    assert [g["blocks"] for g in geometry] == [3, 2, 2]
    assert [g["padded_tokens"] for g in geometry] == [192, 128, 128]
    for record, sample in zip(geometry, SAMPLES):
        assert record["segments"] == [{"kind": s.kind, "size": s.size, "sparsity": s.sparsity} for s in sample]
