"""Real CUDA forward/backward against explicit token-level attention math."""
from functools import partial

import pytest
import torch
from benchmark.compare import prepare_fa4_full
from vsa import vsa_attention
from vsa_layout import Segment, build_vsa_layout
from tests.reference import dense_attention, modality_slices, parity, reference_pack, similarity

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

VIDEO = (Segment("video", (4, 4, 8), .5), Segment("text", 64))
IMAGE = (Segment("video", (1, 8, 8)), Segment("text", 64))
CASES = {
    "mixed": ((Segment("video", (1, 9, 8)), Segment("text", 7)),
              (Segment("video", (5, 4, 8), .5), Segment("text", 65)),
              (Segment("video", (2, 3, 5), .5), Segment("text", 9))),
    "dense": ((Segment("video", (1, 9, 8)), Segment("text", 7)),
              (Segment("video", (3, 5, 7)), Segment("text", 7))),
    "full_blocks": (VIDEO, IMAGE),
    "sparse_image": ((Segment("video", (1, 8, 8), .5), Segment("text", 7)),),
    "dense_reference": ((Segment("video", (3, 5, 7)),
                         Segment("video", (4, 4, 8), .5), Segment("text", 7)),),
    "uniform_videos": (VIDEO, VIDEO, IMAGE),
}


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("dense_query_full", [True, False])
@pytest.mark.parametrize("samples", CASES.values(), ids=CASES.keys())
def test_vsa(heads, dense_query_full, samples):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    layout = build_vsa_layout(samples, head_num=heads, dense_query_full=dense_query_full, device="cuda")
    candidate = partial(vsa_attention, vsa_layout=layout)
    reference = reference_pack(samples, layout.sample_slices, dense_query_full=dense_query_full)
    leaves = [torch.randn(layout.tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
              for _ in range(3 if layout.dense_only else 4)]
    if layout.dense_only:
        gate = torch.full_like(leaves[0], float("nan"))
        candidate, reference = partial(candidate, gate=gate), partial(reference, gate=gate)
    else:
        for group, segments in zip(layout.groups, layout.sample_slices):
            if group.dense_only:
                for rows in segments:
                    leaves[3][rows] = float("nan")  # Fine-only gate must be ignored.
    result = parity(candidate, reference, leaves, torch.randn_like(leaves[0]), modality_slices(layout), 1e-5)
    assert result["passed"], result["failures"]


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("lengths", [(211,), (37, 17, 9)], ids=["single", "packed"])
def test_fa4_full(heads, lengths):
    torch.manual_seed(2027)
    torch.backends.cuda.matmul.allow_tf32 = False
    xs = [torch.randn(sum(lengths), heads, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
          for _ in range(3)]
    dy = torch.randn_like(xs[0])
    out = prepare_fa4_full(lengths)(*xs)
    actual = (out, *torch.autograd.grad(out, xs, dy))
    refs = [x.detach().float().requires_grad_() for x in xs]
    pieces, start = [], 0
    for length in lengths:
        pieces.append(dense_attention(*(x[start:start+length] for x in refs)))
        start += length
    out = torch.cat(pieces)
    expected = (out, *torch.autograd.grad(out, refs, dy.float()))
    # With all video blocks selected and a zero gate, VSA equals full attention.
    samples = [[Segment("video", (1, 1, n), 0)] for n in lengths]
    layout = build_vsa_layout(samples, head_num=heads, device="cuda")
    out = vsa_attention(*xs, torch.zeros_like(xs[0]), layout)
    bsa = (out, *torch.autograd.grad(out, xs, dy))
    for name, result, reference, dense_bsa in zip(("out", "dq", "dk", "dv"), actual, expected, bsa):
        for target in (reference, dense_bsa):
            metrics = similarity(result, target)
            assert min(metrics["cosine"], metrics["tensor_similarity"]) >= .99999, (name, metrics)
