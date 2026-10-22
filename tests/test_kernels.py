"""Slot maps, padding, gather adjoints and coarse merge gradients."""
import pytest
import torch
import vsa_kernels
from vsa_layout import Segment, build_vsa_layout

pytestmark = pytest.mark.usefixtures("cpu_kernels")

SAMPLES = (
    (Segment("video", (2, 3, 5), None), Segment("text", 7)),
    (Segment("video", (5, 4, 8), 0.5), Segment("text", 65)),
    (Segment("video", (1, 9, 8)), Segment("text", 3)),
    (Segment("video", (2, 3, 5), 0.5), Segment("text", 9)),
)


def test_ring_pooling_keeps_fp32_means_and_zero_dummy_gradients():
    sizes = torch.tensor([3, 64, 0], dtype=torch.int32)
    xs = [torch.randn(3 * 64, 2, 8, requires_grad=True) for _ in range(3)]
    valid = torch.arange(64)[None, :] < sizes[:, None]
    masked = [x.view(3, 64, 2, 8).masked_fill(~valid[:, :, None, None], 0).flatten(0, 1)
              for x in xs]
    got = vsa_kernels.ring_block_means.__wrapped__(*masked, sizes)
    expected = tuple(torch.stack((x[:3].mean(0), x[64:128].mean(0), x[128:].sum(0)))
                     for x in masked)
    for actual, target in zip(got, expected):
        assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, target)
    grad = torch.autograd.grad(sum(x.sum() for x in got), xs)
    for x in grad:
        torch.testing.assert_close(x[:3], torch.full_like(x[:3], 1 / 3))
        torch.testing.assert_close(x[64:128], torch.full_like(x[64:128], 1 / 64))
        assert not x[3:64].count_nonzero() and not x[128:].count_nonzero()


def test_sample_major_tiling_maps_and_gather_adjoint():
    # Unequal samples, an image with no text, and a second ragged video.
    samples = [[Segment("video", (4, 4, 8), .9), Segment("text", 3)],
               [Segment("video", (1, 2, 3)), Segment("text", 0)],
               [Segment("video", (2, 1, 2), .9), Segment("text", 2)]]
    layout = build_vsa_layout(samples, head_num=2)
    assert layout.sample_slices == ((slice(0, 128), slice(128, 131)),
                                   (slice(131, 137), slice(137, 137)),
                                   (slice(137, 141), slice(141, 143)))
    assert layout.sample_lengths == (131, 6, 6)
    assert layout.samples == tuple(tuple(sample) for sample in samples)
    assert [[s.tokens for s in sample] for sample in layout.samples] == [[128, 3], [6, 0], [4, 2]]
    # Block order remains dense-prefix, then FHW tiles, within each sample.
    blocks = [[128, 129, 130],
              [f*32+h*8+w for f in range(4) for h in range(4) for w in range(4)],
              [f*32+h*8+w for f in range(4) for h in range(4) for w in range(4, 8)],
              list(range(131, 137)), [141, 142], list(range(137, 141))]
    expected_map = torch.tensor([row for block in blocks for row in block + [-1]*(64-len(block))])
    torch.testing.assert_close(layout.slot_row, expected_map, atol=0, rtol=0)
    x = torch.arange(143, dtype=torch.float64).view(-1, 1, 1).requires_grad_()
    slots = vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot)
    expected = torch.where(expected_map >= 0, expected_map, 0).to(x.dtype)
    torch.testing.assert_close(slots.flatten(), expected, atol=0, rtol=0)
    torch.testing.assert_close(vsa_kernels.from_slots(slots, layout.slot_row, layout.row_slot), x, atol=0, rtol=0)
    weights = torch.randn_like(slots)
    grad, = torch.autograd.grad(slots, (x,), weights)
    expected_grad = torch.empty_like(x)
    for bid, rows in enumerate(blocks):
        expected_grad[rows] = weights[bid*64:bid*64+len(rows)]
    torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)


@pytest.mark.parametrize("samples", [
    [[Segment("video", (4, 4, 16), .5), Segment("text", 7)]],
    [[Segment("video", (1, 1, 3))]],  # Original rows still need a trailing padded block.
    [[Segment("video", (1, 8, 8))]],  # Aligned dense layout.
    [[Segment("text", 3), Segment("video", (5, 4, 8), .5), Segment("audio", 0)],
     [Segment("video", (1, 1, 3)), Segment("text", 0)]],
])
def test_public_layout_slot_round_trip(samples):
    layout = build_vsa_layout(samples, head_num=2)
    x = torch.randn(layout.tokens, 2, 8, requires_grad=True)
    slots = vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot)
    assert slots.shape[0] == layout.padded_tokens
    assert not slots[layout.slot_row < 0].count_nonzero()
    restored = vsa_kernels.from_slots(slots, layout.slot_row, layout.row_slot)
    torch.testing.assert_close(restored, x, atol=0, rtol=0)
    dy = torch.randn_like(x)
    grad, = torch.autograd.grad(restored, x, dy)
    torch.testing.assert_close(grad, dy, atol=0, rtol=0)


def test_segment_pure_layout_and_adjoint():
    layout = build_vsa_layout([(Segment("video", (5, 6, 7), .9), Segment("text", 65), Segment("audio", 67))], head_num=2)
    assert layout.block_sizes[:4].tolist() == [64, 1, 64, 3]
    assert layout.block_sizes.sum().item() == 342
    x = torch.randn(layout.tokens, 2, 8, requires_grad=True)
    slots = vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot)
    assert torch.equal(vsa_kernels.from_slots(slots, layout.slot_row, layout.row_slot), x)
    assert torch.count_nonzero(slots[layout.slot_row < 0]) == 0
    weights = torch.randn_like(x)
    (vsa_kernels.from_slots(slots, layout.slot_row, layout.row_slot) * weights).sum().backward()
    torch.testing.assert_close(x.grad, weights, rtol=0, atol=0)


def test_full_blocks_still_require_reordering():
    full = build_vsa_layout([(Segment("video", (4, 4, 8), .9), Segment("text", 64))], head_num=2)
    ragged = build_vsa_layout([(Segment("video", (5, 4, 8), .9), Segment("text", 64))], head_num=2)
    assert full.all_blocks_full
    assert not ragged.all_blocks_full
    x = torch.randn(full.tokens, 2, 8)
    slots = vsa_kernels.to_slots(x, full.slot_row, full.row_slot)
    assert not torch.equal(slots, x)
    restored = vsa_kernels.from_slots(slots, full.slot_row, full.row_slot)
    torch.testing.assert_close(restored, x, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_block_reduction_merge_adjoint_with_ragged_and_dense_samples(dtype):
    layout = build_vsa_layout(SAMPLES, head_num=2)
    torch.manual_seed(237)
    xs = [torch.randn(layout.tokens, 2, 8, dtype=dtype, requires_grad=True) for _ in range(2)]
    xs.append(torch.randn(layout.coarse_blocks.numel(), 2, 8, dtype=dtype, requires_grad=True))
    refs = [x.detach().clone().requires_grad_() for x in xs]
    actual = vsa_kernels.fine_coarse_gate_add(*xs, layout.row_coarse, layout.coarse_rows)
    # Native PyTorch autograd is independent of the explicit block-reduction adjoint.
    accumulation = refs[2].float() if dtype == torch.bfloat16 else refs[2]
    expected = vsa_kernels._fine_coarse_gate_add_forward(*refs[:2], accumulation, layout.row_coarse)
    dy = torch.randn_like(actual)
    got = torch.autograd.grad(actual, xs, dy)
    want = torch.autograd.grad(expected, refs, dy)
    for a, b in zip((actual, *got), (expected, *want)):
        torch.testing.assert_close(a, b, atol=1e-10 if dtype == torch.float64 else 1e-5,
                                   rtol=1e-10 if dtype == torch.float64 else 1e-5)
