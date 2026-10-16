"""One 720p 5-second video and two independent 720p images, each with text."""
import math
from itertools import accumulate

import torch
from vsa_layout import Segment

RESOLUTION = (720, 1280)
NUM_IMAGES = 2
HEAD_COUNTS, HEAD_DIM = (2, 4), 128
FPS, DURATION_SECONDS = 24, 5
VIDEO_FRAMES = FPS * DURATION_SECONDS + 1
TEMPORAL_COMPRESSION, SPATIAL_COMPRESSION = 4, 16
TEXT_TOKENS, VIDEO_SPARSITY = 2048, 0.9


def make_geometry():
    """One video and two independent images, each with its own text."""
    height, width = RESOLUTION
    h, w = math.ceil(height / SPATIAL_COMPRESSION), math.ceil(width / SPATIAL_COMPRESSION)
    frames = (VIDEO_FRAMES - 1) // TEMPORAL_COMPRESSION + 1
    return [[Segment("video", (t, h, w), sparsity=rate), Segment("text", TEXT_TOKENS)]
            for t, rate in [(frames, VIDEO_SPARSITY), *([(1, None)]*NUM_IMAGES)]]


def make_inputs(heads=2, *, device="cuda", dtype=torch.bfloat16, seed=2026):
    """Return (q, k, v, gate), dy; both backends share the same input leaves.

    Timing uses autograd.grad, so neither backend accumulates into leaf .grad.
    """
    if heads not in HEAD_COUNTS:
        raise ValueError(f"Timed inputs require heads in {HEAD_COUNTS}")
    samples = make_geometry()
    tokens = sum(seg.tokens for sample in samples for seg in sample)
    generator = torch.Generator(device=device).manual_seed(seed)
    xs = [torch.randn(tokens, heads, HEAD_DIM, device=device, dtype=dtype, generator=generator)
          for _ in range(5)]  # q, k, v, gate, dy
    return tuple(x.requires_grad_() for x in xs[:4]), xs[4]


def describe():
    """Serializable input definition for the benchmark evidence."""
    samples = make_geometry()
    t, h, w = samples[0][0].size
    lengths = [sum(seg.tokens for seg in sample) for sample in samples]
    segments, offset = [], 0
    for i, sample in enumerate(samples):
        for seg in sample:
            end = offset + seg.tokens
            segments.append({"sample": i, "kind": seg.kind, "start": offset, "stop": end})
            offset = end
    return {"resolution_hw": list(RESOLUTION),
            "padded_resolution_hw": [h * SPATIAL_COMPRESSION, w * SPATIAL_COMPRESSION],
            "fps": FPS, "duration_seconds": DURATION_SECONDS, "video_frames": VIDEO_FRAMES,
            "temporal_compression": TEMPORAL_COMPRESSION, "spatial_compression": SPATIAL_COMPRESSION,
            "video_grid": [t, h, w], "image_grid": [1, h, w], "num_images": len(samples)-1,
            "text_tokens_per_sample": TEXT_TOKENS, "video_sparsity": VIDEO_SPARSITY,
            "attention_scope": "independent_samples", "layout": "THD",
            "sample_lengths": lengths, "token_order": "sample_major",
            "segments": segments, "cu_seqlens": [0, *accumulate(lengths)]}
