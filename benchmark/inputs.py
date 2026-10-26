"""Shared single-device and distributed VSA workloads and seeded inputs."""
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


CASES = {
    "single64k": ((Segment("video", (16, 60, 64), .9), Segment("text", 4096)),),
    "thd64k": (
        (Segment("video", (12, 48, 64), .9), Segment("text", 2048)),
        (Segment("video", (8, 40, 64), .9), Segment("text", 2048)),
        (Segment("video", (1, 32, 64)), Segment("text", 2048))),
    # 720p, 30 s at 24 fps: 721 frames -> 181 temporal latents;
    # spatial VAE compression 8 and 2x2 patches give 45x80 token rows.
    "single720p30s": (
        (Segment("video", (181, 45, 80), .9), Segment("text", 2048)),),
    "thd720p30s": (
        (Segment("video", (181, 45, 80), .9), Segment("text", 2048)),
        (Segment("video", (1, 45, 80)), Segment("text", 2048))),
    "ragged": (
        (Segment("video", (5, 9, 11), .7), Segment("text", 67)),
        (Segment("text", 3), Segment("video", (1, 3, 5))),
        (Segment("video", (4, 8, 12), .5), Segment("video", (5, 4, 7), .9),
         Segment("audio", 17), Segment("text", 0))),
    "empty_peers": ((Segment("video", (1, 3, 5), .9), Segment("text", 7)),),
    # With ring_size >= 2, rank 1 owns only a fine-only sample; ranks >= 2
    # have no valid tokens. All must participate in the coarse gather adjoint.
    "fine_only_peer": ((Segment("video", (1, 2, 3), .5),),
                       (Segment("video", (1, 2, 3)),)),
    "dense": ((Segment("video", (1, 9, 11)), Segment("text", 67)),
              (Segment("video", (1, 3, 5)), Segment("text", 3))),
}


def case_geometry(name):
    """Resolve a named workload; default matches the FA4 comparison."""
    return make_geometry() if name == "default" else CASES[name]


def make_inputs(heads=2, *, device="cuda", dtype=torch.bfloat16, seed=2026, samples=None):
    """Return (q, k, v, gate), dy; both backends share the same input leaves.

    Timing uses autograd.grad, so neither backend accumulates into leaf .grad.
    """
    if heads not in HEAD_COUNTS:
        raise ValueError(f"Timed inputs require heads in {HEAD_COUNTS}")
    samples = make_geometry() if samples is None else samples
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
