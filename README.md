# cudnn_vsa

Packed multimodal attention for DiT/MMDiT training:
**cuDNN block-sparse fine attention + FA4 varlen coarse attention**.

## Install

With CUDA-enabled PyTorch installed, run from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Use

Inputs are separate, contiguous BF16 `q`, `k`, `v`, `gate` tensors of shape
`[T, H, 128]`. Output preserves the input token order.

```python
import torch
from vsa import vsa_attention
from vsa_layout import Segment, build_vsa_layout

# One 720p 5s video and two independent 720p images, each with text.
H, D = 2, 128
layout = build_vsa_layout(
    samples=[
        [Segment("video", (31, 45, 80), sparsity=0.9), Segment("text", 2048)],
        [Segment("video", (1, 45, 80), sparsity=None), Segment("text", 2048)],
        [Segment("video", (1, 45, 80), sparsity=None), Segment("text", 2048)],
    ],
    head_num=H,
    dense_query_full=False,
    device="cuda",
)
q, k, v, gate = [
    torch.randn(layout.tokens, H, D, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
    for _ in range(4)
]
out = vsa_attention(q, k, v, gate, layout)
out.backward(torch.randn_like(out))
```

Each inner list is one joint sample; different samples cannot attend to each
other. Inputs follow the sample and segment order in `samples`. Each sample
contains at least one video, with optional additional video/image, text, audio
or keyframe segments. Video sizes are `(F, H, W)` token grids in raster order;
`F=1` is an image. Other segment kinds take token counts and use `sparsity=None`.
Reuse the layout across layers with the same geometry and settings.

For training with tokens already in block order, pass `input_order="block"`
to `build_vsa_layout`. Inputs and outputs contain only valid tokens, with no
padding. Each sample uses dense blocks first, followed by 4×4×4 video tiles.
`layout.input_rows` maps this order to the original raster stream; apply the
same order to Q/K/V, gate and position coordinates once at the model boundary.
The attention path only adds/removes padding, packs Q/K/V together, and
skips those copies when all blocks are full. See [the input contract](docs/block_major.md).

Add this directory to `PYTHONPATH`, or copy `vsa.py`, `vsa_layout.py` and
`vsa_kernels.py` into your project and adjust imports.

## Attention semantics

`sparsity` is a video segment's **KV-block drop fraction**. By default, every
query block and head independently selects top-k blocks from **each** video KV
region, including cross-video attention:

```text
k = max(1, ceil((1 - sparsity) * video_region_blocks))
```

The KV segment sets the budget; Q/K determines the selected blocks.
`sparsity=None` makes a region dense: its KV blocks are always visible, with
padding only. Numeric sparsity uses 4×4×4 video tiles; `0` keeps all those blocks.

[FastH3](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/fastvideo/attention/backends/video_sparse_attn_h3.py#L27-L31)
adds a query-side exception: dense queries see **all** KV blocks.
`dense_query_full=True` enables that exception; the default `False` applies
video KV sparsity to every query. Dense queries come from `sparsity=None` regions.

Example fine masks for one sample/head at sparsity 0.5. Both axes follow the
internal block order: **dense first, video second**. `■` visible; `□` masked.
Top-k selections are illustrative.

**`dense_query_full=False` (default)**

```text
           Dense KV    Video KV
           0 1         2 3 4 5
Dense Q0   ■ ■         □ ■ ■ □
Dense Q1   ■ ■         ■ □ □ ■
Video Q2   ■ ■         ■ □ ■ □
Video Q3   ■ ■         □ ■ □ ■
Video Q4   ■ ■         ■ ■ □ □
Video Q5   ■ ■         □ □ ■ ■
```

**`dense_query_full=True` (FastH3 dense-query rule)**

```text
           Dense KV    Video KV
           0 1         2 3 4 5
Dense Q0   ■ ■         ■ ■ ■ ■
Dense Q1   ■ ■         ■ ■ ■ ■
Video Q2   ■ ■         ■ □ ■ □
Video Q3   ■ ■         □ ■ □ ■
Video Q4   ■ ■         ■ ■ □ □
Video Q5   ■ ■         □ □ ■ ■
```

For samples with numeric sparsity, coarse attention covers BF16-pooled blocks
from **all segments**, producing `fine + gate * coarse[its_block]` per token.
A sample whose segments all have `sparsity=None` uses **full attention through
BSA**, with no coarse branch. This is how the independent images above are
handled; a dense image inside a VSA sample still participates in coarse.

## Ring context parallelism

An optional `ring_cp_group` distributes VSA across sequence shards. The caller
owns the process group and any Ulysses all-to-all before/after this call.
Every peer must own the same attention heads.

```python
from vsa_layout import build_ring_vsa_layout

layout = build_ring_vsa_layout(
    samples, head_num=2,
    ring_size=ring_group.size(), ring_rank=ring_group.rank(), device="cuda",
    input_order="block",
)
# Q/K/V/gate are compact owned-block rows, without padding.
# input_rows maps them to original global coordinates for positions/reference.
out = vsa_attention(q, k, v, gate, layout, ring_cp_group=ring_group)
```

Complete 64-token blocks are assigned cyclically across peers. THD samples
remain isolated; ragged lengths, partial blocks and empty local partitions
are represented by the layout. Output and gradients preserve local input
order. Passing `ring_cp_group=None` retains single-device behavior.
`input_order="raster"` remains available for existing restricted-raster callers.
See [ring layout and algorithm](docs/ring_cp.md) for the ownership contract.
Compact inputs pass the complete numerical/group/graph suite. Same-allocation
comparisons improve 64k total MFU by 2.4–2.9%; 720p30s/ring8 stays near
17.0–17.2%, below the parent's approximately 20.3% no-ring target.
See the [compact-input results](docs/ring_cp_benchmark.md#compact-block-major-child-branch).
The prior raster-input ring baseline passes output/gradient and changed-input CUDA Graph checks for single,
packed, ragged, dense-only and empty-peer cases on GB200.
The initial 64k/H2/ring8 implementation reaches 2.04 ms (single sample) and
1.83 ms (packed THD) for complete forward/backward, or 5.05%/2.80%
attention-core MFU. See [ring measurements and scope](docs/ring_cp_benchmark.md).
For the 720p30s workload, a caller-created high-priority NCCL group improves
overlap. The benchmark document compares forward and backward separately
with the historical 720p5s single-device baseline. Group setup and optional
window/scheduler controls are documented in [the ring design](docs/ring_cp.md).
The performance table below describes the existing single-device operator.

```bash
torchrun --standalone --nproc-per-node=8 --module benchmark.ring --input-order block
```

This validates small native references and 65,536-token H2 single/packed cases,
then measures complete forward/backward CUDA Graph latency. Distributed MFU
uses the slowest rank and all eight GPUs' dense BF16 peak.

## Performance

The benchmark uses the sample above: 121 frames,
4× temporal and 16× spatial compression, 2048 text tokens per sample,
124944 tokens total, H=2/4, D=128, video sparsity 0.9, `dense_query_full=False`.

```bash
# VSA forward/backward time, MFU and torch traces.
python -m benchmark.perf_vsa --peak-tflops 2500
# FA4 full vs VSA, alternating order on the same GPU.
python -m benchmark.compare --peak-tflops 2500
```

Both use CUDA Graph and validate replay against eager execution. Timing covers
packing, pooling, routing, fine, coarse, restoration and merge.
`benchmark/results/perf_vsa/` contains `perf.json` and `vsa-h{2,4}.trace.json`;
traces are collected after timing. The comparison writes `benchmark/results/compare.json`.

MFU counts each method's valid attention GEMM FLOPs; backward is twice forward.
Padding, routing and non-GEMM operations are excluded from the FLOP numerator.
Set `--peak-tflops` to your GPU's dense BF16 peak.

Environment: GB200 (2500 TFLOP/s BF16), PyTorch 2.12, CUDA 13.2, cuDNN 9.22 / frontend 1.28.0, Triton 3.6, CuTe 4.6.2, FA4 4.0.0b35.

| Method | H | Forward ms | Forward MFU | Backward ms | Backward MFU | F+B ms | F+B MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| FA4 full | 2 | 9.201 | 57.78% | 25.363 | 41.92% | 34.460 | 46.28% |
| cuDNN VSA | 2 | 2.338 | 25.10% | 6.702 | 17.51% | 9.044 | 19.47% |
| FA4 full | 4 | 18.535 | 57.37% | 50.575 | 42.05% | 69.102 | 46.16% |
| cuDNN VSA | 4 | 4.582 | 25.61% | 13.166 | 17.82% | 17.756 | 19.82% |

CUDA Graph, 5 warmups, 80 timed replays; each time column is an independent
median. FA4 full and VSA were measured in separate runs with the same workload
and environment.

## Tests

```bash
python -m pip install 'pytest>=8'
python -m pytest -q -m 'not fastvideo'

# Optional FastH3 reference
python -m pip install -r tests/requirements-fastvideo.txt
python -m pytest -q tests/test_fastvideo.py
```

[MIT License](LICENSE).

## Acknowledgements

Thanks to [FastVideo / FastH3](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/fastvideo/attention/backends/video_sparse_attn_h3.py)
for the VSA reference and semantics, and
[NVIDIA cuDNN BSA](https://github.com/NVIDIA/cudnn-frontend/blob/v1.28.0/python/cudnn/block_sparse_attention/_interface.py)
for the core block-sparse attention kernels used by this implementation.
