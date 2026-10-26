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
other. With the default raster order, inputs follow the sample and segment
order in `samples`. Each sample contains at least one video, with optional
additional video/image, text, audio
or keyframe segments. Video sizes are `(F, H, W)` token grids in raster order;
`F=1` is an image. Other segment kinds take token counts and use `sparsity=None`.
Reuse the layout across layers with the same geometry and settings.

Add this directory to `PYTHONPATH`, or copy `vsa.py`, `vsa_core.py`,
`vsa_layout.py` and `vsa_kernels.py` into your project and adjust imports.

## Input order

`layout.input_order` explicitly declares the token order:

| `input_order` | Input tokens | Slot conversion |
| --- | --- | --- |
| `"raster"` (default) | Original sample/segment order, video in FHW order | Reorder into blocks and add padding |
| `"block"` | Compact canonical blocks, no padding | Add padding only; skip copies for full blocks |

### Raster input

The default `input_order="raster"` accepts the original FHW sequence, as in
the example above. `vsa_attention` reorders and pads it internally, then
restores the output to the same raster order. No preprocessing is required.

### Block input

Pass `input_order="block"` to `build_vsa_layout` when the caller already
supplies block-ordered tokens. Samples stay in declared order. Within each
sample, dense segments come first in chunks of at most 64 tokens, followed by
numeric-sparsity video segments in 4×4×4 tiles. Segment order is preserved
within each category; blocks never cross segment or sample boundaries.
`layout.input_rows[i]` gives the original raster row for supplied row `i`
(an identity map for raster inputs). Apply the same ordering to Q/K/V, gate,
positional coordinates and dO. Outputs and gradients preserve that order.

For raster data, call `to_block_order` once before the entire VSA stack.
This repository provides the fused reorder kernel separately from
`vsa_attention`: it permutes all supplied tensors in one kernel launch,
preserving their shapes and dtypes and adding **no padding**. Inputs that are
already in block order go directly to attention. Keep that order across layers.

Both `to_block_order` and `to_raster_order` return a tuple, including for one
tensor. They accept contiguous CUDA `[T, ...]` tensors on the layout's device,
with different row shapes and floating-point or integer dtypes. Use a full
block layout. For a complete model, call
`to_block_order(hidden_states, positions, vsa_layout=layout)` before generating
the layers' Q/K/V/gate; those projections then inherit block order. The
operator-level example below starts from already projected raster tensors:

```python
import torch
from vsa import to_block_order, to_raster_order, vsa_attention
from vsa_layout import Segment, build_vsa_layout

H, D = 2, 128
layout = build_vsa_layout(
    [[Segment("video", (5, 9, 11), sparsity=0.9), Segment("text", 67)]],
    head_num=H, input_order="block", device="cuda",
)
q, k, v, gate = [
    torch.randn(layout.tokens, H, D, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
    for _ in range(4)
]
position_ids = torch.arange(layout.tokens, device="cuda")

# One fused reorder before attention; partial blocks add no input padding.
q, k, v, gate, position_ids = to_block_order(
    q, k, v, gate, position_ids, vsa_layout=layout,
)
assert q.shape[0] == layout.tokens  # 562 valid tokens, not 1280 padded slots.
out = vsa_attention(q, k, v, gate, layout)

# Keep block order between layers; restore only if the model output needs it.
(out_raster,) = to_raster_order(out, vsa_layout=layout)
out_raster.backward(torch.randn_like(out_raster))
```

The reorder's backward applies the inverse permutation to gradients. The
inverse helper restores original sample/segment/FHW order after the stack.

Internally, blocks occupy 64 slots: `slot_row` maps slots to input rows, using
`-1` for padding, and `row_slot` is the inverse map for valid rows. Block inputs
use fused Q/K/V padding and fused dQ/dK/dV restoration; padding is zero and
excluded from means and attention. `slots_are_input` skips conversion for
full block inputs. Full raster blocks still require the FHW permutation.

## Code structure

| File | Responsibility |
| --- | --- |
| `vsa.py` | Public attention entry and explicit block/raster reorder helpers |
| `vsa_core.py` | Per-region routing, cuDNN BSA forward/backward, FA4 coarse attention and execution order |
| `vsa_layout.py` | Static sample/block geometry, row maps, routing templates and coarse boundaries |
| `vsa_kernels.py` | Fused token reordering, slot gathers, Q/K/V padding, FP32 pooling and gated merge kernels |

Both input orders share the core's attention math, per-region budgets and
fine-before-coarse schedule. The core does not import the public entry.

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

The table below is a baseline measured before shared-core extraction and
compact block input support.

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

### Shared core and input layouts

The shared-core implementation was compared with the original raster-input
implementation on the same GB200, using separate baseline/candidate/candidate/
baseline runs for raster and block inputs. Each case used 30 warmups and
500 timed CUDA Graph replays. The eight
cases cover single/packed 720p5s, H2/H4 and aligned 64k under both dense-query
policies. Software: PyTorch 2.12.0a0/NV26.04, CUDA 13.2, cuDNN frontend 1.28.0,
CuTe 4.6.2 and FA4 4.0.0b35. Relative MFU changes are:

| Candidate input | Forward | Backward | Total |
| --- | ---: | ---: | ---: |
| Raster | -0.166% to +0.520% | -0.450% to +0.113% | -0.205% to +0.104% |
| Block | +1.690% to +11.735% | +0.163% to +4.312% | +0.527% to +6.313% |

Raster packed 720p5s/H2 with `dense_query_full=True` loses 0.205% total MFU
and 0.450% backward MFU. Its total-time control/repeat drift is 0.029% / 0.083%,
so a universal no-regression claim is not supported. Block timings exclude
the caller's one-time model-boundary permutation; these are standalone
operator measurements, not end-to-end training results.

## Tests

The tests cover both input orders against independent token math, fused
reorder round trips with mixed tensor shapes/dtypes, padding and their adjoints,
and CUDA Graph replay of reorder/attention/restore with original and changed
inputs. GPU tests are skipped when CUDA is unavailable.

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
