# cudnn_vsa

[English](README.md) | [简体中文](README.zh-CN.md)

<!-- Keep README.md and README.zh-CN.md synchronized, including examples and benchmark data. -->

Packed multimodal attention for DiT/MMDiT training:
**cuDNN block-sparse fine attention + FA4 varlen coarse attention**.

## Code structure

| File | Responsibility |
| --- | --- |
| [`vsa.py`](vsa.py) | Public attention entry and explicit block/raster reorder helpers |
| [`vsa_core.py`](vsa_core.py) | Routing, cuDNN BSA/FA4 calls, single-device execution and ring communication/autograd |
| [`vsa_layout.py`](vsa_layout.py) | Static sample/block geometry, row maps, routing templates and coarse boundaries |
| [`vsa_kernels.py`](vsa_kernels.py) | Fused token reordering, slot gathers, Q/K/V padding, FP32 pooling and gated merge kernels |

Ordinary VSA and Ring share these files. Add this directory to `PYTHONPATH`,
or copy these four files into your project and adjust imports.

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

Each inner list is an independent sample; attention never crosses samples.
Inputs follow the declared sample/segment order, with video in FHW raster order.
Each sample needs at least one video: `(F, H, W)` gives its token grid, and
`F=1` denotes an image. Text, audio and keyframe segments use token counts and
`sparsity=None`; additional video/image segments are allowed. Reuse layouts
across layers with the same geometry and settings.

## Input order

The caller declares `layout.input_order`; VSA does not detect the tensor’s actual row order:

| `input_order` | Input tokens | Slot conversion |
| --- | --- | --- |
| `"raster"` (default) | Original sample/segment order, video in FHW order | Reorder into blocks and add padding |
| `"block"` | Valid tokens in the layout's block order, no padding | Add block padding only |

The default raster input needs no preprocessing. Based on the layout,
`vsa_attention` reorders and pads each block to 64 slots internally, then restores
the input order for the output. Padding is excluded from computation.

### Block input

For raster data, reorder **once before the layer stack**, then keep compact block order:

```python
from vsa import raster_to_block_wo_padding

layout = build_vsa_layout(samples, head_num=H, input_order="block", device="cuda")
# [T_global, C] -> [T_global, C]: reorder rows, without padding.
tokens_block = raster_to_block_wo_padding(tokens_global, vsa_layout=layout)
# [T_global, 3] -> [T_global, 3]: preserve each (f, h, w) coordinate.
position_ids_block = raster_to_block_wo_padding(position_ids_global, vsa_layout=layout)
```

The reorder accepts one contiguous CUDA `[T_global, ...]` tensor, preserves dtype
and trailing dimensions, and requires `layout.input_order="block"`.
Positions, targets and masks use the same row mapping. For 3D RoPE, position IDs
have shape `[T_global, 3]` and retain their original `(f, h, w)` coordinates.

The layout defines block order:

- Keep sample order. Within each sample, place `sparsity=None` segments before numeric-sparsity videos, preserving declaration order within each category.
- Split `sparsity=None` segments into consecutive chunks of at most 64 tokens; split numeric-sparsity videos (including `0`) into 4×4×4 tiles.
- Visit tile origins and tokens in F/H/W order (W fastest), keeping only valid tokens in edge tiles. Blocks never cross segment or sample boundaries.

`layout.input_rows` records the corresponding global raster indices.
For Ring, the same reorder also selects this rank’s owned blocks; see the Ring section below.

### Reorder interfaces

| Conversion | Where | Forward interface | Inverse interface |
| --- | --- | --- | --- |
| Raster ↔ unpadded blocks | Layer-stack boundary, called explicitly | `raster_to_block_wo_padding` | `block_wo_padding_to_raster` (full layouts only) |
| Raster ↔ padded blocks | Inside VSA, raster input | `raster_to_block_w_padding` | `block_w_padding_to_raster` |
| Unpadded ↔ padded blocks | Inside VSA, block input | `block_wo_padding_to_block_w_padding` | `block_w_padding_to_block_wo_padding` |

All three conversions support autograd. `w_padding` / `wo_padding` mean with / without
block padding. VSA’s internal `_qkv` variant fuses Q/K/V padding and gradient
restoration to reduce kernel launches; it does not include the QKV linear projection.
See [`vsa_layout.py`](vsa_layout.py) and [`vsa_kernels.py`](vsa_kernels.py)
for internal row maps and kernel details.

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

Example fine masks for one sample/head at sparsity 0.5, using the block order
above. `■` visible; `□` masked.
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

Create an ordinary-priority NCCL group once during parallel initialization,
then pass it to the layout:

```python
import torch.distributed as dist

# Call on all WORLD ranks in the same group-creation order, before building layouts.
group = dist.new_group(ranks=ring_ranks, backend="nccl")
layout = build_vsa_layout(
    samples, head_num=H, input_order="block", device="cuda",
    ring_cp_group=group, ring_partition="cyclic",  # Or "contiguous".
)
# T_ring_local = layout.tokens: Ring-local valid token count; C is hidden size.
# Global [T_global, C] -> Ring-local [T_ring_local, C], reordered without padding.
tokens_local = raster_to_block_wo_padding(tokens_global, vsa_layout=layout)
# Global [T_global, 3] -> Ring-local [T_ring_local, 3], with the same row mapping.
position_ids_local = raster_to_block_wo_padding(position_ids_global, vsa_layout=layout)
# Project tokens_local into Q/K/V/gate, each [T_ring_local, H, D].
out = vsa_attention(q, k, v, gate, layout)  # [T_ring_local, H, D]
```

`H` is the head count seen by VSA; Ring rank and size come from `ring_cp_group`.
Omit the group for ordinary VSA. All group members must use the same global
sample geometry, head count and partition policy. Reuse the group across
micro-batches and layers for forward/backward P2P and pooled collectives;
reuse each layout across layers with the same geometry and settings.

### Block ownership

When calling `vsa_attention` in Ring mode, Q/K/V/gate must follow the layout's
local block order: only valid tokens from this rank's owned blocks, in
increasing global block-ID order, without padding.
`raster_to_block_wo_padding` fuses selection and reordering without a full
block-ordered intermediate or communication. Its backward scatters gradients
to the original raster rows, zeroing unowned rows without cross-rank reduction.

`ring_partition` chooses ownership independently of `input_order`; the choice
is stored in `layout.ring.partition`:

| `ring_partition` | Global block ownership |
| --- | --- |
| `"cyclic"` (default) | Block `b` belongs to rank `b % R`; global numbering continues across samples |
| `"contiguous"` | Each sample contributes at most one consecutive block interval to each rank |

For contiguous ownership, start at rank 0. For each sample's N blocks, compute
`base, extra = divmod(N, R)`: visiting ranks cyclically from the current start,
assign `base + 1` consecutive blocks to the first `extra` ranks and `base` to
the rest. Advance the start by `extra` modulo R, so the next sample begins at
the first rank that received fewer blocks. If `extra=0`, keep the start.

Both policies balance per-sample and cumulative block counts within one.
They do not balance valid-token counts or dynamic top-k work. Cyclic ownership
spreads neighboring blocks across ranks; contiguous ownership makes each
sample's rank-local interval simple to slice. Both use the same attention
kernels. Changing policy requires rebuilding the layout and resharding inputs.

The pack must contain at least R blocks so every Ring rank owns valid data.
Individual samples need no token/block divisibility by R and may be absent
from some ranks. Internally every rank has `C = ceil(global_blocks / R)`
blocks of 64 slots; partial and dummy slots are excluded from the math.

### Worked example: two samples on three ranks

With `R=3`, sample A contains text block `b0` and video blocks `b1..b4`
(`sparsity=0.5`). Sample B has two blocks: `b5` contains dense image tokens,
and `b6` contains text tokens.
Global block IDs continue across samples.

| Policy | Sample / count | Rank 0 | Rank 1 | Rank 2 |
| --- | --- | --- | --- | --- |
| `contiguous` | A | `b0, b1` | `b2, b3` | `b4` |
| `contiguous` | B | `b6` | — | `b5` |
| `contiguous` | Total blocks | 3 | 2 | 2 |
| `cyclic` | A | `b0, b3` | `b1, b4` | `b2` |
| `cyclic` | B | `b6` | — | `b5` |
| `cyclic` | Total blocks | 3 | 2 | 2 |

With `contiguous`, A starts at rank 0 and gives `[2, 2, 1]` blocks to ranks
0/1/2. B starts at rank 2, which received fewer blocks, and gives one block
each to ranks 2 and 0. The next sample starts at rank 1.

### Routing and execution

| Stage | Operation |
| --- | --- |
| Block summaries | Compute local Q/K/V means from valid tokens; all-gather **FP32 K + BF16 V** and restore global block order. |
| Routing | Score with FP32 `Q_mean @ K_mean.T`, select top-k within each video region of the same sample under the attention semantics above, then map to owning ranks and local block IDs. Dense KV is always visible; padding and other samples are excluded. |
| Fine attention | Keep Q local and circulate whole K/V shards around the Ring. Compute selected blocks and merge partial outputs with one softmax normalization over all selected keys. |
| Coarse attention | Attend from local Q means to all K/V means in the same sample, then combine `fine + gate * coarse[query_block]`. Entirely dense samples use fine only. |
| Backward | Traverse the Ring in reverse, accumulate dQ/dK/dV in FP32, and reduce-scatter pooled gradients with FP32 SUM. Discrete top-k selection does not receive gradients. |

For sample A, four video blocks at `sparsity=0.5` retain two.
If top-k selects `b2, b4`, adding text block `b0` gives `[b0, b2, b4]`.

`vsa_attention` handles the internal Ring merging and returns this rank’s valid
tokens. The caller collects token shards only when a full global output is
needed after the layer stack.

VSA enqueues P2P before fine attention to overlap communication and computation;
measure the actual overlap. See [`build_vsa_layout`](vsa_layout.py) for P2P call mode
(`ring_cp_batch_p2p`) and backward compute bucket size (`ring_cp_bwd_bucket_size`).

### Experimental window transport

`ring_cp_transport="ring"` (default) is the step-by-step P2P ring above.
`ring_cp_transport="window"` instead all-gathers every peer's slot-padded K/V
shard into one KV window per direction, in group-rank order, and issues a
single block-sparse call whose routing lists are the global block ids mapped
through `layout.ring.global_to_gather`; no partial-output merge is needed.
Backward re-gathers the window, runs one block-sparse backward and
reduce-scatters FP32 dK/dV partial sums to their owners. Both transports
compute the same attention with the same precision policy (BF16 kernel
partials, FP32 cross-rank accumulation). The window holds `ring_size` shards
per direction, and its two collectives are not overlapped with the kernel.
It is the host-side shape of a fused Ring kernel: one KV tensor and one index
table per launch.

### Local kernel copy

`bsa_cute/` is a vendored copy of cudnn-frontend v1.28.0's
`cudnn.block_sparse_attention` CuTe DSL package (provenance, mechanical import
rewrite and licenses in `bsa_cute/PROVENANCE.md`). `VSA_BSA_IMPL=local`
selects it instead of the installed package (`VSA_BSA_IMPL=cudnn`, the
default); `benchmark.perf_vsa` records the choice as `bsa_impl`.
`tests/test_bsa_cute_cuda.py` checks both packages agree before the local
copy is modified. Fused Ring kernel work edits `bsa_cute/` and keeps the
installed package as the control.

### Why this design

- **Whole blocks, compact inputs**: preserve pooling/routing units without
  adding attention padding to projections or MLPs.
- **Gather summaries, circulate K/V**: enable global routing and coarse
  attention without keeping full token-level K/V on every rank.
- **Equal-sized buffers**: pad internal buffers to the maximum per-rank block
  count for matching P2P shapes and CUDA Graph reuse.

Fine attention sends whole padded K/V shards, so sparsity reduces computation,
not Ring transfer volume. Compare both ownership policies on the actual workload.

## Examples with and without context parallelism

The following example uses two attention layers with residual connections and
one optimizer step. It supports all three modes through the same VSA API.
The entire process world is one context-parallel group; `H` must be divisible
by the Ulysses size `U`, and the world size must equal `U * R`.

| Mode | Input preparation before the layers | Q/K/V/gate passed to VSA |
| --- | --- | --- |
| No CP (`U=1, R=1`) | Keep raster order | All tokens and all heads |
| Ulysses only (`U>1, R=1`) | Split the raster stream into U token slices | All tokens and H/U heads after All-to-All |
| Ring + Ulysses (`U>1, R>1`) | Select whole Ring-owned blocks and reorder once; then split each compact Ring stream into U token slices | This Ring owner's valid tokens and H/U heads after All-to-All |

Ulysses All-to-All reconstructs the token stream before VSA runs: raster order
without Ring, compact block order with Ring.
The demo's `[start:start + A]` slices this stream for Ulysses, independently of
`ring_partition`; a slice may cross a block boundary.
Only trailing **transport padding** is added to equalize Ulysses message sizes;
it is removed before attention and excluded from the loss. This is separate
from VSA's internal 64-slot block padding.

### No CP

Save the shared Python example below as `demo_cp.py` in this repository, then run:

```bash
python demo_cp.py
```

### Ulysses CP only

Use raster input as in the no-CP example. Four GPUs split tokens before the
layers and exchange tokens for heads inside each layer; no input reorder is needed:

```bash
torchrun --standalone --nproc-per-node=4 demo_cp.py --ulysses-size 4
```

### Ring + Ulysses CP

For eight GPUs with `U=4, R=2`, Ulysses groups are `[0,1,2,3]` and
`[4,5,6,7]`; Ring groups are `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`.
Each Ring pair handles the same head slice and complementary block ownership.
Both ownership policies use the same preparation and attention calls:

```bash
torchrun --standalone --nproc-per-node=8 demo_cp.py \
  --ulysses-size 4 --ring-size 2 --ring-partition cyclic
torchrun --standalone --nproc-per-node=8 demo_cp.py \
  --ulysses-size 4 --ring-size 2 --ring-partition contiguous
```

### Shared example

This example uses PyTorch's differentiable All-to-All, replicated weights and
the same full raster pack on every rank. Ulysses, projections, loss normalization
and gradient reduction are caller code. For clarity, it materializes a Ring
shard before the Ulysses slice; data loading can gather the final local rows
directly by composing those maps.

<details>
<summary>Complete Python example: demo_cp.py</summary>

```python
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_to_all_single

from vsa import raster_to_block_wo_padding, vsa_attention
from vsa_layout import Segment, build_vsa_layout


def exchange_heads(mixed, group):
    # [A,H,4,D] -> four [U*A,H/U,D] tensors; backward exchanges gradients.
    if group is None:
        return mixed.unbind(2)
    u = dist.get_world_size(group)
    a, h, n, d = mixed.shape
    send = mixed.reshape(a, u, h // u, n, d).transpose(0, 1).contiguous()  # [U,A,H/U,4,D]
    recv = all_to_all_single(torch.empty_like(send), send, group=group)  # [U,A,H/U,4,D]
    return recv.reshape(u * a, h // u, n, d).unbind(2)


def exchange_tokens(out, group):
    # [U*A,H/U,D] -> [A,H,D], reversing the forward head exchange.
    if group is None:
        return out
    u = dist.get_world_size(group)
    t, h, d = out.shape
    send = out.reshape(u, t // u, h, d).contiguous()  # [U,A,H/U,D]
    recv = all_to_all_single(torch.empty_like(send), send, group=group)  # [U,A,H/U,D]
    return recv.permute(1, 0, 2, 3).reshape(t // u, u * h, d)


parser = argparse.ArgumentParser()
parser.add_argument("--ulysses-size", type=int, default=1)
parser.add_argument("--ring-size", type=int, default=1)
parser.add_argument("--ring-partition", choices=("cyclic", "contiguous"), default="cyclic")
args = parser.parse_args()
U, R, H, D = args.ulysses_size, args.ring_size, 8, 128
input_order = "block" if R > 1 else "raster"
C = H * D  # Hidden size: 1024.
world = int(os.environ.get("WORLD_SIZE", "1"))
if U < 1 or R < 1 or H % U or world != U * R:
    parser.error("Require U >= 1, R >= 1, H % U == 0 and WORLD_SIZE == U * R")
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
torch.cuda.set_device(device)
rank = 0
if world > 1:
    dist.init_process_group("nccl")
    rank = dist.get_rank()

# All ranks create all subgroups in the same order, once before the layers.
ulysses_group = ring_group = None
if U > 1:
    for r in range(R):
        group = dist.new_group(list(range(r * U, (r + 1) * U)))
        if rank // U == r:
            ulysses_group = group
if R > 1:
    for u in range(U):
        group = dist.new_group([r * U + u for r in range(R)])
        if rank % U == u:
            ring_group = group

samples = [
    [Segment("video", (5, 9, 11), sparsity=0.9), Segment("text", 67)],
    [Segment("video", (1, 3, 5)), Segment("text", 7)],
]
T = sum(segment.tokens for sample in samples for segment in sample)  # T_global = 584.
layout = build_vsa_layout(
    samples, head_num=H // U, input_order=input_order, device=device,
    ring_cp_group=ring_group, ring_partition=args.ring_partition,
)
# T_ring_local = layout.tokens: Ring-local valid token count (T if R=1); H is total heads.
# T_ring_local can differ across Ring ranks; it need not equal T/R.
torch.manual_seed(1234)  # Identical mock input and replicated weights on all ranks.
tokens_global = torch.randn(T, C, device=device, dtype=torch.bfloat16)  # [T,C] = [584,1024]
# Example 3D RoPE convention: text uses (1..L, 0, 0), video time starts at L+1.
position_rows = []
for video, text in samples:
    axes = [torch.arange(n, device=device) for n in video.size]
    axes[0] += text.tokens + 1
    video_ids = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    text_ids = torch.zeros(text.tokens, 3, device=device, dtype=torch.long)
    text_ids[:, 0] = torch.arange(1, text.tokens + 1, device=device)
    position_rows.extend((video_ids, text_ids))
position_ids_global = torch.cat(position_rows)  # [T,3] = [584,3]: original (f, h, w).

# Ring sharding + reorder: layout.ring.partition selects cyclic/contiguous blocks.
tokens_attention = tokens_global
position_ids_attention = position_ids_global
if R > 1:
    # Global [T,C] -> Ring-local [T_ring_local,C]: valid tokens in compact block order.
    tokens_attention = raster_to_block_wo_padding(tokens_global, vsa_layout=layout)
    # Global [T,3] -> Ring-local [T_ring_local,3]: select/reorder the same token rows.
    position_ids_attention = raster_to_block_wo_padding(position_ids_global, vsa_layout=layout)

# Ulysses preparation: equal message sizes, then U token slices of this Ring shard.
# Tail padding only; per-block padding is added inside vsa_attention.
max_tokens = layout.tokens
if layout.ring is not None:
    max_tokens = int(layout.ring.peer_block_sizes.sum(dim=1).max().item())
A = (max_tokens + U - 1) // U  # Per-GPU local rows after Ulysses slicing, including padding.
tail = U * A - layout.tokens  # U*A - T_ring_local.
start = (rank % U) * A
# Ring-local [T_ring_local,C] -> padded [U*A,C] -> per-GPU local [A,C].
x = F.pad(tokens_attention, (0, 0, 0, tail))[start:start + A].contiguous()
# Ring-local [T_ring_local,3] -> padded [U*A,3] -> per-GPU local [A,3].
position_ids_local = F.pad(position_ids_attention, (0, 0, 0, tail), value=-1)[start:start + A]
valid = torch.arange(start, start + A, device=device) < layout.tokens  # [A]

layers = torch.nn.ModuleList([
    torch.nn.ModuleList([
        torch.nn.Linear(C, 4 * C, bias=False, device=device, dtype=torch.bfloat16),
        torch.nn.Linear(C, C, bias=False, device=device, dtype=torch.bfloat16),
    ]) for _ in range(2)
])
optimizer = torch.optim.SGD(layers.parameters(), lr=1e-3)
for qkvg_projection, output_projection in layers:
    mixed = qkvg_projection(x).view(A, H, 4, D)  # [A,C] -> [A,4*C] -> [A,H,4,D]
    # All-to-All: four [U*A,H/U,D] tensors; trim each to [T_ring_local,H/U,D].
    q, k, v, gate = (
        t[:layout.tokens].contiguous() for t in exchange_heads(mixed, ulysses_group)
    )
    # RoPE uses position_ids_attention [T_ring_local,3] for these unpadded Q/K rows.
    # VSA adds/removes block padding internally; its public shapes stay compact.
    out = vsa_attention(q, k, v, gate, layout)  # [T_ring_local,H/U,D]
    out = F.pad(out, (0, 0, 0, 0, 0, tail))  # [U*A,H/U,D]
    out = exchange_tokens(out, ulysses_group).reshape(A, C)  # -> [A,H,D] -> [A,C]
    x = (x + output_projection(out)).masked_fill(~valid[:, None], 0)  # [A,C]

# Sum local contributions to a global-token-normalized illustrative loss.
loss = x.float().square().masked_fill(~valid[:, None], 0).sum() / (T * C)
loss.backward()
if world > 1:
    # SUM, not average: each local loss already uses the global denominator.
    for parameter in layers.parameters():
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
optimizer.step()
reported_loss = loss.detach()
if world > 1:
    dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"U={U}, R={R}, order={input_order}, loss={reported_loss.item():.6f}")
if world > 1:
    dist.destroy_process_group()
```

</details>

When using a training framework's parameter-gradient reduction, use that
instead of the example's manual reduction, so gradients are not reduced twice.

## Performance

MFU counts valid attention GEMM FLOPs; backward is twice forward. Padding,
routing and non-GEMM operations are excluded from the FLOP numerator.
TFLOPS is reported per GPU: global FLOPs divided by latency and GPU count.
MFU divides that value
by the dense BF16 peak (`--peak-tflops`). Phase medians are independent.
One-time input sharding/reordering, graph capture and profiling are outside
timing; attention's slot conversion and padding are included.

Environment: GB200 (2500 TFLOPS BF16), PyTorch 2.12.0a0/NV26.04, CUDA 13.2, cuDNN 9.22 / frontend 1.28.0, Triton 3.6, CuTe 4.6.2, FA4 4.0.0b35.

`benchmark.perf_vsa` supports ordinary and distributed workloads (`--ring`).
Both modes share input generation, numerical checks, graph capture and timing.
`--cases` selects workloads, `--heads` selects head counts, `--check-only` runs
reference and retained-backward checks, and `--output` accepts a JSON file or
directory. The `vsa-perf/v5` JSON uses `<case>-h<heads>` keys; each case records
checks, global useful FLOPs and per-GPU throughput/MFU. Ring timing takes the
maximum rank latency for each replay before summarizing. Ring defaults to H=2 and omits traces;
add `--profile` to save a trace per rank.
Use the same explicit `--seed` in both modes when comparing performance:
the defaults are 2026 for ordinary VSA and 937 for Ring. Matched seeds,
geometry and head counts generate identical global Q/K/V/gate tensors and
upstream output gradients (dY).

### Single-device 720p5s: FA4 full and VSA

The benchmark packs one 720p 5-second video and two independent 720p images,
each with text: 121 video frames, 4× temporal and 16× spatial compression,
2048 text tokens per sample, 124944 tokens total, H=2/4, D=128, video sparsity
0.9, `dense_query_full=False`. VSA uses the default `input_order="raster"`.

```bash
# VSA forward/backward time, MFU and torch traces.
python -m benchmark.perf_vsa --peak-tflops 2500
# FA4 full vs VSA, alternating order on the same GPU.
python -m benchmark.compare --peak-tflops 2500 --warmup 30 --repeats 500 --rounds 2
```

Both use CUDA Graph and validate replay against eager execution. VSA timing
covers slot conversion, pooling, routing, fine, coarse, restoration and merge.
`benchmark/results/perf_vsa/` contains `perf.json` and
`perf-default-h{2,4}-rank0.trace.json`; traces are collected after timing.
Use `--input-order block` to measure compact block inputs, `--eager` for eager
execution, or `--no-profile` to omit traces. Input generation/reordering is
outside timing. The comparison writes `benchmark/results/compare.json`.

| Method | H | Forward ms | Forward MFU | Backward ms | Backward MFU | F+B ms | F+B TFLOPS | F+B MFU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FA4 full | 2 | 9.357 | 56.82% | 25.409 | 41.85% | 34.772 | 1146.72 | 45.87% |
| cuDNN VSA (raster) | 2 | 2.366 | 24.81% | 6.746 | 17.40% | 9.114 | 482.91 | 19.32% |
| FA4 full | 4 | 19.001 | 55.96% | 50.988 | 41.71% | 70.019 | 1138.94 | 45.56% |
| cuDNN VSA (raster) | 4 | 4.665 | 25.15% | 13.352 | 17.57% | 18.017 | 488.36 | 19.53% |

CUDA Graph, 30 warmups per measurement, two rounds of 500 timed replays per
method. FA4 full and VSA alternate order on the same GPU. Each time column is
an independent median over 1000 replays. These are standalone operator
measurements; raster-to-slot reordering and padding are included in VSA time.

### Single-head 720p44s: ordinary VSA and Ring4

`thd720p44s` packs a `265 x 45 x 80` video (sparsity 0.9) and an independent
`1 x 45 x 80` image, each with 2048 text tokens: **961696 tokens total**.
All modes use BF16, H=1, D=128, compact block input, seed 937 and
`dense_query_full=False`. Timing covers VSA forward/backward and Ring communication.

The same four-GB200 allocation runs the full sequence on one GPU without Ring,
then compares both Ring4 policies using ordinary-priority NCCL streams:

```bash
vsa_args=(--heads 1 --cases thd720p44s --input-order block --seed 937
          --warmup 10 --repeats 50 --no-profile --peak-tflops 2500)
for vsa_run in single_0 cyclic_0 contiguous_0 contiguous_1 cyclic_1 single_1; do
  vsa_mode="${vsa_run%_*}"
  if [ "$vsa_mode" = single ]; then
    CUDA_VISIBLE_DEVICES=0 python -m benchmark.perf_vsa "${vsa_args[@]}" \
      --output "benchmark/results/720p44s/$vsa_run.json"
  else
    torchrun --standalone --nproc-per-node=4 --module benchmark.perf_vsa \
      --ring --ring-size 4 --ring-partition "$vsa_mode" --batch-p2p \
      --bwd-bucket-size backend "${vsa_args[@]}" \
      --output "benchmark/results/720p44s/$vsa_run.json"
  fi
done
```

| Mode | Forward ms | Forward MFU | Backward ms | Backward MFU | F+B ms | F+B TFLOPS | F+B MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Ring (1 GPU) | 67.460 | 25.17% | 185.257 | 18.33% | 252.572 | 504.15 | 20.17% |
| Ring4, contiguous | 18.077 | 23.48% | 48.780 | 17.40% | 66.356 | 479.74 | 19.19% |
| Ring4, cyclic | 17.914 | 23.69% | 49.173 | 17.26% | 66.733 | 477.03 | 19.08% |

Each mode runs twice, with 10 warmups and 50 timed CUDA Graph replays per run,
in the order above. Each column is the median of two run medians; Ring timing
uses the maximum rank latency per replay. All reference and graph checks passed.

Relative to one GPU, full-pass speedup is **3.806x contiguous / 3.785x cyclic**;
parallel efficiency `T1 / (4 * T4)` is **95.16% / 94.62%**.
These single-host, random-input results include communication and kernel-shape
changes; actual model routing and multi-host placement can change efficiency.

## Tests

The tests cover both input orders against independent token math, fused
single-tensor reorder round trips across row shapes/dtypes, padding and their adjoints,
and CUDA Graph replay of reorder/attention/restore with original and changed
inputs. GPU tests are skipped when CUDA is unavailable.

```bash
python -m pip install 'pytest>=8'
python -m pytest -q -m 'not fastvideo'

# Optional FastH3 reference
python -m pip install -r tests/requirements-fastvideo.txt
python -m pytest -q tests/test_fastvideo.py
```

For ring validation, all workers must have CUDA and NCCL. Small cases compare
output and Q/K/V/gate gradients with independent token math; 64k cases compare
against single-device VSA. Retained backward and CUDA Graph replay with
original/changed inputs are checked before timing. Repeat with
`--dense-query-full` for the alternate dense-query policy.

```bash
# Ragged samples and partial/dummy slots; every rank owns valid blocks.
torchrun --standalone --nproc-per-node=8 --module benchmark.perf_vsa --ring \
  --cases ragged
# Interleaved groups; partial_blocks has three blocks and requires R<=3.
torchrun --standalone --nproc-per-node=8 --module benchmark.perf_vsa --ring \
  --ring-size 2 --cases ragged partial_blocks dense
# Device-bound groups and batched P2P.
torchrun --standalone --nproc-per-node=4 --module benchmark.perf_vsa --ring \
  --eager-group --cases ragged dense
# Single/packed 64k workloads, plus timing after numerical checks.
torchrun --standalone --nproc-per-node=8 --module benchmark.perf_vsa --ring \
  --cases single64k thd64k --warmup 30 --repeats 500
# Strict reproduction of the known fine-only-sample numerical failure.
torchrun --standalone --nproc-per-node=8 --module benchmark.perf_vsa --ring \
  --ring-size 2 --cases fine_only_peer --check-only
```

Ring tests use block input. Defaults are `ragged`, `single64k` and `thd64k`;
each case needs at least R global blocks. The small `partial_blocks`,
`fine_only_peer` and `dense` cases contain 3, 2 and 6 blocks respectively.
For multiple nodes, replace `--standalone` with torchrun rendezvous arguments.
JSON records numerical/graph checks and source hashes; timing and MFU follow
the definitions in **Performance**. Use `thd720p44s` for the long-sequence profile.

The strict BF16 `fine_only_peer` case has an unresolved numerical failure:
dQ and dK for a six-token fine-only sample can fall below the 0.99999 native
reference similarity threshold. The command above reports this failure;
the case and threshold remain unchanged.

## Acknowledgements

Thanks to [FastVideo / FastH3](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/fastvideo/attention/backends/video_sparse_attn_h3.py)
for the VSA reference and semantics, and
[NVIDIA cuDNN BSA](https://github.com/NVIDIA/cudnn-frontend/blob/v1.28.0/python/cudnn/block_sparse_attention/_interface.py)
for the core block-sparse attention kernels used by this implementation.

[MIT License](LICENSE).
