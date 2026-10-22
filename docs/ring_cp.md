# Ring context parallelism

The operator supports single-device and ring execution through one public
`vsa_attention` entry. Both use the same attention math and compact/raster
input contract.

## Module responsibilities

- `vsa.py`: input/group-layout validation, Q/K/V slot conversion, dispatch,
  row restoration and gated coarse merge.
- `vsa_core.py`: shared per-region top-k routing, cuDNN BSA and FA4 calls,
  and the single-device slot-attention autograd path.
- `vsa_ring.py`: caller-group runtime, P2P scheduling, differentiable mean
  gather, and ring fine-attention forward/backward.
- `vsa_kernels.py`: slot maps, shared FP32 pooling, peer-list compaction,
  online-softmax merge, FP32 gradient accumulation and gate kernels.
- `vsa_layout.py`: sample geometry, whole-block ownership and static maps.

The core and kernels do not import the public entry. Backends return fine
slots and optional coarse output; only the entry converts user rows. Ring
peers without coarse queries retain a zero-valued dependency on gathered
means so every peer executes reduce-scatter in backward. The entry also
preserves zero gate gradients on these fine-only rows, including NaN gates.

## Boundary

The caller optionally applies Ulysses all-to-all before VSA and its inverse
after VSA. This library accepts one explicit PyTorch ring process group; it
does not create groups or know about tensor parallelism or Ulysses. Every
ring peer owns the same heads and different sequence blocks. `None` retains
the existing single-device API and never denotes the default world group.

With a logical `[ring, ulysses]` mesh, the input A2A operates among peers
with a fixed ring coordinate; the ring group fixes the Ulysses coordinate.
After A2A, each rank has all rows for its ring coordinate and `H/U` heads.
Q, K, V and the elementwise gate must use this same local THD ordering. The
output uses that ordering, so the external inverse A2A can restore the
projection's distribution. The caller may prepare this distribution at the
data/projection boundary rather than gathering a full sequence in each
attention layer.

## Packed THD ownership

Plan each sample's blocks exactly as for single-device VSA: dense segments
first, followed by sparse video FHW tiles; blocks contain at most 64 valid
tokens. Samples remain isolated. Assign global block `b` to group rank
`b % R`, at local slot block `b // R`. This distributes every sample and
segment across the ring with at most one block of count imbalance, including
the more expensive dense-query-full rows. It avoids splitting tiles,
repeating dense KV, and concentrating a complete short sample on one rank.

Pass `input_order="block"` for compact local THD inputs already arranged
by owned block ID, with each block's valid tokens adjacent and no padding.
`input_rows` still maps these local rows to the original global raster
coordinates. Slot conversion only adds/removes padding; Q/K/V padding and
its adjoint share a fused kernel, and complete physical blocks skip copying.
The default `input_order="raster"` preserves the previous restricted
sample/segment/raster order and its permutation path.
Different ranks can have different valid token counts. Communication uses
`ceil(global_blocks/R)` physical blocks per rank, with zero-filled dummy
blocks. Real partial blocks retain their original valid-token counts.
Padding never enters pooling denominators, routed KV, output rows, or useful
FLOPs. No sample length must be divisible by R or 64.

This ownership is not the same as a contiguous-token or causal zigzag CP
partition. An adapter for such an input must redistribute/permute rows
according to `input_rows` and invert that mapping for outputs. The library
does not silently reinterpret an existing shard. Position IDs and any
caller-owned RoPE must follow the original global token coordinates.

For a `[R, U]` ring/Ulysses mesh, distribute whole blocks to ring
coordinates first, then partition each ring coordinate's compact block
stream among its Ulysses ranks. The external A2A must concatenate those
parts into increasing owned-block order for the selected heads. Equal
token slices of the global compact stream do not implement the cyclic
ownership rule. Variable lengths or caller-side transport padding must
be handled by the external A2A implementation, and removed before VSA.
Maintain original position coordinates and the same Q/K/V/gate order.

For example, R=2 assigns blocks `[0,2,4,6]` and `[1,3,5,7]`; within the
first ring coordinate, U=2 may hold `[0,2]` and `[4,6]`. Input A2A
reconstructs `[0,2,4,6]` with half the heads on each U rank. VSA keeps that
row order throughout its ring; the inverse A2A restores the U partitions.

Global sample/segment boundaries and KV budgets are static metadata. Only
local query routing tables are materialized. A local query may have zero
selected blocks on a particular peer, even if its global attention is
nonempty. Such a partial result is O=0, LSE=-infinity.

## Forward and backward

Pool Q/K/V locally in FP32. All-gather pooled FP32 K/V once; detached Q/K
views select the exact global per-region top-k. Coarse attention uses BF16
means and local queries with global sample-local KV. Its gather has a
reduce-scatter sum adjoint, including peers with no coarse queries.

Partition selected global block IDs by owner and translate to peer-local
IDs. A prefix-scan kernel compacts the already sorted global lists without
sorting each peer list again. Long lists use scans of at most 2048 entries,
carrying the compacted count between chunks and visiting only a row's valid
input entries. This bounds scan state when a few dense queries make the
shared list capacity much wider than most sparse rows. Unused output entries
are zeroed, including empty rows and output lists spanning multiple chunks.
Peer metadata bases are 16-byte aligned, even when a rank has an odd number
of blocks. Fine attention keeps Q local and circulates BF16 KV, prefetching the
next shard while the current BSA call runs. Merge partial O and LSE in FP32.
Each block is visited once; the forward saves local Q, the final received
KV shard, global fine O/LSE, and the selected block lists.

Fine backward starts from the saved final shard and traverses the ring in
reverse with the final fine O/LSE and local dO. Accumulate dQ in FP32.
Pipeline the next KV transfer with the previous step's FP32 dK/dV
accumulation transfer. The last shard belongs to this rank, so no final
gradient-return hop or backward KV exchange prefix is required.
Ping-pong receives use a clone of the saved shard to preserve repeated
backward with `retain_graph=True`. The coarse branch and gate retain their
existing mathematical meaning. No derivative flows through top-k.

A single pointwise kernel accumulates dQ and combines incoming FP32 dK/dV
with the current BF16 BSA gradients, avoiding temporary stacked/cast gradients.
For lazy NCCL groups, individual asynchronous send/receive calls use
odd/even posting order for communicator initialization. Device-bound eager
groups use batched P2P by default, since PyTorch serializes their individual
P2P operations on the parent communicator. `ring_cp_batch_p2p=True/False`
explicitly selects either mode for comparison on the caller's topology. The
library does not change NCCL environment settings or create process groups.
On the measured two-node GB200 topology, a caller-created NCCL group with
`ProcessGroupNCCL.Options().is_high_priority_stream=True` improves ring
overlap substantially. Set this when constructing the group, before using
it; passing a high-priority CUDA stream to VSA does not configure NCCL's
own streams. For example, all world ranks can create their ring groups in
the same order with:

```python
import torch.distributed as dist

options = dist.ProcessGroupNCCL.Options()
options.is_high_priority_stream = True
ring_group = dist.new_group(ranks=ring_ranks, pg_options=options)
```

`ring_cp_bwd_bucket_size="auto"` uses a measured SM100 tuning entry: 64
query blocks per backward bucket for a single sample with two heads and
128 physical local blocks. Other shapes, including packed samples, retain
the backend default. Pass `None` for the backend default everywhere, or a
positive integer to override it. Smaller buckets expose more independent
KV-gradient tasks, at the cost of additional partial gradient accumulation;
the fastest size varies with sample geometry. This entry is intentionally
limited to the measured shape.

The supported ring schedule circulates one KV shard per step. Forward uses
cuDNN's default scheduler and split count. Coalesced-window and forward
scheduler/split sweeps remain in the historical experiment revision; they
are not part of this API because no stable total-time benefit was established.

## Validation and performance

Initial target: BF16, D128, H2, R8, 65,536 valid global tokens. The single
sample is a 16x60x64 video plus 4096 text tokens at video sparsity 0.9.
The packed case has two independent video/text samples and one independent
dense image/text sample, also totaling 65,536 tokens. Small ragged cases
cover uneven blocks, absent local samples, both dense-query policies, and
fine-only samples.

Validate output and Q/K/V/gate gradients against independent token math on
small cases and the existing single-device VSA on the full workload.
Validate CUDA Graph replay with original and changed inputs before timing.
Measure distributed latency using the maximum across ranks for every sample.
Useful-FLOP MFU divides global useful forward/backward FLOPs by that latency
and by R times the GPU's dense BF16 peak. Do not count padding, routing,
communication, or redundant work as useful model FLOPs.

Optimization is evidence driven: compare complete forward/backward time,
then inspect traces for exposed communication, metadata preparation, pooling,
and BSA utilization. Preserve the same sparse math and numerical thresholds.
The shared-core refactor (`1236082`) passes 129 CPU tests (50 CUDA/optional
skips) and 177 tests per GPU node (two optional skips). SLURM 8158753 checked
groups 1/2/4/8, both policies, lazy/eager groups, compact/raster inputs,
retained backward and original/changed-input graphs. Its baseline/refactor
comparison produced 60 complete result files and 2,912 passing rank/check
records, with minimum similarity 0.9999911026. Source hashes and software
versions were verified across all completed files.

An added `fine_only_peer` case
exposes an existing numerical limit: for two six-token samples at R2, the
fine-only peer's dK similarity to native FP32 token math is 0.9999890332,
below the unchanged 0.99999 threshold. The baseline (`b4d92d3`) and refactor
produce bitwise-identical outputs and gradients in both ring and CP1 paths
on this case (40 tensors, both policies, SLURM 8158744). Ring and CP1 match
exactly; the expanded strict numerical suite is therefore not fully passing.
The case fails at R2/4/8 and in the eager-group runs under both policies.
It remains available for reproduction and is not waived: the full job and
analyzer return failure despite completion of the other checks and timings.

Paired before/after/after/before measurements use the same two-node GB200
allocation, 30 warmups and 500 timing repeats per arm. The following values
use the mean of the refactor's two phase medians:

| Refactored dense-query-full workload | Forward MFU | Backward MFU | Total MFU |
| --- | ---: | ---: | ---: |
| Single 720p5s, no ring | 27.272% | 18.070% | 20.360% |
| Single 720p30s, ring8 | 22.943% | 15.388% | 17.371% |
| Packed 720p30s, ring8 | 22.804% | 15.368% | 17.322% |

Against `b4d92d3` on identical inputs, total-MFU relative changes range
from -0.140% to +0.439% for ring and -0.220% to +0.185% for CP1. Maximum
total-time control drift is 0.158% / 0.191%, respectively. Phase-level
results are not uniformly positive: ring backward reaches -0.203%, and
aligned-64k CP1 backward reaches -0.730% (control drift at most 0.252%).
The overall totals are close, but there is no universal no-regression claim.

The no-ring row is a cross-length MFU target, not an equal-workload latency
control. The target remains unmet, including the earlier 20.312% target.
External Ulysses, projections and model-boundary conversion are excluded.
Megatron ring integration and end-to-end training validation remain separate.

Detailed experimental ledgers and the source-isolated block-order driver
are archived in the toolkit under `docs/benchmarks/dit/` and
`scripts/dit/block_major_vsa/`. The published
[pre-refactor ledger](https://github.com/niyunsheng/cudnn_vsa/blob/b4d92d36667981ffbdcf453b7a92d1126eb0ab96/docs/ring_cp_benchmark.md)
retains the complete historical evidence and rejected experiments.

```bash
python -m pytest -q
# Small native-reference, retained-backward and changed-input graph checks.
torchrun --standalone --nproc-per-node=8 --module benchmark.ring \
  --input-order block --cases ragged empty_peers fine_only_peer dense
# Interleaved subgroups; repeat with --ring-size 1 and 4.
torchrun --standalone --nproc-per-node=8 --module benchmark.ring \
  --input-order block --ring-size 2 --dense-query-full
# Long workload, full numerical checks before independent phase timing.
torchrun --standalone --nproc-per-node=8 --module benchmark.ring \
  --input-order block --high-priority-group --dense-query-full \
  --cases single720p30s thd720p30s --warmup 30 --repeats 500
```

Use `--eager-group` to exercise device-bound batched P2P and
`--input-order raster` to validate restricted raster rows. For multiple nodes, replace
`--standalone` with torchrun rendezvous arguments. Each result records source
hashes, numerical checks and graph checks; numerical failures stop timing.

The long-sequence cases `single720p30s` and `thd720p30s` use a 720p,
30-second video at 24 fps: 721 frames, temporal compression 4, and spatial
compression 16 after patchification, giving a `181x45x80` token grid.
Each sample includes 2048 text tokens. The single case has 653648 valid
tokens; the packed case adds an independent dense `1x45x80` image with
its own text, for 659296 valid tokens. Videos retain sparsity 0.9 and the
image uses dense attention. Both use H2/D128 and the same ring ownership
and validation contracts. They are opt-in
benchmark cases and do not change the default 64k regression suite.
