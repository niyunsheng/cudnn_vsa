# Ring context parallelism

This branch is based on `codex/block-major-input` (`0a14382`). It ports the
validated ring implementation from `1f8ed95`, retaining its communication,
attention, reduction and scheduling logic. Compact block-order metadata
and the parent's padding/no-copy paths are the new integration points.
The default raster API remains compatible with the measured ring baseline.

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
Peer metadata bases are 16-byte aligned, even
when a rank has an odd number of blocks. Fine attention keeps Q local and circulates BF16 KV, prefetching the
next shard while the current BSA call runs. Merge partial O and LSE in FP32.
Each block is visited once; the forward saves local Q, the final received
KV window, global fine O/LSE, and the selected block lists.

Fine backward starts from the saved final window and traverses the ring in
reverse with the final fine O/LSE and local dO. Accumulate dQ in FP32.
Pipeline the next KV transfer with the previous step's FP32 dK/dV
accumulation transfer. The last window belongs to this rank, so no final
gradient-return hop or backward window-building prefix is required.
Ping-pong receives use a clone of the saved window to preserve repeated
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

Forward scheduling can be compared separately using
`ring_cp_fwd_use_clc`: `"auto"` currently delegates to cuDNN,
`None` delegates to cuDNN, and `True`/`False` selects CLC/static scheduling.
`ring_cp_fwd_kv_splits` is an integer from 1 to 256 (default 1), passed to
the backend's split-KV forward path. Both controls are part of the cached
runtime key and leave the non-ring operator unchanged. All alternatives
must pass output/gradient and changed-input graph checks before timing.

`ring_cp_chunk_size` optionally coalesces 2 or 4 adjacent KV shards per
BSA call; the default is 1 and the chunk size must divide the ring size.
For chunk size C, rank r first receives C-1 neighboring shards through
ordinary one-hop ring transfers. Its window contains owners
`[r, r-1, ..., r-C+1]`. The windows then circulate by C ranks at each step,
covering every KV owner in R/C BSA calls. Forward still transfers R-1
shards per rank, while buffer capacity increases by C. No new groups are
created, and original sample masks, global routing and public THD rows
are retained. The saved backward KV window also holds C local shards;
the default C=1 holds the same KV capacity as the original local K and V.

Routing assigns a global KV block to window `(r-owner mod R)//C` and slot
`(r-owner mod C)*local_blocks + global_block//R`. Stable compaction keeps
the original selected-key order; it need not sort window-local IDs.
Window block-size tables, indices and counts each use padded storage strides
so every window base is 16-byte aligned, including odd local block counts.
Each backward window first accumulates the Q ranks in its strided cycle,
then adjacent cycles send their remaining components back to the original
KV owners. Those FP32 components are summed before the final BF16 cast.
This final cross-cycle reduction is required for correct dK/dV; simply
changing ring hop distance would omit gradients from other Q ranks.

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
See [the measured validation and tuning results](ring_cp_benchmark.md).

The long-sequence cases `single720p30s` and `thd720p30s` use a 720p,
30-second video at 24 fps: 721 frames, temporal compression 4, and spatial
compression 16 after patchification, giving a `181x45x80` token grid.
Each sample includes 2048 text tokens. The single case has 653648 valid
tokens; the packed case adds an independent dense `1x45x80` image with
its own text, for 659296 valid tokens. Videos retain sparsity 0.9 and the
image uses dense attention. Both use H2/D128 and the same ring ownership
and validation contracts. They are opt-in
benchmark cases and do not change the default 64k regression suite.
