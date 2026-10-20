# Ring CP validation and tuning

Measured on 2026-10-20, branch `codex/ring-cp-thd`.

The latest complete run is SLURM 8093259. See the
[phase comparison](#phase-comparison-with-the-720p5s-reference) and
[final validation](#final-validation) below.

## Workload and measurement boundary

Eight NVIDIA GB200 GPUs on two four-GPU nodes, BF16, two heads per ring peer,
head dimension 128, 65,536 global valid tokens. All peers own the same two
heads. Each has 8,192 valid tokens and 128 physical 64-token blocks.

| Case | Independent samples |
|---|---|
| `single64k` | Video 16×60×64 + 4096 text tokens |
| `thd64k` | Video 12×48×64 + 2048 text; video 8×40×64 + 2048 text; dense image 1×32×64 + 2048 text |

Sparse videos use sparsity 0.9. Default dense queries use the same per-region
selection budgets as video queries. The third packed sample is fine-only.
Packing changes the number of valid attention pairs: the two cases have
different useful FLOPs, despite the same total tokens.

Timing includes local packing, pooling, global block-mean communication,
routing, all ring steps, fine/coarse merge, unpacking, gate, and all gradients.
It excludes external Ulysses A2A, projections, data preparation and optimizer.
This is attention-core MFU, not whole-model training MFU.

CUDA Graph timing uses 10 warmups and 100 replays. For every replay and phase,
take the maximum across all eight ranks, then the median across replays.
Forward, backward and total medians are independent and need not add exactly.
Numerical checks and profiling are outside the timed region.

Useful forward FLOPs count valid fine/coarse attention matrix products;
backward is twice forward. Padding, routing and repeated computation do not
increase the numerator. MFU divides global useful FLOPs by distributed
latency and eight times the configured 2500 TFLOP/s dense BF16 GPU peak.

Software: PyTorch 2.12.0a0 nv26.04, CUDA 13.2, NCCL 2.29.7, cuDNN 9.22,
cuDNN frontend 1.28.0, Triton 3.6.0, CuTe DSL 4.6.2, FA4 4.0.0b35.
Driver 580.126.20; observed SM/memory clocks 2062/3996 MHz.
The runs set `NCCL_MNNVL_ENABLE=1` and `NCCL_CUMEM_ENABLE=1`.
The library itself does not set NCCL environment variables.

## Transport and backward bucket sweep

SLURM job 8089452, all configurations in one allocation:

| P2P mode | Backward query-block bucket | Single F+B ms | Packed F+B ms |
|---|---:|---:|---:|
| Individual, lazy group | Backend default | 2.287 | 1.850 |
| Batched, lazy group | Backend default | 3.842 | 3.732 |
| Individual, lazy group | 16 | 2.348 | 2.062 |
| Individual, lazy group | 32 | 2.125 | 1.952 |
| Individual, lazy group | 64 | **2.059** | 1.852 |
| Individual, lazy group | 128 | 2.270 | 1.871 |

The final automatic defaults were then revalidated in a separate allocation,
SLURM job 8089616:

| Case | Dense-query-full | Forward ms | Backward ms | F+B ms | F+B TFLOP/s/GPU | F+B MFU |
|---|---|---:|---:|---:|---:|---:|
| Single, bucket 64 | False | 0.637 | 1.431 | 2.043 | 126.31 | 5.05% |
| Packed, backend default | False | 0.586 | 1.263 | 1.831 | 69.97 | 2.80% |
| Single, bucket 64 | True | 0.646 | 1.522 | 2.153 | 160.26 | 6.41% |
| Packed, backend default | True | 0.636 | 1.273 | 1.893 | 89.13 | 3.57% |

Dense-query-full performs more useful attention work and therefore has a
different MFU numerator. Its higher MFU is not a speedup of the default math.

The first validated ring implementation (job 8089061) took 3.890/3.736 ms
for single/packed inputs. The selected configurations are approximately
1.90×/2.04× faster. This comparison includes transport and kernel changes;
it does not attribute the entire improvement to an individual optimization.

PyTorch device-bound eager groups reuse the parent NCCL communicator for
individual P2P operations. On the tested stack this serialized communication:
job 8089274 measured 5.181/5.133 ms with individual P2P versus 3.835/3.754 ms
with batched P2P. The implementation therefore selects batched P2P for groups
with `bound_device_id`, and individual P2P otherwise. This is a measured
choice for this stack/topology; `ring_cp_batch_p2p` permits an override.
The relevant implementation is PyTorch's
[ProcessGroupNCCL point-to-point path](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp).

The automatic backward bucket choice is deliberately narrow: SM100,
one sample, two heads and 128 physical local blocks use bucket 64.
Other shapes retain cuDNN's default. It does not extrapolate this sweep
to unrelated head counts, sequence lengths or accelerators.

## Correctness and remaining costs

Every sweep configuration passes output, dQ, dK, dV and dgate checks on all
ranks, plus CUDA Graph replay checks with original and changed inputs.
Both cosine and tensor similarity must be at least 0.99999.
Large cases compare against the original single-device VSA. Small ragged,
dense-only and empty-peer cases compare against independent token math.
Source fingerprints are unchanged throughout each measured subprocess.
The final defaults also pass all these checks, with minimum similarity
0.9999916846 across both policies, all ranks, all output/gradient tensors,
and both graph replay variants. Their fingerprints match the checked-in
Python sources.

The local CPU suite has 101 passing tests. Six GPU unit tests validate
partial-result merging and peer-list compaction/alignment, including odd
block counts and empty lists. Distributed regression also covers both
dense-query policies and ranks with no valid local tokens.
Final subgroup checks pass for four concurrent, non-contiguous two-rank
groups `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]` across nodes. The separate
device-bound world-group check passes and selects batched P2P automatically.

The tuned single-sample rank-0 trace has about 0.240 ms of fine-forward BSA
kernels and 0.739 ms of fine-backward BSA kernels. The timeline analysis
finds 0.236 ms of exposed communication and 0.820 ms overlapped with compute;
summed P2P kernel durations overlap each other and must not be added to
wall time. Repeated per-step BSA metadata construction, output merging and
gradient accumulation remain significant. Two heads and roughly 20 selected
KV blocks per peer give small individual BSA problems.

These are initial measured optimizations, not a claim of global optimality.
Further work should target the repeated backward metadata/finalization and
the ring transfer schedule, while preserving global top-k, sample isolation
and the same numerical threshold.

## 720p, 30-second video, ring 8

The opt-in long cases match the toolkit's `mock_dit_720p30s` geometry:
24 fps, 721 frames, a `181x45x80` video token grid after VAE compression and
patchification, and 2048 text tokens. H2/D128/BF16, sparsity 0.9, ring 8,
hardware/software and measurement boundaries are as above. Each ring peer
owns the same two heads. The dense image is an independent packed sample.

| Case | Valid tokens | Global layout slots | Physical slots per rank |
|---|---:|---:|---:|
| `single720p30s`: video + text | 653,648 | 708,608 | 88,576 |
| `thd720p30s`: video + text; dense image + text | 659,296 | 714,304 | 89,344 |

THD's uniform ring buffers add seven dummy blocks to the global layout.
Valid local token counts are 82,792 on ranks 0–3 and 80,620 on ranks 4–7
for the single sample. Packed counts are 83,512 on rank 0, 83,496 on
ranks 1–3, and 81,324 on ranks 4–7.

The initial run, SLURM 8091895, measured:

| Case | Dense-query-full | Forward ms / MFU | Backward ms / MFU | F+B ms / MFU |
|---|---|---:|---:|---:|
| Single | False | 9.688 / 20.49% | 28.857 / 13.76% | 38.291 / 15.55% |
| Packed | False | 9.732 / 20.41% | 29.087 / 13.66% | 38.539 / 15.46% |
| Single | True | 15.841 / 12.92% | 29.394 / 13.93% | 44.983 / 13.65% |
| Packed | True | 15.978 / 12.82% | 29.629 / 13.83% | 45.327 / 13.56% |

All output/gradient and original/changed-input graph checks pass across all
eight ranks; minimum cosine/tensor similarity is 0.9999931048. Source
fingerprints remain unchanged. Both long cases use the backend's default
backward bucket.

The packed rank-0 traces locate the dense-query-full forward penalty in
`_partition_kernel`: 0.075 ms with sparse dense-query budgets versus 6.131 ms
with full dense-query rows. Fine BSA forward changes only from 6.415 to
6.574 ms. The original partition kernel scans the shared maximum capacity
in one vector for every row and peer, even though most rows have only
1136 valid entries. Full text rows raise that capacity to 11,072.

The bounded scan implementation limits each scan to 2048 entries, visits
only valid chunks, and carries the per-peer output count between chunks.
It preserves sorted IDs, exact counts, zero tails, and 16-byte peer-base
alignment. CUDA regression covers zero/full/sparse counts, 2048-entry
boundaries, odd peer sizes, and output tails spanning multiple chunks.

Final validation, SLURM 8092110, uses the same software and two other GB200
nodes. Useful FLOP counts match the first run exactly:

| Case | Dense-query-full | Forward ms / MFU | Backward ms / MFU | F+B ms / MFU |
|---|---|---:|---:|---:|
| Single | False | 9.690 / 20.48% | 28.914 / 13.73% | 38.343 / 15.53% |
| Packed | False | 9.729 / 20.42% | 28.962 / 13.72% | 38.427 / 15.51% |
| Single | True | 9.851 / 20.78% | 28.914 / 14.16% | 38.515 / 15.95% |
| Packed | True | 9.940 / 20.61% | 29.534 / 13.88% | 39.197 / 15.68% |

Dense-query-full F+B improves by 1.168x / 1.156x for single/packed samples;
default-policy time changes by less than 0.3%. Packed dense-query-full
partition time falls to 0.114 ms, explaining most of the improvement.
These comparisons use separate allocations rather than paired alternating
measurements on the same nodes.

Both policies pass output/dQ/dK/dV/dgate and original/changed-input graph
checks on all eight ranks for both long cases. Minimum similarity is
0.9999934629. The same allocation revalidates ragged, empty-peer, dense and
both 64k cases under both policies; minimum similarity over that whole suite
is 0.9999916846. Source fingerprints match the final Python sources.
Eight CUDA unit tests pass on each node, and the CPU suite has 101 passes.
The final 64k single/packed F+B times are 1.999/1.812 ms (default) and
2.164/1.844 ms (dense-query-full), consistent with the earlier measurements.

The final packed dense-query-full rank-0 trace has 6.585 ms fine BSA forward,
23.135 ms fine BSA backward, 3.038 ms exposed communication and 18.637 ms
communication overlapped with compute. The default trace has 2.946 ms
exposed communication. Fine backward remains the largest optimization
target; summed NCCL durations are not the exposed communication cost.

For context, the historical 113,648-token H2 single-device run with
dense-query-full measured F+B MFU 20.16%. The 653,648-token ring-8 result
reaches 15.95%, about 79% of that MFU. Different sequence lengths and
allocations mean this is not a strong-scaling efficiency measurement.
The historical artifact is
`runtime/cudnn-vsa/thd-graph-20261015/benchmark.json`, case `thd1-h2`.

Initial raw JSON, all-rank traces and rank-0 HTML reports are retained in
`runtime/cudnn-vsa/ring-8091895/` in the toolkit. Final results and reports
are under `runtime/cudnn-vsa/ring-8092110/`; `perf.json` and
`perf-dense-query-full.json` contain complete validation, timing samples,
environment and source hashes.

## Phase comparison with the 720p5s reference

Use the same dense-query-full policy for this comparison. The historical
720p5s case has 113,648 tokens and runs on one GB200; the 720p30s case has
653,648 tokens across eight GB200s. These are attention-core measurements,
not a strong-scaling experiment or end-to-end model MFU.

| Phase | 720p5s, no ring: ms / MFU | 720p30s, high-priority ring8: ms / MFU (8093259) | 30s time needed for the 5s MFU |
|---|---:|---:|---:|
| Forward | 2.486 / 25.98% | 9.114 / 22.46% | 7.881 ms |
| Backward | 7.120 / 18.14% | 26.907 / 15.22% | 22.573 ms |
| F+B | 9.605 / 20.16% | 35.817 / 17.15% | 30.455 ms |

The useful-FLOP counts and the minimum 0.99999 cosine/tensor similarity
requirement remain unchanged in the following sweeps.

### Backward bucket sweep

Job 8092233 tests the long dense-query-full cases with backend forward
scheduling. All settings pass the all-rank reference and both graph checks.

| Bucket size | Single backward ms | Packed backward ms |
|---|---:|---:|
| Backend (1088) | 29.434 | 29.649 |
| 512 | 30.514 | 30.595 |
| 256 | 33.116 | 33.038 |
| 128 | 37.897 | 38.082 |
| 64 | 48.511 | 48.876 |
| 1024 | 29.555 | 29.666 |
| Backend, repeated control | 29.505 | 29.608 |

The single-sample trace's main BSA backward kernels increase from 22.926 ms
with the backend default to 41.695 ms with bucket 64. CSR metadata remains
about 1.6 ms. This is a kernel execution regression, not an increase in
metadata cost. The successful small 64k bucket policy does not generalize
to this geometry.

### Forward scheduler and split-KV sweep

Job 8092399 also tests a full-local-Q backward bucket of 2048. Forward
alternatives are compared in the same allocation with a repeated control.
All settings pass numerical and graph checks.

| Forward policy | Single F / B ms | Packed F / B ms |
|---|---:|---:|
| CLC | 9.867 / 30.700 | 9.953 / 30.721 |
| Static | 10.797 / 30.775 | 10.901 / 30.906 |
| Static, split 2 | 11.796 / 30.733 | 11.895 / 30.837 |
| Static, split 4 | 13.533 / 30.686 | 13.721 / 30.773 |
| CLC, repeated control | 9.869 / 30.677 | 9.951 / 30.731 |

Neither the larger backward bucket nor the forward alternatives improves
the original defaults. These experiments retain their exact Python source
snapshots and source-fingerprinted results under
`runtime/cudnn-vsa/ring-8092399/`. Job 8092233's Python hashes match commit
`dbe5d3d`; its raw results and profile reports are under
`runtime/cudnn-vsa/ring-8092233/`.

### Coalesced KV windows

Job 8092592 tests one, two or four shards per window, preserving the exact
global routing and useful FLOPs. This version constructs a fresh window at
backward entry and traverses it in the same direction as forward.

| Shards per window | Single F / B / F+B ms | Packed F / B / F+B ms |
|---|---:|---:|
| 1 | 9.854 / 29.107 / 38.711 | 9.933 / 29.607 / 39.282 |
| 2 | 9.431 / 30.382 / 39.592 | 9.497 / 30.926 / 40.206 |
| 4 | 9.978 / 30.475 / 40.145 | 10.068 / 30.380 / 40.134 |

Two-shard windows improve forward, but regress backward and total latency.
The single-sample fine BSA kernels change from 6.491 to 6.228 ms forward
and 23.117 to 23.575 ms backward; forward partial merging falls from
0.618 to 0.302 ms. All seven cases pass all-rank output/gradient and
original/changed-input graph checks for every window size, including
ragged and empty partitions. The minimum similarity is 0.9999916330.
Twenty CUDA metadata/merge tests pass on each node. An earlier attempt,
job 8092485, found misaligned window block-size tables for odd block counts;
the fixed layout pads each window's storage stride to 16 bytes, and the
CUDA regression verifies the sizes as well as routing indices/counts.

The complete results, source snapshot and profile reports are under
`runtime/cudnn-vsa/ring-8092592/`. These measurements do not justify
changing the default window size from one.

### Reverse backward and high-priority NCCL groups

Job 8092693 reuses the final forward KV window and traverses windows in
reverse during backward. This removes backward window construction and
the final gradient-return hop; saved windows remain immutable for retained
autograd graphs. A separate comparison changes the caller-owned NCCL
group's `is_high_priority_stream` option.

| Window | NCCL priority | Single F / B / F+B ms | Packed F / B / F+B ms |
|---|---|---:|---:|
| 1 | Normal | 9.820 / 29.205 / 38.798 | 9.910 / 29.065 / 38.721 |
| 2 | Normal | 9.373 / 30.209 / 39.355 | 9.454 / 30.289 / 39.521 |
| 4 | Normal | 9.990 / 29.269 / 38.938 | 10.066 / 29.357 / 39.078 |
| 1 | High | 9.047 / 26.580 / 35.450 | 9.111 / 26.676 / 35.614 |
| 2 | High | 9.127 / 26.908 / 35.429 | 9.158 / 27.042 / 35.603 |

The one-window high-priority single-sample run reaches forward/backward/
total MFU 22.63% / 15.40% / 17.32%. Packed reaches 22.49% / 15.36% /
17.26%. The rank-0 timeline shows exposed communication falling from
2.958 ms to 0.395 ms. Fine BSA forward changes from 6.460 to 6.885 ms
and backward from 23.100 to 21.894 ms: communication priority also changes
the kernels' scheduling and resource contention.

Two windows do not improve total time meaningfully under high priority
(less than 0.1% difference), so one remains the default. The ordinary
priority reverse-backward measurement alone does not establish a
long-sequence speedup over job 8092592, which used different nodes.

All five settings pass every reference, repeated retained-graph backward,
and both graph replay checks on all seven cases and all eight ranks.
Minimum cosine/tensor similarity is 0.9999916330. Each setting records
224 passing rank/check records, covering output, dQ, dK, dV and dgate.
Source fingerprints are unchanged and match the retained Python snapshot.
Results and HTML/JSON profile reports are under
`runtime/cudnn-vsa/ring-8092693/`.

### P2P channel comparison

Job 8093099 retains high-priority groups, reverse backward and one shard
per window. It sets both `NCCL_MIN_P2P_NCHANNELS` and
`NCCL_MAX_P2P_NCHANNELS` to each requested value. Actual SendRecv launch
grids are checked in the traces; requested channel counts are not assumed
to equal the number of CTAs launched for each transfer.

| Requested channels | Observed SendRecv CTAs | Single F / B / F+B ms | Packed F / B / F+B ms |
|---|---:|---:|---:|
| Default | 16 | 9.189 / 27.172 / 36.143 | 9.206 / 26.985 / 35.976 |
| 4 | 4 | 9.054 / 26.956 / 35.869 | 9.220 / 27.165 / 36.202 |
| 8 | 8 | 9.093 / 27.123 / 36.013 | 9.183 / 27.161 / 36.078 |
| 32 | 16 | 9.156 / 26.924 / 35.834 | 9.254 / 27.105 / 36.080 |
| 64 | 32 | 9.185 / 26.877 / 35.894 | 9.251 / 26.946 / 36.031 |
| Default, repeated control | 16 | 9.122 / 26.924 / 35.831 | 9.224 / 27.021 / 35.997 |

No override consistently improves on the repeated default for both cases.
Keep NCCL's channel defaults. All six settings pass reference,
retained-backward and both graph checks, with minimum similarity
0.9999931048; source hashes match job 8092693. Raw results, traces and NCCL
initialization logs are under `runtime/cudnn-vsa/ring-8093099/`.
Across the two allocations the high-priority default totals about
17.0–17.3% MFU, still below the historical 20.16% target.

## Final validation

SLURM 8093259 uses the same Python source, a caller-owned high-priority
NCCL group, one shard per window, backend forward scheduling and the
existing automatic backward bucket policy. NCCL channel counts are left
at their defaults. Both long-case useful FLOP dictionaries exactly match
job 8092110; the MFU numerator and validation threshold are unchanged.

| Case | Dense-query-full | Forward ms / MFU | Backward ms / MFU | F+B ms / MFU |
|---|---|---:|---:|---:|
| Single | False | 8.930 / 22.23% | 26.416 / 15.03% | 35.137 / 16.95% |
| Packed | False | 8.918 / 22.28% | 26.372 / 15.07% | 35.124 / 16.97% |
| Single | True | 9.114 / 22.46% | 26.907 / 15.22% | 35.817 / 17.15% |
| Packed | True | 9.201 / 22.27% | 27.053 / 15.15% | 36.030 / 17.06% |

All 25 result files pass: 1,216 rank/check records, each checking output
and four gradients against references, repeated retained backward, or
original/changed-input graph replay. The minimum cosine/tensor similarity
is 0.9999916846. Coverage includes both query policies, all seven main
cases, singleton groups, non-contiguous two- and four-rank groups,
coalesced windows spanning a whole subgroup or multiple ring steps, and
device-bound eager groups with window sizes 1/2/4. Source fingerprints
remain unchanged and match the committed operator and benchmark code.
CPU tests: 101 passed; CUDA metadata/merge tests: 20 passed per node.

The 64k single/packed F+B times are 1.923/1.729 ms with default budgets,
and 2.090/1.757 ms with dense-query-full. The final long single/packed
rank-0 traces show BSA backward at 22.016/22.058 ms, backward CSR metadata
at 1.654/1.617 ms, and exposed communication at 0.594/0.938 ms.
The remaining gap includes BSA backward computation and repeated
metadata/preprocessing/finalization, beyond communication alone.

Forward and backward reach approximately 86% and 84% of the historical
5s phase MFUs, respectively. Total MFU remains about 15% below the 20.16%
target; this validation does not claim the target is met.
Raw results, all-rank traces, and `single720p30s-perf.html` /
`thd720p30s-perf.html` reports are retained under
`runtime/cudnn-vsa/ring-8093259/`.

## Reproduce

Use a caller-created lazy NCCL ring group, or run:

```bash
torchrun --standalone --nproc-per-node=8 --module benchmark.ring \
  --warmup 10 --repeats 100 --profile
torchrun --standalone --nproc-per-node=8 --module benchmark.ring \
  --dense-query-full --warmup 10 --repeats 100
torchrun --standalone --nproc-per-node=8 --module benchmark.ring \
  --ring-size 2 --cases ragged empty_peers dense
torchrun --standalone --nproc-per-node=8 --module benchmark.ring \
  --cases single720p30s thd720p30s --warmup 10 --repeats 100 --profile
torchrun --standalone --nproc-per-node=8 --module benchmark.ring \
  --cases single720p30s thd720p30s --dense-query-full --warmup 10 --repeats 100
torchrun --standalone --nproc-per-node=8 --module benchmark.ring \
  --cases single720p30s thd720p30s --dense-query-full --high-priority-group \
  --warmup 10 --repeats 100 --profile
```

For multi-node runs use torchrun's `--nnodes`, `--node-rank`,
`--master-addr` and `--master-port`. The toolkit wrapper is
`scripts/dit/ring_vsa/run.sh`; `--tune` runs the bucket sweep and
`--verify-groups` runs subgroup and final-default regressions.

Raw tuning artifacts are retained under the toolkit's
`runtime/cudnn-vsa/ring-8089452/`: `perf.json`, `perf-batch-p2p.json`,
`perf-bucket{16,32,64,128}.json`, rank-0 traces, and
`single64k-perf.html` / `thd64k-perf.html`.
Final regression artifacts are under `runtime/cudnn-vsa/ring-8089616/`:
`perf.json`, `perf-dense-query-full.json`, `eager-group.json`,
`subgroups-group{0,1,2,3}.json`, rank-0 traces and HTML reports.

## Compact block-major child branch

The child of `0a14382` accepts unpadded owned-block inputs and reuses the
parent fused-padding/identity paths. SLURM 8096942 compares immutable
raster-ring `1f8ed95` and this implementation in A/B/B/A order on the same
eight GB200 GPUs, using high-priority lazy NCCL groups, H2/D128/BF16,
10 warmups and 100 graph replays. Timing reduces each observation by the
maximum across ranks before taking independent phase medians.

127 CPU tests pass. GPU regression: 73 passes on each node. All 25 result
files and 2,272 rank/check records pass, minimum similarity 0.9999916846.
This covers both dense policies, native and single-device references,
retained backward, original/changed-input graphs, empty peers, singleton
and interleaved subgroups, eager groups and coalesced windows. Source
fingerprints match all runtime/benchmark files and useful FLOPs are unchanged.

| Case | Raster-ring F/B/total MFU (%) | Block-ring F/B/total MFU (%) |
|---|---|---|
| single64k-full0 | 5.833/5.059/5.322 | 6.066/5.156/5.454 |
| thd64k-full0 | 2.993/2.896/2.946 | 3.127/2.948/3.027 |
| single720p30s-full0 | 22.323/15.080/16.997 | 22.373/15.067/17.012 |
| thd720p30s-full0 | 22.182/15.034/16.952 | 22.272/15.013/16.961 |
| single64k-full1 | 7.411/6.277/6.633 | 7.697/6.387/6.794 |
| thd64k-full1 | 3.818/3.779/3.812 | 3.962/3.869/3.921 |
| single720p30s-full1 | 22.436/15.201/17.150 | 22.471/15.183/17.154 |
| thd720p30s-full1 | 22.234/15.186/17.101 | 22.225/15.161/17.105 |

64k total MFU improves 2.42–2.86%. Long-input totals are effectively
unchanged (0.02–0.09% higher), with backward point estimates 0.08–0.17%
lower. Do not claim that all long-workload phases improve. The single
720p30s dense-query-full result is 22.471% forward, 15.183% backward and
17.154% total MFU. The validated parent 720p5s/no-ring reference is
27.177%, 18.035% and 20.312%, respectively. The no-ring MFU target is
**not met** by changing input order and padding alone.

Raw evidence: `runtime/cudnn-vsa/block-ring-8096942/` in the toolkit.
Reproduction: `scripts/dit/block_major_vsa/run_ring.sh` and
`analyze_ring.py`. Neither feature branch is merged. Further scheduling
experiments must preserve these checks and compare repeated controls.
