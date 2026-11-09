# KV-window prototype versus merged Ring VSA

Measured on 2026-11-09, one node with four NVIDIA GB200 GPUs, SLURM
8775247. Baseline: `main@03942631443504fbe177261f3b64f1e5ae741c97`. Measured candidate:
`feat/fused-ring-bsa@7e14f16e189d28919c1ec78a182b55e9ee2e66a4`. Subsequent changes only document the
results and normalize whitespace in copied license files.

This prototype replaces R fine-attention BSA calls with one call over
all-gathered K/V windows. Backward re-gathers K/V and reduce-scatters FP32
dK/dV. It moves whole peer shards, including blocks selected by other query
blocks, and does not overlap these collectives with the BSA kernels.
The local CuTe DSL kernels are unchanged apart from import paths; persistent
scheduling, arrival flags and transport inside the kernel remain future work.
The default is still Ring.

The window's long-sequence F+B latency is 1.26% higher than main with cyclic
ownership and 1.90% higher with contiguous ownership on this allocation.
It approaches Ring performance but does not demonstrate a long-sequence
speedup. The 64k cases have 16–27% lower latency, while the three-block
Ring2 cases have 3–6% higher latency. These results support keeping the
window experimental and retaining Ring as the default.

## Long sequence: 720p44s packed attention

Fixed global work: 961,696 valid tokens in two samples, 16,201 blocks,
BF16, H1/D128, video sparsity 0.9, `dense_query_full=False`, block inputs,
Ring4, seed 937, batched P2P, ordinary-priority NCCL and backend-default
backward buckets. Each run uses 10 warmups and 50 timed CUDA Graph replays.
Run order per ownership: main, branch Ring, window, window, branch Ring, main.

| Ownership | Code / transport | Forward ms | Backward ms | F+B ms | Change vs main |
| --- | --- | ---: | ---: | ---: | ---: |
| cyclic | main / Ring | 17.969 | 49.456 | 67.310 | +0.00% |
| cyclic | branch / Ring | 17.995 | 49.330 | 67.266 | -0.06% |
| cyclic | branch / window | 17.668 | 51.247 | 68.159 | +1.26% |
| contiguous | main / Ring | 18.347 | 48.945 | 66.783 | +0.00% |
| contiguous | branch / Ring | 18.332 | 49.258 | 67.088 | +0.46% |
| contiguous | branch / window | 17.165 | 52.129 | 68.052 | +1.90% |

Lower is better. Each table entry is the median of the two run medians.
The benchmark takes the maximum across ranks for each timing observation
before computing independent forward, backward and full-step medians.
Consequently the F+B column need not equal the sum of the phase columns.
Compilation, validation, graph capture and one-time boundary conversion are
excluded; routing, coarse attention and fine-path communication are included.

| Ownership | Transport | Round 0 F+B ms | Round 1 F+B ms |
| --- | --- | ---: | ---: |
| cyclic | main / Ring | 67.227 | 67.392 |
| cyclic | branch / Ring | 67.162 | 67.370 |
| cyclic | branch / window | 67.853 | 68.465 |
| contiguous | main / Ring | 66.577 | 66.989 |
| contiguous | branch / Ring | 66.834 | 67.342 |
| contiguous | branch / window | 68.293 | 67.811 |

## Small and 64k packs

These compare the rebased branch's unchanged Ring path with the window path,
using H2, Ring4, 3 warmups and 10 timed replays. Unlike the long comparison,
these are single runs and should be read as indicative microbenchmarks.

| Ownership | Case (H2, Ring4) | Ring F+B ms | Window F+B ms | Change |
| --- | --- | ---: | ---: | ---: |
| cyclic | ragged | 0.442 | 0.370 | -16.29% |
| cyclic | dense | 0.217 | 0.143 | -34.20% |
| cyclic | single64k | 2.413 | 2.024 | -16.12% |
| cyclic | thd64k | 2.051 | 1.649 | -19.60% |
| contiguous | ragged | 0.441 | 0.371 | -15.93% |
| contiguous | dense | 0.216 | 0.143 | -33.83% |
| contiguous | single64k | 2.623 | 1.915 | -26.98% |
| contiguous | thd64k | 2.160 | 1.583 | -26.70% |

The partial-block fixture has only three global blocks, so four workers form
two interleaved Ring2 groups. This checks unequal ownership and partial blocks:

| Ownership | Interleaved Ring2 group | Ring F+B ms | Window F+B ms | Change |
| --- | --- | ---: | ---: | ---: |
| cyclic | 0 | 0.247 | 0.261 | +5.63% |
| cyclic | 1 | 0.247 | 0.254 | +3.03% |
| contiguous | 0 | 0.253 | 0.260 | +2.94% |
| contiguous | 1 | 0.241 | 0.256 | +6.15% |

## Validation and reproducibility

- 458 tests passed, two optional FastVideo tests deselected. The local/installed
  BSA parity tests require bitwise-identical output/LSE and gradient similarity
  at least 0.99999. All 41 upstream Python files match their recorded hashes
  after reversing the mechanical import rewrite.
- All four complete README program executions passed: Use, Shared example,
  and the Shared example on Ring4 with cyclic and contiguous ownership.
- All 36 measured case/group runs passed reference comparison, retained-graph
  backward and original/changed-input CUDA Graph replay. Large cases use the
  single-device VSA reference; small cases use native token math. The minimum
  cosine/tensor similarity was 0.9999924394, above the
  unchanged 0.99999 threshold.
- All 25 runner steps exited zero; candidate and baseline Python source hashes
  were unchanged throughout the job. The [machine-readable evidence](window_transport_benchmark_20261109.json)
  includes raw timing samples, per-rank numerical metrics, environment snapshots,
  geometry and source hashes.

Runtime: PyTorch 2.12.0a0+0291f960b6.nv26.04.48445190, CUDA 13.2,
Triton 3.6.0, cuDNN backend 92200, cuDNN frontend 1.28.0, CuTe DSL 4.6.2,
FA4 4.0.0b35. Ring runs use the installed BSA; window runs use `bsa_cute`.
The local copy and installed BSA have the same kernel code. Environment
snapshots reported 1950–2062 MHz SM / 3996 MHz memory and a 1200 W power limit;
SM clocks were not locked. These snapshots bracket each benchmark and are
not a trace of clock frequency during timing.
`CUDA_DEVICE_MAX_CONNECTIONS=32`, `NCCL_GRAPH_REGISTER=0`.

Reproduce from the candidate checkout with the dependencies above:

```bash
mkdir -p /tmp/cudnn-vsa-main-0394263
git archive 03942631443504fbe177261f3b64f1e5ae741c97 | tar -x -C /tmp/cudnn-vsa-main-0394263
export CUDA_DEVICE_MAX_CONNECTIONS=32 NCCL_GRAPH_REGISTER=0
python -m benchmark.compare_window \
  --baseline-root /tmp/cudnn-vsa-main-0394263 --output /tmp/vsa-window-comparison
```

The toolkit launcher is `scripts/dit/ring_vsa/run_fused_ring_rebase.sh`;
raw logs and manifest are retained at
`runtime/cudnn-vsa/fused-ring-rebase-8775247/` in the toolkit workspace.
An earlier job, 8775098, passed the unit suite and examples but used Ring4 for
the three-block partial fixture. The layout correctly rejected it. The runner
was fixed to use Ring2 and this entire matrix was rerun; that failed job's
timings are excluded here.

## Limits

This is a single-node standalone attention comparison, not an end-to-end
training result or a multi-node transport result. The two long runs are not
a statistical confidence interval. For this long shape, the full padded BF16
K/V windows occupy 530,972,672 bytes and the packed FP32 dK/dV partial buffer
occupies 1,061,945,344 bytes per rank. These are buffer sizes, not measured
peak memory or incremental memory versus Ring. Backward currently rebuilds
the windows, so it pays their communication cost again.
