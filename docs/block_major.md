# Compact block-major inputs

This branch is based on main, independently of ring CP. The public builder
accepts `input_order="block"` for valid tokens already arranged by VSA block,
without padding. The default `"raster"` preserves the published API.

Within each sample, the canonical block order remains dense segments first,
then sparse video segments in 4x4x4 tile order. Tokens within a block retain
their original raster order. Samples and segments retain their mathematical
boundaries and KV budgets; routing and attention math are unchanged.

`layout.input_rows[i]` identifies input row i in the original global raster
stream. Q, K, V, gate, position coordinates and dO must share that permutation.
The operator returns output in the supplied order. `sample_slices` continues
to describe the original raster stream; sample lengths are unchanged.
One-time conversion at the model boundary is outside attention timing.

For block-major inputs, slot maps only insert/remove padding. Fully populated
blocks already have the kernel layout, so those gathers are skipped. Ragged
blocks still get explicit zero padding before pooling; no caller padding is
accepted or assumed to contain zeros.

Validation compares output and all four gradients with raster/native
references, checks exact slot values and inverse mappings, and validates
original/changed-input graph replay. Benchmarks compare an immutable main
checkout and this branch in alternating order on the same GPU allocation,
with identical valid inputs, useful FLOPs, software and phase timing.
Report forward, backward and F+B separately. Do not claim a strict
performance guarantee from noisy measurements.

Only after the single-device gate passes, create a child ring branch.
It will preserve compact block-major local inputs and whole-block ownership,
and compare against the existing validated raster-input ring branch.
Neither feature branch is merged into main.

## Initial measurement

SLURM 8096459, one GB200, BF16/D128, 2500 TFLOP/s dense peak.
Immutable main `e5976aa` and this branch use A/B/B/A, 10 warmups and
100 graph replays per case. Each value below averages the two arm medians;
MFU counts valid attention pairs only. All source fingerprints match.
GPU tests: 47 passed. Benchmark output/gradient and original/changed-input
graph checks pass, minimum similarity 0.9999999727.

| Case | Main F/B/total MFU (%) | Block F/B/total MFU (%) |
|---|---|---|
| aligned64k-h2-full0 | 26.764/20.504/22.236 | 29.951/21.380/23.633 |
| aligned64k-h2-full1 | 30.672/21.947/24.250 | 33.809/22.704/25.497 |
| single5s-h2-full0 | 25.000/17.133/19.144 | 25.011/17.137/19.149 |
| single5s-h2-full1 | 26.801/17.924/20.148 | 26.788/17.927/20.149 |
| thd5s-h2-full0 | 25.296/17.453/19.465 | 25.303/17.434/19.450 |
| thd5s-h2-full1 | 26.911/18.195/20.398 | 26.909/18.199/20.399 |
| thd5s-h4-full0 | 25.794/17.847/19.888 | 25.823/17.866/19.909 |
| thd5s-h4-full1 | 27.286/18.455/20.678 | 27.279/18.480/20.697 |

Aligned inputs improve total MFU by 5.15–6.28%. Ragged inputs are nearly
unchanged; the largest negative phase difference is 0.11% relative.
A longer 30-warmup/500-replay A/B/B/A run is required to resolve this
small difference before accepting the parent performance gate.
Raw evidence: `runtime/cudnn-vsa/block-order-8096459/` in the toolkit.

The longer repeat (SLURM 8096610) reproduces small negative differences
for ragged THD, up to 0.27% in total MFU. The initial implementation does
not pass the strict no-regression gate. Preserve both runs as evidence.

For ragged block-major inputs, fuse Q/K/V padding into one gather kernel
and one autograd operation; fuse its three inverse gathers in backward.
All three tensors share the same bijective valid-row maps. Reuse each row
index and launch across Q/K/V, store separate contiguous output views, and
zero padding explicitly. Keep raster inputs on the original path and
aligned block-major inputs on the no-copy path. Validate exact slot values
and first-order gradients, including noncontiguous upstream gradients,
then rerun the complete numerical and A/B/B/A performance gate.

## Validated fused-padding result

SLURM 8096754 uses the same A/B/B/A protocol with 30 warmups and 500
replays. All source hashes match the measured code. CPU regression: 99
passed, 42 GPU/optional skips. GPU regression: 53 passed, including exact
fused-padding values and gradients, native attention references, and graph
replay. The 32 benchmark cases pass output/all-gradient and original/changed
graph checks; minimum similarity is 0.9999999734.

| Case | Main F/B/total MFU (%) | Block F/B/total MFU (%) |
|---|---|---|
| aligned64k-h2-full0 | 26.621/20.449/22.163 | 29.766/21.324/23.552 |
| aligned64k-h2-full1 | 30.448/21.911/24.167 | 33.568/22.668/25.417 |
| single5s-h2-full0 | 24.908/17.107/19.101 | 25.548/17.241/19.338 |
| single5s-h2-full1 | 26.539/17.908/20.085 | 27.177/18.035/20.312 |
| thd5s-h2-full0 | 25.147/17.433/19.419 | 25.826/17.583/19.677 |
| thd5s-h2-full1 | 26.643/18.182/20.334 | 27.288/18.312/20.566 |
| thd5s-h4-full0 | 25.568/17.809/19.806 | 26.058/17.808/19.912 |
| thd5s-h4-full1 | 26.765/18.259/20.430 | 27.456/18.396/20.678 |

Every H2 forward/backward phase improves: ragged total MFU gains 1.13–1.33%,
and aligned 64k gains 5.17–6.27% relative. The additional THD H4/default
backward result is effectively unchanged: 13.175176 vs 13.176120 ms
(-0.0072% relative MFU), against 0.144% control drift and 0.203% candidate
repeat drift. Preserve that point estimate rather than claiming every
observed phase improves. Other H4 phases improve. Accept the measured H2
performance gate; these results are not a guarantee for arbitrary hardware
or workloads.

Raw results and complete arm medians: `runtime/cudnn-vsa/block-order-8096754/`.
Reproduce the source-isolated driver with:

```bash
python benchmark/compare_order.py --source-root /path/to/main --order raster \
  --warmup 30 --repeats 500 --output main-a.json
python benchmark/compare_order.py --source-root /path/to/this/branch --order block \
  --warmup 30 --repeats 500 --output block-a.json
# Repeat the second command as block-b, then the first as main-b.
```

The toolkit runner is `scripts/dit/block_major_vsa/run_single.sh`, with
`ORDER_WARMUP=30 ORDER_REPEATS=500`; `analyze.py` validates all four
artifacts, counts unchanged useful FLOPs and reports every phase.
