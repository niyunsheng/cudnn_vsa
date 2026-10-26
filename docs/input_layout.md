# Input layouts and shared VSA core

The public entry is `vsa.vsa_attention(q, k, v, gate, layout, *, scale=None)`.
Q/K/V/gate are contiguous CUDA BF16 tensors with shape `[T, H, 128]`.
Output and gradients use the supplied token order. The default layout keeps
the original sample-major FHW contract.

## Module responsibilities

- `vsa.py` validates tensors, converts them to slots, calls the attention
  core, restores input rows and applies the gated coarse contribution.
- `vsa_core.py` owns per-region top-k routing, cuDNN BSA forward/backward,
  FA4 coarse attention and their single-device execution order.
- `vsa_layout.py` plans sample-local blocks and prepares routing templates,
  row maps and coarse boundaries before attention runs.
- `vsa_kernels.py` implements slot gathers, fused Q/K/V padding, FP32
  pooling and gated coarse merge with their backward operations.

The core does not import the public entry. Both input orders use the same
attention math, per-region budgets and fine-before-coarse schedule. Pooling
and gated merge retain the original single-device implementation.

## Token order

`build_vsa_layout(..., input_order="raster")` keeps samples and their segments
in declared order. Video rows use FHW order. `slot_row` converts that stream
to block slots and inserts padding on each call; `row_slot` restores it.

`build_vsa_layout(..., input_order="block")` describes compact valid rows in
canonical block order. Samples remain in declared order. Within a sample,
dense segments come first, each split into chunks of at most 64 tokens;
numeric-sparsity video segments follow as 4x4x4 FHW tiles. Segment order is
preserved within each category. Blocks never cross segment or sample bounds.
Boundary blocks contain only their valid tokens in the external input.

`input_rows[i]` gives the original raster row for supplied row `i`. It is
an identity map for raster input and a permutation for block input. The caller
performs the one-time permutation before the whole VSA stack and keeps
activations, gate, positions and upstream gradients in that order. This
library does not perform dataset preprocessing or model-boundary conversion.

## Internal slots and gradients

Every physical slot block has 64 rows. `slot_row` uses -1 for padding;
`row_slot` maps valid input rows back to those slots. In block mode the
nonnegative entries of `slot_row` are sequential: slot conversion only pads.
One fused kernel gathers Q/K/V into a shared allocation; its backward gathers
dQ/dK/dV together through the inverse map. Padding is zero and contributes
neither to block means nor to valid-token attention.

When all blocks are full, `slots_are_input` allows block inputs and fine
outputs to bypass conversion entirely. Full raster blocks still require the
FHW permutation. Coarse rows and gating are mapped to each input order.

## Validation

Run `python -m pytest -q -m 'not fastvideo'` after installing the runtime and
pytest dependencies. `tests/test_block_major.py` checks exact maps and gather
adjoints, independent native attention for both dense-query policies, both
CUDA input orders, and original/changed-input CUDA Graph replay. Full and
partial blocks, multiple samples, multiple video segments and dense samples
are covered. Existing routing, attention and kernel tests exercise the
extracted core. These correctness checks do not establish MFU parity.

### PR A validation, 2026-10-25

Runtime revision `9cf46a2` passes 99 CPU tests (40 CUDA skips, two optional
FastVideo tests deselected) and 139 GPU tests (two optional FastVideo tests
deselected). Forty additional CPU routing comparisons match original main
exactly. SLURM 8296435 completes with exit 0 on GB200, including both input
orders, gradients and original/changed-input CUDA Graph replay.

Two separate main/candidate/candidate/main comparisons measure candidate
raster and candidate block against original main `e5976aa` raster. All arms
run serially on the same GPU, with 30 warmups and 500 timed replays per case.
Cases cover single/packed 720p5s, H2/H4 and aligned 64k under both dense-query
policies. Software is PyTorch 2.12.0a0/NV26.04, CUDA 13.2, cuDNN frontend
1.28.0, CuTe 4.6.2 and FA4 4.0.0b35. Source hashes match the tested revisions.

Relative MFU changes across the eight cases are:

| Candidate input | Forward | Backward | Total |
| --- | ---: | ---: | ---: |
| Raster | -0.166% to +0.520% | -0.450% to +0.113% | -0.205% to +0.104% |
| Block | +1.690% to +11.735% | +0.163% to +4.312% | +0.527% to +6.313% |

Raster packed 720p5s/H2 with `dense_query_full=True` loses 0.205% total MFU
and 0.450% backward MFU. Its total-time control/repeat drift is 0.029% / 0.083%,
so a universal no-regression claim is not supported. Block timings exclude
the caller's one-time model-boundary permutation; these are standalone
operator measurements, not end-to-end training results.
