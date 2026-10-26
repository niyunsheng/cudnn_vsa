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
