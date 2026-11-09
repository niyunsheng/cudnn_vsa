# bsa_cute provenance

`bsa_cute/` is a vendored copy of the CuTe DSL block-sparse attention (BSA)
kernels that ship inside the `nvidia-cudnn-frontend` Python package. It exists
so that the Ring VSA kernels can be modified locally (fused Ring transport
experiments) while the installed package remains available as the control.

| Field | Value |
| --- | --- |
| Upstream repository | https://github.com/NVIDIA/cudnn-frontend |
| Tag / commit | `v1.28.0` / `91159779637b672a3f768738f5b30eee33f2180d` |
| Upstream path | `python/cudnn/block_sparse_attention/` |
| Matching wheel | `nvidia_cudnn_frontend-1.28.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`, sha256 `060b0c02…da075`; its extracted package is byte-identical to the tag |
| Per-file upstream hashes | `UPSTREAM_MANIFEST.json` |

## Mechanical changes at import

- The package is imported as `bsa_cute` instead of `cudnn.block_sparse_attention`
  (83 absolute imports in 21 files rewritten; relative imports unchanged).
- `api.py` imported `TupleDict` from `cudnn.api_base`; that class is copied
  verbatim into `_tupledict.py` so the vendored package never imports `cudnn`.
- No kernel, scheduler or host-interface logic is changed in this draft.
  The window transport lives in `vsa_core.py`; device-side fusion remains
  future work. Diff against a fresh sparse checkout of the tag to audit the
  mechanical import changes:

```text
git clone --depth 1 --branch v1.28.0 --filter=blob:none --sparse https://github.com/NVIDIA/cudnn-frontend.git fe
git -C fe sparse-checkout set python/cudnn/block_sparse_attention
diff -ru fe/python/cudnn/block_sparse_attention bsa_cute -x __pycache__ -x licenses -x '*.md' -x '*.json'
```

## Runtime requirements

Identical to the upstream package at the same version: `nvidia-cutlass-dsl`
(4.5.0 minimum, 4.6.2 used by the toolkit runners), `cuda-python` bindings,
`torch`, and an SM100 GPU for the blk64 BF16 paths used by VSA. The installed
`nvidia-cudnn-frontend` is still required by the rest of the runtime for
`cudnn.backend_version()` reporting and as the default BSA implementation.

## Licenses

cudnn-frontend is distributed primarily under Apache-2.0 with some files under
MIT, and the BSA kernels derive from FlashAttention (BSD-3-Clause). Each source
file keeps its SPDX tag and copyright header. The upstream `LICENSE.txt`,
`LICENSE-MIT.txt`, `LICENSING.md`, `THIRD_PARTY_LICENSES.txt` and `NOTICE` are
reproduced under `licenses/`, with trailing whitespace and extra blank lines
at EOF removed; license text is unchanged.
