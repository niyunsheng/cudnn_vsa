"""Parity of the vendored bsa_cute kernels against the installed cudnn-frontend BSA.

Both packages compile the same CuTe DSL source at this revision. Output and
LSE are produced once per query tile and must match bitwise. The backward
accumulates dQ across KV blocks and dK/dV across bucketed query groups with
reduce-add in scheduling order, so GPU job 8563118 showed dK differing
bitwise between two compilations of the same source; the gradients are
therefore compared with the suite's 0.99999 similarity threshold.
"""
import importlib

import pytest
import torch

from benchmark.utils import similarity

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _run(bsa, q, k, v, index, counts, sizes, grad):
    scale = q.shape[-1] ** -0.5
    qh, kh, vh = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
    fwd = bsa.block_sparse_attention_forward(
        qh, kh, vh, index, None, sizes, counts, sparse_block_size=64, softmax_scale=scale,
        layout="bhsd", allow_empty_block_nums=True)
    out = fwd["o_tensor"].squeeze(0).transpose(0, 1).contiguous()
    bwd = bsa.block_sparse_attention_backward(
        grad.unsqueeze(0), q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), out.unsqueeze(0),
        fwd["lse_tensor"], index, None, sizes, counts, softmax_scale=scale,
        sparse_block_size=64, layout="bshd")
    return {"out": out, "lse": fwd["lse_tensor"],
            **{name[:2]: bwd[name].squeeze(0) for name in ("dq_tensor", "dk_tensor", "dv_tensor")}}


@pytest.mark.parametrize("heads,q_blocks,kv_blocks,keys", [(2, 128, 128, 20), (1, 37, 96, 9)])
def test_local_bsa_matches_installed_package(heads, q_blocks, kv_blocks, keys):
    installed = importlib.import_module("cudnn").BSA
    local = importlib.import_module("bsa_cute").BSA
    torch.manual_seed(7)
    q, k, v, grad = (torch.randn(blocks * 64, heads, 128, device="cuda", dtype=torch.bfloat16)
                     for blocks in (q_blocks, kv_blocks, kv_blocks, q_blocks))
    index = torch.rand(1, heads, q_blocks, kv_blocks, device="cuda").argsort(-1)[..., :keys]
    index = index.sort(-1).values.to(torch.int32).contiguous()
    counts = torch.randint(0, keys + 1, (1, heads, q_blocks), device="cuda", dtype=torch.int32)
    counts[..., 0] = keys  # Keep at least one full row; zero rows exercise allow_empty.
    sizes = torch.randint(1, 65, (kv_blocks,), device="cuda", dtype=torch.int32)
    reference = _run(installed, q, k, v, index, counts, sizes, grad)
    candidate = _run(local, q, k, v, index, counts, sizes, grad)
    for name in ("out", "lse"):
        assert torch.equal(candidate[name], reference[name]), name
    rows = counts.sum(1).squeeze(0) > 0
    for name in ("dq", "dk", "dv"):
        a, b = candidate[name], reference[name]
        if name == "dq":
            a, b = (x.view(q_blocks, 64, heads, 128)[rows] for x in (a, b))
        metrics = similarity(a, b)
        assert min(metrics["cosine"], metrics["tensor_similarity"]) >= 1 - 1e-5, (name, metrics)
