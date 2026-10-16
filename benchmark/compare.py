"""Compare FA4 full attention and VSA with identical CUDA Graph inputs."""
import argparse
import json
from functools import partial
from itertools import accumulate
from pathlib import Path

import torch
from flash_attn.cute.interface import flash_attn_varlen_func
from vsa import vsa_attention
from vsa_layout import build_vsa_layout

from .inputs import HEAD_COUNTS, HEAD_DIM, describe, make_geometry, make_inputs
from .utils import (capture_graph, environment, measure_graph,
                    packed_geometry_and_flops, summarize)


def prepare_fa4_full(lengths, device="cuda"):
    """Prepare sample boundaries outside timing; return fn(q, k, v) -> o."""
    lengths = tuple(lengths)
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("FA4 sample lengths must be positive")
    cu_seqlens = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)
    return partial(_fa4_full_attention, cu_seqlens=cu_seqlens,
                   max_seqlen=max(lengths), total_tokens=sum(lengths))


def _fa4_full_attention(q, k, v, *, cu_seqlens, max_seqlen, total_tokens):
    """Apply full attention independently within each prepared sample."""
    if q.ndim != 3 or any(n <= 0 for n in q.shape):
        raise ValueError("FA4 expects [tokens, heads, dim] with positive dimensions")
    for x in (q, k, v):
        if x.shape != q.shape or x.dtype != q.dtype or x.device != q.device:
            raise ValueError("Q/K/V must have identical shape, dtype and device")
        if not x.is_contiguous():
            raise ValueError("Q/K/V must be contiguous")
    if q.shape[0] != total_tokens:
        raise ValueError("THD token count does not match the sample boundaries")
    if not q.is_cuda:
        raise ValueError("FA4 expects CUDA [tokens, heads, dim]")
    out, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, causal=False)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, nargs="+", choices=HEAD_COUNTS, default=list(HEAD_COUNTS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--peak-tflops", type=float, default=2500, help="Dense BF16 peak per GPU")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/compare.json"))
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.rounds, args.peak_tflops) <= 0:
        raise ValueError("Counts and dense peak must be positive")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {"schema": "vsa-compare/v2", "config": {k: v for k, v in vars(args).items() if k != "output"},
              "input": describe(), "environment": environment(), "cases": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for heads in args.heads:
        xs, dy = make_inputs(heads)
        layout = build_vsa_layout(make_geometry(), head_num=heads, device="cuda")
        fns = {"fa4_full": prepare_fa4_full(layout.sample_lengths),
               "cudnn_vsa": partial(vsa_attention, vsa_layout=layout)}
        geometry, vsa_flops = packed_geometry_and_flops(layout, xs, HEAD_DIM)
        full_fwd = 4 * heads * HEAD_DIM * sum(n**2 for n in layout.sample_lengths)
        flops = {"fa4_full": {"forward": full_fwd, "backward": 2*full_fwd, "total": 3*full_fwd},
                 "cudnn_vsa": vsa_flops}
        captures, parity = {}, {}
        for name, fn in fns.items():
            leaves = xs[:3] if name == "fa4_full" else xs
            captures[name], parity[name] = capture_graph(fn, leaves, dy, args.warmup, 1e-5)
        samples, orders = {name: [] for name in fns}, []
        for round_id in range(args.rounds):
            order = list(fns)[::1 if round_id % 2 == 0 else -1]
            orders.append(order)
            for name in order:
                samples[name].extend(measure_graph(captures[name], args.warmup, args.repeats))
        summary = {}
        print(f"\nH={heads}\nBackend      Phase          ms       MFU", flush=True)
        for name in fns:
            summary[name] = summarize(samples[name], flops[name], args.peak_tflops)
            for phase, row in summary[name].items():
                print(f"{name:<12} {phase:<9} {row['median_ms']:8.3f} {row['mfu_percent']:8.2f}%", flush=True)
        result["cases"][f"h{heads}"] = {"dense_query_full": layout.dense_query_full,
                    "sample_geometry": geometry, "flops": flops,
                    "graph_parity": parity, "round_order": orders, "samples_ms": samples, "summary": summary}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        del captures, xs, dy, leaves, fns, fn, layout
        torch.cuda.empty_cache()
    result["environment_after"] = environment()
    if result["environment"]["source_sha256"] != result["environment_after"]["source_sha256"]:
        raise AssertionError("Sources changed during measurement")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Measurements: {args.output}", flush=True)


if __name__ == "__main__":
    main()
