"""Source-isolated main/block-order benchmark, driven in A/B/B/A order."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--order", choices=("raster", "block"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    import torch
    from benchmark.inputs import make_geometry
    from benchmark.utils import (capture_graph, environment, measure_graph,
                                 packed_geometry_and_flops, similarity, summarize)
    from vsa import vsa_attention
    from vsa_layout import Segment, build_vsa_layout

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = {
        "single5s-h2": (((Segment("video", (31, 45, 80), .9), Segment("text", 2048)),), 2),
        "thd5s-h2": (make_geometry(), 2),
        "thd5s-h4": (make_geometry(), 4),
        "aligned64k-h2": (((Segment("video", (16, 60, 64), .9), Segment("text", 4096)),), 2),
    }
    result = {"schema": "vsa-input-order/v1", "environment": environment(),
              "source_root": str(root), "input_order": args.order,
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "warmup": args.warmup, "repeats": args.repeats, "peak_tflops": 2500, "cases": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, (samples, heads) in cases.items():
        for full in (False, True):
            key = f"{name}-full{int(full)}"
            raster = build_vsa_layout(samples, head_num=heads, dense_query_full=full, device="cuda")
            options = {} if args.order == "raster" else {"input_order": "block"}
            layout = build_vsa_layout(samples, head_num=heads, dense_query_full=full,
                                      device="cuda", **options)
            torch.manual_seed(2026)
            original = [torch.randn(raster.tokens, heads, 128, device="cuda",
                                    dtype=torch.bfloat16).requires_grad_() for _ in range(4)]
            original_dy = torch.randn_like(original[0])
            rows = (torch.arange(raster.tokens, device="cuda") if args.order == "raster"
                    else layout.input_rows)
            xs = [x.detach()[rows].requires_grad_() for x in original]
            dy = original_dy[rows]
            expected = vsa_attention(*original, vsa_layout=raster)
            expected_grads = torch.autograd.grad(expected, original, original_dy)
            fn = functools.partial(vsa_attention, vsa_layout=layout)
            actual = fn(*xs)
            actual_grads = torch.autograd.grad(actual, xs, dy)
            checks = {n: similarity(a, b[rows]) for n, a, b in zip(
                ("out", "dq", "dk", "dv", "dgate"),
                (actual, *actual_grads), (expected, *expected_grads))}
            assert all(min(v["cosine"], v["tensor_similarity"]) >= .99999 for v in checks.values()), checks
            _, flops = packed_geometry_and_flops(layout, xs, 128)
            _, raster_flops = packed_geometry_and_flops(raster, original, 128)
            assert flops == raster_flops
            del original, original_dy, expected, expected_grads, actual, actual_grads
            captured, graph_checks = capture_graph(fn, xs, dy, args.warmup, 1e-5)
            durations = measure_graph(captured, args.warmup, args.repeats)
            timing = summarize(durations, flops, 2500)
            result["cases"][key] = {"tokens": layout.tokens, "heads": heads,
                "dense_query_full": full, "parity": checks, "graph_parity": graph_checks,
                "flops": flops, "timing": timing, "samples_ms": durations}
            print(json.dumps({"case": key, "order": args.order, "timing": timing}), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            del captured, xs, dy
            torch.cuda.empty_cache()
    result["environment_after"] = environment()
    result["source_unchanged"] = (result["environment"]["source_sha256"] ==
                                 result["environment_after"]["source_sha256"])
    assert result["source_unchanged"]
    assert result["driver_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
