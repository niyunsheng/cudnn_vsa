"""Measure VSA CUDA Graph forward/backward time and MFU; save torch traces."""
import argparse
import functools
import json
from pathlib import Path

import torch
from .inputs import HEAD_COUNTS, HEAD_DIM, describe, make_geometry, make_inputs
from .utils import (capture_graph, environment, measure_graph,
                    packed_geometry_and_flops, summarize)
from vsa import vsa_attention
from vsa_layout import build_vsa_layout


def measure_case(heads, output, warmup, repeats, peak_tflops):
    xs, dy = make_inputs(heads)
    layout = build_vsa_layout(make_geometry(), head_num=heads, device="cuda")
    fn = functools.partial(vsa_attention, vsa_layout=layout)
    geometry, flops = packed_geometry_and_flops(layout, xs, HEAD_DIM)
    captured, parity = capture_graph(fn, xs, dy, warmup, 1e-5)
    samples = measure_graph(captured, warmup, repeats)
    timing = summarize(samples, flops, peak_tflops)
    print(f"\nVSA H={heads}: CUDA Graph", flush=True)
    print("Phase           ms    TFLOP/s      MFU", flush=True)
    for phase, row in timing.items():
        print(f"{phase:<9} {row['median_ms']:8.4f} {row['tflops']:10.2f} {row['mfu_percent']:7.2f}%", flush=True)

    # Profile an additional replay of the same graph after all timing samples.
    # Profiler overhead never enters the time/MFU measurements above.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("ProfilerStep#0"):
            captured[0].replay()
            torch.cuda.synchronize()
    trace = output / f"vsa-h{heads}.trace.json"
    prof.export_chrome_trace(str(trace))
    print(f"Torch trace: {trace}", flush=True)
    return {"heads": heads, "tokens": layout.tokens, "dense_query_full": layout.dense_query_full,
            "input": describe(),
            "sample_geometry": geometry, "flops": flops, "graph_parity": parity,
            "timing": timing, "timing_samples_ms": samples, "trace": trace.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", nargs="+", type=int, choices=HEAD_COUNTS, default=list(HEAD_COUNTS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--peak-tflops", type=float, default=2500, help="Dense BF16 peak per GPU; default GB200")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/perf_vsa"))
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.peak_tflops) <= 0:
        raise ValueError("Warmup, repeats and dense peak must be positive")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "vsa-perf/v3", "environment": environment(), "cases": {},
              "config": {k: v for k, v in vars(args).items() if k != "output"},
              "scope": "CUDA Graph time/MFU, followed by one profiled replay of the same graph"}
    result_path = args.output / "perf.json"
    for heads in args.heads:
        result["cases"][f"h{heads}"] = measure_case(heads, args.output, args.warmup,
                                                  args.repeats, args.peak_tflops)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        torch.cuda.empty_cache()
    result["environment_after"] = environment()
    if result["environment"]["source_sha256"] != result["environment_after"]["source_sha256"]:
        raise AssertionError("Sources changed during measurement")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Measurements: {result_path}", flush=True)


if __name__ == "__main__":
    main()
