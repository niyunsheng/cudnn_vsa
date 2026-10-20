"""Distributed ring VSA parity, CUDA Graph timing and useful-FLOP MFU."""
import argparse
from datetime import timedelta
import functools
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

from benchmark.utils import environment, packed_geometry_and_flops, similarity
from tests.reference import reference_pack
from vsa import vsa_attention
from vsa_layout import Segment, build_ring_vsa_layout, build_vsa_layout


CASES = {
    "single64k": ((Segment("video", (16, 60, 64), .9), Segment("text", 4096)),),
    "thd64k": (
        (Segment("video", (12, 48, 64), .9), Segment("text", 2048)),
        (Segment("video", (8, 40, 64), .9), Segment("text", 2048)),
        (Segment("video", (1, 32, 64)), Segment("text", 2048))),
    # 720p, 30 s at 24 fps: 721 frames -> 181 temporal latents;
    # spatial VAE compression 8 and 2x2 patches give 45x80 token rows.
    "single720p30s": (
        (Segment("video", (181, 45, 80), .9), Segment("text", 2048)),),
    "thd720p30s": (
        (Segment("video", (181, 45, 80), .9), Segment("text", 2048)),
        (Segment("video", (1, 45, 80)), Segment("text", 2048))),
    "ragged": (
        (Segment("video", (5, 9, 11), .7), Segment("text", 67)),
        (Segment("text", 3), Segment("video", (1, 3, 5))),
        (Segment("video", (4, 8, 12), .5), Segment("video", (5, 4, 7), .9),
         Segment("audio", 17), Segment("text", 0))),
    "empty_peers": ((Segment("video", (1, 3, 5), .9), Segment("text", 7)),),
    "dense": ((Segment("video", (1, 9, 11)), Segment("text", 67)),
              (Segment("video", (1, 3, 5)), Segment("text", 3))),
}


def _step(fn, xs, dy, events=None, *, retain_graph=False):
    if events:
        events[0].record()
    out = fn(*xs)
    if events:
        events[1].record()
    grads = torch.autograd.grad(out, xs, dy, allow_unused=True, retain_graph=retain_graph)
    if events:
        events[2].record()
    return (out, *(torch.zeros_like(x) if g is None else g for x, g in zip(xs, grads)))


def _compare(actual, expected, eps):
    result = {}
    for name, a, b in zip(("out", "dq", "dk", "dv", "dgate"), actual, expected):
        if not a.numel():
            result[name] = {"cosine": 1., "tensor_similarity": 1., "max_abs": 0., "relative_l2": 0.}
        else:
            result[name] = similarity(a, b)
    passed = all(min(m["cosine"], m["tensor_similarity"]) >= 1 - eps for m in result.values())
    return {"passed": passed, "metrics": result}


def _collect(value, group):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _barrier(group):
    # Identifies the barrier's GPU without forcing eager communicator creation.
    dist.barrier(group=group, device_ids=[torch.cuda.current_device()])


def _bucket_argument(value):
    if value == "auto":
        return value
    if value == "backend":
        return None
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected auto, backend, or a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("Bucket size must be positive")
    return number


def _assert_all(check, group, label):
    checks = _collect(check, group)
    if not all(c["passed"] for c in checks):
        raise AssertionError(f"{label}: {checks}")
    return checks


def run_case(name, args, group):
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    samples = CASES[name]
    full = build_vsa_layout(samples, head_num=args.heads, device="cuda",
                            dense_query_full=args.dense_query_full)
    local = build_ring_vsa_layout(samples, head_num=args.heads, device="cuda",
                                  dense_query_full=args.dense_query_full, ring_size=size, ring_rank=rank,
                                  input_order=args.input_order)
    torch.manual_seed(937)
    global_xs = [torch.randn(full.tokens, args.heads, 128, device="cuda",
                            dtype=torch.bfloat16).requires_grad_() for _ in range(4)]
    global_dy = torch.randn_like(global_xs[0])
    xs = [x.detach()[local.input_rows].contiguous().requires_grad_() for x in global_xs]
    dy = global_dy[local.input_rows].contiguous()
    fn = functools.partial(vsa_attention, vsa_layout=local, ring_cp_group=group,
                           ring_cp_batch_p2p=args.batch_p2p,
                           ring_cp_bwd_bucket_size=args.bwd_bucket_size,
                           ring_cp_fwd_use_clc={"auto": "auto", "backend": None,
                                               "clc": True, "static": False}[args.fwd_scheduler],
                           ring_cp_fwd_kv_splits=args.fwd_kv_splits,
                           ring_cp_chunk_size=args.chunk_size)
    ref = (functools.partial(vsa_attention, vsa_layout=full) if full.tokens > 4096
           else reference_pack(samples, full.sample_slices, dense_query_full=args.dense_query_full))
    if rank == 0:
        print(f"CASE {name} tokens={full.tokens} H={args.heads} ring={size} "
              f"dense_query_full={args.dense_query_full}", flush=True)
    expected = _step(ref, global_xs, global_dy)
    actual = _step(fn, xs, dy, retain_graph=True)
    expected_local = tuple(x[local.input_rows] for x in expected)
    checks = _assert_all(_compare(actual, expected_local, args.eps), group, name)
    geometry, flops = packed_geometry_and_flops(full, global_xs, 128)
    record = {"tokens": full.tokens, "heads": args.heads, "ring_size": size,
              "reference": "single_device_vsa" if full.tokens > 4096 else "native_token_math",
              "threshold": 1-args.eps, "parity": checks, "geometry": geometry, "global_flops": flops}
    repeated_grads = torch.autograd.grad(actual[0], xs, dy, allow_unused=True)
    repeated = (actual[0], *(torch.zeros_like(x) if g is None else g
                             for x, g in zip(xs, repeated_grads)))
    record["retained_backward_parity"] = _assert_all(
        _compare(repeated, actual, args.eps), group, f"{name} retained backward")
    del repeated, repeated_grads
    record["bwd_bucket_size_effective"] = next(iter(local._runtime.values())).bwd_bucket_size
    record["fwd_use_clc_effective"] = next(iter(local._runtime.values())).fwd_use_clc
    if rank == 0:
        minimum = min(min(m["cosine"], m["tensor_similarity"])
                      for check in checks for m in check["metrics"].values())
        print(f"PARITY {name} PASS min_similarity={minimum:.9f}", flush=True)
    del global_xs, global_dy, expected, expected_local, actual
    if args.check_only:
        return record
    stream = torch.cuda.current_stream() if args.eager else torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        # Create capture leaves on the stream that owns their autograd nodes.
        xs = [x.detach().clone().requires_grad_() for x in xs]
        dy = dy.clone()
        for _ in range(args.warmup):
            _step(fn, xs, dy)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    _barrier(group)
    if args.eager:
        captured = None
    else:
        with torch.cuda.graph(graph, stream=stream):
            captured = _step(fn, xs, dy, events)
        torch.cuda.current_stream().wait_stream(stream)
        original = [x.detach().clone() for x in (*xs, dy)]
        graph_checks = []
        for changed in (False, True):
            if changed:
                with torch.no_grad():
                    for x in (*xs, dy):
                        x.normal_()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                eager = _step(fn, xs, dy)
            torch.cuda.current_stream().wait_stream(stream)
            graph.replay()
            graph_checks.append(_assert_all(_compare(captured, eager, args.eps), group,
                                             f"{name} graph changed={changed}"))
        with torch.no_grad():
            for x, source in zip((*xs, dy), original):
                x.copy_(source)
        record["graph_parity"] = graph_checks
        del original, eager
    if full.tokens <= 4096:
        return record
    for _ in range(args.warmup):
        if captured is None:
            _step(fn, xs, dy)
        else:
            graph.replay()
    torch.cuda.synchronize()
    _barrier(group)
    timings = []
    for _ in range(args.repeats):
        if captured is None:
            _step(fn, xs, dy, events)
        else:
            graph.replay()
        events[-1].synchronize()
        timings.append([events[0].elapsed_time(events[1]), events[1].elapsed_time(events[2]),
                        events[0].elapsed_time(events[2])])
    # Reduce each observation, not the per-rank medians.
    durations = torch.tensor(timings, device="cuda", dtype=torch.float64)
    dist.all_reduce(durations, op=dist.ReduceOp.MAX, group=group)
    records = durations.cpu().tolist()
    record["mode"] = "eager" if captured is None else "cuda_graph"
    record["samples_ms_max_across_ranks"] = records
    record["timing"] = {}
    for index, phase in enumerate(("forward", "backward", "total")):
        ms = statistics.median(row[index] for row in records)
        tflops = flops[phase] / ms / 1e9 / size
        record["timing"][phase] = {"median_ms": ms, "tflops_per_gpu": tflops,
                                  "mfu_percent": 100*tflops/args.peak_tflops}
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            # Exclude rank-skew during profiler startup from the profiled step.
            _barrier(group)
            torch.cuda.synchronize()
            with torch.profiler.record_function("ProfilerStep#0"):
                if captured is None:
                    _step(fn, xs, dy)
                else:
                    graph.replay()
                torch.cuda.synchronize()
        trace = args.output.parent / f"{args.output.stem}-{name}-rank{rank}.trace.json"
        prof.export_chrome_trace(str(trace))
    if rank == 0:
        print(json.dumps({"case": name, "mode": record["mode"], "timing": record["timing"]}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES,
                        default=["ragged", "empty_peers", "dense", "single64k", "thd64k"])
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--input-order", choices=("raster", "block"), default="raster")
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--batch-p2p", action="store_true", default=None)
    transport.add_argument("--unbatched-p2p", dest="batch_p2p", action="store_false")
    parser.add_argument("--eager-group", action="store_true",
                        help="Bind device_id when creating WORLD; default uses lazy NCCL groups")
    parser.add_argument("--high-priority-group", action="store_true",
                        help="Create caller-owned NCCL groups with high-priority streams")
    parser.add_argument("--bwd-bucket-size", type=_bucket_argument, default="auto",
                        help="auto (default), backend, or a positive number of query blocks")
    parser.add_argument("--fwd-scheduler", choices=("auto", "backend", "clc", "static"), default="auto")
    parser.add_argument("--fwd-kv-splits", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--ring-size", type=int,
                        help="Test interleaved subgroups instead of WORLD; must divide world size")
    parser.add_argument("--dense-query-full", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--peak-tflops", type=float, default=2500.)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/ring/perf.json"))
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    init_options = ({"device_id": torch.device("cuda", int(os.environ["LOCAL_RANK"]))}
                    if args.eager_group else {})
    pg_options = None
    if args.high_priority_group:
        pg_options = dist.ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = True
        init_options["pg_options"] = pg_options
    dist.init_process_group("nccl", timeout=timedelta(minutes=15), **init_options)
    group = dist.group.WORLD
    world_rank, world_size = dist.get_rank(), dist.get_world_size()
    if args.ring_size is not None:
        if args.ring_size < 1 or world_size % args.ring_size:
            raise ValueError("--ring-size must be positive and divide world size")
        group_count = world_size // args.ring_size
        for index in range(group_count):
            ranks = list(range(index, world_size, group_count))
            candidate = dist.new_group(ranks=ranks, pg_options=pg_options)
            if world_rank in ranks:
                group = candidate
                args.output = args.output.with_stem(f"{args.output.stem}-group{index}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _barrier(group)
    result = {"schema": "vsa-ring/v1", "environment": environment(),
              "config": {k: v for k, v in vars(args).items() if k != "output"}, "cases": {}}
    result["batch_p2p_effective"] = (
        group.bound_device_id is not None if args.batch_p2p is None else args.batch_p2p)
    result["nccl_environment"] = {key: os.environ.get(key) for key in (
        "NCCL_MNNVL_ENABLE", "NCCL_CUMEM_ENABLE", "NCCL_P2P_DISABLE",
        "NCCL_MIN_P2P_NCHANNELS", "NCCL_MAX_P2P_NCHANNELS",
        "NCCL_NCHANNELS_PER_NET_PEER", "NCCL_NVLS_ENABLE", "NCCL_IB_DISABLE")}
    for name in args.cases:
        result["cases"][name] = run_case(name, args, group)
        if dist.get_rank(group) == 0:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        torch.cuda.synchronize()
        _barrier(group)
    result["environment_after"] = environment()
    result["source_unchanged"] = (
        result["environment"]["source_sha256"] == result["environment_after"]["source_sha256"])
    if not result["source_unchanged"]:
        raise RuntimeError("Benchmark source changed during execution")
    if dist.get_rank(group) == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
