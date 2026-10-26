"""VSA parity, forward/backward timing, useful-FLOP MFU and optional traces."""
import argparse
from datetime import timedelta
import functools
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from .inputs import CASES, HEAD_COUNTS, HEAD_DIM, case_geometry, describe, make_inputs
from .utils import (backward, barrier, capture_graph, check_tensors, environment,
                    forward_backward, measure_eager, measure_graph,
                    packed_geometry_and_flops, profile_step, summarize)
from tests.reference import reference_pack
from vsa import vsa_attention
from vsa_layout import build_vsa_layout


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



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring", action="store_true", help="Run under torchrun with explicit NCCL groups")
    parser.add_argument("--cases", nargs="+", choices=("default", *CASES))
    parser.add_argument("--heads", nargs="+", type=int)
    parser.add_argument("--input-order", choices=("raster", "block"))
    parser.add_argument("--dense-query-full", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--peak-tflops", type=float, default=2500, help="Dense BF16 peak per GPU; default GB200")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--eager", action="store_true", help="Time eager execution instead of CUDA Graph")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/perf_vsa"),
                        help="JSON file or output directory")
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--batch-p2p", action="store_true", default=None)
    transport.add_argument("--unbatched-p2p", dest="batch_p2p", action="store_false")
    parser.add_argument("--eager-group", action="store_true", help="Bind device_id when creating WORLD")
    parser.add_argument("--high-priority-group", action="store_true")
    parser.add_argument("--bwd-bucket-size", type=_bucket_argument, default="auto",
                        help="auto (default), backend, or a positive number of query blocks")
    parser.add_argument("--ring-size", type=int, help="Interleaved subgroup size; must divide world size")
    args = parser.parse_args(argv)
    if not args.ring and (args.ring_size is not None or args.eager_group or args.high_priority_group
                         or args.batch_p2p is not None or args.bwd_bucket_size != "auto"):
        parser.error("Ring transport options require --ring")
    args.cases = args.cases or (["ragged", "empty_peers", "dense", "single64k", "thd64k"]
                               if args.ring else ["default"])
    args.heads = args.heads or ([2] if args.ring else list(HEAD_COUNTS))
    if min(args.heads) < 1:
        parser.error("Head counts must be positive")
    args.input_order = args.input_order or ("block" if args.ring else "raster")
    if args.ring and args.input_order != "block":
        parser.error("--ring requires --input-order block")
    if args.ring_size is not None and args.ring_size < 1:
        parser.error("--ring-size must be positive")
    args.warmup = (3 if args.ring else 5) if args.warmup is None else args.warmup
    args.repeats = (30 if args.ring else 80) if args.repeats is None else args.repeats
    if min(args.warmup, args.repeats, args.peak_tflops) <= 0 or not 0 < args.eps < 1:
        parser.error("Warmup, repeats and peak must be positive; eps must be in (0, 1)")
    args.profile = not args.ring if args.profile is None else args.profile
    if args.output.suffix != ".json":
        args.output = args.output / "perf.json"
    return args


def init_ring(args):
    """Create caller-owned WORLD or interleaved groups on every worker."""
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
    return group


def run_case(name, heads, args, group):
    rank, size = (dist.get_rank(group), dist.get_world_size(group)) if group is not None else (0, 1)
    samples = case_geometry(name)
    kwargs = dict(head_num=heads, device="cuda", dense_query_full=args.dense_query_full)
    full = build_vsa_layout(samples, **kwargs)
    if group is not None:
        local = build_vsa_layout(samples, **kwargs, ring_size=size, ring_rank=rank,
                                input_order=args.input_order)
    else:
        local = full if args.input_order == "raster" else build_vsa_layout(samples, **kwargs, input_order="block")
    global_xs, global_dy = make_inputs(heads, samples=samples, seed=937 if args.ring else 2026)
    if name == "fine_only_peer":
        with torch.no_grad():
            global_xs[3][full.sample_slices[1][0]] = float("nan")
    xs = tuple(x.detach()[local.input_rows].contiguous().requires_grad_() for x in global_xs)
    dy = global_dy[local.input_rows].contiguous()
    fn = functools.partial(vsa_attention, vsa_layout=local, ring_cp_group=group,
                           ring_cp_batch_p2p=args.batch_p2p, ring_cp_bwd_bucket_size=args.bwd_bucket_size)
    ref = (functools.partial(vsa_attention, vsa_layout=full) if full.tokens > 4096
           else reference_pack(samples, full.sample_slices, dense_query_full=args.dense_query_full))
    expected = tuple(x[local.input_rows] for x in forward_backward(ref, global_xs, global_dy))
    actual = forward_backward(fn, xs, dy, retain_graph=True)
    parity = check_tensors(actual, expected, args.eps, group=group, label=name)
    repeated = (actual[0], *backward(actual[0], xs, dy))
    retained = check_tensors(repeated, actual, args.eps, group=group, label=f"{name} retained backward")
    geometry, flops = packed_geometry_and_flops(full, global_xs, HEAD_DIM)
    record = {"tokens": full.tokens, "heads": heads, "ring_size": size, "input_order": args.input_order,
              "dense_query_full": args.dense_query_full, "threshold": 1-args.eps,
              "reference": "single_device_vsa" if full.tokens > 4096 else "native_token_math",
              "parity": parity, "retained_backward_parity": retained,
              "sample_geometry": geometry, "flops": flops}
    if name == "default":
        record["input"] = describe()
    if group is not None:
        record["bwd_bucket_size_effective"] = next(iter(local._runtime.values())).bwd_bucket_size
    del global_xs, global_dy, expected, actual, repeated
    if rank == 0:
        print(f"PARITY {name} H={heads} ring={size} input={args.input_order} PASS", flush=True)
    if args.check_only:
        return record
    if args.eager:
        timings = measure_eager(fn, xs, dy, args.warmup, args.repeats, group=group)
        replay = lambda: forward_backward(fn, xs, dy)
    else:
        captured, record["graph_parity"] = capture_graph(fn, xs, dy, args.warmup, args.eps, group=group)
        timings = measure_graph(captured, args.warmup, args.repeats, group=group)
        replay = captured.graph.replay
    record["mode"] = "eager" if args.eager else "cuda_graph"
    record["timing_samples_ms"] = timings
    record["timing"] = summarize(timings, flops, args.peak_tflops, world_size=size)
    if args.profile:
        trace = args.output.with_name(f"{args.output.stem}-{name}-h{heads}-rank{rank}.trace.json")
        profile_step(replay, trace, group=group)
        record["trace"] = trace.name
    if rank == 0:
        print(json.dumps({"case": name, "heads": heads, "mode": record["mode"],
                          "timing": record["timing"]}), flush=True)
    return record


def main():
    args = parse_args()
    if args.ring:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    group = init_ring(args) if args.ring else None
    writer = group is None or dist.get_rank(group) == 0
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        barrier(group)
        result = {"schema": "vsa-perf/v4", "environment": environment(), "cases": {},
                  "config": {k: v for k, v in vars(args).items() if k != "output"},
                  "scope": "Full forward/backward; timing excludes validation, capture and profiling"}
        if group is not None:
            result["batch_p2p_effective"] = (group.bound_device_id is not None
                                             if args.batch_p2p is None else args.batch_p2p)
            result["nccl_environment"] = {key: os.environ.get(key) for key in (
                "NCCL_MNNVL_ENABLE", "NCCL_CUMEM_ENABLE", "NCCL_P2P_DISABLE",
                "NCCL_MIN_P2P_NCHANNELS", "NCCL_MAX_P2P_NCHANNELS",
                "NCCL_NCHANNELS_PER_NET_PEER", "NCCL_NVLS_ENABLE", "NCCL_IB_DISABLE")}
        for name in args.cases:
            for heads in args.heads:
                result["cases"][f"{name}-h{heads}"] = run_case(name, heads, args, group)
                if writer:
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                torch.cuda.synchronize()
                barrier(group)
                torch.cuda.empty_cache()
        result["environment_after"] = environment()
        if result["environment"]["source_sha256"] != result["environment_after"]["source_sha256"]:
            raise AssertionError("Sources changed during measurement")
        if writer:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Measurements: {args.output}", flush=True)
    finally:
        if group is not None:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
