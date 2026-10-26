"""Shared numerical metrics, useful-FLOP accounting and full-core timing."""
import hashlib
import importlib.metadata
import statistics
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import torch
import torch.distributed as dist
from vsa_core import route
import vsa_kernels

ROOT = Path(__file__).resolve().parent.parent

def similarity(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} != {b.shape}")
    if not a.numel():
        return {"cosine": 1., "tensor_similarity": 1., "max_abs": 0., "relative_l2": 0.}
    a, b = a.detach().flatten().double(), b.detach().flatten().double()
    aa, bb, ab = (a*a).sum().item(), (b*b).sum().item(), (a*b).sum().item()
    return {
        "cosine": ab / (aa * bb)**0.5 if aa and bb else float(aa == bb),
        "tensor_similarity": 2 * ab / (aa + bb) if aa + bb else 1.0,
        "max_abs": (a-b).abs().max().item(),
        "relative_l2": ((a-b).square().sum().item() / max(bb, 1e-30))**0.5,
    }


@torch.no_grad()
def packed_geometry_and_flops(layout, xs, dim):
    """Count actual routed valid pairs and BF16 coarse GEMMs; no padding pairs."""
    heads = layout.head_num
    indices = layout.routing_indices
    if not layout.dense_only:
        q_slots, k_slots, v_slots = (
            vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot) for x in xs[:3]
        )
        q_route, k_route, _, _, _ = vsa_kernels.routing_and_coarse_means(
            q_slots, k_slots, v_slots, layout.block_sizes, layout.coarse_blocks)
        indices = route(q_route, k_route, layout.groups, indices,
            dense_query_full=layout.dense_query_full)
    # Count index lists directly: no pack-wide dense NxN mask or pair tensor.
    records = []
    for i, g in enumerate(layout.groups):
        sample = layout.samples[i]
        index = indices[0, :, g.start:g.end]
        count = layout.routing_counts[0, :, g.start:g.end]
        valid = torch.arange(index.shape[-1], device=index.device) < count[..., None]
        sizes = layout.block_sizes.double()
        selected_sizes = sizes[index.long()] * valid
        pairs = int((selected_sizes.sum(-1) * sizes[g.start:g.end]).sum().item())
        length = layout.sample_lengths[i]
        fine = 4 * dim * pairs
        coarse_blocks = 0 if g.dense_only else g.blocks
        coarse = 4 * heads * dim * coarse_blocks**2
        # Routing QK is forward-only; record separately, excluding recomputation
        # from useful FLOPs because coarse QK already includes these pairs.
        routing = 2 * heads * dim * g.blocks * sum(
            end-start for start, end, keep in g.video_spans if keep < end-start)
        flops = {"forward": fine+coarse, "backward": 2*(fine+coarse),
                 "total": 3*(fine+coarse), "fine_forward": fine,
                 "coarse_forward": coarse, "routing_forward": routing,
                 "valid_pairs_all_heads": pairs, "dense_pairs_all_heads": heads * length**2,
                 "selected_block_pairs_all_heads": int(count.sum().item())}
        records.append({"sample": i, "backend": "cudnn_vsa", "mode": "dense" if g.dense_only else "vsa",
                        "tokens": length, "segments": [
                            {"kind": s.kind, "size": s.size, "sparsity": s.sparsity} for s in sample],
                        "blocks": g.blocks, "coarse_blocks": coarse_blocks,
                        "padded_tokens": g.blocks * 64, "flops": flops})
    totals = {key: sum(record["flops"][key] for record in records) for key in records[0]["flops"]}
    return records, totals


def barrier(group):
    if group is not None:
        # Identify the GPU without forcing eager communicator creation.
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])


def check_tensors(actual, expected, eps, *, group=None, label="parity"):
    if len(actual) != len(expected):
        raise ValueError("Different numbers of tensors to compare")
    metrics = {name: similarity(a, b) for name, a, b in
               zip(("out", "dq", "dk", "dv", "dgate"), actual, expected)}
    passed = all(min(m["cosine"], m["tensor_similarity"]) >= 1-eps for m in metrics.values())
    check = {"passed": passed, "metrics": metrics}
    if group is not None:
        ranks = [None] * dist.get_world_size(group)
        dist.all_gather_object(ranks, check, group=group)
        check = {**check, "passed": all(c["passed"] for c in ranks), "ranks": ranks}
    if not check["passed"]:
        raise AssertionError(f"{label}: {check}")
    return check


def backward(out, leaves, dy, *, retain_graph=False):
    grads = torch.autograd.grad(out, leaves, dy, allow_unused=True, retain_graph=retain_graph)
    return tuple(torch.zeros_like(x) if g is None else g for x, g in zip(leaves, grads))


def forward_backward(fn, leaves, dy, events=None, *, retain_graph=False):
    if events:
        events[0].record()
    out = fn(*leaves)
    if events:
        events[1].record()
    grads = backward(out, leaves, dy, retain_graph=retain_graph)
    if events:
        events[2].record()
    return (out, *grads)


class CapturedGraph(NamedTuple):
    graph: torch.cuda.CUDAGraph
    out: torch.Tensor
    grads: tuple
    events: list
    leaves: tuple
    dy: torch.Tensor


def capture_graph(fn, leaves, dy, warmup, eps, *, group=None):
    """Own capture buffers and check replay against eager with changed inputs."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        # Create fresh leaves on their autograd stream, even after eager checks.
        leaves = tuple(x.detach().clone().requires_grad_() for x in leaves)
        dy = dy.clone()
        for _ in range(warmup):
            forward_backward(fn, leaves, dy)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    barrier(group)
    with torch.cuda.graph(graph, stream=stream):
        actual = forward_backward(fn, leaves, dy, events)
    torch.cuda.current_stream().wait_stream(stream)
    saved = [x.detach().clone() for x in (*leaves, dy)]
    checks = []
    try:
        for changed in (False, True):
            if changed:
                with torch.no_grad():
                    for x in (*leaves, dy):
                        x.normal_()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                expected = forward_backward(fn, leaves, dy)
            torch.cuda.current_stream().wait_stream(stream)
            graph.replay()
            check = check_tensors(actual, expected, eps, group=group,
                                  label=f"graph changed={changed}")
            checks.append({"changed_inputs": changed, **check})
    finally:
        with torch.no_grad():
            for x, original in zip((*leaves, dy), saved):
                x.copy_(original)
    graph.replay()
    torch.cuda.synchronize()
    return CapturedGraph(graph, actual[0], actual[1:], events, leaves, dy), checks


def _measure(replay, events, warmup, repeats, group):
    for _ in range(warmup):
        replay()
    torch.cuda.synchronize()
    barrier(group)
    timings = []
    for _ in range(repeats):
        replay()
        events[-1].synchronize()
        start, middle, end = events
        timings.append([start.elapsed_time(middle), middle.elapsed_time(end), start.elapsed_time(end)])
    if group is not None:
        # Reduce each observation before taking medians, not the rank medians.
        durations = torch.tensor(timings, device="cuda", dtype=torch.float64)
        dist.all_reduce(durations, op=dist.ReduceOp.MAX, group=group)
        timings = durations.cpu().tolist()
    return [dict(zip(("forward", "backward", "total"), row)) for row in timings]


def measure_graph(captured, warmup, repeats, *, group=None):
    """CUDA-event times inside one full F+B graph; compilation/capture excluded."""
    return _measure(captured.graph.replay, captured.events, warmup, repeats, group)


def measure_eager(fn, leaves, dy, warmup, repeats, *, group=None):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    return _measure(lambda: forward_backward(fn, leaves, dy, events), events, warmup, repeats, group)


def profile_step(replay, path, *, group=None):
    """Profile an additional step after timing, excluding profiler startup skew."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        barrier(group)
        torch.cuda.synchronize()
        with torch.profiler.record_function("ProfilerStep#0"):
            replay()
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))


def summarize(samples, flops, peak_tflops, *, world_size=1):
    """Independent phase medians and useful-FLOP throughput/MFU per GPU."""
    result = {}
    for phase in ("forward", "backward", "total"):
        elapsed = statistics.median(row[phase] for row in samples)
        tflops = flops[phase] / elapsed / 1e9 / world_size
        result[phase] = {"median_ms": elapsed, "tflops": tflops,
                         "mfu_percent": 100 * tflops / peak_tflops}
    return result


def environment():
    import cudnn
    import triton

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        smi = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit",
            "--format=csv"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        smi = str(exc)
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "triton": triton.__version__, "cudnn_frontend": importlib.metadata.version("nvidia-cudnn-frontend"),
            "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"),
            "flash_attn_4": importlib.metadata.version("flash-attn-4"),
            "cudnn_backend": cudnn.backend_version(), "gpu": props.name,
            "compute_capability": [props.major, props.minor], "gpu_memory_bytes": props.total_memory,
            "python": sys.version, "nvidia_smi": smi,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in (".", "benchmark")
                              for p in sorted((ROOT / folder).iterdir()) if p.suffix == ".py"}}
