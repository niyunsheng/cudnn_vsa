"""Shared numerical metrics, useful-FLOP accounting and full-core timing."""
import hashlib
import importlib.metadata
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from vsa import route
import vsa_kernels

ROOT = Path(__file__).resolve().parent.parent

def similarity(a, b):
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
        q_route, k_route, _, _, _ = vsa_kernels.block_means(
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


def capture_graph(fn, leaves, dy, warmup, eps):
    """Capture complete forward/backward; validate replay with changed inputs."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            out = fn(*leaves)
            grads = torch.autograd.grad(out, leaves, dy)
            del out, grads
        with torch.cuda.graph(graph, stream=stream):
            events[0].record()
            out = fn(*leaves)
            events[1].record()
            grads = torch.autograd.grad(out, leaves, dy)
            events[2].record()
    torch.cuda.current_stream().wait_stream(stream)
    saved = [x.detach().clone() for x in (*leaves, dy)]
    checks = []
    try:
        for changed in (False, True):
            if changed:
                with torch.no_grad():
                    for x in (*leaves, dy):
                        x.normal_()
            expected = fn(*leaves)
            expected_grads = torch.autograd.grad(expected, leaves, dy)
            graph.replay()
            metrics = {name: similarity(a, b) for name, a, b in
                       zip(("out", "dq", "dk", "dv", "dgate"), (out, *grads), (expected, *expected_grads))}
            passed = all(min(m["cosine"], m["tensor_similarity"]) >= 1-eps for m in metrics.values())
            checks.append({"changed_inputs": changed, "passed": passed, "metrics": metrics})
            if not passed:
                raise AssertionError(checks[-1])
            del expected, expected_grads
    finally:
        with torch.no_grad():
            for x, original in zip((*leaves, dy), saved):
                x.copy_(original)
    graph.replay()
    torch.cuda.synchronize()
    return (graph, out, grads, events), checks


def measure_graph(captured, warmup, repeats):
    """CUDA-event times inside one full F+B graph; compilation/capture excluded."""
    graph, out, grads, events = captured  # Keep output/gradient storage alive.
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        graph.replay()
        events[-1].synchronize()
        start, middle, end = events
        samples.append({"forward": start.elapsed_time(middle),
                        "backward": middle.elapsed_time(end), "total": start.elapsed_time(end)})
    return samples


def summarize(samples, flops, peak_tflops):
    """Independent phase medians and useful-FLOP MFU against a dense GPU peak."""
    result = {}
    for phase in ("forward", "backward", "total"):
        elapsed = statistics.median(row[phase] for row in samples)
        tflops = flops[phase] / elapsed / 1e9
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
