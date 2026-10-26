"""Optional pip-installed FastVideo H3 oracle; no upstream source is vendored."""
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vsa import vsa_attention
from vsa_layout import Segment, build_vsa_layout

from tests.reference import dense_attention, modality_slices, parity, routing_mask, sample_input, similarity

PIN = {'commit': 'd5253fd71055927f12aa6c4aa679b93a00238851',
 'kernel_version': '0.3.5',
 'files_sha256': {'fastvideo/attention/backends/video_sparse_attn_h3.py': 'cb08b7c40a5e33d64643cd4e052b26c9c71d0a17531faabb53b70c107698c105',
                  'fastvideo/attention/backends/video_sparse_attn.py': '6f05e429129bb131532f42e24aa868bd7752f6600bc958ea66934b6390157fc2',
                  'fastvideo/attention/backends/abstract.py': '6c7ed11857958fa332e375bf3acecc700735a6830a4ce64f989a32bbdc5b9b20'}}


@functools.cache
def load_h3():
    pin = PIN
    dist = importlib.metadata.distribution("fastvideo")
    for name, digest in pin["files_sha256"].items():
        if hashlib.sha256(Path(dist.locate_file(name)).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Install the pinned H3 reference from tests/requirements-fastvideo.txt")
    if importlib.metadata.version("fastvideo-kernel") != pin["kernel_version"]:
        raise RuntimeError("The correctness oracle requires the pinned fastvideo-kernel wheel")
    # Deliberately use upstream Triton for correctness, never for benchmarking.
    os.environ["FASTVIDEO_VSA_TRITON"] = "1"
    os.environ["FASTVIDEO_VSA_SM100A"] = "0"
    module = importlib.import_module("fastvideo.attention.backends.video_sparse_attn_h3")
    expected = Path(dist.locate_file("fastvideo/attention/backends/video_sparse_attn_h3.py")).resolve()
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError("An unexpected FastVideo installation was imported")
    return module


class FastVideoReference:
    def __init__(self, segments, heads, dim=128, device="cuda"):
        self.h3 = load_h3()
        segments = tuple(segments)
        videos = [s for s in segments if s.sparsity is not None]
        if len(videos) != 1:
            raise ValueError("The oracle expects one VSA region per independent sample")
        self.meta = self.h3.MiniMaxH3VSAMetadataBuilder().build(
            current_timestep=0, patch_size=(1, 1, 1), VSA_sparsity=videos[0].sparsity,
            packed_segments=tuple(s.size if s.sparsity is not None else s.tokens for s in segments),
            device=torch.device(device),
            tile_size=64, exempt=True)
        self.impl = self.h3.MiniMaxH3VSAImpl(heads, dim, causal=False, softmax_scale=dim**-0.5)

    def tiled(self, tensors):
        # Upstream no-grad tile() reuses a scratch buffer. Tile all operands
        # once with a call-owned holder so Q/K/V/gate cannot alias, including
        # successive forwards whose autograd graphs are simultaneously alive.
        meta = replace(self.meta, tile_buf_holder=type(self.meta.tile_buf_holder)())
        width = tensors[0].shape[-1]
        tiled = self.impl.tile(torch.cat(tensors, dim=-1).unsqueeze(0), meta)
        return tuple(x.contiguous() for x in tiled.split(width, dim=-1))

    def __call__(self, q, k, v, gate):
        q, k, v, gate = self.tiled((q, k, v, gate))
        out = self.impl.forward(q, k, v, gate, self.meta)
        return self.impl.postprocess_output(out, self.meta).squeeze(0).contiguous()

    @torch.no_grad()
    def check_routes(self, q, k, v, layout):
        from vsa_core import route
        import vsa_kernels
        torch.testing.assert_close(layout.block_sizes.long(), self.meta.variable_block_sizes.long(), rtol=0, atol=0)
        torch.testing.assert_close(layout.row_slot, self.meta.untile_combined_index, rtol=0, atol=0)
        q_slots, k_slots, v_slots = (
            vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot) for x in (q, k, v)
        )
        q_route, k_route, _, _, _ = vsa_kernels.routing_and_coarse_means(
            q_slots, k_slots, v_slots, layout.block_sizes, layout.coarse_blocks)
        indices = route(q_route, k_route, layout.groups, layout.routing_indices,
            dense_query_full=layout.dense_query_full)
        actual = routing_mask(indices, layout.routing_counts)
        tq, tk = self.tiled((q, k))
        means = [self.h3._pool_tiles(x, self.meta.variable_block_sizes, 64) for x in (tq, tk)]
        scores = means[0] @ means[1].transpose(-1, -2) / q.shape[-1]**0.5
        m = self.meta
        expected = self.h3._build_block_mask(scores, m.num_prefix_tiles, m.VSA_sparsity,
                                              m.exempt, m.video_tile_spans, m.span_sparsities)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def build_packed_reference(samples, heads, dim=128, device="cuda"):
    """Loop independent samples and preserve [v0,t0,v1,t1,...] ordering."""
    rows = build_vsa_layout(samples, head_num=heads, dense_query_full=True).sample_slices
    refs = [FastVideoReference(sample, heads, dim, device) if any(s.sparsity is not None for s in sample) else None
            for sample in samples]

    def run(q, k, v, gate):
        outputs = []
        for ref, pair in zip(refs, rows):
            xs = [sample_input(x, pair) for x in (q, k, v)]
            outputs.append(dense_attention(*xs) if ref is None else
                           ref(*xs, sample_input(gate, pair)))
        return torch.cat(outputs)

    run.references = refs
    return run

def check_streams(heads, eps):
    segments = (Segment("video", (8, 8, 8), 0.75), Segment("text", 77))
    layout = build_vsa_layout([segments], head_num=heads, dense_query_full=True, device="cuda")
    implementations = [partial(vsa_attention, vsa_layout=layout),
                       FastVideoReference(segments, heads)]
    hidden, width = heads * 128, 4 * heads * 128
    result = {}
    for mode in ("single", "dual"):
        torch.manual_seed(91)
        weight_count = 1 if mode == "single" else 2
        base = [torch.randn(layout.tokens, hidden, device="cuda", dtype=torch.bfloat16)]
        base += [torch.randn(width, hidden, device="cuda", dtype=torch.bfloat16) / hidden**0.5
                 for _ in range(weight_count)]
        dy = torch.randn(layout.tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        runs = []
        for impl in implementations:
            leaves = [x.detach().clone().requires_grad_() for x in base]
            x, weights = leaves[0], leaves[1:]
            if mode == "single":
                projected = F.linear(x, weights[0])
            else:
                projected = torch.cat([F.linear(x[rows], w) for (_, rows), w in
                                       zip(modality_slices(layout), weights)])
            q, k, v, gate = [x.contiguous() for x in projected.view(layout.tokens, 4, heads, 128).unbind(1)]
            out = impl(q, k, v, gate)
            grads = torch.autograd.grad(out, leaves, dy)
            runs.append((out.detach(), *grads))
        names = ("output", "input_grad", *(f"projection_{i}_grad" for i in range(weight_count)))
        metrics = {name: similarity(a, b) for name, a, b in zip(names, *runs)}
        passed = all(m["cosine"] >= 1-eps and m["tensor_similarity"] >= 1-eps for m in metrics.values())
        result[mode] = {"passed": passed, "metrics": metrics}
    return result

def run(heads, eps):
    checks = {}
    cases = {
        "video_text": (Segment("video", (8, 8, 8), 0.75), Segment("text", 77)),
        "ragged_video_text_audio": (Segment("video", (5, 6, 7), 0.5),
                                    Segment("text", 65), Segment("audio", 67)),
        "dense_reference_video": (Segment("video", (2, 5, 7), None), Segment("text", 7),
                                  Segment("video", (4, 4, 8), .5)),
        "sparse_image": (Segment("video", (1, 8, 8), .5), Segment("text", 7)),
    }
    for name, segments in cases.items():
        layout = build_vsa_layout([segments], head_num=heads, dense_query_full=True, device="cuda")
        candidate = partial(vsa_attention, vsa_layout=layout)
        reference = FastVideoReference(segments, heads)
        xs = [torch.randn(layout.tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
              for _ in range(4)]
        dy = torch.randn_like(xs[0])
        reference.check_routes(*xs[:3], layout)
        for gate_scale in (0.0, 0.2):
            leaves = (*xs[:3], xs[3] * gate_scale)
            check = parity(candidate, reference, leaves, dy, modality_slices(layout), eps)
            with torch.no_grad():
                # Also exercise upstream's cached no-grad tiling contract.
                check["metrics"]["no_grad/output"] = similarity(candidate(*leaves), reference(*leaves))
            checks[f"{name}/gate_{gate_scale}"] = check

    # Two videos of different lengths and one independent image. The image
    # oracle is FP32 full attention; FastVideo is called once for each video.
    samples = (cases["video_text"],
               (Segment("video", (5, 4, 7), 0.5), Segment("text", 19)),
               (Segment("video", (1, 3, 5)), Segment("text", 11)))
    layout = build_vsa_layout(samples, head_num=heads, dense_query_full=True, device="cuda")
    candidate = partial(vsa_attention, vsa_layout=layout)
    reference = build_packed_reference(samples, heads)
    xs = [torch.randn(sum(layout.sample_lengths), heads, 128, device="cuda", dtype=torch.bfloat16)
          for _ in range(4)]
    for i, ref in enumerate(reference.references):
        if ref is not None:
            ref.check_routes(*(sample_input(x, layout.sample_slices[i]) for x in xs[:3]),
                             build_vsa_layout([samples[i]], head_num=heads, dense_query_full=True, device="cuda"))
    checks["packed_vsa"] = parity(candidate, reference, xs, torch.randn_like(xs[0]),
                                  modality_slices(layout), eps)
    checks.update({f"{mode}_stream": check for mode, check in check_streams(heads, eps).items()})
    for check in checks.values():
        check["failures"] = {k: v for k, v in check["metrics"].items()
                             if not (v["cosine"] >= 1-eps and v["tensor_similarity"] >= 1-eps)}
        check["passed"] = not check["failures"]
    return checks

@dataclass
class Meta:
    tile_buf_holder: object = field(default_factory=SimpleNamespace)


class ScratchImpl:
    def tile(self, x, meta):
        holder = meta.tile_buf_holder
        if not hasattr(holder, "buffer"):
            holder.buffer = torch.zeros_like(x)
        holder.buffer.copy_(x)
        return holder.buffer

    def forward(self, q, k, v, gate, meta):
        return q * k + v * gate

    def postprocess_output(self, out, meta):
        return out


def test_reference_keeps_operands_and_live_graphs_independent():
    ref = FastVideoReference.__new__(FastVideoReference)
    ref.meta, ref.impl = Meta(), ScratchImpl()
    xs = [torch.randn(7, 2, 8, requires_grad=True) for _ in range(4)]
    ys = [torch.randn_like(x, requires_grad=True) for x in xs]
    a, b = ref(*xs), ref(*ys)
    expected = xs[0] * xs[1] + xs[2] * xs[3]
    torch.testing.assert_close(a, expected)
    torch.testing.assert_close(b, ys[0] * ys[1] + ys[2] * ys[3])
    grads = torch.autograd.grad(a.sum() + b.sum(), xs + ys)
    for actual, wanted in zip(grads, (xs[1], xs[0], xs[3], xs[2], ys[1], ys[0], ys[3], ys[2])):
        torch.testing.assert_close(actual, wanted)
    with torch.no_grad():
        torch.testing.assert_close(ref(*xs), expected)

@pytest.mark.fastvideo
@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("fastvideo") is None,
                    reason="CUDA and optional tests/requirements-fastvideo.txt required")
def test_fastvideo(heads):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_h3()
    checks = run(heads, 1e-5)
    assert all(c["passed"] for c in checks.values()), checks
