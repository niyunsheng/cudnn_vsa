"""CPU core harness, independent reference math and correctness fixtures."""
from unittest.mock import patch

import torch
import torch.nn.functional as F
import vsa as core
import vsa_kernels
from vsa_layout import build_vsa_layout
from benchmark.utils import similarity


def modality_slices(layout):
    """Per-modality labels and rows for numerical error reporting only."""
    return [(f"sample_{i}/{seg.kind}", rows)
            for i, (sample, slices) in enumerate(zip(layout.samples, layout.sample_slices))
            for seg, rows in zip(sample, slices)]


def cpu_gather(x, row_map):
    """Test substitute for the Triton gather, including zero-filled padding."""
    return x[row_map.clamp_min(0)].masked_fill((row_map < 0)[:, None, None], 0)


def cpu_merge_forward(fine, gate, coarse, row_coarse):
    coarse = coarse.float() if coarse.dtype in (torch.float16, torch.bfloat16) else coarse
    # Fine-only samples do not consume gate, even if those rows are nonfinite.
    gate = gate.masked_fill((row_coarse == coarse.shape[0])[:, None, None], 0)
    # Append an explicit zero row: fine-only samples have no coarse value/gradient.
    coarse = torch.cat((coarse, coarse.new_zeros((1, *coarse.shape[1:]))))
    return (fine + gate * coarse[row_coarse]).to(fine.dtype)


def cpu_merge_backward(grad, gate, coarse, row_coarse, coarse_rows):
    coarse_dtype = coarse.dtype
    coarse = coarse.float() if coarse.dtype in (torch.float16, torch.bfloat16) else coarse
    padded = torch.cat((coarse, coarse.new_zeros((1, *coarse.shape[1:]))))
    dgate = (grad.to(coarse.dtype) * padded[row_coarse]).to(gate.dtype)
    rows = coarse_rows.clamp_min(0)
    # Gather each block's rows and reduce in FP32 instead of contended scatter
    # atomics. Static inverse maps exclude padding and fine-only samples.
    products = grad[rows].to(coarse.dtype) * gate[rows].to(coarse.dtype)
    dcoarse = products.masked_fill((coarse_rows < 0)[:, :, None, None], 0).sum(1)
    return grad, dgate, dcoarse.to(coarse_dtype)


def routing_mask(indices, counts):
    """Diagnostics/reference only: convert block lists to a boolean map."""
    n = indices.shape[-2]
    valid = torch.arange(indices.shape[-1], device=indices.device) < counts[..., None]
    # scatter_add prevents ignored duplicate zero indices from overwriting valid entries.
    mask = torch.zeros((*indices.shape[:-1], n), dtype=torch.int32, device=indices.device)
    mask.scatter_add_(-1, indices.long(), valid.int())
    return mask.bool()


def dense_fine(q, k, v, indices, counts, sizes, scale):
    """Test substitute for the cuDNN BSA call, including padded KV masking."""
    mask = routing_mask(indices, counts)[0]
    mask = mask.repeat_interleave(64, -2).repeat_interleave(64, -1)
    if sizes is not None:
        valid = torch.arange(k.shape[0], device=k.device) % 64
        mask &= (valid < sizes.repeat_interleave(64))[None, None, :]
    out = F.scaled_dot_product_attention(
        q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2),
        attn_mask=mask, scale=scale)
    return out.permute(1, 0, 2).contiguous()


def dense_coarse(q, k, v, cu_seqlens, max_seqlen, scale=None):
    """Test substitute for FA4 varlen over independent pooled samples."""
    scale = q.shape[-1]**-0.5 if scale is None else scale
    outputs, boundaries = [], cu_seqlens.tolist()
    assert max(b-a for a, b in zip(boundaries, boundaries[1:])) == max_seqlen
    for start, end in zip(boundaries, boundaries[1:]):
        qs, ks, vs = (x[start:end].transpose(0, 1) for x in (q, k, v))
        outputs.append(F.scaled_dot_product_attention(qs, ks, vs, scale=scale).transpose(0, 1))
    return torch.cat(outputs)


def cpu_attention(q, k, v, gate, vsa_layout, *, scale=None):
    """Exercise core orchestration with test-only GPU-kernel substitutes.

    Requires the cpu_kernels fixture. This is not the public entry: its CUDA/BF16
    contract is tested separately.
    """
    core._validate(q, k, v, gate)
    layout = vsa_layout
    slots = tuple(vsa_kernels.to_slots(x, layout.slot_row, layout.row_slot)
                  for x in (q, k, v))
    with patch.object(core.BlockSparseAttention, "apply", dense_fine), \
         patch.object(core, "coarse_attention", dense_coarse):
        fine, coarse = core._attention_slots(*slots, layout, scale=scale)
    out = vsa_kernels.from_slots(fine, layout.slot_row, layout.row_slot)
    if coarse is not None:
        out = vsa_kernels.fine_coarse_gate_add(
            out, gate, coarse, layout.row_coarse, layout.coarse_rows)
    return out.contiguous()


def sample_input(x, rows):
    return x[rows[0].start:rows[-1].stop]


def pool(x, layout):
    """Diagnostic all-block FP32 means in [H,N,D], independent of runtime pooling."""
    means = x.view(layout.blocks, 64, *x.shape[1:]).sum(1, dtype=torch.float32)
    return (means / layout.block_sizes[:, None, None]).permute(1, 0, 2)


def dense_attention(q, k, v):
    """Explicit full softmax reference, promoting low-precision inputs to FP32."""
    dtype = torch.float32 if q.dtype in (torch.bfloat16, torch.float16) else q.dtype
    qf, kf, vf = (x.to(dtype).transpose(0, 1) for x in (q, k, v))
    out = ((qf @ kf.transpose(-1, -2) / q.shape[-1]**0.5).softmax(-1) @ vf)
    return out.transpose(0, 1).to(q.dtype).contiguous()


def parity(candidate, reference, leaves, dy, modality_slices, eps):
    inputs = [[x.detach().clone().requires_grad_() for x in leaves] for _ in range(2)]
    outputs, gradients = [], []
    for fn, xs in zip((candidate, reference), inputs):
        output = fn(*xs)
        assert output.shape == xs[0].shape and output.dtype == xs[0].dtype
        assert output.is_contiguous(), "The common entry must return contiguous O"
        outputs.append(output.detach())
        gradients.append(torch.autograd.grad(output, xs, dy))
    metrics = {}
    for name, a, b in zip(("out", "dq", "dk", "dv", "dgate"),
                           (outputs[0], *gradients[0]), (outputs[1], *gradients[1])):
        for number, (kind, rows) in enumerate(modality_slices):
            key = f"{name}/{kind}/{number}"
            metrics[key] = similarity(a[rows], b[rows])
    failures = {key: value for key, value in metrics.items()
                if not (value["cosine"] >= 1-eps and value["tensor_similarity"] >= 1-eps)}
    return {"passed": not failures, "threshold": 1-eps, "metrics": metrics, "failures": failures}

def reference_sample(sample, q, k, v, gate, *, dense_query_full=False):
    """Independent token softmax and BF16-pooled coarse; diagnostic sizes only."""
    dense_only = all(s.sparsity is None for s in sample)
    layout = build_vsa_layout([sample], head_num=q.shape[1], dense_query_full=dense_query_full)
    rows = [[r for r in block if r >= 0] for block in layout.slot_row.view(-1, 64).tolist()]
    means32 = [torch.stack([x[ids].float().mean(0) for ids in rows]) for x in (q, k, v)]
    h, d, n = q.shape[1], q.shape[2], q.shape[0]
    mask = torch.zeros(h, n, n, dtype=torch.bool, device=q.device)
    for head in range(h):
        for bid, query_rows in enumerate(rows):
            if dense_only or (dense_query_full and bid < layout.groups[0].dense_blocks_num):
                keys = list(range(n))
            else:
                keys = [r for block in rows[:layout.groups[0].dense_blocks_num] for r in block]
                for start, end, keep in layout.groups[0].video_spans:
                    scores = means32[1][start:end, head].detach() @ means32[0][bid, head].detach()
                    selected = scores.topk(keep).indices.tolist()
                    keys += [r for b in selected for r in rows[start+b]]
            mask[head, torch.tensor(query_rows, device=q.device)[:, None], keys] = True
    qf, kf, vf = (x.float().transpose(0, 1) for x in (q, k, v))
    scores = (qf @ kf.transpose(-1, -2) / d**0.5).masked_fill(~mask, -torch.inf)
    fine = (scores.softmax(-1) @ vf).transpose(0, 1).to(q.dtype)
    if dense_only:
        return fine.contiguous()
    qp, kp, vp = (x.to(q.dtype).float().transpose(0, 1) for x in means32)
    coarse = ((qp @ kp.transpose(-1, -2) / d**0.5).softmax(-1) @ vp).transpose(0, 1).to(q.dtype)
    bid = (layout.row_slot // 64).to(q.device)
    return (fine.float() + gate.float() * coarse.float()[bid]).to(q.dtype).contiguous()


def reference_pack(samples, slices, *, dense_query_full=False):
    def run(q, k, v, gate):
        outputs = [reference_sample(sample, *(sample_input(x, rows) for x in (q, k, v)),
                                    sample_input(gate, rows), dense_query_full=dense_query_full)
                   for sample, rows in zip(samples, slices)]
        return torch.cat(outputs)
    return run
