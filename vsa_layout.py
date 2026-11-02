"""Static sample geometry, token maps and BSA routing templates for VSA."""
import math
from dataclasses import dataclass, field

import torch

BLOCK = 64


@dataclass(frozen=True)
class Segment:
    """An ordered token region; video sparsity is its KV-block drop fraction.

    None (the default) means dense, with raster chunks and no FHW reordering.
    A numeric video sparsity enables FHW tiles and VSA, even for F=1 or rate 0.
    """
    kind: str
    size: int | tuple[int, int, int]
    sparsity: float | None = None

    def __post_init__(self):
        if self.kind not in {"video", "text", "audio", "keyframe"}:
            raise ValueError(f"Unknown segment kind: {self.kind}")
        if self.kind == "video":
            if not isinstance(self.size, tuple) or len(self.size) != 3:
                raise ValueError("Video size must be a (frames, height, width) tuple")
            dims = self.size
            if self.sparsity is not None and (
                not isinstance(self.sparsity, (int, float)) or not 0 <= self.sparsity <= 1
            ):
                raise ValueError("Video sparsity must be None or a finite number between 0 and 1")
        else:
            if not isinstance(self.size, int):
                raise ValueError("Dense segment size must be an integer")
            if self.sparsity is not None:
                raise ValueError("sparsity is only supported for video segments")
            dims = (self.size,)
        if any(not isinstance(n, int) or n < (1 if self.kind == "video" else 0) for n in dims):
            raise ValueError("Video dimensions must be positive; dense lengths must be nonnegative integers")

    @property
    def tokens(self):
        return math.prod(self.size) if isinstance(self.size, tuple) else self.size


@dataclass(frozen=True)
class SampleBlocks:
    """One sample's global block range and per-region KV budgets.

    Blocks always use [dense, VSA] order. dense_blocks_num counts the leading
    blocks from None-sparsity segments; video_spans covers the remaining blocks.
    """
    start: int
    end: int
    dense_blocks_num: int
    video_spans: tuple[tuple[int, int, int], ...]

    @property
    def dense_only(self):
        return not self.video_spans

    @property
    def blocks(self):
        return self.end - self.start

    @property
    def kept(self):
        return self.dense_blocks_num + sum(k for _, _, k in self.video_spans)


@dataclass
class _RingMetadata:
    """Distribution metadata only; all local attention geometry lives in VSALayout."""
    size: int
    rank: int
    global_blocks: int
    global_tokens: int
    local_groups: tuple[tuple[SampleBlocks, int, int], ...]
    block_ids: torch.Tensor
    peer_block_sizes: torch.Tensor
    global_to_gather: torch.Tensor
    _runtime: dict = field(default_factory=dict, repr=False)


@dataclass
class VSALayout:
    """Static sample/block geometry, token maps and prepared attention metadata.

    Construction lives in build_vsa_layout. __post_init__ builds contiguous
    BSA routing templates for head_num heads and inverse coarse-row groups.
    Dynamic Q/K routing lives in vsa_core.route and does not modify these templates.
    input_order="raster" means original sample/segment order (FHW for video):
    slot conversion permutes and pads. "block" means compact canonical blocks:
    slot conversion only pads, or is an identity when every block is full.
    ring=None describes a full input; otherwise the same maps describe the
    local compact shard. input_rows always indexes the global raster stream.
    groups and sample_slices describe global samples in both modes.
    """
    samples: tuple[tuple[Segment, ...], ...]
    head_num: int
    block_sizes: torch.Tensor
    row_slot: torch.Tensor
    slot_row: torch.Tensor
    groups: tuple[SampleBlocks, ...]
    coarse_blocks: torch.Tensor
    coarse_kv_blocks: torch.Tensor
    coarse_cu_q: torch.Tensor
    coarse_cu_kv: torch.Tensor
    coarse_max_q: int
    coarse_max_kv: int
    row_coarse: torch.Tensor
    input_rows: torch.Tensor
    input_order: str = "raster"
    dense_query_full: bool = False
    sample_slices: tuple = ()
    ring: _RingMetadata | None = None
    routing_indices: torch.Tensor = field(init=False, repr=False)
    routing_counts: torch.Tensor = field(init=False, repr=False)
    coarse_rows: torch.Tensor = field(init=False, repr=False)
    dense_only: bool = field(init=False)

    def __post_init__(self):
        """Build routing templates and coarse-row groups before attention runs."""
        if self.input_order not in ("raster", "block"):
            raise ValueError('input_order must be "raster" or "block"')
        if self.ring is not None and self.input_order != "block":
            raise ValueError('Ring input_order must be "block" (compact, without padding)')
        if not isinstance(self.head_num, int) or isinstance(self.head_num, bool) or self.head_num <= 0:
            raise ValueError("head_num must be a positive integer")
        if not isinstance(self.dense_query_full, bool):
            raise ValueError("dense_query_full must be a bool")
        device = self.block_sizes.device
        # Only exempt dense queries need capacity for every KV block.
        capacity = max(g.blocks if self.dense_query_full and g.dense_blocks_num else g.kept
                       for g in self.groups)
        index = torch.zeros((1, self.head_num, self.blocks, capacity), device=device, dtype=torch.int32)
        counts = torch.zeros((1, self.head_num, self.blocks), device=device, dtype=torch.int32)
        local_groups = (self.ring.local_groups if self.ring is not None else
                        ((g, g.start, g.end) for g in self.groups))
        for g, begin, end in local_groups:
            ids = torch.arange(g.start, g.end, device=device, dtype=torch.int32)
            if g.kept == g.blocks:
                index[:, :, begin:end, :g.blocks] = ids
                counts[:, :, begin:end] = g.blocks
                continue
            dense_end = g.start + g.dense_blocks_num
            if self.ring is not None:
                dense_end = min(end, max(begin, (
                    dense_end - self.ring.rank + self.ring.size - 1) // self.ring.size))
            routed_start = dense_end if self.dense_query_full else begin
            if g.dense_blocks_num:
                if self.dense_query_full:
                    index[:, :, begin:dense_end, :g.blocks] = ids
                    counts[:, :, begin:dense_end] = g.blocks
                index[:, :, routed_start:end, :g.dense_blocks_num] = ids[:g.dense_blocks_num]
            counts[:, :, routed_start:end] = g.kept
            cursor = g.dense_blocks_num
            for start, span_end, k in g.video_spans:
                if k == span_end - start:
                    index[:, :, routed_start:end, cursor:cursor+k] = ids[start-g.start:span_end-g.start]
                cursor += k
        self.routing_indices, self.routing_counts = index, counts
        # Input token rows per local coarse block; -1 marks padding for the merge adjoint.
        self.coarse_rows = self.slot_row.view(self.blocks, BLOCK)[self.coarse_blocks]
        # A rank with no local coarse queries must still join pooled collectives
        # when another rank owns a sample with VSA blocks.
        self.dense_only = all(g.dense_only for g in self.groups)

    @property
    def tokens(self):
        return self.row_slot.numel()

    @property
    def blocks(self):
        return self.block_sizes.numel()

    @property
    def padded_tokens(self):
        return self.blocks * BLOCK

    @property
    def all_blocks_full(self):
        # Full blocks can still require FHW reordering.
        return self.padded_tokens == self.tokens

    @property
    def slots_are_input(self):
        return self.input_order == "block" and self.all_blocks_full

    @property
    def sample_lengths(self):
        """Global sample token counts, including for a Ring shard."""
        return tuple(sum(r.stop-r.start for r in rows) for rows in self.sample_slices)


def _finalize_layout(samples, global_rows, groups, sample_slices, *, head_num,
                     dense_query_full, device, input_order, ring_size, ring_rank) -> VSALayout:
    """Materialize padding, inverse maps and coarse metadata from planned blocks.

    Each block lists its valid original token rows. Block order, sample bounds
    and KV budgets are already fixed; this stage only derives runtime metadata.
    """
    def tensor(values, dtype=torch.int64):
        return torch.tensor(values, device=device, dtype=dtype)

    ring = None
    block_rows = global_rows
    local_groups = [(g, g.start, g.end) for g in groups]
    if ring_size is not None:
        total = len(global_rows)
        capacity = (total + ring_size - 1) // ring_size
        owned = list(range(ring_rank, total, ring_size))
        block_rows = [global_rows[bid] for bid in owned]
        block_rows += [[] for _ in range(capacity - len(owned))]
        local_groups = []
        for g in groups:
            begin = max(0, (g.start - ring_rank + ring_size - 1) // ring_size)
            end = max(0, (g.end - ring_rank + ring_size - 1) // ring_size)
            if begin < end:
                local_groups.append((g, begin, end))
        sizes = [len(rows) for rows in global_rows]
        peer_sizes = [[sizes[rank + i * ring_size] if rank + i * ring_size < total else 0
                       for i in range(capacity)] for rank in range(ring_size)]
        ring = _RingMetadata(
            size=ring_size, rank=ring_rank, global_blocks=total, global_tokens=sum(sizes),
            local_groups=tuple(local_groups),
            block_ids=tensor(owned + [-1] * (capacity - len(owned))),
            # Each peer slice needs a 16-byte-aligned base for the CuTe frontend.
            peer_block_sizes=tensor([row + [0] * (-capacity % 4) for row in peer_sizes],
                                    torch.int32)[:, :capacity],
            global_to_gather=tensor([(bid % ring_size) * capacity + bid // ring_size
                                     for bid in range(total)]))
    token_count = sum(map(len, block_rows))
    input_rows = (list(range(token_count)) if input_order == "raster" else
                  [row for rows in block_rows for row in rows])
    if input_order == "block":
        offset = 0
        compact_blocks = []
        for rows in block_rows:
            compact_blocks.append(list(range(offset, offset + len(rows))))
            offset += len(rows)
        block_rows = compact_blocks
    slots = [r for rows in block_rows for r in rows + [-1] * (BLOCK-len(rows))]
    inverse = [0] * token_count
    for slot, row in enumerate(slots):
        if row >= 0:
            inverse[row] = slot
    coarse_q, coarse_kv, cu_q, cu_kv = [], [], [0], [0]
    for g, begin, end in local_groups:
        if not g.dense_only:
            coarse_q.extend(range(begin, end))
            coarse_kv.extend(range(g.start, g.end))
            cu_q.append(len(coarse_q))
            cu_kv.append(len(coarse_kv))
    # The final coarse row is a zero sentinel for every fine-only sample row.
    block_to_coarse = [len(coarse_q)] * len(block_rows)
    for i, bid in enumerate(coarse_q):
        block_to_coarse[bid] = i
    coarse_blocks = tensor(coarse_q)
    coarse_cu_q = tensor(cu_q, torch.int32)
    return VSALayout(
        samples=samples,
        head_num=head_num,
        dense_query_full=dense_query_full,
        block_sizes=tensor([len(rows) for rows in block_rows], torch.int32),
        row_slot=tensor(inverse, torch.int64),
        slot_row=tensor(slots, torch.int64),
        groups=tuple(groups),
        coarse_blocks=coarse_blocks,
        coarse_kv_blocks=coarse_blocks if ring is None else tensor(coarse_kv),
        coarse_cu_q=coarse_cu_q,
        coarse_cu_kv=coarse_cu_q if ring is None else tensor(cu_kv, torch.int32),
        coarse_max_q=max((b-a for a, b in zip(cu_q, cu_q[1:])), default=0),
        coarse_max_kv=max((b-a for a, b in zip(cu_kv, cu_kv[1:])), default=0),
        row_coarse=tensor([block_to_coarse[s//BLOCK] for s in inverse], torch.int64),
        input_rows=tensor(input_rows, torch.int64),
        input_order=input_order,
        sample_slices=tuple(sample_slices),
        ring=ring,
    )


def _segment_blocks(seg, offset):
    """Original token rows per block: dense chunks or numeric-sparsity FHW tiles."""
    if seg.sparsity is None:
        return [list(range(offset+p, offset+min(p+BLOCK, seg.tokens)))
                for p in range(0, seg.tokens, BLOCK)]
    t, h, w = seg.size
    return [[offset + (tt*h+hh)*w+ww
             for tt in range(ti, min(ti+4, t))
             for hh in range(hi, min(hi+4, h))
             for ww in range(wi, min(wi+4, w))]
            for ti in range(0, t, 4) for hi in range(0, h, 4) for wi in range(0, w, 4)]


def _build_sample_blocks(segments, *, token_offset, block_offset):
    """Return block rows, routing group and segment slices for one sample.

    Segment slices follow input order; blocks put dense chunks before VSA
    tiles. Both token rows and block spans use the supplied pack-wide offsets.
    """
    segment_slices = []
    for seg in segments:
        segment_slices.append(slice(token_offset, token_offset+seg.tokens))
        token_offset += seg.tokens
    block_rows, video_spans = [], []
    for dense in (True, False):
        for seg, token_slice in zip(segments, segment_slices):
            if (seg.sparsity is None) != dense:
                continue
            first = block_offset + len(block_rows)
            block_rows.extend(_segment_blocks(seg, token_slice.start))
            if not dense:
                end = block_offset + len(block_rows)
                keep = max(1, math.ceil((1-seg.sparsity)*(end-first)))
                video_spans.append((first, end, keep))
        if dense:
            dense_blocks_num = len(block_rows)
    group = SampleBlocks(start=block_offset, end=block_offset+len(block_rows),
                         dense_blocks_num=dense_blocks_num, video_spans=tuple(video_spans))
    return block_rows, group, tuple(segment_slices)


def build_vsa_layout(samples, *, head_num, dense_query_full=False, device="cpu",
                     input_order="raster", ring_size=None, ring_rank=None) -> VSALayout:
    """Build a full layout or an explicitly requested ring shard from Segments.

    input_order="raster" (the default) concatenates samples/segments
    in their original order, with FHW raster order inside each video.
    input_order="block" accepts valid rows in canonical block order without
    padding; the caller reorders once before the stack of VSA layers.
    input_rows maps either ordering to the original raster stream. The layout
    records the input order explicitly; it is never inferred from tensor shape.
    Samples never attend to each other.
    Each sample requires at least one video segment. None sparsity preserves
    segment-internal row order and adds only padding, regardless of frame count.
    Numeric sparsity enables FHW tiling and an independent KV budget per region.
    Samples with only None segments use full fine without pooling/coarse/gate;
    otherwise coarse includes all segments, including dense regions.
    dense_query_full=False (default) routes every query by the per-region KV
    budgets. True lets None-sparsity queries see all KV in their sample.
    None-sparsity KV is always fully visible. Coarse and gate rules are unchanged.
    head_num fixes the Q/K/V head count and the prepared BSA template shapes.
    Reuse the layout across calls with the same geometry, policy, heads and device.

    Omit ring_size/ring_rank for a full VSALayout. Provide both as group-local
    integers to populate layout.ring, including ring_size=1. Ring shards require
    input_order="block" and contain only valid owned-block rows without padding.
    The global block count must be at least ring_size, so every rank owns data.
    Individual samples may have fewer blocks; no divisibility is required.
    Reorder/distribute whole blocks at the dataset/model boundary. The builder
    needs no process group or distributed initialization; pass the matching
    explicit ring_cp_group to vsa_attention. Boundary reorder helpers still
    require a full block VSALayout.
    """
    if input_order not in ("raster", "block"):
        raise ValueError('input_order must be "raster" or "block"')
    if (ring_size is None) != (ring_rank is None):
        raise ValueError("ring_size and ring_rank must be provided together")
    is_ring = ring_size is not None
    if is_ring:
        if input_order != "block":
            raise ValueError('Ring input_order must be "block" (compact, without padding)')
        for name, number, minimum in (("head_num", head_num, 1), ("ring_size", ring_size, 1),
                                      ("ring_rank", ring_rank, 0)):
            if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if ring_rank >= ring_size:
            raise ValueError("ring_rank must be smaller than ring_size")
        if not isinstance(dense_query_full, bool):
            raise ValueError("dense_query_full must be a bool")
    samples, block_rows, groups, sample_slices = _plan_samples(samples)
    if is_ring:
        if len(block_rows) < ring_size:
            raise ValueError(
                "Ring requires at least ring_size global blocks; "
                f"got global_blocks={len(block_rows)}, ring_size={ring_size}")
    return _finalize_layout(samples, block_rows, groups, sample_slices,
                            head_num=head_num, dense_query_full=dense_query_full, device=device,
                            input_order=input_order, ring_size=ring_size, ring_rank=ring_rank)


def _plan_samples(samples):
    """Plan global blocks on the host without allocating a routing table."""
    try:
        samples = tuple(tuple(sample) for sample in samples)
    except TypeError as error:
        raise ValueError("samples must be a nested sequence of Segment objects") from error
    if not samples:
        raise ValueError("samples must be nonempty")
    block_rows, groups, sample_slices = [], [], []
    token_offset = 0
    for sample in samples:
        if not sample or any(not isinstance(seg, Segment) for seg in sample):
            raise ValueError("Each sample must be a nonempty sequence of Segment objects")
        if not any(seg.kind == "video" for seg in sample):
            raise ValueError("Each sample requires at least one video segment")
        sample_block_rows, group, segment_slices = _build_sample_blocks(
            sample, token_offset=token_offset, block_offset=len(block_rows))
        block_rows.extend(sample_block_rows)
        groups.append(group)
        sample_slices.append(segment_slices)
        token_offset = segment_slices[-1].stop
    return samples, block_rows, groups, sample_slices
