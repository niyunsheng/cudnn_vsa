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
class VSALayout:
    """Static sample/block geometry, token maps and prepared attention metadata.

    Construction lives in build_vsa_layout. __post_init__ builds contiguous
    BSA routing templates for head_num heads and inverse coarse-row groups.
    Dynamic Q/K routing lives in vsa_core.route and does not modify these templates.
    """
    samples: tuple[tuple[Segment, ...], ...]
    head_num: int
    block_sizes: torch.Tensor
    row_slot: torch.Tensor
    slot_row: torch.Tensor
    groups: tuple[SampleBlocks, ...]
    coarse_blocks: torch.Tensor
    coarse_cu_seqlens: torch.Tensor
    coarse_lengths: tuple[int, ...]
    row_coarse: torch.Tensor
    input_rows: torch.Tensor
    input_order: str = "raster"
    dense_query_full: bool = False
    sample_slices: tuple = ()
    routing_indices: torch.Tensor = field(init=False, repr=False)
    routing_counts: torch.Tensor = field(init=False, repr=False)
    coarse_rows: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        """Build routing templates and coarse-row groups before attention runs."""
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
        for g in self.groups:
            ids = torch.arange(g.start, g.end, device=device, dtype=torch.int32)
            if g.kept == g.blocks:
                index[:, :, g.start:g.end, :g.blocks] = ids
                counts[:, :, g.start:g.end] = g.blocks
                continue
            dense_end = g.start + g.dense_blocks_num
            routed_start = dense_end if self.dense_query_full else g.start
            if g.dense_blocks_num:
                if self.dense_query_full:
                    index[:, :, g.start:dense_end, :g.blocks] = ids
                    counts[:, :, g.start:dense_end] = g.blocks
                index[:, :, routed_start:g.end, :g.dense_blocks_num] = ids[:g.dense_blocks_num]
            counts[:, :, routed_start:g.end] = g.kept
            cursor = g.dense_blocks_num
            for start, end, k in g.video_spans:
                if k == end - start:
                    index[:, :, routed_start:g.end, cursor:cursor+k] = ids[start-g.start:end-g.start]
                cursor += k
        self.routing_indices, self.routing_counts = index, counts
        # Original token rows per coarse block; -1 marks padding for the merge adjoint.
        self.coarse_rows = self.slot_row.view(self.blocks, BLOCK)[self.coarse_blocks]

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
    def dense_only(self):
        return not self.coarse_lengths

    @property
    def sample_lengths(self):
        return tuple(sum(r.stop-r.start for r in rows) for rows in self.sample_slices)


def _finalize_layout(samples, block_rows, groups, sample_slices, *, head_num,
                     dense_query_full, device, input_order="raster") -> VSALayout:
    """Materialize padding, inverse maps and coarse metadata from planned blocks.

    Each block lists its valid original token rows. Block order, sample bounds
    and KV budgets are already fixed; this stage only derives runtime metadata.
    """
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
    inverse = [0] * sum(len(rows) for rows in block_rows)
    for slot, row in enumerate(slots):
        if row >= 0:
            inverse[row] = slot
    coarse_ids, cu = [], [0]
    for g in groups:
        if not g.dense_only:
            coarse_ids.extend(range(g.start, g.end))
            cu.append(len(coarse_ids))
    # The final coarse row is a zero sentinel for every fine-only sample row.
    block_to_coarse = [len(coarse_ids)] * len(block_rows)
    for i, bid in enumerate(coarse_ids):
        block_to_coarse[bid] = i
    def tensor(x, dtype):
        return torch.tensor(x, device=device, dtype=dtype)
    return VSALayout(
        samples=samples,
        head_num=head_num,
        dense_query_full=dense_query_full,
        block_sizes=tensor([len(rows) for rows in block_rows], torch.int32),
        row_slot=tensor(inverse, torch.int64),
        slot_row=tensor(slots, torch.int64),
        groups=tuple(groups),
        coarse_blocks=tensor(coarse_ids, torch.int64),
        coarse_cu_seqlens=tensor(cu, torch.int32),
        coarse_lengths=tuple(b-a for a, b in zip(cu, cu[1:])),
        row_coarse=tensor([block_to_coarse[s//BLOCK] for s in inverse], torch.int64),
        input_rows=tensor(input_rows, torch.int64),
        input_order=input_order,
        sample_slices=tuple(sample_slices),
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
                     input_order="raster") -> VSALayout:
    """Public layout builder: ordered samples of Segments -> VSALayout.

    input_order="raster" concatenates samples/segments in their original order.
    input_order="block" accepts valid rows in canonical block order without
    padding. input_rows maps either ordering to the original raster stream.
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
    """
    if input_order not in ("raster", "block"):
        raise ValueError('input_order must be "raster" or "block"')
    samples, block_rows, groups, sample_slices = _plan_samples(samples)
    return _finalize_layout(samples, block_rows, groups, sample_slices,
                            head_num=head_num, dense_query_full=dense_query_full, device=device,
                            input_order=input_order)


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


@dataclass
class RingVSALayout:
    """Packed THD ownership after any external Ulysses all-to-all.

    ``input_rows`` indexes the original global sample-major raster stream.
    All other row maps index the rank's compact THD input. Communication
    buffers have ``blocks * 64`` rows, including any dummy trailing block.
    """
    samples: tuple
    head_num: int
    ring_size: int
    ring_rank: int
    global_blocks: int
    global_tokens: int
    groups: tuple
    sample_slices: tuple
    local_groups: tuple
    input_rows: torch.Tensor
    block_ids: torch.Tensor
    block_sizes: torch.Tensor
    peer_block_sizes: torch.Tensor
    global_to_gather: torch.Tensor
    slot_row: torch.Tensor
    row_slot: torch.Tensor
    routing_indices: torch.Tensor
    routing_counts: torch.Tensor
    coarse_blocks: torch.Tensor
    coarse_kv_blocks: torch.Tensor
    coarse_cu_q: torch.Tensor
    coarse_cu_kv: torch.Tensor
    coarse_max_q: int
    coarse_max_kv: int
    row_coarse: torch.Tensor
    coarse_rows: torch.Tensor
    dense_only: bool
    dense_query_full: bool
    input_order: str = "raster"
    _runtime: dict = field(default_factory=dict, repr=False)

    @property
    def tokens(self):
        return self.input_rows.numel()

    @property
    def blocks(self):
        return self.block_ids.numel()

    @property
    def padded_tokens(self):
        return self.blocks * BLOCK

    @property
    def slots_are_input(self):
        return self.input_order == "block" and self.tokens == self.padded_tokens


def build_ring_vsa_layout(samples, *, head_num, ring_size, ring_rank,
                          dense_query_full=False, device="cpu", input_order="raster") -> RingVSALayout:
    """Distribute complete sample-local VSA blocks cyclically across a ring.

    This builder requires no distributed initialization. Pass group-local rank
    and size, then pass that same explicit process group to ``vsa_attention``.
    A peer can own no rows of a particular sample, or even no real blocks.
    "block" accepts compact owned-block order without padding; "raster" keeps
    original raster order restricted to owned blocks. input_rows maps either
    local ordering to global raster coordinates.
    """
    if input_order not in ("raster", "block"):
        raise ValueError('input_order must be "raster" or "block"')
    for name, number, minimum in (("head_num", head_num, 1), ("ring_size", ring_size, 1),
                                  ("ring_rank", ring_rank, 0)):
        if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if ring_rank >= ring_size:
        raise ValueError("ring_rank must be smaller than ring_size")
    if not isinstance(dense_query_full, bool):
        raise ValueError("dense_query_full must be a bool")
    samples, global_rows, groups, sample_slices = _plan_samples(samples)
    total = len(global_rows)
    capacity = (total + ring_size - 1) // ring_size
    owned = list(range(ring_rank, total, ring_size))
    input_rows = [row for bid in owned for row in global_rows[bid]]
    if input_order == "raster":
        input_rows.sort()
    local_row = {row: i for i, row in enumerate(input_rows)}
    blocks = [[local_row[row] for row in global_rows[bid]] for bid in owned]
    blocks += [[] for _ in range(capacity - len(blocks))]
    slots = [row for rows in blocks for row in rows + [-1] * (BLOCK - len(rows))]
    inverse = [0] * len(input_rows)
    for slot, row in enumerate(slots):
        if row >= 0:
            inverse[row] = slot
    max_keys = max(g.blocks if dense_query_full and g.dense_blocks_num else g.kept
                   for g in groups)
    index = torch.zeros((1, head_num, capacity, max_keys), dtype=torch.int32)
    counts = torch.zeros((1, head_num, capacity), dtype=torch.int32)
    local_groups, coarse_q, coarse_kv, cu_q, cu_kv = [], [], [], [0], [0]
    for g in groups:
        local_start = max(0, (g.start - ring_rank + ring_size - 1) // ring_size)
        local_end = max(0, (g.end - ring_rank + ring_size - 1) // ring_size)
        if local_start == local_end:
            continue
        local_groups.append((g, local_start, local_end))
        for local_bid in range(local_start, local_end):
            global_bid = ring_rank + ring_size * local_bid
            full = g.kept == g.blocks or (
                dense_query_full and global_bid < g.start + g.dense_blocks_num)
            counts[:, :, local_bid] = g.blocks if full else g.kept
            if full:
                index[:, :, local_bid, :g.blocks] = torch.arange(g.start, g.end)
            else:
                cursor = g.dense_blocks_num
                index[:, :, local_bid, :cursor] = torch.arange(g.start, g.start + cursor)
                for start, end, keep in g.video_spans:
                    if keep == end - start:
                        index[:, :, local_bid, cursor:cursor + keep] = torch.arange(start, end)
                    cursor += keep
        if not g.dense_only:
            coarse_q.extend(range(local_start, local_end))
            coarse_kv.extend(range(g.start, g.end))
            cu_q.append(len(coarse_q))
            cu_kv.append(len(coarse_kv))
    block_to_coarse = [len(coarse_q)] * capacity
    for i, bid in enumerate(coarse_q):
        block_to_coarse[bid] = i
    sizes = [len(rows) for rows in global_rows]
    peer_sizes = [[sizes[rank + i * ring_size] if rank + i * ring_size < total else 0
                   for i in range(capacity)] for rank in range(ring_size)]
    def tensor(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device=device)
    slot_row = tensor(slots)
    coarse_blocks = tensor(coarse_q)
    return RingVSALayout(
        samples=samples, head_num=head_num, ring_size=ring_size, ring_rank=ring_rank,
        global_blocks=total, global_tokens=sum(len(rows) for rows in global_rows),
        groups=tuple(groups), sample_slices=tuple(sample_slices), local_groups=tuple(local_groups),
        input_rows=tensor(input_rows), block_ids=tensor(owned + [-1] * (capacity - len(owned))),
        block_sizes=tensor(peer_sizes[ring_rank], torch.int32),
        # Each peer slice needs a 16-byte-aligned base for the CuTe frontend.
        peer_block_sizes=tensor([row + [0] * (-capacity % 4) for row in peer_sizes],
                                torch.int32)[:, :capacity],
        global_to_gather=tensor([(bid % ring_size) * capacity + bid // ring_size for bid in range(total)]),
        slot_row=slot_row, row_slot=tensor(inverse),
        routing_indices=index.to(device), routing_counts=counts.to(device),
        coarse_blocks=coarse_blocks, coarse_kv_blocks=tensor(coarse_kv),
        coarse_cu_q=tensor(cu_q, torch.int32), coarse_cu_kv=tensor(cu_kv, torch.int32),
        coarse_max_q=max((b-a for a, b in zip(cu_q, cu_q[1:])), default=0),
        coarse_max_kv=max((b-a for a, b in zip(cu_kv, cu_kv[1:])), default=0),
        row_coarse=tensor([block_to_coarse[s // BLOCK] for s in inverse]),
        coarse_rows=slot_row.view(capacity, BLOCK)[coarse_blocks],
        dense_only=all(g.dense_only for g in groups), dense_query_full=dense_query_full,
        input_order=input_order)
