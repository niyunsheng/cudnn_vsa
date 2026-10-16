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
    Dynamic Q/K routing lives in vsa.route and does not modify these templates.
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
    def dense_only(self):
        return not self.coarse_lengths

    @property
    def sample_lengths(self):
        return tuple(sum(r.stop-r.start for r in rows) for rows in self.sample_slices)


def _finalize_layout(samples, block_rows, groups, sample_slices, *, head_num,
                     dense_query_full, device) -> VSALayout:
    """Materialize padding, inverse maps and coarse metadata from planned blocks.

    Each block lists its valid original token rows. Block order, sample bounds
    and KV budgets are already fixed; this stage only derives runtime metadata.
    """
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


def build_vsa_layout(samples, *, head_num, dense_query_full=False, device="cpu") -> VSALayout:
    """Public layout builder: ordered samples of Segments -> VSALayout.

    Concatenate samples in list order and segments in their original order;
    video/image rows use FHW raster order. Samples never attend to each other.
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
    return _finalize_layout(samples, block_rows, groups, sample_slices,
                            head_num=head_num, dense_query_full=dense_query_full, device=device)
