"""Memory-efficient reverse index builder for block-sparse varlen backward.

Forward PackGQA CSR (k_block_m=128, k_block_n=64), KV-head organized:
    fwd segment id = total_q_tiles * h_kv + (cu_q_tiles[b] + packed_m128)
    value = local k64 block id, ascending per segment

Backward K-centric index consumed by the SM90 bwd kernel:
    bwd segment id = (h_kv * batch + b) * max_k_tiles + k64_block
    value = local packed group id (e), each group covering ``bwd_block_m``
    packed rows (the bwd kernel's kBlockM, default 64), **sorted ascending
    within each segment** so entries sharing the same q tile
    (m_block = e // gqa_ratio) are contiguous.

Each forward q128 tile expands to EXPAND = 128 // bwd_block_m consecutive
packed groups. The bwd kernel (per-Q-head CTA) merges the up-to-gqa_ratio
entries of one q tile into a single bwd_step with a row bitmap; a row q_pos
of head h_q = h_kv*ratio + hig is selected iff its packed group
    e = (q_pos * gqa_ratio + hig) // bwd_block_m
is present in the segment (bitmap bit e % gqa_ratio).

CUDA path is count + fill + one global sort (no host sync).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bs_bwd_count_kernel(SEG_HKV, SEG_B, BS_IDX, BWD_CU, batch, max_k_tiles, E, EXPAND: tl.constexpr):
    j = tl.program_id(0)
    if j >= E:
        return
    h_kv = tl.load(SEG_HKV + j).to(tl.int64)
    b = tl.load(SEG_B + j).to(tl.int64)
    k_block = tl.load(BS_IDX + j).to(tl.int64)
    sid = (h_kv * batch + b) * max_k_tiles + k_block
    tl.atomic_add(BWD_CU + sid + 1, EXPAND)


@triton.jit
def _bs_bwd_fill_kernel(SEG_HKV, SEG_B, LOCAL_P, BS_IDX, OFFSET, BWD_IDX, batch, max_k_tiles, E, EXPAND: tl.constexpr):
    j = tl.program_id(0)
    if j >= E:
        return
    h_kv = tl.load(SEG_HKV + j).to(tl.int64)
    b = tl.load(SEG_B + j).to(tl.int64)
    k_block = tl.load(BS_IDX + j).to(tl.int64)
    local_p = tl.load(LOCAL_P + j)
    sid = (h_kv * batch + b) * max_k_tiles + k_block
    pos = tl.atomic_add(OFFSET + sid, EXPAND)
    for s in tl.static_range(EXPAND):
        tl.store(BWD_IDX + pos + s, local_p * EXPAND + s)


def _entry_metadata(block_sparse_cu, block_sparse_idx, cu_q_tiles, num_heads_kv, total_q_tiles):
    device = block_sparse_cu.device
    counts = (block_sparse_cu[1:] - block_sparse_cu[:-1]).to(torch.int64)
    num_fwd_segments = num_heads_kv * total_q_tiles
    seg_id = torch.repeat_interleave(
        torch.arange(num_fwd_segments, dtype=torch.int64, device=device), counts
    )
    h_kv = (seg_id // total_q_tiles).to(torch.int32)
    global_p = seg_id % total_q_tiles
    b = torch.searchsorted(cu_q_tiles.to(torch.int64), global_p, right=True) - 1
    local_p = (global_p - cu_q_tiles.to(torch.int64)[b]).to(torch.int32)
    return h_kv, b.to(torch.int32), local_p


def _segment_ids_from_cu(bwd_cu, num_segments, device):
    counts = (bwd_cu[1:] - bwd_cu[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(num_segments, dtype=torch.int64, device=device), counts
    )


def _sort_entries_within_segments(bwd_cu, bwd_idx, num_segments):
    """Sort entries ascending within each segment (stable grouping by m_block).

    Uses one global int64 sort on key = sid * 2^20 + value, which preserves
    segment boundaries (bwd_cu unchanged) and sorts values inside segments.
    """
    if bwd_idx.numel() == 0:
        return bwd_idx
    device = bwd_idx.device
    sids = _segment_ids_from_cu(bwd_cu, num_segments, device)
    key = sids * (1 << 20) + bwd_idx.to(torch.int64)
    _, order = torch.sort(key)
    return bwd_idx[order]


def _build_cpu_fallback(
    block_sparse_cu, block_sparse_idx, cu_q_tiles, batch_size,
    num_heads_kv, total_q_tiles, max_k_tiles, expand,
):
    device = block_sparse_cu.device
    num_bwd_segments = num_heads_kv * batch_size * max_k_tiles
    h_kv, b, local_p = _entry_metadata(
        block_sparse_cu, block_sparse_idx, cu_q_tiles, num_heads_kv, total_q_tiles
    )
    e = block_sparse_idx.numel()
    if e == 0:
        return (
            torch.zeros(num_bwd_segments + 1, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            max_k_tiles,
        )
    sid = (h_kv.to(torch.int64) * batch_size + b.to(torch.int64)) * max_k_tiles + block_sparse_idx.to(torch.int64)
    # Each forward entry expands to `expand` consecutive packed groups.
    sid_e = sid[:, None].expand(-1, expand).reshape(-1)
    val_e = (local_p.to(torch.int64)[:, None] * expand
             + torch.arange(expand, device=device, dtype=torch.int64)[None, :]).reshape(-1)
    key = torch.unique(sid_e * (1 << 20) + val_e)
    sid_sorted = key >> 20
    bwd_idx = (key & ((1 << 20) - 1)).to(torch.int32)
    counts = torch.bincount(sid_sorted, minlength=num_bwd_segments).to(torch.int32)
    bwd_cu = torch.zeros(num_bwd_segments + 1, dtype=torch.int32, device=device)
    bwd_cu[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    return bwd_cu, bwd_idx, max_k_tiles


def build_block_sparse_bwd_index(
    block_sparse_cu: torch.Tensor,
    block_sparse_idx: torch.Tensor,
    cu_q_tiles: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    total_q_tiles: int,
    max_seqlen_k: int,
    k_block_m: int = 128,
    k_block_n: int = 64,
    bwd_block_m: int = 64,
):
    """Returns (bwd_cu, bwd_idx, max_k_tiles).

    bwd segment id = (h_kv * batch + b) * max_k_tiles + k64_block.
    bwd_idx stores packed group ids of ``bwd_block_m`` packed rows each,
    **sorted ascending within each segment** so entries of the same q tile
    (m_block = entry // gqa_ratio) are contiguous for in-kernel merging.
    ``bwd_block_m`` must match the backward kernel's kBlockM.
    """
    device = block_sparse_cu.device
    if k_block_m != 128 or k_block_n != 64:
        raise NotImplementedError("block sparse backward index currently supports k_block_m=128, k_block_n=64 only")
    if k_block_m % bwd_block_m != 0:
        raise ValueError("bwd_block_m must divide k_block_m")
    if num_heads_q % num_heads_kv != 0:
        raise NotImplementedError("num_heads_q must be divisible by num_heads_kv")
    expand = k_block_m // bwd_block_m

    batch_size = cu_seqlens_q.numel() - 1
    total_q_tiles = int(total_q_tiles)
    max_k_tiles = (int(max_seqlen_k) + k_block_n - 1) // k_block_n
    if block_sparse_cu.numel() != num_heads_kv * total_q_tiles + 1:
        raise ValueError("block_sparse_cu does not match PackGQA forward CSR shape")
    if cu_q_tiles.numel() != batch_size + 1:
        raise ValueError("cu_q_tiles batch mismatch")

    if device.type != "cuda":
        return _build_cpu_fallback(
            block_sparse_cu, block_sparse_idx, cu_q_tiles, batch_size,
            num_heads_kv, total_q_tiles, max_k_tiles, expand,
        )

    e = int(block_sparse_idx.numel())
    num_bwd_segments = num_heads_kv * batch_size * max_k_tiles
    if e == 0:
        return (
            torch.zeros(num_bwd_segments + 1, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            max_k_tiles,
        )

    # Single-pass torch path: expand to (sid, value) keys, then one unique()
    # produces both the ascending order and dedup, replacing the previous
    # count-kernel + fill-kernel + global-sort pipeline.
    h_kv, b, local_p = _entry_metadata(
        block_sparse_cu, block_sparse_idx, cu_q_tiles, num_heads_kv, total_q_tiles
    )
    sid = (h_kv.to(torch.int64) * batch_size + b.to(torch.int64)) * max_k_tiles \
        + block_sparse_idx.to(torch.int64)
    sid_e = sid[:, None].expand(-1, expand).reshape(-1)
    val_e = (local_p.to(torch.int64)[:, None] * expand
             + torch.arange(expand, device=device, dtype=torch.int64)[None, :]).reshape(-1)
    key = torch.unique(sid_e * (1 << 20) + val_e)
    sids_sorted = key >> 20
    bwd_idx = (key & ((1 << 20) - 1)).to(torch.int32)
    counts = torch.bincount(sids_sorted, minlength=num_bwd_segments).to(torch.int32)
    bwd_cu = torch.zeros(num_bwd_segments + 1, dtype=torch.int32, device=device)
    bwd_cu[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    return bwd_cu, bwd_idx, max_k_tiles
