"""PackGQA block-sparse index builder for paged FlashPrefill KV caches.

The returned tuple is directly accepted by ``flash_attn_with_kvcache``::

    block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles

CSR segment layout::

    segment = kv_head * total_q_tiles + cu_q_tiles[batch] + m_block

A PackGQA M tile contains ``k_block_m`` packed rows. Packed row ``r`` maps to
``q_position = r // gqa_ratio`` and ``q_head = kv_head * gqa_ratio + r % gqa_ratio``.
K indices are logical ``k_block_n``-sized blocks, not physical page IDs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)

# NVIDIA H20 global winners over BF16/FP8, Q/KV heads 32/8, 16/4, 24/2,
# and sequence lengths 500, 4096, 16384, and 65536.
_MEAN_NUM_WARPS = 2
_MEAN_NUM_STAGES = 5
_PAGE1_BF16_MEAN_NUM_WARPS = 2
_PAGE1_BF16_MEAN_NUM_STAGES = 4
_PAGE1_FP8_MEAN_NUM_WARPS = 1
_PAGE1_FP8_MEAN_NUM_STAGES = 4
_SCORE_K_TILE = 16
_SCORE_NUM_WARPS = 4
_SCORE_NUM_STAGES = 1
_NORMALIZE_K_TILE = 256
_NORMALIZE_NUM_WARPS = 1
_NORMALIZE_NUM_STAGES = 4
# The attention kernel (mainloop_fwd_sm90) consumes each CSR entry as one
# 64-token n_block. Selection may use a larger logical block
# k_block_n = _ATTN_TILE_N * n; each selected logical block is expanded into
# n consecutive physical tiles when compacted to CSR.
_ATTN_TILE_N = 64


@triton.jit
def _fill_cu_q_tiles_kernel(
    cu_seqlens_q_ptr,
    cu_q_tiles_ptr,
    batch_size: tl.constexpr,
    gqa_ratio: tl.constexpr,
    K_BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < batch_size
    q_begin = tl.load(cu_seqlens_q_ptr + offsets, mask=valid, other=0)
    q_end = tl.load(cu_seqlens_q_ptr + offsets + 1, mask=valid, other=0)
    q_tiles = tl.where(
        valid,
        (q_end - q_begin) * gqa_ratio + K_BLOCK_M - 1,
        0,
    ) // K_BLOCK_M
    tl.store(cu_q_tiles_ptr, 0)
    tl.store(
        cu_q_tiles_ptr + offsets + 1,
        tl.cumsum(q_tiles, axis=0),
        mask=valid,
    )


@triton.jit
def _paged_k_mean_kernel(
    k_ptr,
    k_mean_ptr,
    v_ptr,
    v_mean_ptr,
    page_table_ptr,
    kv_seqlens_ptr,
    cu_seqlens_q_ptr,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_pt_batch,
    stride_pt_page,
    stride_mean_batch,
    stride_mean_block,
    stride_mean_head,
    stride_mean_dim,
    stride_vmean_batch,
    stride_vmean_block,
    stride_vmean_head,
    stride_vmean_dim,
    num_kv_heads: tl.constexpr,
    gqa_ratio: tl.constexpr,
    page_size: tl.constexpr,
    max_k_blocks: tl.constexpr,
    min_sparse_q_len,
    last_n_blocks,
    K_BLOCK_M: tl.constexpr,
    K_BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    COMPUTE_V_MEAN: tl.constexpr,
):
    k_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_kv_heads
    kv_head = batch_head % num_kv_heads

    q_begin = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    q_len = q_end - q_begin
    q_tile_count = tl.cdiv(q_len * gqa_ratio, K_BLOCK_M)
    if q_len <= min_sparse_q_len or q_tile_count <= last_n_blocks:
        return

    kv_len = tl.load(kv_seqlens_ptr + batch)
    if k_block * K_BLOCK_N >= kv_len:
        return
    token_offsets = k_block * K_BLOCK_N + tl.arange(0, K_BLOCK_N)
    dim_offsets = tl.arange(0, HEAD_DIM)
    valid_token = token_offsets < kv_len

    logical_page = token_offsets // page_size
    page_offset = token_offsets % page_size
    physical_page = tl.load(
        page_table_ptr + batch * stride_pt_batch + logical_page * stride_pt_page,
        mask=valid_token,
        other=0,
    )

    k = tl.load(
        k_ptr
        + physical_page[:, None] * stride_k_page
        + page_offset[:, None] * stride_k_token
        + kv_head * stride_k_head
        + dim_offsets[None, :] * stride_k_dim,
        mask=valid_token[:, None],
        other=0.0,
    )
    # Mean pool over the block (per-dim): invalid rows are zero-filled by
    # the masked load above, so a plain sum / valid count gives the mean.
    token_count = tl.sum(valid_token.to(tl.float32), axis=0)
    denom = tl.maximum(token_count, 1.0)
    k_f32 = k.to(tl.float32)
    pooled = tl.sum(k_f32, axis=0) / denom

    store_mask = (k_block < max_k_blocks) & (dim_offsets < HEAD_DIM)
    tl.store(
        k_mean_ptr
        + batch * stride_mean_batch
        + k_block * stride_mean_block
        + kv_head * stride_mean_head
        + dim_offsets * stride_mean_dim,
        pooled,
        mask=store_mask,
    )
    if COMPUTE_V_MEAN:
        # Mean pool V over the same block for the zero-order correction of
        # unselected blocks in the attention epilogue.
        vdim_offsets = tl.arange(0, HEAD_DIM_V)
        v = tl.load(
            v_ptr
            + physical_page[:, None] * stride_v_page
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + vdim_offsets[None, :] * stride_v_dim,
            mask=valid_token[:, None],
            other=0.0,
        )
        v_pooled = tl.sum(v.to(tl.float32), axis=0) / denom
        vstore_mask = (k_block < max_k_blocks) & (vdim_offsets < HEAD_DIM_V)
        tl.store(
            v_mean_ptr
            + batch * stride_vmean_batch
            + k_block * stride_vmean_block
            + kv_head * stride_vmean_head
            + vdim_offsets * stride_vmean_dim,
            v_pooled,
            mask=vstore_mask,
        )


@triton.jit
def _packgqa_score_select_kernel(
    q_ptr,
    k_mean_ptr,
    out_index_ptr,
    out_count_ptr,
    chunk_max_ptr,
    cu_seqlens_q_ptr,
    kv_seqlens_ptr,
    cu_q_tiles_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_mean_batch,
    stride_mean_block,
    stride_mean_head,
    stride_mean_dim,
    stride_index_head,
    stride_index_tile,
    stride_index_k,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_ratio: tl.constexpr,
    total_q_tiles,
    max_k_blocks: tl.constexpr,
    min_sparse_q_len,
    last_n_blocks,
    scale_log2: tl.constexpr,
    abs_threshold,
    attention_sink,
    window_size,
    IS_CAUSAL: tl.constexpr,
    K_BLOCK_M: tl.constexpr,
    K_BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_TILE: tl.constexpr,
):
    # Fused per-tile scoring + max-based block selection (paper Sec. 3.2/3.4).
    # Block scores are tile-level softmax energies summed over packed rows,
    #   S[b] = sum_rows 2^(qk[row, b] - M_tile),
    # accumulated online in the SINGLE GEMM pass with a running tile max
    # (per-chunk scalar corrections recover the exact tile-referenced sums),
    # and selected with thresh = abs_threshold * max_b S[b]. Per-block sums
    # are bit-cast into the compact index buffer; no score buffer and no
    # second GEMM pass are needed.
    m_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_kv_heads
    kv_head = batch_head % num_kv_heads

    q_tile_begin = tl.load(cu_q_tiles_ptr + batch)
    q_tile_end = tl.load(cu_q_tiles_ptr + batch + 1)
    q_tile_count = q_tile_end - q_tile_begin
    if m_block >= q_tile_count:
        return

    q_begin = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(kv_seqlens_ptr + batch)
    prefix_len = kv_len - q_len
    global_q_tile = q_tile_begin + m_block

    # A PackGQA tile may begin or end in the middle of a GQA group. Derive
    # the first/last query positions from the exact packed-row interval
    # rather than assuming K_BLOCK_M is divisible by gqa_ratio.
    packed_begin = m_block * K_BLOCK_M
    packed_end = tl.minimum((m_block + 1) * K_BLOCK_M, q_len * gqa_ratio)
    q_first = packed_begin // gqa_ratio
    q_last = (packed_end - 1) // gqa_ratio
    first_k = (prefix_len + q_first) // K_BLOCK_N
    last_k = (prefix_len + q_last) // K_BLOCK_N
    full_last_tile = m_block >= tl.maximum(q_tile_count - last_n_blocks, 0)

    out_base = (
        out_index_ptr
        + kv_head * stride_index_head
        + global_q_tile * stride_index_tile
    )
    out_count_base = out_count_ptr + kv_head * total_q_tiles + global_q_tile

    # Physical-tile accounting (see _compact_to_csr_kernel): the sequence-tail
    # logical block is tagged with its valid tile count in the high bits
    # (e_tail << 20) and counts are stored as physical tile counts, so the
    # CSR never contains a fully-invalid tile.
    n_sub: tl.constexpr = K_BLOCK_N // 64
    last_logical = (kv_len - 1) // K_BLOCK_N
    e_tail = (kv_len + 63) // 64 - last_logical * n_sub

    # Dense path: short sequences and the last last_n_blocks q tiles keep all
    # causally visible blocks without scoring.
    if q_len <= min_sparse_q_len or full_last_tile:
        count = tl.zeros((), tl.int32)
        tail_cnt = tl.zeros((), tl.int32)
        for k_base in range(0, max_k_blocks, K_TILE):
            if k_base * K_BLOCK_N < kv_len:
                k_blocks = k_base + tl.arange(0, K_TILE)
                keep = (k_blocks < max_k_blocks) & (k_blocks * K_BLOCK_N < kv_len)
                if IS_CAUSAL:
                    keep &= k_blocks <= last_k
                keep_i32 = keep.to(tl.int32)
                tail_cnt += tl.sum(tl.where((k_blocks == last_logical) & keep, 1, 0), axis=0)
                local_offset = tl.cumsum(keep_i32, axis=0) - 1
                tl.store(
                    out_base + (count + local_offset) * stride_index_k,
                    tl.where(k_blocks == last_logical, k_blocks | (e_tail << 20), k_blocks),
                    mask=keep,
                )
                count += tl.sum(keep_i32, axis=0)
        tl.store(out_count_base, count * n_sub - tail_cnt * (n_sub - e_tail))
        return

    packed = packed_begin + tl.arange(0, K_BLOCK_M)
    q_pos = packed // gqa_ratio
    q_head_in_group = packed % gqa_ratio
    q_head = kv_head * gqa_ratio + q_head_in_group
    valid_q = (q_pos < q_len) & (q_head < num_q_heads)
    dims = tl.arange(0, HEAD_DIM)

    q = tl.load(
        q_ptr
        + (q_begin + q_pos)[:, None] * stride_q_token
        + q_head[:, None] * stride_q_head
        + dims[None, :] * stride_q_dim,
        mask=valid_q[:, None],
        other=0.0,
    )
    max_visible_token = prefix_len + q_last
    if IS_CAUSAL:
        active_k_blocks = (max_visible_token + 1) // K_BLOCK_N
    else:
        active_k_blocks = tl.cdiv(kv_len, K_BLOCK_N)

    # Paper-form block scoring (Sec. 3.2, fused single GEMM pass). A k block
    # contributes to a row iff its FIRST token is visible to that row
    # (prefix_len + q_pos >= k_block * K_BLOCK_N). The block score is the
    # tile-level softmax energy
    #   S[b] = sum_rows exp2(qk[row, b] - M_tile),
    # accumulated online with a running tile max: each chunk's sums are
    # computed against the running max M_c, bit-cast into the compact index
    # buffer, and M_c is stored per chunk in chunk_max_ptr. Since every block
    # belongs to exactly one chunk, the selection side recovers the exact
    # tile-referenced sums via the per-chunk correction exp2(M_c - M_final);
    # no second GEMM pass and no rescale of previously stored data. run_max
    # starts at -1e30 (finite) so fully-masked lanes contribute exp2(-inf)=0.
    chunk_stride: tl.constexpr = tl.cdiv(max_k_blocks, K_TILE)
    chunk_base = chunk_max_ptr + (kv_head * total_q_tiles + global_q_tile) * chunk_stride
    run_max = tl.full((), -1e30, tl.float32)
    for k_base in range(0, active_k_blocks, K_TILE):
        k_blocks = k_base + tl.arange(0, K_TILE)
        valid_k = (
            (k_blocks < active_k_blocks)
            & (k_blocks < max_k_blocks)
            & (k_blocks * K_BLOCK_N < kv_len)
        )
        k = tl.load(
            k_mean_ptr
            + batch * stride_mean_batch
            + k_blocks[:, None] * stride_mean_block
            + kv_head * stride_mean_head
            + dims[None, :] * stride_mean_dim,
            mask=valid_k[:, None],
            other=0.0,
        )
        # q and k remain in their input dtype here. tl.dot therefore selects
        # BF16 tensor-core math for BF16 and FP8 tensor-core math for E4M3.
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale_log2
        if IS_CAUSAL:
            score_mask = valid_q[:, None] & valid_k[None, :] & (
                prefix_len + q_pos[:, None] >= k_blocks[None, :] * K_BLOCK_N
            )
        else:
            score_mask = valid_q[:, None] & valid_k[None, :]
        qk = tl.where(score_mask, qk, float("-inf"))
        run_max = tl.maximum(run_max, tl.max(qk))
        block_energy = tl.sum(tl.exp2(qk - run_max), axis=0)
        tl.store(
            out_base + k_blocks * stride_index_k,
            block_energy.to(tl.int32, bitcast=True),
            mask=valid_k,
        )
        tl.store(chunk_base + k_base // K_TILE, run_max)
    # Make the buffered energies and chunk maxima visible block-wide before
    # the read-back passes below.
    tl.debug_barrier()

    # Max-based dynamic thresholding (paper Sec. 3.4): keep blocks whose
    # tile-referenced energy S'[b] = S[b] * exp2(M_c(b) - M_final) satisfies
    # S'[b] >= abs_threshold * max_b S'[b]; abs_threshold is the paper's alpha
    # in (0, 1]. This micro pass only re-reads the buffered energies and
    # per-chunk scalars (no GEMM).
    mass_max = tl.zeros((), tl.float32)
    for k_base in range(0, active_k_blocks, K_TILE):
        k_blocks = k_base + tl.arange(0, K_TILE)
        valid_k = (
            (k_blocks < active_k_blocks)
            & (k_blocks < max_k_blocks)
            & (k_blocks * K_BLOCK_N < kv_len)
        )
        m_c = tl.load(chunk_base + k_base // K_TILE)
        corr = tl.exp2(tl.minimum(m_c - run_max, 0.0))
        block_energy = tl.load(
            out_base + k_blocks * stride_index_k,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32, bitcast=True)
        mass_max = tl.maximum(
            mass_max, tl.max(tl.where(valid_k, block_energy * corr, 0.0), axis=0)
        )

    threshold = abs_threshold * mass_max
    count = tl.zeros((), tl.int32)
    tail_cnt = tl.zeros((), tl.int32)
    # The diagonal boundary block (j == last_k when last_k == active_k_blocks)
    # has no score (energy 0); it is still visited so sink/local selection
    # can keep it. Compaction writes always trail unread energy positions
    # (count <= blocks scanned), so the in-place buffer reuse is safe.
    for k_base in range(0, last_k + 1, K_TILE):
        k_blocks = k_base + tl.arange(0, K_TILE)
        in_range = k_blocks < max_k_blocks
        valid = in_range & (k_blocks * K_BLOCK_N < kv_len)
        scored_range = k_blocks < active_k_blocks
        m_c = tl.load(
            chunk_base + k_base // K_TILE,
            mask=k_base < active_k_blocks,
            other=0.0,
        )
        corr = tl.exp2(tl.minimum(m_c - run_max, 0.0))
        block_energy = tl.load(
            out_base + k_blocks * stride_index_k,
            mask=valid & scored_range,
            other=0.0,
        ).to(tl.float32, bitcast=True)
        scored = valid & scored_range & (block_energy * corr >= threshold)
        sink = k_blocks < attention_sink
        local = (k_blocks >= first_k - window_size + 1) & (k_blocks <= last_k)
        keep = valid & (scored | sink | local)
        if IS_CAUSAL:
            keep &= k_blocks <= last_k
        keep_i32 = keep.to(tl.int32)
        tail_cnt += tl.sum(tl.where((k_blocks == last_logical) & keep, 1, 0), axis=0)
        local_offset = tl.cumsum(keep_i32, axis=0) - 1
        tl.store(
            out_base + (count + local_offset) * stride_index_k,
            tl.where(k_blocks == last_logical, k_blocks | (e_tail << 20), k_blocks),
            mask=keep,
        )
        count += tl.sum(keep_i32, axis=0)

    tl.store(out_count_base, count * n_sub - tail_cnt * (n_sub - e_tail))


@triton.jit
def _compact_to_csr_kernel(
    compact_ptr,
    csr_ptr,
    csr_cu_ptr,
    counts_ptr,
    stride_compact_head,
    stride_compact_tile,
    stride_compact_k,
    total_q_tiles,
    max_k_blocks: tl.constexpr,
    N_SUB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # counts are PHYSICAL tile counts; compact values are logical block ids,
    # with the sequence-tail block tagged by its valid tile count in the high
    # bits (e_tail << 20). Each logical block j expands to emit(j) consecutive
    # tiles j*N_SUB .. j*N_SUB+emit(j)-1 (emit = N_SUB except for the tail),
    # written at compact (gap-free) positions, so the CSR remains ascending
    # and never contains a fully-invalid tile.
    segment = tl.program_id(0)
    kv_head = segment // total_q_tiles
    q_tile = segment % total_q_tiles
    offsets = tl.arange(0, BLOCK)
    count_phys = tl.load(counts_ptr + segment)
    count_logical = (count_phys + N_SUB - 1) // N_SUB
    values = tl.load(
        compact_ptr
        + kv_head * stride_compact_head
        + q_tile * stride_compact_tile
        + offsets * stride_compact_k,
        mask=offsets < count_logical,
        other=0,
    )
    e_tag = values >> 20
    emit = tl.where(e_tag > 0, e_tag, N_SUB)
    block_id = values & 0xFFFFF
    output_begin = tl.load(csr_cu_ptr + segment)
    # Exclusive prefix of per-block emit counts gives gap-free store positions.
    pos = tl.cumsum(emit, axis=0) - emit
    lanes = tl.arange(0, N_SUB)
    store_mask = (offsets < count_logical)[:, None] & (lanes[None, :] < emit[:, None])
    tl.store(
        csr_ptr + output_begin + pos[:, None] + lanes[None, :],
        block_id[:, None] * N_SUB + lanes[None, :],
        mask=store_mask,
    )


def _check_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_block_m: int,
    k_block_n: int,
) -> None:
    if not (q.is_cuda and k_cache.is_cuda and page_table.is_cuda):
        raise ValueError("q, k_cache, and page_table must be CUDA tensors")
    if q.dtype != k_cache.dtype or q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("q and k_cache must have the same dtype: bfloat16 or float8_e4m3fn")
    if q.ndim != 3:
        raise ValueError("q must have shape (total_q, num_q_heads, head_dim)")
    if k_cache.ndim != 4:
        raise ValueError("k_cache must have shape (num_pages, page_size, num_kv_heads, head_dim)")
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("q and k_cache head dimensions must match")
    if q.shape[1] % k_cache.shape[2] != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if page_table.ndim != 2 or page_table.dtype != torch.int32:
        raise TypeError("page_table must be a 2-D int32 tensor")
    if cache_seqlens.dtype != torch.int32 or cu_seqlens_q.dtype != torch.int32:
        raise TypeError("cache_seqlens and cu_seqlens_q must be int32")
    if cache_seqlens.ndim != 1 or cu_seqlens_q.ndim != 1:
        raise ValueError("cache_seqlens and cu_seqlens_q must be 1-D")
    if cu_seqlens_q.numel() != cache_seqlens.numel() + 1:
        raise ValueError("cu_seqlens_q must have batch_size + 1 elements")
    if k_block_m <= 0 or k_block_n <= 0:
        raise ValueError("k_block_m and k_block_n must be positive")
    if k_block_m % 16 != 0 or k_block_n & (k_block_n - 1):
        raise ValueError("k_block_m must be a multiple of 16 and k_block_n must be a power of two")
    if k_block_n % _ATTN_TILE_N != 0:
        raise ValueError(f"k_block_n must be a multiple of {_ATTN_TILE_N} (the attention kernel tile size)")



class SparseIndexWorkspace:
    """Reusable storage and launch metadata for the asynchronous fast path."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        total_q_tiles: int,
        max_q_tiles: int,
        max_k_blocks: int,
        dtype: torch.dtype,
        device: torch.device,
        cu_q_tiles: torch.Tensor,
        n_sub: int = 1,
        use_mean_correction: bool = False,
        head_dim_v: Optional[int] = None,
    ) -> None:
        self.batch_capacity = batch_size
        self.total_q_tiles_capacity = total_q_tiles
        self.max_q_tiles_capacity = max_q_tiles
        self.max_k_blocks_capacity = max_k_blocks
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim if head_dim_v is None else head_dim_v
        self.total_q_tiles = total_q_tiles
        self.max_q_tiles = max_q_tiles
        self.max_k_blocks = max_k_blocks
        self.cu_q_tiles = cu_q_tiles
        # n_sub: physical 64-token attention tiles per logical selection block.
        self.n_sub = n_sub
        self.use_mean_correction = use_mean_correction
        self.k_mean = torch.empty(
            (batch_size, max_k_blocks, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        # V block means feed the zero-order correction for unselected blocks
        # in the attention kernel epilogue (Sol-Attn style).
        self.v_mean = (
            torch.empty(
                (batch_size, max_k_blocks, num_kv_heads, self.head_dim_v),
                dtype=dtype,
                device=device,
            )
            if use_mean_correction
            else self.k_mean
        )
        shape = (num_kv_heads, total_q_tiles, max_k_blocks)
        self.compact_index = torch.empty(shape, dtype=torch.int32, device=device)
        self.counts = torch.empty(num_kv_heads * total_q_tiles, dtype=torch.int32, device=device)
        # One scalar per scoring chunk (running tile max), used by the fused
        # single-pass scorer to recover exact tile-referenced block energies.
        # K_TILE >= 16 in all configs, so cdiv(max_k_blocks, 16) is a safe bound.
        self.chunk_max = torch.empty(
            (num_kv_heads * total_q_tiles, (max_k_blocks + 15) // 16),
            dtype=torch.float32,
            device=device,
        )
        self.block_sparse_cu = torch.empty(
            num_kv_heads * total_q_tiles + 1, dtype=torch.int32, device=device
        )
        self.block_sparse_cu[0] = 0
        self.block_sparse_idx = torch.empty(
            num_kv_heads * total_q_tiles * max_k_blocks * n_sub,
            dtype=torch.int32,
            device=device,
        )

    def activate(
        self,
        *,
        batch_size: int,
        total_q_tiles: int,
        max_q_tiles: int,
        max_k_blocks: int,
    ) -> None:
        if (
            batch_size > self.batch_capacity
            or total_q_tiles > self.total_q_tiles_capacity
            or max_q_tiles > self.max_q_tiles_capacity
            or max_k_blocks > self.max_k_blocks_capacity
        ):
            raise ValueError("active shape exceeds SparseIndexWorkspace capacity")
        self.batch_size = batch_size
        self.total_q_tiles = total_q_tiles
        self.max_q_tiles = max_q_tiles
        self.max_k_blocks = max_k_blocks


def build_block_sparse_index_fast(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    workspace: SparseIndexWorkspace,
    *,
    v_cache: Optional[torch.Tensor] = None,
    k_block_m: int = 128,
    k_block_n: int = 64,
    abs_threshold: float = 1.0,
    attention_sink: int = 2,
    window_size: int = 4,
    last_n_blocks: int = 2,
    min_sparse_q_len: int = 0,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    q_descale: float = 1.0,
    k_descale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Asynchronous allocation-free path; valid idx length is ``cu[-1]``."""
    if abs_threshold < 0.0:
        raise ValueError("abs_threshold must be non-negative")
    use_mean_correction = workspace.use_mean_correction
    if use_mean_correction and v_cache is None:
        raise ValueError("v_cache is required when the workspace enables use_mean_correction")
    if use_mean_correction and (
        v_cache.shape != k_cache.shape or v_cache.dtype != k_cache.dtype
    ):
        raise ValueError("v_cache must match k_cache in shape and dtype")
    if use_mean_correction and workspace.v_mean.shape[-1] != v_cache.shape[-1]:
        raise ValueError("workspace v_mean head_dim_v does not match v_cache")
    if k_block_n % _ATTN_TILE_N != 0:
        raise ValueError(f"k_block_n must be a multiple of {_ATTN_TILE_N} (the attention kernel tile size)")
    if workspace.n_sub != k_block_n // _ATTN_TILE_N:
        raise ValueError(
            f"workspace was allocated with n_sub={workspace.n_sub} but k_block_n={k_block_n} "
            f"requires n_sub={k_block_n // _ATTN_TILE_N}"
        )
    n_sub = workspace.n_sub
    batch_size = workspace.batch_size
    num_q_heads = q.shape[1]
    num_kv_heads = workspace.num_kv_heads
    head_dim = workspace.head_dim
    page_size = k_cache.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads
    total_q_tiles = workspace.total_q_tiles
    max_k_blocks = workspace.max_k_blocks
    max_q_tiles = workspace.max_q_tiles
    cu_q_tiles = workspace.cu_q_tiles

    cu_q_tiles_block = triton.next_power_of_2(batch_size)
    _fill_cu_q_tiles_kernel[(1,)](
        cu_seqlens_q,
        cu_q_tiles,
        batch_size=batch_size,
        gqa_ratio=gqa_ratio,
        K_BLOCK_M=k_block_m,
        BLOCK=cu_q_tiles_block,
        num_warps=1,
    )

    if page_size == 1:
        mean_num_warps = _PAGE1_FP8_MEAN_NUM_WARPS if k_cache.dtype == torch.float8_e4m3fn else _PAGE1_BF16_MEAN_NUM_WARPS
        mean_num_stages = _PAGE1_FP8_MEAN_NUM_STAGES if k_cache.dtype == torch.float8_e4m3fn else _PAGE1_BF16_MEAN_NUM_STAGES
    else:
        mean_num_warps = _MEAN_NUM_WARPS
        mean_num_stages = _MEAN_NUM_STAGES

    _paged_k_mean_kernel[(max_k_blocks, batch_size * num_kv_heads)](
        k_cache,
        workspace.k_mean,
        v_cache if use_mean_correction else k_cache,
        workspace.v_mean,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        *k_cache.stride(),
        *(v_cache.stride() if use_mean_correction else k_cache.stride()),
        *page_table.stride(),
        *workspace.k_mean.stride(),
        *workspace.v_mean.stride(),
        num_kv_heads=num_kv_heads,
        gqa_ratio=gqa_ratio,
        page_size=page_size,
        max_k_blocks=max_k_blocks,
        min_sparse_q_len=min_sparse_q_len,
        last_n_blocks=last_n_blocks,
        K_BLOCK_M=k_block_m,
        K_BLOCK_N=k_block_n,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=workspace.head_dim_v,
        COMPUTE_V_MEAN=use_mean_correction,
        num_warps=mean_num_warps,
        num_stages=mean_num_stages,
    )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    eff_scale = float(scale) * float(q_descale) * float(k_descale)
    scale_log2 = eff_scale * math.log2(math.e)
    use_long_score_config = page_size == 1 and max_k_blocks >= 256
    score_k_tile = 32 if use_long_score_config else _SCORE_K_TILE
    score_num_stages = 1 if use_long_score_config else _SCORE_NUM_STAGES
    _packgqa_score_select_kernel[(max_q_tiles, batch_size * num_kv_heads)](
        q,
        workspace.k_mean,
        workspace.compact_index,
        workspace.counts,
        workspace.chunk_max,
        cu_seqlens_q,
        cache_seqlens,
        cu_q_tiles,
        *q.stride(),
        *workspace.k_mean.stride(),
        *workspace.compact_index.stride(),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_ratio=gqa_ratio,
        total_q_tiles=total_q_tiles,
        max_k_blocks=max_k_blocks,
        min_sparse_q_len=min_sparse_q_len,
        last_n_blocks=last_n_blocks,
        scale_log2=scale_log2,
        abs_threshold=abs_threshold,
        attention_sink=attention_sink,
        window_size=window_size,
        IS_CAUSAL=causal,
        K_BLOCK_M=k_block_m,
        K_BLOCK_N=k_block_n,
        HEAD_DIM=head_dim,
        K_TILE=score_k_tile,
        num_warps=_SCORE_NUM_WARPS,
        num_stages=score_num_stages,
    )

    segments = num_kv_heads * total_q_tiles
    active_counts = workspace.counts[:segments]
    active_cu = workspace.block_sparse_cu[: segments + 1]
    # block_sparse_cu[0] was set to 0 at workspace creation and is never
    # written elsewhere; counts are already physical tile counts
    # (tail-trimmed by the fused selector).
    torch.cumsum(active_counts, dim=0, out=active_cu[1:])
    copy_block = triton.next_power_of_2(max_k_blocks)
    _compact_to_csr_kernel[(segments,)](
        workspace.compact_index,
        workspace.block_sparse_idx,
        active_cu,
        active_counts,
        *workspace.compact_index.stride(),
        total_q_tiles,
        max_k_blocks=max_k_blocks,
        N_SUB=n_sub,
        BLOCK=copy_block,
        num_warps=1,
    )
    return (
        active_cu,
        workspace.block_sparse_idx[: segments * max_k_blocks * n_sub],
        total_q_tiles,
        cu_q_tiles[: batch_size + 1],
    )

def build_block_sparse_index(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    k_block_m: int = 128,
    k_block_n: int = 64,
    abs_threshold: float = 1.0,
    attention_sink: int = 2,
    window_size: int = 4,
    last_n_blocks: int = 2,
    min_sparse_q_len: int = 0,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    q_descale: float = 1.0,
    k_descale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Build FlashPrefill PackGQA CSR indices from varlen Q and paged K.

    ``q`` has shape ``(total_q, num_q_heads, head_dim)``. ``k_cache`` has
    shape ``(num_pages, page_size, num_kv_heads, head_dim)``. The page table
    maps each logical KV page to a physical page in ``k_cache``.

    BF16 inputs use BF16 Q/K operands in ``tl.dot``. E4M3 inputs use FP8 Q/K
    operands. Means are accumulated in FP32 and quantized back to the input
    dtype before the dot product; dot accumulators and normalized sparse scores are FP32.

    A segment covers the exact packed-row interval, including a query split by
    an M-tile boundary when ``k_block_m % gqa_ratio != 0``. The local window is
    the union needed by the first through last query positions in that segment.
    """
    _check_inputs(q, k_cache, page_table, cache_seqlens, cu_seqlens_q, k_block_m, k_block_n)
    if abs_threshold < 0.0:
        raise ValueError("abs_threshold must be non-negative")
    if min(attention_sink, window_size, last_n_blocks) < 0:
        raise ValueError("attention_sink, window_size, and last_n_blocks must be non-negative")
    if min_sparse_q_len < 0:
        raise ValueError("min_sparse_q_len must be non-negative")

    batch_size = cache_seqlens.numel()
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[2]
    page_size = k_cache.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    if bool(torch.any(cache_seqlens < q_lens).item()):
        raise ValueError("each cache sequence length must be >= its query length for bottom-right causal alignment")

    q_tiles_per_batch = torch.div(
        q_lens * gqa_ratio + k_block_m - 1, k_block_m, rounding_mode="floor"
    ).to(torch.int32)
    cu_q_tiles = torch.empty(batch_size + 1, dtype=torch.int32, device=q.device)
    cu_q_tiles[0] = 0
    cu_q_tiles[1:] = torch.cumsum(q_tiles_per_batch, dim=0)
    total_q_tiles = int(cu_q_tiles[-1].item())

    max_k_len = int(cache_seqlens.max().item()) if batch_size else 0
    max_k_blocks = triton.cdiv(max_k_len, k_block_n)
    if total_q_tiles == 0 or max_k_blocks == 0:
        cu = torch.zeros(num_kv_heads * total_q_tiles + 1, dtype=torch.int32, device=q.device)
        idx = torch.empty(0, dtype=torch.int32, device=q.device)
        return cu, idx, total_q_tiles, cu_q_tiles

    k_mean = torch.empty(
        (batch_size, max_k_blocks, num_kv_heads, head_dim), dtype=k_cache.dtype, device=q.device
    )
    if page_size == 1:
        mean_num_warps = _PAGE1_FP8_MEAN_NUM_WARPS if k_cache.dtype == torch.float8_e4m3fn else _PAGE1_BF16_MEAN_NUM_WARPS
        mean_num_stages = _PAGE1_FP8_MEAN_NUM_STAGES if k_cache.dtype == torch.float8_e4m3fn else _PAGE1_BF16_MEAN_NUM_STAGES
    else:
        mean_num_warps = _MEAN_NUM_WARPS
        mean_num_stages = _MEAN_NUM_STAGES

    _paged_k_mean_kernel[(max_k_blocks, batch_size * num_kv_heads)](
        k_cache,
        k_mean,
        k_cache,  # v placeholder: the slow reference path does not pool V
        k_mean,   # v_mean placeholder
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        *k_cache.stride(),
        *k_cache.stride(),
        *page_table.stride(),
        *k_mean.stride(),
        *k_mean.stride(),
        num_kv_heads=num_kv_heads,
        gqa_ratio=gqa_ratio,
        page_size=page_size,
        max_k_blocks=max_k_blocks,
        min_sparse_q_len=min_sparse_q_len,
        last_n_blocks=last_n_blocks,
        K_BLOCK_M=k_block_m,
        K_BLOCK_N=k_block_n,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim,
        COMPUTE_V_MEAN=False,
        num_warps=mean_num_warps,
        num_stages=mean_num_stages,
    )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    eff_scale = float(scale) * float(q_descale) * float(k_descale)
    scale_log2 = eff_scale * math.log2(math.e)
    max_q_tiles = int(q_tiles_per_batch.max().item())
    use_long_score_config = page_size == 1 and max_k_blocks >= 256
    score_k_tile = 32 if use_long_score_config else _SCORE_K_TILE
    score_num_stages = 1 if use_long_score_config else _SCORE_NUM_STAGES

    out_index = torch.full(
        (num_kv_heads, total_q_tiles, max_k_blocks), -1, dtype=torch.int32, device=q.device
    )
    counts = torch.zeros(num_kv_heads * total_q_tiles, dtype=torch.int32, device=q.device)
    # One scalar per scoring chunk (see SparseIndexWorkspace.chunk_max).
    chunk_max = torch.empty(
        (num_kv_heads * total_q_tiles, (max_k_blocks + 15) // 16),
        dtype=torch.float32,
        device=q.device,
    )
    _packgqa_score_select_kernel[(max_q_tiles, batch_size * num_kv_heads)](
        q,
        k_mean,
        out_index,
        counts,
        chunk_max,
        cu_seqlens_q,
        cache_seqlens,
        cu_q_tiles,
        *q.stride(),
        *k_mean.stride(),
        *out_index.stride(),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_ratio=gqa_ratio,
        total_q_tiles=total_q_tiles,
        max_k_blocks=max_k_blocks,
        min_sparse_q_len=min_sparse_q_len,
        last_n_blocks=last_n_blocks,
        scale_log2=scale_log2,
        abs_threshold=abs_threshold,
        attention_sink=attention_sink,
        window_size=window_size,
        IS_CAUSAL=causal,
        K_BLOCK_M=k_block_m,
        K_BLOCK_N=k_block_n,
        HEAD_DIM=head_dim,
        K_TILE=score_k_tile,
        num_warps=_SCORE_NUM_WARPS,
        num_stages=score_num_stages,
    )

    n_sub = k_block_n // _ATTN_TILE_N
    block_sparse_cu = torch.empty(counts.numel() + 1, dtype=torch.int32, device=q.device)
    block_sparse_cu[0] = 0
    # counts are already physical tile counts (tail-trimmed by the selector).
    torch.cumsum(counts, dim=0, out=block_sparse_cu[1:])
    block_sparse_idx_capacity = torch.empty(
        num_kv_heads * total_q_tiles * max_k_blocks * n_sub,
        dtype=torch.int32,
        device=q.device,
    )
    copy_block = triton.next_power_of_2(max_k_blocks)
    _compact_to_csr_kernel[(counts.numel(),)](
        out_index,
        block_sparse_idx_capacity,
        block_sparse_cu,
        counts,
        *out_index.stride(),
        total_q_tiles,
        max_k_blocks=max_k_blocks,
        N_SUB=n_sub,
        BLOCK=copy_block,
        num_warps=1,
    )
    selected = int(block_sparse_cu[-1].item())
    block_sparse_idx = block_sparse_idx_capacity[:selected]

    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


__all__ = [
    "SparseIndexWorkspace",
    "build_block_sparse_index",
    "build_block_sparse_index_fast",
]