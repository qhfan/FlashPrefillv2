"""
Test block sparse attention with FlashPrefill_block_sparse.
Tests PagedKVNonTMA path with page_size=1, bf16/fp8, GQA, varlen.

Sparse pattern (matching Triton flashprefill):
1. Attention sink: first N blocks always selected
2. Local window: current Q-block's diagonal K-block + window-1 blocks before it
3. Last N Q-blocks: full attention (all causal K-blocks)
4. Remaining: randomly selected

When kBlockM != kBlockN (or PackGQA packs Q heads), the Q-to-K block mapping
requires careful position calculation.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_attn_interface import flash_attn_with_kvcache

torch.manual_seed(42)
device = "cuda"


def get_tile_sizes(head_dim, element_size=2, is_causal=False, is_local=False, paged_kv_non_tma=True):
    """Replicate tile_size_fwd_sm90 logic to get kBlockM and kBlockN."""
    if element_size == 2:  # bf16/fp16
        if head_dim <= 128:
            use_blockN_128 = is_causal or is_local or paged_kv_non_tma
            return 128, 64 if use_blockN_128 else 176
        elif head_dim <= 192:
            return 128, 96 if paged_kv_non_tma else 128
        else:  # 256
            return 128, 64
    else:  # fp8 (element_size=1)
        if head_dim <= 128:
            return 128, 64
        elif head_dim <= 192:
            return 128, 160
        else:  # 256
            return 128, 64


def m_block_to_q_pos(m_block, kBlockM, gqa_ratio):
    """Map PackGQA m_block to the first Q sequence position it covers."""
    return (m_block * kBlockM) // gqa_ratio


def m_block_to_q_pos_end(m_block, kBlockM, gqa_ratio, seqlen_q):
    """Map PackGQA m_block to the last Q sequence position it covers (exclusive)."""
    end = ((m_block + 1) * kBlockM) // gqa_ratio
    return min(end, seqlen_q)


def q_pos_to_k_block(q_pos, prefix_len, kBlockN):
    """Map a Q sequence position to its diagonal K-block index (logical KV space)."""
    return (prefix_len + q_pos) // kBlockN


def build_compact_index_triton_style(
    batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
    kBlockM, kBlockN,
    attention_sink=2, window=4, last_n_blocks=2,
    num_random_blocks=2,
    causal=True,
    device="cuda",
    rng_seed=42,
):
    """
    Build compact index using Triton flashprefill-style sparse pattern.

    Rules (adapted for PackGQA where m_block iterates over seqlen_q * gqa_ratio):
    1. Attention sink: first `attention_sink` K-blocks always selected
    2. Local window: Q-block's diagonal K-block + (window-1) blocks before it
    3. Last N Q-blocks: full attention (all causal K-blocks)
    4. Remaining causal K-blocks: randomly select `num_random_blocks`

    For GQA union: since the pattern depends on Q position (not Q head), all Q heads
    in the same group select the same K-blocks. Union is trivially the same set.
    (If random selection differed per Q head, we'd take the union here.)

    Layout:
      block_sparse_cu: [num_kv_heads * total_q_tiles + 1]  (int32)
      block_sparse_idx: [block_sparse_cu[-1]]              (int32, ascending per segment)
      total_q_tiles: ceil_div(seqlen_q * gqa_ratio, kBlockM) per batch, total across batch
      cu_q_tiles: [batch_size + 1]  (int32, prefix sum of q tiles per batch)

    Segment ID: g = total_q_tiles * h_kv + (cu_q_tiles[b] + m_block)
    """
    gqa_ratio = nheads_q // nheads_kv
    n_q_tiles_per_batch = (seqlen_q * gqa_ratio + kBlockM - 1) // kBlockM
    n_k_tiles_per_batch = (seqlen_k + kBlockN - 1) // kBlockN
    total_q_tiles = n_q_tiles_per_batch * batch_size

    cu_q_tiles = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        cu_q_tiles[b + 1] = cu_q_tiles[b] + n_q_tiles_per_batch

    rng = np.random.RandomState(rng_seed)

    # Determine which Q-blocks are "last N" in Q-position space
    # Q positions go from 0 to seqlen_q-1
    # Q-blocks in Q-position space: cdiv(seqlen_q, kBlockM_effective)
    # where kBlockM_effective = kBlockM // gqa_ratio (positions per m_block)
    positions_per_m_block = max(1, kBlockM // gqa_ratio)
    n_q_pos_blocks = (seqlen_q + positions_per_m_block - 1) // positions_per_m_block
    last_n_q_pos_start = max(0, n_q_pos_blocks - last_n_blocks)

    all_indices = []
    cu_offsets = [0]

    # Segment order: g = total_q_tiles * h_kv + (cu_q_tiles[b] + m_block)
    for h_kv in range(nheads_kv):
        for b in range(batch_size):
            prefix_len = seqlen_k - seqlen_q  # bottom-right causal alignment

            for m_block in range(n_q_tiles_per_batch):
                q_pos_start = m_block_to_q_pos(m_block, kBlockM, gqa_ratio)
                q_pos_end = m_block_to_q_pos_end(m_block, kBlockM, gqa_ratio, seqlen_q)

                if q_pos_start >= seqlen_q:
                    # This m_block is padding
                    all_indices.extend([])
                    cu_offsets.append(len(all_indices))
                    continue

                # Diagonal K-block for this Q position
                q_k_blk = q_pos_to_k_block(q_pos_start, prefix_len, kBlockN)

                # Causal range: K-blocks 0 to q_k_blk (inclusive)
                causal_max_n = min(q_k_blk + 1, n_k_tiles_per_batch) if causal else n_k_tiles_per_batch

                # Check if this is a "last N" Q-block (by Q position block index)
                q_pos_blk_idx = q_pos_start // positions_per_m_block
                is_last_n = q_pos_blk_idx >= last_n_q_pos_start

                selected = set()

                # Rule 1: Attention sink
                sink_end = min(attention_sink, causal_max_n)
                selected.update(range(sink_end))

                # Rule 2: Local window (diagonal block + window-1 blocks before)
                window_start = max(0, q_k_blk - window + 1)
                window_end = min(q_k_blk + 1, causal_max_n)
                selected.update(range(window_start, window_end))

                # Rule 3: Last N Q-blocks → full attention
                if is_last_n:
                    selected.update(range(causal_max_n))
                else:
                    # Rule 4: Random selection from remaining causal blocks
                    remaining = [k for k in range(causal_max_n) if k not in selected]
                    if len(remaining) > 0 and num_random_blocks > 0:
                        n_select = min(num_random_blocks, len(remaining))
                        chosen = rng.choice(remaining, size=n_select, replace=False)
                        selected.update(chosen.tolist())

                # Sort ascending (kernel expects ascending order)
                selected_sorted = sorted(selected)
                all_indices.extend(selected_sorted)
                cu_offsets.append(len(all_indices))

    block_sparse_cu = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(all_indices, dtype=torch.int32, device=device)

    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


def build_full_compact_index(
    batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
    kBlockM, kBlockN,
    causal=True,
    device="cuda",
):
    """
    Build compact index that includes ALL causal K tiles (i.e., dense attention).
    This should produce identical results to dense FA3.
    """
    gqa_ratio = nheads_q // nheads_kv
    n_q_tiles_per_batch = (seqlen_q * gqa_ratio + kBlockM - 1) // kBlockM
    n_k_tiles_per_batch = (seqlen_k + kBlockN - 1) // kBlockN
    total_q_tiles = n_q_tiles_per_batch * batch_size

    cu_q_tiles = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        cu_q_tiles[b + 1] = cu_q_tiles[b] + n_q_tiles_per_batch

    all_indices = []
    cu_offsets = [0]

    for h_kv in range(nheads_kv):
        for b in range(batch_size):
            prefix_len = seqlen_k - seqlen_q
            for m_block in range(n_q_tiles_per_batch):
                if causal:
                    q_pos_start = m_block_to_q_pos(m_block, kBlockM, gqa_ratio)
                    q_pos_end = m_block_to_q_pos_end(m_block, kBlockM, gqa_ratio, seqlen_q)
                    if q_pos_start >= seqlen_q:
                        all_indices.extend([])
                        cu_offsets.append(len(all_indices))
                        continue
                    q_k_blk = q_pos_to_k_block(q_pos_end - 1, prefix_len, kBlockN)
                    causal_max_n = min(q_k_blk + 1, n_k_tiles_per_batch)
                    selected = list(range(causal_max_n))
                else:
                    selected = list(range(n_k_tiles_per_batch))

                all_indices.extend(selected)
                cu_offsets.append(len(all_indices))

    block_sparse_cu = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(all_indices, dtype=torch.int32, device=device)

    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


def reference_attention_sparse(
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
    nheads_q, nheads_kv, head_dim, softmax_scale,
    block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles,
    kBlockM, kBlockN,
    causal=True,
):
    """
    Reference attention with sparse pattern, applying element-level causal masking.
    Matches FA3 kernel behavior: block sparse selects K-tiles, causal mask applied within.
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = (cu_seqlens_q[1] - cu_seqlens_q[0]).item()
    gqa_ratio = nheads_q // nheads_kv
    n_q_tiles_per_batch = (seqlen_q * gqa_ratio + kBlockM - 1) // kBlockM

    out_ref = torch.zeros(q.shape, dtype=torch.float32, device=device)

    for b in range(batch_size):
        kv_len = cache_seqlens[b].item()
        page_indices = page_table[b, :kv_len]
        k_b = k_cache[page_indices, 0]  # (kv_len, nheads_kv, head_dim)
        v_b = v_cache[page_indices, 0]
        prefix_len = kv_len - seqlen_q

        for h_kv in range(nheads_kv):
            for m_block in range(n_q_tiles_per_batch):
                g = total_q_tiles * h_kv + (cu_q_tiles[b].item() + m_block)
                lo = block_sparse_cu[g].item()
                hi = block_sparse_cu[g + 1].item()
                selected_k_tiles = block_sparse_idx[lo:hi].tolist()

                if len(selected_k_tiles) == 0:
                    continue

                # Gather selected K/V
                k_tiles_data = []
                v_tiles_data = []
                k_pos_ranges = []
                for n_block in selected_k_tiles:
                    k_start = n_block * kBlockN
                    k_end = min((n_block + 1) * kBlockN, kv_len)
                    k_tiles_data.append(k_b[k_start:k_end, h_kv, :])
                    v_tiles_data.append(v_b[k_start:k_end, h_kv, :])
                    k_pos_ranges.append((k_start, k_end))

                k_selected = torch.cat(k_tiles_data, dim=0)
                v_selected = torch.cat(v_tiles_data, dim=0)

                # Map m_block to seq positions (PackGQA: row -> seq_pos = row // gqa_ratio)
                row_start = m_block * kBlockM
                row_end = min((m_block + 1) * kBlockM, seqlen_q * gqa_ratio)

                for row in range(row_start, row_end):
                    seq_pos = row // gqa_ratio
                    gqa_idx = row % gqa_ratio
                    h_q = h_kv * gqa_ratio + gqa_idx
                    q_row = b * seqlen_q + seq_pos

                    if seq_pos >= seqlen_q:
                        continue

                    q_vec = q[q_row, h_q, :].float().unsqueeze(0)
                    scores = torch.matmul(q_vec, k_selected.float().T) * softmax_scale

                    # Element-level causal mask (bottom-right alignment)
                    # Causal condition: q_logical >= k_logical, i.e., prefix_len + seq_pos >= k_pos
                    # So K positions > (prefix_len + seq_pos) should be masked
                    if causal:
                        k_mask_threshold = prefix_len + seq_pos  # K positions > this are masked
                        col = 0
                        for (k_s, k_e) in k_pos_ranges:
                            for k_pos in range(k_s, k_e):
                                if k_pos > k_mask_threshold:
                                    scores[0, col] = float('-inf')
                                col += 1

                    # Softmax (handle all-masked rows)
                    max_score = scores.max(dim=-1, keepdim=True).values
                    if max_score.item() == float('-inf'):
                        # All masked - skip (output stays 0)
                        continue
                    scores = scores - max_score
                    scores = torch.exp(scores)
                    scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)

                    out_tile = torch.matmul(scores, v_selected.float())
                    out_ref[q_row, h_q, :] = out_tile[0].to(out_ref.dtype)

    return out_ref


def reference_attention_dense(q, k_cache, v_cache, page_table, cache_seqlens,
                               nheads_q, nheads_kv, head_dim, softmax_scale, causal=True):
    """Dense reference attention."""
    batch_size = page_table.shape[0]
    seqlen_q = q.shape[0] // batch_size
    gqa_ratio = nheads_q // nheads_kv
    out_ref = torch.zeros_like(q)

    for b in range(batch_size):
        kv_len = cache_seqlens[b].item()
        page_indices = page_table[b, :kv_len]
        k_b = k_cache[page_indices, 0]
        v_b = v_cache[page_indices, 0]
        prefix_len = kv_len - seqlen_q

        for h_kv in range(nheads_kv):
            k_b_h = k_b[:, h_kv, :].float()  # (kv_len, head_dim)
            v_b_h = v_b[:, h_kv, :].float()

            for gqa_idx in range(gqa_ratio):
                h_q = h_kv * gqa_ratio + gqa_idx
                for i in range(seqlen_q):
                    q_row = b * seqlen_q + i
                    q_vec = q[q_row, h_q, :].float().unsqueeze(0)

                    scores = torch.matmul(q_vec, k_b_h.T) * softmax_scale

                    if causal:
                        k_thresh = prefix_len + i
                        scores[0, k_thresh+1:] = float('-inf')

                    scores = scores - scores.max(dim=-1, keepdim=True).values
                    scores = torch.exp(scores)
                    scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)

                    out_ref[q_row, h_q, :] = torch.matmul(scores, v_b_h).to(q.dtype)[0]

    return out_ref


def create_test_inputs(batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device):
    """Create standard test inputs."""
    total_q = batch_size * seqlen_q
    q = torch.randn(total_q, nheads_q, head_dim, dtype=torch.float32, device=device).to(
        torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    )

    num_pages = batch_size * seqlen_k * 2
    page_size = 1
    scale = 0.5
    if dtype == torch.float8_e4m3fn:
        k_cache = (torch.randn(num_pages, page_size, nheads_kv, head_dim, dtype=torch.float32, device=device) * scale).to(dtype)
        v_cache = (torch.randn(num_pages, page_size, nheads_kv, head_dim, dtype=torch.float32, device=device) * scale).to(dtype)
    else:
        k_cache = torch.randn(num_pages, page_size, nheads_kv, head_dim, dtype=dtype, device=device) * scale
        v_cache = torch.randn(num_pages, page_size, nheads_kv, head_dim, dtype=dtype, device=device) * scale

    page_table = torch.zeros(batch_size, seqlen_k, dtype=torch.int32, device=device)
    for b in range(batch_size):
        page_table[b] = torch.arange(b * seqlen_k, (b + 1) * seqlen_k, dtype=torch.int32, device=device)

    cache_seqlens = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0] + [seqlen_q * (i+1) for i in range(batch_size)], dtype=torch.int32, device=device)

    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q


def run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
            seqlen_q, head_dim, softmax_scale, causal=True,
            block_sparse_cu=None, block_sparse_idx=None, total_q_tiles=None, cu_q_tiles=None,
            num_splits=0):
    """Run FA3 with or without block sparse."""
    kwargs = dict(
        q=q, k_cache=k_cache, v_cache=v_cache,
        page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=seqlen_q,
        softmax_scale=softmax_scale, causal=causal,
        num_splits=num_splits,
    )
    if block_sparse_cu is not None:
        kwargs.update(
            block_sparse_cu=block_sparse_cu,
            block_sparse_idx=block_sparse_idx,
            total_q_tiles=total_q_tiles,
            cu_q_tiles=cu_q_tiles,
        )
    return flash_attn_with_kvcache(**kwargs)


def test_dense_vs_dense(head_dim=128, dtype=torch.bfloat16, batch_size=2,
                         seqlen_q=256, seqlen_k=512, nheads_q=8, nheads_kv=2):
    """Test that full compact index (dense) matches dense FA3 output."""
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

    print(f"\n{'='*60}")
    print(f"Test: dense compact index vs dense FA3")
    print(f"  head_dim={head_dim}, dtype={dtype}, kBlockM={kBlockM}, kBlockN={kBlockN}")
    print(f"  batch={batch_size}, q={seqlen_q}, k={seqlen_k}, q_heads={nheads_q}, kv_heads={nheads_kv}")
    print(f"{'='*60}")

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device)
    softmax_scale = head_dim ** (-0.5)

    block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles = build_full_compact_index(
        batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN, causal=True, device=device)

    print(f"  Index: cu={block_sparse_cu.shape}, idx={block_sparse_idx.shape}, total_q_tiles={total_q_tiles}")

    out_sparse = run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                         seqlen_q, head_dim, softmax_scale, causal=True,
                         block_sparse_cu=block_sparse_cu, block_sparse_idx=block_sparse_idx,
                         total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles)

    out_dense = run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                        seqlen_q, head_dim, softmax_scale, causal=True)

    max_diff = (out_sparse.float() - out_dense.float()).abs().max().item()
    mean_diff = (out_sparse.float() - out_dense.float()).abs().mean().item()

    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    if max_diff < 0.05:
        print(f"  ✅ PASS")
        return True
    else:
        print(f"  ❌ FAIL")
        print(f"  sparse[0,0,:4]: {out_sparse[0,0,:4]}")
        print(f"  dense[0,0,:4]:  {out_dense[0,0,:4]}")
        return False


def test_triton_style_sparse(head_dim=128, dtype=torch.bfloat16, batch_size=2,
                              seqlen_q=256, seqlen_k=512, nheads_q=8, nheads_kv=2,
                              attention_sink=2, window=4, last_n_blocks=2, num_random_blocks=2):
    """Test Triton-style sparse pattern against reference implementation."""
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

    print(f"\n{'='*60}")
    print(f"Test: Triton-style sparse pattern vs reference")
    print(f"  head_dim={head_dim}, dtype={dtype}, kBlockM={kBlockM}, kBlockN={kBlockN}")
    print(f"  batch={batch_size}, q={seqlen_q}, k={seqlen_k}, q_heads={nheads_q}, kv_heads={nheads_kv}")
    print(f"  sink={attention_sink}, window={window}, last_n={last_n_blocks}, random={num_random_blocks}")
    print(f"{'='*60}")

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device)
    # For fp8, Q must match K/V dtype (create_test_inputs creates Q as bf16 for fp8)
    if dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
    softmax_scale = head_dim ** (-0.5)

    block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles = build_compact_index_triton_style(
        batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN,
        attention_sink=attention_sink, window=window, last_n_blocks=last_n_blocks,
        num_random_blocks=num_random_blocks, causal=True, device=device)

    print(f"  Index: cu={block_sparse_cu.shape}, idx={block_sparse_idx.shape}, total_q_tiles={total_q_tiles}")
    print(f"  Total selected K-tiles: {block_sparse_idx.shape[0]}")

    # Print sparsity stats
    n_k_tiles = (seqlen_k + kBlockN - 1) // kBlockN
    gqa_ratio = nheads_q // nheads_kv
    n_q_tiles = (seqlen_q * gqa_ratio + kBlockM - 1) // kBlockM
    total_possible = n_q_tiles * batch_size * nheads_kv * n_k_tiles
    actual = block_sparse_idx.shape[0]
    print(f"  Sparsity: {actual}/{total_possible} = {actual/total_possible*100:.1f}% selected")

    out_sparse = run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                         seqlen_q, head_dim, softmax_scale, causal=True,
                         block_sparse_cu=block_sparse_cu, block_sparse_idx=block_sparse_idx,
                         total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles)

    out_ref = reference_attention_sparse(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
        nheads_q, nheads_kv, head_dim, softmax_scale,
        block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles,
        kBlockM, kBlockN, causal=True)

    max_diff = (out_sparse.float() - out_ref).abs().max().item()
    mean_diff = (out_sparse.float() - out_ref).abs().mean().item()
    # Exclude NaN rows (all-masked) from comparison
    valid_mask = out_ref.abs().sum(dim=-1) > 0
    if valid_mask.any():
        valid_diff = (out_sparse.float()[valid_mask] - out_ref[valid_mask]).abs().max().item()
    else:
        valid_diff = float('inf')

    print(f"  Max diff (all): {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    print(f"  Max diff (valid rows only): {valid_diff:.6e}")
    print(f"  Valid rows: {valid_mask.sum().item()}/{valid_mask.numel()}")

    if valid_diff < 0.1:
        print(f"  ✅ PASS")
        return True
    else:
        print(f"  ❌ FAIL (valid diff too large)")
        # Find worst row
        diffs_per_row = (out_sparse.float() - out_ref.float()).abs().mean(dim=-1)
        worst_idx = diffs_per_row.argmax().item()
        worst_row = worst_idx // nheads_q
        worst_head = worst_idx % nheads_q
        print(f"  Worst at row={worst_row}, head={worst_head}")
        print(f"    sparse: {out_sparse[worst_row, worst_head, :4]}")
        print(f"    ref:    {out_ref[worst_row, worst_head, :4]}")
        return False


def test_fp8(head_dim=128, batch_size=2, seqlen_q=256, seqlen_k=512,
             nheads_q=8, nheads_kv=2):
    """Test fp8 block sparse attention (dense index)."""
    dtype = torch.float8_e4m3fn
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size=1, is_causal=True, paged_kv_non_tma=True)

    print(f"\n{'='*60}")
    print(f"Test: fp8 block sparse (dense index)")
    print(f"  head_dim={head_dim}, kBlockM={kBlockM}, kBlockN={kBlockN}")
    print(f"{'='*60}")

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device)
    # For fp8, Q must match K/V dtype
    q = q.to(torch.float8_e4m3fn)
    softmax_scale = head_dim ** (-0.5)

    block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles = build_full_compact_index(
        batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN, causal=True, device=device)

    try:
        out_sparse = run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                             seqlen_q, head_dim, softmax_scale, causal=True,
                             block_sparse_cu=block_sparse_cu, block_sparse_idx=block_sparse_idx,
                             total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles)
        out_dense = run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                            seqlen_q, head_dim, softmax_scale, causal=True)

        max_diff = (out_sparse.float() - out_dense.float()).abs().max().item()
        print(f"  Max diff: {max_diff:.6e}")
        if max_diff < 0.05:
            print(f"  ✅ PASS")
            return True
        else:
            print(f"  ❌ FAIL")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hdim256(head_dim=256, dtype=torch.bfloat16, batch_size=2, seqlen_q=256, seqlen_k=512,
                 nheads_q=8, nheads_kv=2):
    """Test hdim=256 block sparse (dense index)."""
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

    print(f"\n{'='*60}")
    print(f"Test: hdim=256 block sparse (dense index)")
    print(f"  dtype={dtype}, kBlockM={kBlockM}, kBlockN={kBlockN}")
    print(f"{'='*60}")

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device)
    softmax_scale = head_dim ** (-0.5)

    block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles = build_full_compact_index(
        batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN, causal=True, device=device)

    try:
        out_sparse = run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                             seqlen_q, head_dim, softmax_scale, causal=True,
                             block_sparse_cu=block_sparse_cu, block_sparse_idx=block_sparse_idx,
                             total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles)
        out_dense = run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                            seqlen_q, head_dim, softmax_scale, causal=True)

        max_diff = (out_sparse.float() - out_dense.float()).abs().max().item()
        print(f"  Max diff: {max_diff:.6e}")
        if max_diff < 0.05:
            print(f"  ✅ PASS")
            return True
        else:
            print(f"  ❌ FAIL")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_split_kv_block_sparse(head_dim=128, dtype=torch.bfloat16, batch_size=2,
                                seqlen_q=256, seqlen_k=2048, nheads_q=16, nheads_kv=4,
                                attention_sink=2, window=4, last_n_blocks=2, num_random_blocks=4,
                                num_splits=4):
    """Test that Split-KV + block sparse produces correct results matching non-split."""
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

    print(f"\n{'='*60}")
    print(f"Test: Split-KV + block sparse (num_splits={num_splits})")
    print(f"  head_dim={head_dim}, dtype={dtype}, kBlockM={kBlockM}, kBlockN={kBlockN}")
    print(f"  batch={batch_size}, q={seqlen_q}, k={seqlen_k}, q_heads={nheads_q}, kv_heads={nheads_kv}")
    print(f"  sink={attention_sink}, window={window}, last_n={last_n_blocks}, random={num_random_blocks}")
    print(f"{'='*60}")

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device)
    # For fp8, Q must match K/V dtype (create_test_inputs creates Q as bf16 for fp8)
    if dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
    softmax_scale = head_dim ** (-0.5)

    block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles = build_compact_index_triton_style(
        batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN,
        attention_sink=attention_sink, window=window, last_n_blocks=last_n_blocks,
        num_random_blocks=num_random_blocks, causal=True, device=device)

    print(f"  Index: cu={block_sparse_cu.shape}, idx={block_sparse_idx.shape}, total_q_tiles={total_q_tiles}")

    # Run with split-KV (num_splits > 1)
    out_split = run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                        seqlen_q, head_dim, softmax_scale, causal=True,
                        block_sparse_cu=block_sparse_cu, block_sparse_idx=block_sparse_idx,
                        total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles,
                        num_splits=num_splits)

    # Run without split (num_splits=0 → heuristic, should be 1 for small workloads)
    out_nosplit = run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                          seqlen_q, head_dim, softmax_scale, causal=True,
                          block_sparse_cu=block_sparse_cu, block_sparse_idx=block_sparse_idx,
                          total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles,
                          num_splits=1)

    max_diff = (out_split.float() - out_nosplit.float()).abs().max().item()
    mean_diff = (out_split.float() - out_nosplit.float()).abs().mean().item()

    print(f"  Max diff (split vs nosplit): {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    if max_diff < 0.05:
        print(f"  ✅ PASS")
        return True
    else:
        print(f"  ❌ FAIL (split vs nosplit mismatch)")
        # Find worst row
        diffs_per_row = (out_split.float() - out_nosplit.float()).abs().mean(dim=-1)
        worst_idx = diffs_per_row.argmax().item()
        worst_row = worst_idx // nheads_q
        worst_head = worst_idx % nheads_q
        print(f"  Worst at row={worst_row}, head={worst_head}")
        print(f"    split:   {out_split[worst_row, worst_head, :4]}")
        print(f"    nosplit: {out_nosplit[worst_row, worst_head, :4]}")
        return False


if __name__ == "__main__":
    print("FlashPrefill Block Sparse Attention Test Suite")
    print("=" * 60)

    results = []

    # Test 1: bf16, hdim=128, dense compact index vs dense FA3
    results.append(("bf16 hdim128 dense-vs-dense", test_dense_vs_dense(
        head_dim=128, dtype=torch.bfloat16, batch_size=2,
        seqlen_q=256, seqlen_k=512, nheads_q=8, nheads_kv=2
    )))

    # Test 2: bf16, hdim=128, Triton-style sparse pattern vs reference
    results.append(("bf16 hdim128 triton-sparse", test_triton_style_sparse(
        head_dim=128, dtype=torch.bfloat16, batch_size=2,
        seqlen_q=256, seqlen_k=512, nheads_q=8, nheads_kv=2,
        attention_sink=2, window=4, last_n_blocks=2, num_random_blocks=2
    )))

    # Test 3: fp8, hdim=128, dense index
    results.append(("fp8 hdim128 dense", test_fp8(
        head_dim=128, batch_size=2, seqlen_q=256, seqlen_k=512,
        nheads_q=8, nheads_kv=2
    )))

    # Test 4: bf16, hdim=256, dense index
    results.append(("bf16 hdim256 dense", test_hdim256(
        head_dim=256, dtype=torch.bfloat16, batch_size=2,
        seqlen_q=256, seqlen_k=512, nheads_q=8, nheads_kv=2
    )))

    # Test 5: bf16, hdim=128, Triton-style sparse with different params
    results.append(("bf16 hdim128 triton-sparse v2", test_triton_style_sparse(
        head_dim=128, dtype=torch.bfloat16, batch_size=1,
        seqlen_q=128, seqlen_k=512, nheads_q=8, nheads_kv=2,
        attention_sink=1, window=2, last_n_blocks=1, num_random_blocks=1
    )))

    # Test 6: Split-KV + block sparse correctness
    results.append(("bf16 hdim128 split-kv block sparse", test_split_kv_block_sparse(
        head_dim=128, dtype=torch.bfloat16, batch_size=2,
        seqlen_q=256, seqlen_k=2048, nheads_q=16, nheads_kv=4,
        attention_sink=2, window=4, last_n_blocks=2, num_random_blocks=4,
        num_splits=4
    )))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
