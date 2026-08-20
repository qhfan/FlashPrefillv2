"""
Compare output consistency of pack_gqa=True vs pack_gqa=False in block_sparse mode.

Key differences:
- pack_gqa=True:  total_q_tiles = ceil_div(seqlen_q * gqa_ratio, kBlockM) * batch, bidh iterates h_kv
- pack_gqa=False: total_q_tiles = ceil_div(seqlen_q, kBlockM) * batch, bidh iterates h_q, segment = bidh // gqa_ratio

The two modes have different sparse index structures, but the selected K-blocks (per Q position) should be identical.
"""

import torch
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_block_sparse import (
    get_tile_sizes, create_test_inputs,
    m_block_to_q_pos, m_block_to_q_pos_end, q_pos_to_k_block,
)
from flash_attn_interface import flash_attn_with_kvcache

def run_fa3_packgqa(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
            seqlen_q, head_dim, softmax_scale, causal=True,
            block_sparse_cu=None, block_sparse_idx=None, total_q_tiles=None, cu_q_tiles=None,
            num_splits=0):
    """Run flashprefill with or without block sparse."""
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

torch.manual_seed(42)
device = "cuda"


def build_compact_index_packgqa(
    batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
    kBlockM, kBlockN,
    attention_sink=2, window=4, last_n_blocks=2, num_random_blocks=2,
    causal=True, device="cuda", rng_seed=42,
):
    """Build compact index for PackGQA=True mode.
    
    In PackGQA mode, m_block iterates over seqlen_q * gqa_ratio rows.
    total_q_tiles = ceil_div(seqlen_q * gqa_ratio, kBlockM) * batch_size
    Segment: g = total_q_tiles * h_kv + (cu_q_tiles[b] + m_block)
    """
    gqa_ratio = nheads_q // nheads_kv
    n_q_tiles_per_batch = (seqlen_q * gqa_ratio + kBlockM - 1) // kBlockM
    n_k_tiles_per_batch = (seqlen_k + kBlockN - 1) // kBlockN
    total_q_tiles = n_q_tiles_per_batch * batch_size

    cu_q_tiles = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        cu_q_tiles[b + 1] = cu_q_tiles[b] + n_q_tiles_per_batch

    rng = np.random.RandomState(rng_seed)
    positions_per_m_block = max(1, kBlockM // gqa_ratio)
    n_q_pos_blocks = (seqlen_q + positions_per_m_block - 1) // positions_per_m_block
    last_n_q_pos_start = max(0, n_q_pos_blocks - last_n_blocks)

    all_indices = []
    cu_offsets = [0]

    for h_kv in range(nheads_kv):
        for b in range(batch_size):
            prefix_len = seqlen_k - seqlen_q
            for m_block in range(n_q_tiles_per_batch):
                q_pos_start = m_block_to_q_pos(m_block, kBlockM, gqa_ratio)
                q_pos_end = m_block_to_q_pos_end(m_block, kBlockM, gqa_ratio, seqlen_q)

                if q_pos_start >= seqlen_q:
                    all_indices.extend([])
                    cu_offsets.append(len(all_indices))
                    continue

                q_k_blk = q_pos_to_k_block(q_pos_start, prefix_len, kBlockN)
                causal_max_n = min(q_k_blk + 1, n_k_tiles_per_batch) if causal else n_k_tiles_per_batch

                q_pos_blk_idx = q_pos_start // positions_per_m_block
                is_last_n = q_pos_blk_idx >= last_n_q_pos_start

                selected = set()
                sink_end = min(attention_sink, causal_max_n)
                selected.update(range(sink_end))
                window_start = max(0, q_k_blk - window + 1)
                window_end = min(q_k_blk + 1, causal_max_n)
                selected.update(range(window_start, window_end))
                if is_last_n:
                    selected.update(range(causal_max_n))
                else:
                    remaining = [k for k in range(causal_max_n) if k not in selected]
                    if len(remaining) > 0 and num_random_blocks > 0:
                        n_select = min(num_random_blocks, len(remaining))
                        chosen = rng.choice(remaining, size=n_select, replace=False)
                        selected.update(chosen.tolist())

                selected_sorted = sorted(selected)
                all_indices.extend(selected_sorted)
                cu_offsets.append(len(all_indices))

    block_sparse_cu = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(all_indices, dtype=torch.int32, device=device)
    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


def build_compact_index_no_packgqa(
    batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
    kBlockM, kBlockN,
    attention_sink=2, window=4, last_n_blocks=2, num_random_blocks=2,
    causal=True, device="cuda", rng_seed=42,
):
    """Build compact index for PackGQA=False mode.
    
    In non-PackGQA mode, m_block iterates over seqlen_q rows directly.
    total_q_tiles = ceil_div(seqlen_q, kBlockM) * batch_size
    Segment: g = total_q_tiles * (bidh // gqa_ratio) + (cu_q_tiles[b] + m_block)
    All Q heads in the same group share the same segment.
    """
    gqa_ratio = nheads_q // nheads_kv
    n_q_tiles_per_batch = (seqlen_q + kBlockM - 1) // kBlockM
    n_k_tiles_per_batch = (seqlen_k + kBlockN - 1) // kBlockN
    total_q_tiles = n_q_tiles_per_batch * batch_size

    cu_q_tiles = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        cu_q_tiles[b + 1] = cu_q_tiles[b] + n_q_tiles_per_batch

    rng = np.random.RandomState(rng_seed)
    n_q_pos_blocks = (seqlen_q + kBlockM - 1) // kBlockM
    last_n_q_pos_start = max(0, n_q_pos_blocks - last_n_blocks)

    all_indices = []
    cu_offsets = [0]

    # Segments are organized per KV head (h_kv heads, not h_q)
    for h_kv in range(nheads_kv):
        for b in range(batch_size):
            prefix_len = seqlen_k - seqlen_q
            for m_block in range(n_q_tiles_per_batch):
                # Direct Q position mapping (no GQA packing)
                q_pos_start = m_block * kBlockM
                q_pos_end = min((m_block + 1) * kBlockM, seqlen_q)

                if q_pos_start >= seqlen_q:
                    all_indices.extend([])
                    cu_offsets.append(len(all_indices))
                    continue

                q_k_blk = q_pos_to_k_block(q_pos_start, prefix_len, kBlockN)
                causal_max_n = min(q_k_blk + 1, n_k_tiles_per_batch) if causal else n_k_tiles_per_batch

                q_pos_blk_idx = q_pos_start // kBlockM
                is_last_n = q_pos_blk_idx >= last_n_q_pos_start

                selected = set()
                sink_end = min(attention_sink, causal_max_n)
                selected.update(range(sink_end))
                window_start = max(0, q_k_blk - window + 1)
                window_end = min(q_k_blk + 1, causal_max_n)
                selected.update(range(window_start, window_end))
                if is_last_n:
                    selected.update(range(causal_max_n))
                else:
                    remaining = [k for k in range(causal_max_n) if k not in selected]
                    if len(remaining) > 0 and num_random_blocks > 0:
                        n_select = min(num_random_blocks, len(remaining))
                        chosen = rng.choice(remaining, size=n_select, replace=False)
                        selected.update(chosen.tolist())

                selected_sorted = sorted(selected)
                all_indices.extend(selected_sorted)
                cu_offsets.append(len(all_indices))

    block_sparse_cu = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(all_indices, dtype=torch.int32, device=device)
    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


def test_block_sparse_packgqa_vs_nopackgqa(
    head_dim=128, dtype=torch.bfloat16, batch_size=2,
    seqlen_q=256, seqlen_k=512, nheads_q=8, nheads_kv=2,
    attention_sink=2, window=4, last_n_blocks=2, num_random_blocks=2,
    num_splits=0,
):
    """Compare block_sparse output between pack_gqa=True and pack_gqa=False."""
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

    print(f"\n{'='*70}")
    print(f"Block Sparse: pack_gqa=True vs pack_gqa=False")
    print(f"  head_dim={head_dim}, dtype={dtype}, kBlockM={kBlockM}, kBlockN={kBlockN}")
    print(f"  batch={batch_size}, q={seqlen_q}, k={seqlen_k}, q_heads={nheads_q}, kv_heads={nheads_kv}")
    print(f"  sink={attention_sink}, window={window}, last_n={last_n_blocks}, random={num_random_blocks}")
    print(f"  num_splits={num_splits}")
    print(f"{'='*70}")

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
    softmax_scale = head_dim ** (-0.5)

    # Build index for PackGQA=True
    bs_cu_pg, bs_idx_pg, total_q_tiles_pg, cu_q_tiles_pg = build_compact_index_packgqa(
        batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN,
        attention_sink=attention_sink, window=window, last_n_blocks=last_n_blocks,
        num_random_blocks=num_random_blocks, causal=True, device=device)

    # Build index for PackGQA=False
    bs_cu_np, bs_idx_np, total_q_tiles_np, cu_q_tiles_np = build_compact_index_no_packgqa(
        batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN,
        attention_sink=attention_sink, window=window, last_n_blocks=last_n_blocks,
        num_random_blocks=num_random_blocks, causal=True, device=device)

    print(f"  PackGQA=True:  cu={bs_cu_pg.shape}, idx={bs_idx_pg.shape}, total_q_tiles={total_q_tiles_pg}")
    print(f"  PackGQA=False: cu={bs_cu_np.shape}, idx={bs_idx_np.shape}, total_q_tiles={total_q_tiles_np}")

    # Run with pack_gqa mode index
    out_pg = run_fa3_packgqa(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                     seqlen_q, head_dim, softmax_scale, causal=True,
                     block_sparse_cu=bs_cu_pg, block_sparse_idx=bs_idx_pg,
                     total_q_tiles=total_q_tiles_pg, cu_q_tiles=cu_q_tiles_pg,
                     num_splits=num_splits)

    # Run with no-pack_gqa mode index
    out_np = run_fa3_packgqa(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                     seqlen_q, head_dim, softmax_scale, causal=True,
                     block_sparse_cu=bs_cu_np, block_sparse_idx=bs_idx_np,
                     total_q_tiles=total_q_tiles_np, cu_q_tiles=cu_q_tiles_np,
                     num_splits=num_splits)

    max_diff = (out_pg.float() - out_np.float()).abs().max().item()
    mean_diff = (out_pg.float() - out_np.float()).abs().mean().item()

    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    if max_diff < 0.05:
        print(f"  PASS")
        return True
    else:
        print(f"  FAIL")
        diffs_per_row = (out_pg.float() - out_np.float()).abs().mean(dim=-1)
        worst_idx = diffs_per_row.argmax().item()
        worst_row = worst_idx // nheads_q
        worst_head = worst_idx % nheads_q
        print(f"  Worst at row={worst_row}, head={worst_head}")
        print(f"    pack_gqa=True:  {out_pg[worst_row, worst_head, :4]}")
        print(f"    pack_gqa=False: {out_np[worst_row, worst_head, :4]}")
        return False


def bench_block_sparse_packgqa_vs_nopackgqa(
    head_dim=128, dtype=torch.bfloat16, batch_size=2,
    seqlen_q=4096, seqlen_k=4096, nheads_q=8, nheads_kv=2,
    attention_sink=2, window=4, last_n_blocks=2, num_random_blocks=4,
    num_splits=0, warmup=5, reps=20,
):
    """Benchmark block_sparse speed: pack_gqa=True vs pack_gqa=False."""
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

    print(f"\n{'='*70}")
    print(f"Benchmark: block_sparse pack_gqa=True vs False")
    print(f"  head_dim={head_dim}, dtype={dtype}, kBlockM={kBlockM}, kBlockN={kBlockN}")
    print(f"  batch={batch_size}, q={seqlen_q}, k={seqlen_k}, q_heads={nheads_q}, kv_heads={nheads_kv}")
    print(f"  sink={attention_sink}, window={window}, last_n={last_n_blocks}, random={num_random_blocks}")
    print(f"{'='*70}")

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
    softmax_scale = head_dim ** (-0.5)

    bs_cu_pg, bs_idx_pg, total_q_tiles_pg, cu_q_tiles_pg = build_compact_index_packgqa(
        batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN,
        attention_sink=attention_sink, window=window, last_n_blocks=last_n_blocks,
        num_random_blocks=num_random_blocks, causal=True, device=device)

    bs_cu_np, bs_idx_np, total_q_tiles_np, cu_q_tiles_np = build_compact_index_no_packgqa(
        batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN,
        attention_sink=attention_sink, window=window, last_n_blocks=last_n_blocks,
        num_random_blocks=num_random_blocks, causal=True, device=device)

    def run_packgqa(pg):
        bs_cu = bs_cu_pg if pg else bs_cu_np
        bs_idx = bs_idx_pg if pg else bs_idx_np
        total_q_tiles = total_q_tiles_pg if pg else total_q_tiles_np
        cu_q_tiles = cu_q_tiles_pg if pg else cu_q_tiles_np
        return run_fa3_packgqa(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                       seqlen_q, head_dim, softmax_scale, causal=True,
                       block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
                       total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles,
                       num_splits=num_splits)

    # Warmup + correctness check
    out_pg = run_packgqa(True)
    out_np = run_packgqa(False)
    max_diff = (out_pg.float() - out_np.float()).abs().max().item()
    print(f"  Correctness: max_diff={max_diff:.6e} {'PASS' if max_diff < 0.05 else 'FAIL'}")

    # Benchmark
    for label, pg in [("pack_gqa=True", True), ("pack_gqa=False", False)]:
        times = []
        for _ in range(warmup):
            run_packgqa(pg)
        torch.cuda.synchronize()
        for _ in range(reps):
            t0 = time.perf_counter()
            run_packgqa(pg)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        mean_t = np.mean(times)
        std_t = np.std(times)
        print(f"  {label:20s}: {mean_t:.3f} +- {std_t:.3f} ms")

    # Speedup
    torch.cuda.synchronize()
    times_pg, times_np = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_packgqa(True)
        torch.cuda.synchronize()
        times_pg.append((time.perf_counter() - t0) * 1000)
    for _ in range(reps):
        t0 = time.perf_counter()
        run_packgqa(False)
        torch.cuda.synchronize()
        times_np.append((time.perf_counter() - t0) * 1000)
    speedup = np.mean(times_np) / np.mean(times_pg)
    print(f"  Speedup (pack_gqa=True / pack_gqa=False): {speedup:.2f}x")


if __name__ == "__main__":
    print("Block Sparse: pack_gqa=True vs pack_gqa=False")
    print("=" * 70)

    results = []

    # --- Correctness tests ---
    configs = [
        # (head_dim, dtype, batch, q, k, q_heads, kv_heads, sink, window, last_n, random, splits)
        (128, torch.bfloat16, 2, 256, 512, 8, 2, 2, 4, 2, 2, 0),
        (128, torch.bfloat16, 2, 256, 512, 8, 1, 2, 4, 2, 2, 0),
        (128, torch.bfloat16, 1, 512, 1024, 16, 2, 2, 4, 2, 4, 0),
        (128, torch.bfloat16, 2, 1024, 2048, 8, 2, 2, 4, 2, 4, 0),
        (128, torch.bfloat16, 2, 1024, 2048, 32, 8, 2, 4, 2, 4, 0),
        (128, torch.bfloat16, 2, 4096, 4096, 8, 2, 2, 4, 2, 4, 0),
        (128, torch.bfloat16, 2, 4096, 4096, 8, 2, 2, 4, 2, 4, 4),
        (256, torch.bfloat16, 2, 256, 512, 8, 2, 2, 4, 2, 2, 0),
        (256, torch.bfloat16, 2, 1024, 2048, 8, 2, 2, 4, 2, 4, 0),
        (128, torch.bfloat16, 2, 256, 512, 4, 2, 2, 4, 2, 2, 0),
        (128, torch.bfloat16, 2, 256, 512, 16, 4, 2, 4, 2, 2, 0),
        (128, torch.bfloat16, 2, 256, 512, 32, 4, 2, 4, 2, 2, 0),
    ]

    for i, (hd, dt, bs, sq, sk, hq, hkv, sink, win, ln, rand, splits) in enumerate(configs):
        name = f"cfg{i}: hd={hd} {dt} bs={bs} q={sq} k={sk} hq={hq} hkv={hkv} sink={sink} win={win} ln={ln} rand={rand} splits={splits}"
        try:
            ok = test_block_sparse_packgqa_vs_nopackgqa(
                head_dim=hd, dtype=dt, batch_size=bs,
                seqlen_q=sq, seqlen_k=sk, nheads_q=hq, nheads_kv=hkv,
                attention_sink=sink, window=win, last_n_blocks=ln, num_random_blocks=rand,
                num_splits=splits)
            results.append((name, ok))
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # --- Benchmark ---
    bench_configs = [
        (128, torch.bfloat16, 2, 4096, 4096, 8, 2, 2, 4, 2, 4, 0),
        (128, torch.bfloat16, 2, 8192, 8192, 8, 2, 2, 4, 2, 4, 0),
        (128, torch.bfloat16, 2, 8192, 8192, 8, 2, 2, 4, 2, 4, 4),
        (128, torch.bfloat16, 2, 16384, 16384, 8, 2, 2, 4, 2, 4, 0),
        (256, torch.bfloat16, 2, 4096, 4096, 8, 2, 2, 4, 2, 4, 0),
        (256, torch.bfloat16, 2, 8192, 8192, 8, 2, 2, 4, 2, 4, 0),
        (128, torch.bfloat16, 2, 4096, 4096, 16, 2, 2, 4, 2, 4, 0),
        (128, torch.bfloat16, 2, 4096, 4096, 32, 8, 2, 4, 2, 4, 0),
    ]

    for i, (hd, dt, bs, sq, sk, hq, hkv, sink, win, ln, rand, splits) in enumerate(bench_configs):
        try:
            bench_block_sparse_packgqa_vs_nopackgqa(
                head_dim=hd, dtype=dt, batch_size=bs,
                seqlen_q=sq, seqlen_k=sk, nheads_q=hq, nheads_kv=hkv,
                attention_sink=sink, window=win, last_n_blocks=ln, num_random_blocks=rand,
                num_splits=splits)
        except Exception as e:
            print(f"  Bench FAIL: {e}")
            import traceback
            traceback.print_exc()

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY (Correctness)")
    print(f"{'='*70}")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
