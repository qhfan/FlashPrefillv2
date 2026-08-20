"""Score builder stage breakdown + K_TILE sweep + fp8/bf16 comparison.

Break down the runtime of the two stages k_mean / fused score+select, to verify:
1. The effect of K_TILE on the fused score+select kernel (tensor core N-dim utilization)
2. fp8 vs bf16 score time (whether the fp8 benefit is realized)
"""
import os
import time
import torch
import triton

import flash_block_sparse_index_triton as M
import test_block_sparse

device = "cuda"
HQ, HKV = 32, 8


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return sorted(ts)[len(ts) // 2]


def run(sq, head_dim, dtype, k_tiles=(16, 32, 64)):
    torch.manual_seed(0)
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = \
        test_block_sparse.create_test_inputs(1, sq, sq, HQ, HKV, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(dtype)
    num_q_heads, num_kv_heads = HQ, HKV
    gqa = HQ // HKV
    page_size = k_cache.shape[1]
    batch = 1

    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    n_tiles = int(((q_lens * gqa + 127) // 128).max().item())
    total_q_tiles = n_tiles * batch
    max_k_blocks = triton.cdiv(sq, 64)
    cu_q_tiles = torch.tensor([0, n_tiles], dtype=torch.int32, device=device)

    k_mean = torch.empty((batch, max_k_blocks, HKV, head_dim), dtype=dtype, device=device)
    out_index = torch.full((HKV, total_q_tiles, max_k_blocks), -1, dtype=torch.int32, device=device)
    counts = torch.zeros(HKV * total_q_tiles, dtype=torch.int32, device=device)
    chunk_max = torch.empty((HKV * total_q_tiles, (max_k_blocks + 15) // 16), dtype=torch.float32, device=device)
    scale_log2 = (head_dim ** -0.5) * 1.4426950408889634

    def fn_mean():
        M._paged_k_mean_kernel[(max_k_blocks, batch * HKV)](
            k_cache, k_mean, k_cache, k_mean, page_table, cache_seqlens, cu_seqlens_q,
            *k_cache.stride(), *k_cache.stride(), *page_table.stride(),
            *k_mean.stride(), *k_mean.stride(),
            num_kv_heads=HKV, gqa_ratio=gqa, page_size=page_size,
            max_k_blocks=max_k_blocks, min_sparse_q_len=0, last_n_blocks=2,
            K_BLOCK_M=128, K_BLOCK_N=64, HEAD_DIM=head_dim, HEAD_DIM_V=head_dim,
            COMPUTE_V_MEAN=False,
            num_warps=M._MEAN_NUM_WARPS, num_stages=M._MEAN_NUM_STAGES)

    t_mean = bench(fn_mean)

    print(f"sq={sq} hd={head_dim} {str(dtype).split('.')[-1]}: k_mean={t_mean:.3f}ms", end="")
    for k_tile in k_tiles:
        def fn_score_select():
            M._packgqa_score_select_kernel[(n_tiles, batch * HKV)](
                q, k_mean, out_index, counts, chunk_max, cu_seqlens_q, cache_seqlens, cu_q_tiles,
                *q.stride(), *k_mean.stride(), *out_index.stride(),
                num_q_heads=HQ, num_kv_heads=HKV, gqa_ratio=gqa,
                total_q_tiles=total_q_tiles, max_k_blocks=max_k_blocks,
                min_sparse_q_len=0, last_n_blocks=2,
                scale_log2=scale_log2, abs_threshold=1.0, attention_sink=2,
                window_size=4, IS_CAUSAL=True,
                K_BLOCK_M=128, K_BLOCK_N=64, HEAD_DIM=head_dim, K_TILE=k_tile,
                num_warps=M._SCORE_NUM_WARPS, num_stages=M._SCORE_NUM_STAGES)
        t_score = bench(fn_score_select)
        # GFLOPs: 2 passes x tiles x hkv x active/2 (average) x 128x64xhdimx2
        avg_blocks = max_k_blocks / 2
        flops = 2 * total_q_tiles * HKV * avg_blocks * 128 * 64 * head_dim * 2
        print(f" | score+select(K{k_tile})={t_score:.3f}ms/{flops/t_score/1e9:.0f}GF", end="")
    print()


def main():
    print(f"device: {torch.cuda.get_device_name(0)}, heads {HQ}/{HKV}")
    for sq in [8192, 32768]:
        for head_dim in [128, 256]:
            run(sq, head_dim, torch.bfloat16)
            run(sq, head_dim, torch.float8_e4m3fn)


if __name__ == "__main__":
    main()
