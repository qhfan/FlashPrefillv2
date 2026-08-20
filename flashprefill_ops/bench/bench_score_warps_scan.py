"""Sweep num_warps x num_stages of _packgqa_score_select_kernel (tuning the 2-pass fused variant)."""
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


def run(sq, head_dim, dtype):
    torch.manual_seed(0)
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = \
        test_block_sparse.create_test_inputs(1, sq, sq, HQ, HKV, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(dtype)
    gqa = HQ // HKV
    page_size = k_cache.shape[1]
    n_tiles = (sq * gqa + 127) // 128
    total_q_tiles = n_tiles
    max_k_blocks = triton.cdiv(sq, 64)
    cu_q_tiles = torch.tensor([0, n_tiles], dtype=torch.int32, device=device)

    k_mean = torch.empty((1, max_k_blocks, HKV, head_dim), dtype=dtype, device=device)
    out_index = torch.full((HKV, total_q_tiles, max_k_blocks), -1, dtype=torch.int32, device=device)
    counts = torch.zeros(HKV * total_q_tiles, dtype=torch.int32, device=device)
    chunk_max = torch.empty((HKV * total_q_tiles, (max_k_blocks + 15) // 16), dtype=torch.float32, device=device)
    scale_log2 = (head_dim ** -0.5) * 1.4426950408889634

    M._paged_k_mean_kernel[(max_k_blocks, HKV)](
        k_cache, k_mean, k_cache, k_mean, page_table, cache_seqlens, cu_seqlens_q,
        *k_cache.stride(), *k_cache.stride(), *page_table.stride(),
        *k_mean.stride(), *k_mean.stride(),
        num_kv_heads=HKV, gqa_ratio=gqa, page_size=page_size,
        max_k_blocks=max_k_blocks, min_sparse_q_len=0, last_n_blocks=2,
        K_BLOCK_M=128, K_BLOCK_N=64, HEAD_DIM=head_dim, HEAD_DIM_V=head_dim,
        COMPUTE_V_MEAN=False,
        num_warps=M._MEAN_NUM_WARPS, num_stages=M._MEAN_NUM_STAGES)
    torch.cuda.synchronize()

    dname = str(dtype).split(".")[-1].replace("float8_e4m3fn", "fp8").replace("bfloat16", "bf16")
    print(f"sq={sq} hd={head_dim} {dname}:")
    results = []
    for nw in (1, 2, 4, 8):
        row = []
        for ns in (1, 2, 3, 4, 5):
            def fn():
                M._packgqa_score_select_kernel[(n_tiles, HKV)](
                    q, k_mean, out_index, counts, chunk_max, cu_seqlens_q, cache_seqlens, cu_q_tiles,
                    *q.stride(), *k_mean.stride(), *out_index.stride(),
                    num_q_heads=HQ, num_kv_heads=HKV, gqa_ratio=gqa,
                    total_q_tiles=total_q_tiles,
                    max_k_blocks=max_k_blocks, min_sparse_q_len=0, last_n_blocks=2,
                    scale_log2=scale_log2, abs_threshold=1.0,
                    attention_sink=2, window_size=4,
                    IS_CAUSAL=True,
                    K_BLOCK_M=128, K_BLOCK_N=64, HEAD_DIM=head_dim, K_TILE=16,
                    num_warps=nw, num_stages=ns)
            try:
                t = bench(fn)
            except Exception:
                t = float("inf")
            row.append(t)
            results.append((t, nw, ns))
        print("  warps=%d: %s" % (nw, "  ".join(f"ns{ns}={t:.3f}" for ns, t in zip((1, 2, 3, 4, 5), row))))
    results.sort()
    t, nw, ns = results[0]
    print(f"  BEST: warps={nw} stages={ns} {t:.3f}ms")


def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    for sq in [8192, 32768]:
        for head_dim in [128, 256]:
            run(sq, head_dim, torch.bfloat16)
            run(sq, head_dim, torch.float8_e4m3fn)


if __name__ == "__main__":
    main()
