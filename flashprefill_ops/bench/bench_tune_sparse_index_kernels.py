from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import torch
import triton


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("flash_block_sparse_index_triton", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def gmean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bench(fn, warmup, rep):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def launch_mean(module, tensors, warps, stages, compute_v_mean):
    k_cache, v_cache, k_mean, v_mean, page_table, cache_seqlens, cu_q = tensors
    max_k_blocks = k_mean.shape[1]
    module._paged_k_mean_kernel[(max_k_blocks, k_cache.shape[2])](
        k_cache,
        k_mean,
        v_cache if compute_v_mean else k_cache,
        v_mean,
        page_table,
        cache_seqlens,
        cu_q,
        *k_cache.stride(),
        *v_cache.stride(),
        *page_table.stride(),
        *k_mean.stride(),
        *v_mean.stride(),
        num_kv_heads=k_cache.shape[2],
        gqa_ratio=1,
        page_size=k_cache.shape[1],
        max_k_blocks=max_k_blocks,
        min_sparse_q_len=0,
        last_n_blocks=2,
        K_BLOCK_M=128,
        K_BLOCK_N=64,
        HEAD_DIM=k_cache.shape[3],
        HEAD_DIM_V=k_cache.shape[3],
        COMPUTE_V_MEAN=compute_v_mean,
        num_warps=warps,
        num_stages=stages,
    )


def launch_score_select(module, tensors, k_tile, warps, stages):
    q, k_mean, out_index, counts, chunk_max, cu_q, cache_seqlens, cu_q_tiles = tensors
    num_q_heads = q.shape[1]
    num_kv_heads = k_mean.shape[2]
    max_q_tiles = int((cu_q_tiles[1] - cu_q_tiles[0]).item())
    module._packgqa_score_select_kernel[(max_q_tiles, num_kv_heads)](
        q,
        k_mean,
        out_index,
        counts,
        chunk_max,
        cu_q,
        cache_seqlens,
        cu_q_tiles,
        *q.stride(),
        *k_mean.stride(),
        *out_index.stride(),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_ratio=num_q_heads // num_kv_heads,
        total_q_tiles=out_index.shape[1],
        max_k_blocks=out_index.shape[2],
        min_sparse_q_len=0,
        last_n_blocks=2,
        scale_log2=math.log2(math.e) / math.sqrt(q.shape[2]),
        abs_threshold=1.0,
        attention_sink=2,
        window_size=4,
        IS_CAUSAL=True,
        K_BLOCK_M=128,
        K_BLOCK_N=64,
        HEAD_DIM=q.shape[2],
        K_TILE=k_tile,
        num_warps=warps,
        num_stages=stages,
    )


def make_case(seq_len, num_q_heads, num_kv_heads, dtype, head_dim=128, page_size=64):
    max_k_blocks = triton.cdiv(seq_len, 64)
    gqa_ratio = num_q_heads // num_kv_heads
    q_tiles = triton.cdiv(seq_len * gqa_ratio, 128)
    num_pages = triton.cdiv(seq_len, page_size)
    q_source = torch.randn((seq_len, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_source = torch.randn(
        (num_pages, page_size, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_source = torch.randn(
        (num_pages, page_size, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    q = q_source if dtype == torch.bfloat16 else q_source.to(dtype)
    k_cache = k_source if dtype == torch.bfloat16 else k_source.to(dtype)
    v_cache = v_source if dtype == torch.bfloat16 else v_source.to(dtype)
    k_mean = torch.empty(
        (1, max_k_blocks, num_kv_heads, head_dim), device="cuda", dtype=dtype
    )
    v_mean = torch.empty_like(k_mean)
    page_table = torch.arange(num_pages, device="cuda", dtype=torch.int32).unsqueeze(0)
    cache_seqlens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    cu_q_tiles = torch.tensor([0, q_tiles], device="cuda", dtype=torch.int32)
    out_index = torch.full((num_kv_heads, q_tiles, max_k_blocks), -1, device="cuda", dtype=torch.int32)
    counts = torch.empty(num_kv_heads * q_tiles, device="cuda", dtype=torch.int32)
    chunk_max = torch.empty(
        (num_kv_heads * q_tiles, (max_k_blocks + 15) // 16), device="cuda", dtype=torch.float32
    )
    return {
        "mean": (k_cache, v_cache, k_mean, v_mean, page_table, cache_seqlens, cu_q),
        "score_select": (q, k_mean, out_index, counts, chunk_max, cu_q, cache_seqlens, cu_q_tiles),
    }


def record(rows, kernel, dtype_name, q_heads, kv_heads, seq_len, page_size, v_mean, k_tile, warps, stages, ms):
    rows.append(
        {
            "kernel": kernel,
            "dtype": dtype_name,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "seq_len": seq_len,
            "page_size": page_size,
            "v_mean": int(v_mean),
            "k_tile": "" if k_tile is None else k_tile,
            "num_warps": warps,
            "num_stages": stages,
            "median_ms": ms,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--output", default="sparse_index_all_kernel_tuning.csv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    module = load_module(root / "flash_block_sparse_index_triton.py")

    lengths = [500, 4096, 16384, 65536]
    heads = [(32, 8)]
    dtypes = [("bf16", torch.bfloat16), ("fp8_e4m3", torch.float8_e4m3fn)]
    page_sizes = [1]           # 1 = bench/production pagedKV path (_PAGE1_*); 64 = regular paging (_MEAN_*)
    v_flags = [False, True]    # True = use_mean_correction path
    warp_stage = [(w, s) for w in (1, 2, 4, 8) for s in (1, 2, 3, 4, 5)]
    score_configs = [(k, w, s) for k in (16, 32, 64) for w, s in warp_stage]
    rows = []

    def report(tag, timings):
        complete = [(gmean(values), config, values) for config, values in timings.items() if len(values) == len(lengths)]
        complete.sort()
        if not complete:
            print(f"NO_VALID_CONFIG {tag}", flush=True)
            return
        best, (k_tile, warps, stages), values = complete[0]
        print(
            f"BEST {tag} k_tile={str(k_tile):>4s} warps={warps} stages={stages} gmean_ms={best:.6f} "
            f"times={[round(x, 6) for x in values]}",
            flush=True,
        )
        print(
            "TOP3 " + ", ".join(
                f"k={config[0]},w={config[1]},s={config[2]}:{value:.6f}"
                for value, config, _ in complete[:3]
            ),
            flush=True,
        )

    print(f"GPU={torch.cuda.get_device_name()} lengths={lengths} page_sizes={page_sizes}", flush=True)
    for dtype_name, dtype in dtypes:
        for q_heads, kv_heads in heads:
            # score_select is independent of page_size / v_mean; sweep only once
            score_agg = {}
            for seq_len in lengths:
                case = make_case(seq_len, q_heads, kv_heads, dtype)
                launch_score_select(module, case["score_select"], 16, 4, 1)
                torch.cuda.synchronize()
                for k_tile, warps, stages in score_configs:
                    try:
                        ms = bench(
                            lambda k=k_tile, w=warps, s=stages: launch_score_select(module, case["score_select"], k, w, s),
                            args.warmup,
                            args.rep,
                        )
                    except Exception as exc:
                        print(f"SKIP score_select {dtype_name} {q_heads}/{kv_heads} L={seq_len} k={k_tile} w={warps} s={stages}: {exc}", flush=True)
                        continue
                    score_agg.setdefault((k_tile, warps, stages), []).append(ms)
                    record(rows, "score_select", dtype_name, q_heads, kv_heads, seq_len, "", False, k_tile, warps, stages, ms)
                del case
                torch.cuda.empty_cache()
            report(f"kernel=score_select dtype={dtype_name} heads={q_heads}/{kv_heads}", score_agg)

            # mean: sweep over page_size x v_mean combinations
            for page_size in page_sizes:
                for v_mean in v_flags:
                    mean_agg = {}
                    for seq_len in lengths:
                        case = make_case(seq_len, q_heads, kv_heads, dtype, page_size=page_size)
                        launch_mean(module, case["mean"], 1, 2, v_mean)
                        torch.cuda.synchronize()
                        for warps, stages in warp_stage:
                            try:
                                ms = bench(
                                    lambda w=warps, s=stages: launch_mean(module, case["mean"], w, s, v_mean),
                                    args.warmup,
                                    args.rep,
                                )
                            except Exception as exc:
                                print(f"SKIP mean {dtype_name} {q_heads}/{kv_heads} L={seq_len} ps={page_size} v={int(v_mean)} w={warps} s={stages}: {exc}", flush=True)
                                continue
                            mean_agg.setdefault((None, warps, stages), []).append(ms)
                            record(rows, "mean", dtype_name, q_heads, kv_heads, seq_len, page_size, v_mean, None, warps, stages, ms)
                        del case
                        torch.cuda.empty_cache()
                    report(f"kernel=mean dtype={dtype_name} heads={q_heads}/{kv_heads} ps={page_size} v_mean={int(v_mean)}", mean_agg)

    output = root / args.output
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
