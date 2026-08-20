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


def make_inputs(seq_len: int, num_kv_heads: int, dtype: torch.dtype, head_dim: int, page_size: int):
    num_pages = triton.cdiv(seq_len, page_size)
    source = torch.randn(
        (num_pages, page_size, num_kv_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = source if dtype == torch.bfloat16 else source.to(dtype)
    page_table = torch.arange(num_pages, device="cuda", dtype=torch.int32).unsqueeze(0)
    cache_seqlens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    return k_cache, page_table, cache_seqlens


def launch(kernel, k_cache, page_table, cache_seqlens, output, seq_len, block_n, head_dim, warps, stages):
    num_kv_heads = k_cache.shape[2]
    max_k_blocks = triton.cdiv(seq_len, block_n)
    kernel[(max_k_blocks, num_kv_heads)](
        k_cache,
        output,
        page_table,
        cache_seqlens,
        *k_cache.stride(),
        *page_table.stride(),
        *output.stride(),
        num_kv_heads=num_kv_heads,
        page_size=k_cache.shape[1],
        max_k_blocks=max_k_blocks,
        K_BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paged_k_mean_tuning.csv")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=60)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    module = load_module(root / "flash_block_sparse_index_triton.py")
    kernel = module._paged_k_mean_kernel
    lengths = [512, 4096, 16384, 65536]
    head_configs = [(32, 8), (16, 4), (24, 2)]
    dtypes = [("bf16", torch.bfloat16), ("fp8_e4m3", torch.float8_e4m3fn)]
    configs = [(w, s) for w in (1, 2, 4, 8) for s in (1, 2, 3, 4)]
    head_dim = 128
    page_size = 64
    block_n = 64
    rows = []

    print(f"gpu={torch.cuda.get_device_name()} head_dim={head_dim} page_size={page_size} block_n={block_n}")
    for dtype_name, dtype in dtypes:
        for num_q_heads, num_kv_heads in head_configs:
            timings = {config: [] for config in configs}
            for seq_len in lengths:
                k_cache, page_table, cache_seqlens = make_inputs(
                    seq_len, num_kv_heads, dtype, head_dim, page_size
                )
                output = torch.empty(
                    (1, triton.cdiv(seq_len, block_n), num_kv_heads, head_dim),
                    device="cuda",
                    dtype=dtype,
                )
                baseline = None
                for warps, stages in configs:
                    launch(
                        kernel, k_cache, page_table, cache_seqlens, output,
                        seq_len, block_n, head_dim, warps, stages,
                    )
                    torch.cuda.synchronize()
                    current = output.float().clone()
                    if baseline is None:
                        baseline = current
                    elif not torch.equal(current, baseline):
                        raise RuntimeError(
                            f"output mismatch: {dtype_name} {num_q_heads}/{num_kv_heads} "
                            f"len={seq_len} warps={warps} stages={stages}"
                        )
                    ms = triton.testing.do_bench(
                        lambda: launch(
                            kernel, k_cache, page_table, cache_seqlens, output,
                            seq_len, block_n, head_dim, warps, stages,
                        ),
                        warmup=args.warmup,
                        rep=args.rep,
                        return_mode="median",
                    )
                    timings[(warps, stages)].append(ms)
                    rows.append({
                        "dtype": dtype_name,
                        "q_heads": num_q_heads,
                        "kv_heads": num_kv_heads,
                        "seq_len": seq_len,
                        "num_warps": warps,
                        "num_stages": stages,
                        "median_ms": ms,
                    })
                del k_cache, page_table, cache_seqlens, output, baseline
            ranked = []
            for config, values in timings.items():
                geometric_mean = math.exp(sum(math.log(x) for x in values) / len(values))
                ranked.append((geometric_mean, config, values))
            ranked.sort()
            best_gmean, (best_warps, best_stages), best_values = ranked[0]
            print(
                f"BEST dtype={dtype_name:9s} heads={num_q_heads:2d}/{num_kv_heads:<2d} "
                f"warps={best_warps} stages={best_stages} gmean={best_gmean:.6f} ms "
                f"length_ms={dict(zip(lengths, [round(x, 6) for x in best_values]))}"
            )
            print("  top3=" + ", ".join(
                f"w{w}s{s}:{g:.6f}" for g, (w, s), _ in ranked[:3]
            ))

    output_path = root / args.output
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
