"""
Comparison test: flashprefill (this repo) vs flash_attn_interface (upstream FA3)
Only dense attention is compared; upstream FA3 has no block sparse
"""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os
import torch
import numpy as np

# Set up the environment
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/python3.12/dist-packages/torch/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from flashprefill import flash_attn_interface as my_impl
from flash_attn_interface import flash_attn_func as std_impl

torch.manual_seed(42)
device = "cuda"

def run_comparison(dtype, head_dim, seq_len, batch=1, nheads=8, causal=True):
    """Compare a single configuration"""
    # Generate inputs
    if dtype == torch.bfloat16:
        q = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.bfloat16, device=device)
        dt_str = "bf16"
    elif dtype == torch.float8_e4m3fn:
        # FA3 fp8 inputs are fp8, but scale is None (default 1.0)
        q_f32 = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.float32, device=device)
        k_f32 = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.float32, device=device)
        v_f32 = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.float32, device=device)
        q = q_f32.to(torch.float8_e4m3fn)
        k = k_f32.to(torch.float8_e4m3fn)
        v = v_f32.to(torch.float8_e4m3fn)
        dt_str = "fp8"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Run this repo's implementation
    try:
        out_my = my_impl.flash_attn_func(
            q, k, v, causal=causal, num_splits=1
        )
    except Exception as e:
        print(f"  [{dt_str} hdim={head_dim} seq={seq_len}] my_impl ERROR: {e}")
        return None

    # Run the standard implementation
    try:
        out_std = std_impl(
            q, k, v, causal=causal, num_splits=1
        )
    except Exception as e:
        print(f"  [{dt_str} hdim={head_dim} seq={seq_len}] std_impl ERROR: {e}")
        return None

    # Compare results
    out_my_f = out_my.float()
    out_std_f = out_std.float()

    max_diff = (out_my_f - out_std_f).abs().max().item()
    mean_diff = (out_my_f - out_std_f).abs().mean().item()
    max_abs = out_std_f.abs().max().item()
    rel_diff = max_diff / max_abs if max_abs > 0 else float('inf')

    # Also compare the mean/std of the outputs
    my_mean = out_my_f.mean().item()
    std_mean = out_std_f.mean().item()
    my_std = out_my_f.std().item()
    std_std = out_std_f.std().item()

    status = "PASS" if rel_diff < 0.05 else "FAIL"
    print(f"  [{dt_str} hdim={head_dim} seq={seq_len}] {status}  "
          f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
          f"rel_diff={rel_diff:.4f}  max_abs={max_abs:.4f}  "
          f"my_mean={my_mean:.4f} std_mean={std_mean:.4f}  "
          f"my_std={my_std:.4f} std_std={std_std:.4f}")

    return {"max_diff": max_diff, "mean_diff": mean_diff, "rel_diff": rel_diff, "status": status}


def main():
    print("=" * 100)
    print("Comparison test: flashprefill (this repo) vs flash_attn_interface (upstream FA3)")
    print("Mode: dense attention, causal=True")
    print("=" * 100)

    configs = [
        # (dtype, head_dim, seq_len)
        (torch.bfloat16, 128, 4096),
        (torch.bfloat16, 128, 16384),
        (torch.bfloat16, 128, 65536),
        (torch.bfloat16, 256, 4096),
        (torch.bfloat16, 256, 16384),
        (torch.bfloat16, 256, 65536),
        (torch.float8_e4m3fn, 128, 4096),
        (torch.float8_e4m3fn, 128, 16384),
        (torch.float8_e4m3fn, 128, 65536),
        (torch.float8_e4m3fn, 256, 4096),
        (torch.float8_e4m3fn, 256, 16384),
        (torch.float8_e4m3fn, 256, 65536),
    ]

    results = []
    for dtype, head_dim, seq_len in configs:
        result = run_comparison(dtype, head_dim, seq_len, batch=1, nheads=8, causal=True)
        if result:
            results.append({
                "dtype": "bf16" if dtype == torch.bfloat16 else "fp8",
                "head_dim": head_dim,
                "seq_len": seq_len,
                **result
            })

    print()
    print("=" * 100)
    print("Summary:")
    print(f"{'dtype':<6} {'hdim':<6} {'seq_len':<10} {'status':<8} {'max_diff':<12} {'mean_diff':<12} {'rel_diff':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['dtype']:<6} {r['head_dim']:<6} {r['seq_len']:<10} {r['status']:<8} "
              f"{r['max_diff']:<12.6f} {r['mean_diff']:<12.6f} {r['rel_diff']:<10.4f}")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\nPASS: {n_pass}/{len(results)}  FAIL: {n_fail}/{len(results)}")


if __name__ == "__main__":
    main()
