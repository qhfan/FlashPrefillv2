"""
Speed comparison: flashprefill (this repo) vs flash_attn_interface (upstream FA3)
Dense attention, causal=True, bf16 and fp8, hdim=128 and 256
"""
import sys, os, time
import torch
import numpy as np

os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/python3.12/dist-packages/torch/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flashprefill import flash_attn_interface as my_impl
from flash_attn_interface import flash_attn_func as std_impl

torch.manual_seed(42)
device = "cuda"

SEQ_LENS = [4096, 8192, 16384, 32768, 65536]
WARMUP = 3
ITERS = 15


def benchmark_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = sorted(times)
    return times[len(times) // 2]


def run_comparison(dtype, head_dim, seq_len, batch=1, nheads=8, causal=True):
    """Compare a single configuration"""
    if dtype == torch.bfloat16:
        q = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.bfloat16, device=device)
        dt_str = "bf16"
    elif dtype == torch.float8_e4m3fn:
        q_f32 = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.float32, device=device)
        k_f32 = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.float32, device=device)
        v_f32 = torch.randn(batch, seq_len, nheads, head_dim, dtype=torch.float32, device=device)
        q = q_f32.to(torch.float8_e4m3fn)
        k = k_f32.to(torch.float8_e4m3fn)
        v = v_f32.to(torch.float8_e4m3fn)
        dt_str = "fp8"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Benchmark my impl
    fn_my = lambda: my_impl.flash_attn_func(q, k, v, causal=causal, num_splits=1)
    med_my = benchmark_fn(fn_my)

    # Benchmark std impl
    fn_std = lambda: std_impl(q, k, v, causal=causal, num_splits=1)
    med_std = benchmark_fn(fn_std)

    speedup = med_std / med_my if med_my > 0 else 0

    print(f"  [{dt_str:>4} hdim={head_dim:>3} seq={seq_len:>6}]  "
          f"my={med_my*1000:>8.3f}ms  std={med_std*1000:>8.3f}ms  "
          f"ratio(std/my)={speedup:>5.2f}x  "
          f"diff={(med_std-med_my)*1000:>+8.3f}ms")

    return {
        "dtype": dt_str, "hdim": head_dim, "seq": seq_len,
        "my_ms": med_my * 1000, "std_ms": med_std * 1000, "speedup": speedup
    }


def main():
    print("=" * 110)
    print("Speed comparison: flashprefill (this repo) vs flash_attn_interface (upstream FA3)")
    print("Mode: dense attention, causal=True, MHA, batch=1, nheads=8")
    print(f"Warmup={WARMUP}, Iters={ITERS}, median timing")
    print("=" * 100)

    all_results = []

    for head_dim in [128, 256]:
        for dtype, dt_name in [(torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
            print(f"\n--- {dt_name} hdim={head_dim} ---")
            print(f"  {'config':>30}  {'my_impl':>10}  {'std_FA3':>10}  {'std/my':>8}  {'diff':>10}")
            print(f"  {'-'*85}")

            for sq in SEQ_LENS:
                r = run_comparison(dtype, head_dim, sq)
                all_results.append(r)
                torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*110}")
    print("Summary:")
    print(f"  {'dtype':>5} {'hdim':>5} {'seq':>7}  {'my_impl(ms)':>12} {'std_FA3(ms)':>12} {'std/my':>8}  {'diff(ms)':>10}")
    print(f"  {'-'*75}")
    for r in all_results:
        print(f"  {r['dtype']:>5} {r['hdim']:>5} {r['seq']:>7}  {r['my_ms']:>12.3f} {r['std_ms']:>12.3f} {r['speedup']:>7.2f}x  {(r['std_ms']-r['my_ms']):>+9.3f}")

    print(f"\n{'='*110}")
    print("Notes:")
    print("  - std/my > 1.0: flashprefill is faster")
    print("  - std/my < 1.0: standard FA3 is faster")
    print("  - diff = std - my (positive = my is faster)")


if __name__ == "__main__":
    main()
