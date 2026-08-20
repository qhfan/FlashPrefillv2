"""GPU smoke test for flash_block_sparse_bwd_index.

Compares the CUDA count+fill reverse-index builder against the CPU torch
fallback. Run on a machine with CUDA:

    python test_block_sparse_bwd_index_gpu.py
"""

import os
import sys

import torch

if not torch.cuda.is_available():
    print("SKIP: CUDA is not available")
    raise SystemExit(0)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
from flash_block_sparse_bwd_index import build_block_sparse_bwd_index

K_BLOCK_M = 128
K_BLOCK_N = 64


def _prefix_i32(values):
    out = torch.zeros(len(values) + 1, dtype=torch.int32)
    if len(values):
        out[1:] = torch.cumsum(torch.tensor(values, dtype=torch.int32), dim=0)
    return out


def build_forward_csr(seqlens_q, seqlens_k, num_heads_q, num_heads_kv, seed=0, max_random=6, empty=False):
    assert num_heads_q % num_heads_kv == 0
    gqa_ratio = num_heads_q // num_heads_kv
    batch = len(seqlens_q)
    q_tiles = [(sq * gqa_ratio + K_BLOCK_M - 1) // K_BLOCK_M for sq in seqlens_q]
    k_tiles = [(sk + K_BLOCK_N - 1) // K_BLOCK_N for sk in seqlens_k]
    total_q_tiles = sum(q_tiles)
    cu_q_tiles = _prefix_i32(q_tiles)
    rng = np.random.RandomState(seed)

    all_idx = []
    cu = [0]
    for h_kv in range(num_heads_kv):
        for b in range(batch):
            sq = seqlens_q[b]
            sk = seqlens_k[b]
            n_k = k_tiles[b]
            prefix_len = sk - sq
            for p in range(q_tiles[b]):
                if empty or n_k <= 0:
                    cu.append(len(all_idx))
                    continue
                selected = set()
                # sink
                selected.update(range(min(2, n_k)))
                # diagonal/local block, using the same floor mapping as forward PackGQA
                q_pos = (p * K_BLOCK_M) // gqa_ratio
                if q_pos < sq:
                    diag = (prefix_len + q_pos) // K_BLOCK_N
                    if 0 <= diag < n_k:
                        selected.add(diag)
                        if diag > 0:
                            selected.add(diag - 1)
                remaining = [x for x in range(n_k) if x not in selected]
                if remaining and max_random > 0:
                    n_sel = int(rng.randint(0, min(max_random, len(remaining)) + 1))
                    if n_sel > 0:
                        selected.update(rng.choice(remaining, size=n_sel, replace=False).tolist())
                all_idx.extend(sorted(selected))
                cu.append(len(all_idx))

    block_sparse_cu = torch.tensor(cu, dtype=torch.int32)
    block_sparse_idx = torch.tensor(all_idx, dtype=torch.int32)
    assert block_sparse_cu.numel() == num_heads_kv * total_q_tiles + 1
    return block_sparse_cu, block_sparse_idx, cu_q_tiles, total_q_tiles


def _seg_values(bwd_cu, bwd_idx, sid):
    s = int(bwd_cu[sid].item())
    e = int(bwd_cu[sid + 1].item())
    return sorted(bwd_idx[s:e].cpu().tolist())


def compare_cpu_gpu(name, seqlens_q, seqlens_k, num_heads_q, num_heads_kv, seed, max_random=6, empty=False, pad_max_k=None):
    block_sparse_cu, block_sparse_idx, cu_q_tiles, total_q_tiles = build_forward_csr(
        seqlens_q, seqlens_k, num_heads_q, num_heads_kv, seed=seed, max_random=max_random, empty=empty
    )
    cu_seqlens_q = _prefix_i32(seqlens_q)
    cu_seqlens_k = _prefix_i32(seqlens_k)
    max_seqlen_k = max(seqlens_k) if pad_max_k is None else pad_max_k
    assert max_seqlen_k >= max(seqlens_k)

    cpu_out = build_block_sparse_bwd_index(
        block_sparse_cu, block_sparse_idx, cu_q_tiles, cu_seqlens_q, cu_seqlens_k,
        num_heads_q, num_heads_kv, total_q_tiles, max_seqlen_k,
    )
    gpu_args = tuple(x.cuda() for x in (block_sparse_cu, block_sparse_idx, cu_q_tiles, cu_seqlens_q, cu_seqlens_k))
    gpu_out = build_block_sparse_bwd_index(
        *gpu_args,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        total_q_tiles=total_q_tiles,
        max_seqlen_k=max_seqlen_k,
    )
    torch.cuda.synchronize()

    cpu_cu, cpu_idx, cpu_mk = cpu_out
    gpu_cu, gpu_idx, gpu_mk = gpu_out
    assert cpu_mk == gpu_mk
    assert cpu_cu.cpu().tolist() == gpu_cu.cpu().tolist(), f"{name}: bwd_cu mismatch"
    # EXPAND=2: each forward entry expands to 2 packed_m64 entries.
    assert int(gpu_cu[-1].item()) == 2 * block_sparse_idx.numel()

    num_segments = cpu_cu.numel() - 1
    for sid in range(num_segments):
        cpu_vals = _seg_values(cpu_cu, cpu_idx, sid)
        gpu_vals = _seg_values(gpu_cu, gpu_idx, sid)
        if cpu_vals != gpu_vals:
            raise AssertionError(f"{name}: segment {sid} mismatch\n cpu={cpu_vals[:16]}\n gpu={gpu_vals[:16]}")
    print(f"OK {name}: segments={num_segments}, slots={int(gpu_cu[-1].item())}, max_k_tiles={cpu_mk}")


def main():
    torch.manual_seed(0)
    compare_cpu_gpu("explicit-g7", [128], [192], 7, 1, seed=1, max_random=0)
    compare_cpu_gpu("empty", [128, 64], [192, 320], 8, 2, seed=2, empty=True)
    compare_cpu_gpu("random-g1", [37, 128, 200], [100, 300, 512], 2, 2, seed=3)
    compare_cpu_gpu("random-g4-varlen", [513, 1024, 777], [2048, 4096, 1536], 8, 2, seed=4, max_random=24)
    compare_cpu_gpu("random-g7", [37, 128, 200], [100, 300, 512], 14, 2, seed=5)
    compare_cpu_gpu("random-g12", [37, 128, 200], [100, 300, 512], 24, 2, seed=6)
    compare_cpu_gpu("padded-maxk", [128, 64], [192, 320], 8, 2, seed=7, pad_max_k=1024)
    print("All GPU reverse-index smoke tests passed")


if __name__ == "__main__":
    main()
