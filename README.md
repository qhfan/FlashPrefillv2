# FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving

This repository contains the code for **FlashPrefill V2**: a block-sparse
prefill attention system for long-context LLM inference, with

- an **FA3/4-aligned sparse attention operator** (CUDA/CuTe, Hopper SM90) with
  PackGQA memory access, warp specialization, pingpong pipelining, and BF16/FP8
  support;
- a **zero-order mean correction term** that compensates pruned blocks with
  their pooled K/V statistics inside the softmax computation;
- **native paged-KV-cache and continuous-batching integration**, demonstrated
  with an SGLang attention backend.

Paper: link to be added.

## Repository Layout

```
FlashPrefillv2/
├── flashprefill_ops/      # the block-sparse attention operator (CUDA/CuTe C++, SM90)
│   ├── setup.py, generate_kernels.py, instantiations/   # build system
│   ├── flashprefill/      # installed Python package (FlashPrefill, interfaces, config)
│   ├── flash_block_sparse_index_triton.py               # Triton index-selection stage
│   ├── test_*.py          # correctness tests (dense/FA3 comparison, FP8, mean correction, ...)
│   └── bench/             # benchmark scripts
├── csrc/cutlass/          # vendored CUTLASS 4.3.4 headers (compile-time only)
├── sglang_int/            # SGLang 0.5.10 source tree with the FlashPrefill V2
│                          # attention backend integrated (--prefill-attention-backend flashprefill)
├── install_ops.sh         # one-shot operator build & install
└── eval_install.py        # installation smoke test covering the main APIs
```

## Requirements

- NVIDIA Hopper GPU (H20 / H100, sm_90a)
- CUDA 12.x toolkit with `nvcc` on `PATH`
- PyTorch (built against the same CUDA major version), `triton` (ships with
  the torch CUDA wheel)

## Build & Install the Operator

CUTLASS headers are vendored under `csrc/cutlass/`; `setup.py` picks them up
automatically.

```bash
bash install_ops.sh
```

The script cleans stale build artifacts, sets the build flags used for the
paper's H20 setup (forward only, BF16+FP8, head-dim 128/256, SM90), runs
`pip install . --no-build-isolation -v` inside `flashprefill_ops/`, and ends
with an import smoke test. The installed artifacts are the `flashprefill`
package (including the compiled `_C.abi3.so` extension) and the top-level
module `flash_block_sparse_index_triton`.

Three common pitfalls:
- **`--no-build-isolation` is required** (already in `install_ops.sh`): the
  build imports `torch`, which does not exist in an isolated build env.
- **`FLASH_ATTENTION_FORCE_BUILD=TRUE` is required**: without it, `setup.py`
  falls back to downloading an official upstream FlashAttention-3 wheel and
  installs the wrong package.
- When the system CUDA is neither 12.8 nor >= 13.0, `setup.py` downloads an
  nvcc 12.6 + ptxas 12.8 toolchain from `developer.download.nvidia.com`
  (cached under `~/.flashattn`). For offline machines, copy `~/.flashattn`
  from a machine that already has it, or point `FLASH_ATTENTION_HOME` at a
  shared cache.

### Verify the Installation

```bash
python eval_install.py
```

Covers, on a paged KV cache (token-level paging): `flash_attn_func`,
`flash_attn_with_kvcache` (dense), `FlashPrefill.index_select`,
`block_sparse_attention`, the full sparse pipeline with and without mean
correction, the two-step high-level call, the low-level direct call, and the
FP8 pipeline. All checks are smoke tests (run + shape/dtype/finiteness); for
numerical validation use `flashprefill_ops/test_compare_fa3.py` and
`flashprefill_ops/test_mean_correction.py`.

## Usage

### High-level pipeline (recommended)

`FlashPrefill` bundles index selection and sparse attention, and manages the
index workspace and correction statistics internally:

```python
import torch
from flashprefill import FlashPrefill

fp = FlashPrefill(
    k_block_m=128, k_block_n=128,   # query-tile / selection-block sizes
    abs_threshold=0.1,              # max-based selection threshold (higher = sparser)
    attention_sink=2,               # always-selected sink blocks
    window_size=4,                  # always-selected local window blocks
    last_n_blocks=8,
    use_mean_correction=True,       # zero-order correction of pruned blocks
)

# q:        (total_q, num_q_heads, head_dim)   varlen-packed queries
# k/v_cache:(num_pages, page_size, num_kv_heads, head_dim)   paged KV cache
# page_table: (batch, max_pages) int32;  cache_seqlens: (batch,) int32
# cu_seqlens_q: (batch+1,) int32
out = fp(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
         q_lens=q_lens)            # -> (total_q, num_q_heads, head_dim)
```

To reuse an index across calls, split the pipeline:

```python
index = fp.index_select(q, k_cache, page_table, cache_seqlens, cu_seqlens_q,
                        v_cache=v_cache, q_lens=q_lens)
out = fp.block_sparse_attention(q, k_cache, v_cache, page_table,
                                cache_seqlens, cu_seqlens_q, index,
                                max_seqlen_q=max(q_lens))
```

`k_mean` / `v_mean` / `block_sparse_*` are produced together with the index
and wired automatically — do not pass them manually.

### Low-level direct call

For full control over the index/workspace lifecycle:

```python
from flash_block_sparse_index_triton import (
    SparseIndexWorkspace, build_block_sparse_index_fast)
from flashprefill.flash_attn_interface import flash_attn_with_kvcache

ws = SparseIndexWorkspace(
    batch_size=B, num_kv_heads=NKV, head_dim=D,
    total_q_tiles=total_tiles, max_q_tiles=max_tiles,
    max_k_blocks=max_k_blocks, dtype=torch.bfloat16, device="cuda",
    cu_q_tiles=cu_q_tiles, n_sub=k_block_n // 64,
    use_mean_correction=True)

cu, idx, total_tiles, cu_q_tiles = build_block_sparse_index_fast(
    q, k_cache, page_table, cache_seqlens, cu_seqlens_q, ws,
    v_cache=v_cache, k_block_m=128, k_block_n=128,
    abs_threshold=0.1, attention_sink=2, window_size=4, last_n_blocks=8,
    causal=True)

out = flash_attn_with_kvcache(
    q, k_cache, v_cache, page_table=page_table, cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_q_len, causal=True,
    block_sparse_cu=cu, block_sparse_idx=idx,
    total_q_tiles=total_tiles, cu_q_tiles=cu_q_tiles,
    k_mean=ws.k_mean, v_mean=ws.v_mean,
    mean_k_block_size=128)          # must equal the selection k_block_n; 0 disables correction
```

### Serving with SGLang

`sglang_int/` is an SGLang 0.5.10 source tree with the FlashPrefill V2
backend integrated. Route Python to it and launch a server:

```bash
REPO=/path/to/FlashPrefillv2
export PYTHONPATH=${REPO}/sglang_int/python:${PYTHONPATH}

python -m sglang.launch_server --model-path <MODEL> \
    --tensor-parallel-size 4 \
    --prefill-attention-backend flashprefill \
    --decode-attention-backend fa3 \
    --flashprefill-attention-sink 2 \
    --flashprefill-window 4 \
    --flashprefill-abs-threshold 0.1 \
    --flashprefill-full-attention-layers 4 \
    --flashprefill-last-n-blocks 8 \
    --flashprefill-k-block-n 128 \
    --flashprefill-min-sparse-q-len 0 \
    --flashprefill-use-mean-correction
```

The operator must be installed first (`bash install_ops.sh`); the backend
imports `flashprefill` and `flash_block_sparse_index_triton` at runtime.
`--flashprefill-full-attention-layers N` routes the first N layers to the
decode (dense) backend.

## Tests & Benchmarks

```bash
cd flashprefill_ops
python test_compare_fa3.py        # dense attention vs FA3 (requires flash-attn 3)
python test_mean_correction.py    # corrected vs uncorrected sparse pipeline
python test_block_sparse.py       # block-sparse forward correctness
ls bench/                         # operator- and index-level benchmark scripts
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). It bundles
or derives from the following third-party projects, which retain their own
licenses:

- `flashprefill_ops/` derives from FlashAttention-3 (BSD-3-Clause);
- `csrc/cutlass/` contains NVIDIA CUTLASS headers (BSD-3-Clause);
- `sglang_int/` contains the SGLang source tree (Apache-2.0).

## Acknowledgements

FlashPrefill V2 builds on the following open-source projects:

- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — our attention operator is derived from the FlashAttention-3 Hopper implementation (TMA, warp specialization, pingpong scheduling), extended to block-sparse computation.
- [FlashPrefill](https://github.com/qhfan/FlashPrefill) — the block-level score estimation and max-based dynamic thresholding this work evolves from.
- [SGLang](https://github.com/sgl-project/sglang) — the serving framework our attention backend plugs into (`sglang_int/` is an SGLang 0.5.10 tree with the backend integrated).

## Citation

Paper link to be added.
