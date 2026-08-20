# SGLang Integration of FlashPrefill V2

This directory is an **SGLang 0.5.10 source tree** with the FlashPrefill V2
attention backend integrated. Only four touch points differ from upstream
0.5.10, all additive (no upstream code is modified or removed):

| File | Change |
|---|---|
| `python/sglang/srt/layers/attention/flashprefill_backend.py` | **New.** The backend itself (552 LoC). Wraps the `flashprefill` operator: builds block-sparse metadata for extend/prefill batches and delegates decode steps verbatim to a standard FA3 backend (`FlashAttentionBackend`), since a single query token per step leaves no room for block sparsity. Sliding-window layers and the first `--flashprefill-full-attention-layers` layers are also delegated to FA3. |
| `python/sglang/srt/layers/attention/attention_registry.py` | Registers `create_flashprefill_backend` under the name `"flashprefill"`, wiring the CLI hyper-parameters (`--flashprefill-*`) from `runner.server_args` into the backend constructor. |
| `python/sglang/srt/server_args.py` | Adds 8 dataclass fields with defaults (`flashprefill_attention_sink=2`, `flashprefill_window=4`, `flashprefill_abs_threshold=1.0`, `flashprefill_full_attention_layers=0`, `flashprefill_last_n_blocks=2`, `flashprefill_min_sparse_q_len=4096`, `flashprefill_k_block_n=64`, `flashprefill_use_mean_correction=False`), the matching `--flashprefill-*` argparse options, and `"flashprefill"` in `ATTENTION_BACKEND_CHOICES`. |

## Architecture

```
extend / prefill batch                decode step
        │                                   │
FlashPrefillAttnBackend          FlashAttentionBackend (FA3)
        │                                   │
  FlashPrefill (operator)          dense paged attention
        │
  Stage 1: index selection (Triton, flash_block_sparse_index_triton)
           -> CSR index per (KV head, packed Q tile)
  Stage 2: warp-specialized block-sparse attention kernel
           over the selected blocks (+ mean correction)
```

## Usage

Install the operator first (`bash install_ops.sh` at the repo root — the
backend imports `flashprefill` and `flash_block_sparse_index_triton` at
runtime), then route Python to this tree and launch:

```bash
export PYTHONPATH=/path/to/FlashPrefillv2/sglang_int/python:${PYTHONPATH}

python -m sglang.launch_server --model-path <MODEL> \
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

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--flashprefill-attention-sink` | 2 | Number of always-selected sink blocks at sequence start. |
| `--flashprefill-window` | 4 | Local sliding-window size (in blocks) always selected. |
| `--flashprefill-abs-threshold` | 1.0 | Max-based dynamic threshold alpha in (0, 1]; keeps blocks whose tile-level score energy >= alpha * max block energy in the same (KV head, Q tile) segment. Larger = sparser. |
| `--flashprefill-full-attention-layers` | 0 | Keep the first N layers (0-based layer_id < N) on full dense attention. |
| `--flashprefill-last-n-blocks` | 2 | Number of trailing blocks always selected. |
| `--flashprefill-min-sparse-q-len` | 4096 | Batches whose max Q length is at or below this fall back to dense attention entirely. Set to 0 to always allow sparsity. |
| `--flashprefill-k-block-n` | 64 | Logical K-block size in tokens for scoring/selection; power-of-two multiple of 64, expanded into 64-token attention tiles by the index builder. |
| `--flashprefill-use-mean-correction` | off | Enable the zero-order mean correction: unselected KV blocks contribute their pooled K/V means inside the attention epilogue (forward only). |
