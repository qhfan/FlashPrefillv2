#!/usr/bin/env bash
# One-shot build & install of the FlashPrefill V2 operator (Hopper SM90: H20/H100).
# Installs via pip: the built wheel already contains _C.abi3.so and the
# top-level module, so no manual copying is needed.
#
# Usage:  bash install_ops.sh   (run from the repository root)
set -euo pipefail
cd "$(dirname "$0")/flashprefill_ops"

# ---- 0. Dependency check (torch must be installed beforehand, matching the
#         system CUDA major version) ----
# ninja/packaging/einops are declared in setup.py install_requires and are
# installed automatically by pip; triton ships with the torch CUDA wheel, so
# installing it manually could pull a version incompatible with torch. Don't.
python3 -c "import torch, triton" || { echo "ERROR: install a torch build matching your system CUDA first (triton ships with torch)"; exit 1; }

# ---- 1. Clean stale build artifacts (objects cached with old flags would
#         silently mix into the new build) ----
rm -rf build/ dist/ *.egg-info
find . -path ./third_party -prune -o -name "*.so" -exec rm -f {} +
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ---- 2. Build configuration (paper's H20 setup: forward only, BF16+FP8,
#         head-dim 128/256, SM90 only) ----
# NOTE: without FLASH_ATTENTION_FORCE_BUILD, setup.py downloads an official
# upstream FlashAttention-3 wheel instead — you would install the wrong package!
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE    # the variable setup.py reads is SM80 (not SM8x)
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export MAX_JOBS="${MAX_JOBS:-8}"            # nvcc parallelism; each TU needs ~4-6GB RAM, lower it if OOM

# ---- 3. Build & install (--no-build-isolation is required: the build imports torch) ----
# If the system CUDA is neither 12.8 nor >= 13.0, setup.py downloads an
# nvcc 12.6 + ptxas 12.8 toolchain from developer.download.nvidia.com (cached
# under ~/.flashattn) and needs network access. For offline machines, copy
# ~/.flashattn from an existing machine or set FLASH_ATTENTION_HOME to a shared cache.
pip install . --no-build-isolation -v

# ---- 4. Smoke test ----
cd ..
python3 -c "import torch; import flashprefill; import flashprefill._C; import flash_block_sparse_index_triton; print('flashprefill', flashprefill.__version__, 'installed OK')"
echo "Done. Full functional verification (needs a GPU): python eval_install.py"
