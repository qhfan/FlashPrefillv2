from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flashprefill")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "3.0.0"

# Import _C to register torch ops
import flashprefill._C  # noqa: E402,F401

# Expose block sparse index builder (disabled - kernel has bug)
# from flash_attn_interface import build_block_sparse_index  # noqa: E402,F401

from flashprefill.prefill import FlashPrefill, SparseIndex  # noqa: E402,F401
