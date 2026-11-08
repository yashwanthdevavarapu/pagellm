"""pagellm: a vLLM-style LLM inference server built from scratch in numpy."""

from __future__ import annotations

import os

# Pin BLAS to one thread before numpy loads (ADR 0011). This model's matmuls are
# tiny, and waking a BLAS thread pool for each one costs far more than the math:
# measured 9.5x slower for an 80-row matmul with the default thread pool.
# setdefault keeps any explicit user choice; it has no effect if numpy was
# imported before pagellm.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("PAGELLM_BLAS_THREADS", "1"))

from .engine import (  # noqa: E402
    CompletionOutput,
    EngineConfig,
    LLMEngine,
    SamplingParams,
    SpeculativeDecoder,
    StepOutput,
)
from .model import WEIGHTS_DIR, CharTokenizer, ModelConfig, Transformer, load_pretrained  # noqa: E402

__version__ = "1.0.0"

__all__ = [
    "WEIGHTS_DIR",
    "CharTokenizer",
    "CompletionOutput",
    "EngineConfig",
    "LLMEngine",
    "ModelConfig",
    "SamplingParams",
    "SpeculativeDecoder",
    "StepOutput",
    "Transformer",
    "__version__",
    "load_pretrained",
]
