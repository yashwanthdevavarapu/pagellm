from __future__ import annotations

import numpy as np
import pytest

from helpers import tiny_model
from pagellm.model.tokenizer import CharTokenizer
from pagellm.model.transformer import Transformer
from pagellm.model.weights import DATA_PATH, WEIGHTS_DIR


@pytest.fixture(scope="session")
def tok() -> CharTokenizer:
    return CharTokenizer.from_text(DATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def model(tok) -> Transformer:
    return tiny_model(tok.vocab_size)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


needs_weights = pytest.mark.skipif(
    not (WEIGHTS_DIR / "target.npz").exists(), reason="bundled weights not present"
)
