"""Locations of the bundled assets and a loader for the pretrained models."""

from __future__ import annotations

from pathlib import Path

from .tokenizer import CharTokenizer
from .transformer import Transformer

PACKAGE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PACKAGE_DIR / "weights"
DATA_PATH = PACKAGE_DIR / "data" / "tinyshakespeare.txt"
PRETRAINED = ("target", "draft")


def load_pretrained(name: str = "target") -> tuple[Transformer, CharTokenizer]:
    """Load a bundled char-level Shakespeare model: ``"target"`` or ``"draft"``."""
    path = WEIGHTS_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no bundled model {name!r} (looked for {path})")
    model, meta = Transformer.load(path)
    return model, CharTokenizer(meta["tokenizer"])
