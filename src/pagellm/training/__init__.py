"""Numpy training loop with a hand-derived backward pass."""

from .trainer import PRESETS, AdamW, loss_and_grads, train

__all__ = ["PRESETS", "AdamW", "loss_and_grads", "train"]
