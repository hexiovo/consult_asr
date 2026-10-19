"""Resolve offline pyannote checkpoints relative to their pipeline directory."""
from __future__ import annotations

from pathlib import Path


def resolve_checkpoint(reference: str, model_root: Path) -> Path:
    root = model_root.expanduser().resolve()
    configured = Path(reference).expanduser()
    checkpoint = (configured if configured.is_absolute() else root / configured).resolve()
    try:
        checkpoint.relative_to(root)
    except ValueError:
        raise ValueError("checkpoint must remain inside the local pyannote model directory") from None
    if checkpoint.name.casefold() != "pytorch_model.bin" or not any(
        part.casefold() == "pyannote" for part in checkpoint.parts
    ):
        raise ValueError("expected a local pyannote pytorch_model.bin checkpoint")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError("local pyannote checkpoint is missing or empty")
    return checkpoint
