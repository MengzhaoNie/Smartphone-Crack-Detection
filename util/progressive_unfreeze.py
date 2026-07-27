
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch.nn as nn


STAGE_ORDER = ("decoder", "cnn_kan", "full")


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def count_trainable(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def apply_unfreeze_stage(model: nn.Module, stage: str) -> Dict[str, int]:

    stage = stage.lower().strip()
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage={stage}. Choose from {STAGE_ORDER}")


    if hasattr(model, "ukan") and hasattr(model, "vssm") and hasattr(model, "decoder"):
        _set_requires_grad(model, False)

        _set_requires_grad(model.decoder, True)

        for name in ("fusion0", "fusion1", "fusion2", "fusion3", "fusion4"):
            if hasattr(model, name):
                _set_requires_grad(getattr(model, name), True)

        if stage in ("cnn_kan", "full"):
            _set_requires_grad(model.ukan, True)
        if stage == "full":
            _set_requires_grad(model.vssm, True)

        n_train, n_all = count_trainable(model)
        return {"stage": stage, "trainable": n_train, "total": n_all, "kind": "dual_encoder"}


    if hasattr(model, "encoder") and hasattr(model, "decoder"):
        _set_requires_grad(model, False)
        _set_requires_grad(model.decoder, True)
        if hasattr(model, "head"):
            _set_requires_grad(model.head, True)
        if stage != "decoder":
            _set_requires_grad(model.encoder, True)
        n_train, n_all = count_trainable(model)
        return {"stage": stage, "trainable": n_train, "total": n_all, "kind": "encoder_decoder"}


    children = list(model.named_children())
    if not children:
        _set_requires_grad(model, True)
        n_train, n_all = count_trainable(model)
        return {"stage": stage, "trainable": n_train, "total": n_all, "kind": "fallback_all"}

    _set_requires_grad(model, False)

    n = len(children)
    cut = max(1, n // 2)
    decoder_names = [n for n, _ in children[cut:]]
    encoder_names = [n for n, _ in children[:cut]]
    for name in decoder_names:
        _set_requires_grad(dict(children)[name], True)
    if stage != "decoder":

        enc_cut = len(encoder_names) if stage == "full" else max(1, len(encoder_names) // 2)
        for name in encoder_names[:enc_cut] if stage == "cnn_kan" else encoder_names:

            pass
        if stage == "cnn_kan":
            for name in encoder_names[len(encoder_names) // 2 :]:
                _set_requires_grad(dict(children)[name], True)
        else:
            for name in encoder_names:
                _set_requires_grad(dict(children)[name], True)

    n_train, n_all = count_trainable(model)
    return {"stage": stage, "trainable": n_train, "total": n_all, "kind": "fallback"}


def parse_stage_epochs(spec: str, total_epochs: int) -> List[Tuple[str, int, int]]:

    parts = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) == 1:
        parts = [parts[0], 0, max(0, total_epochs - parts[0])]
    while len(parts) < 3:
        parts.append(0)
    parts = parts[:3]
    s = sum(parts)
    if s <= 0:
        parts = [total_epochs // 3, total_epochs // 3, total_epochs - 2 * (total_epochs // 3)]
    elif s != total_epochs:
        parts[-1] = max(0, total_epochs - parts[0] - parts[1])

    schedule = []
    cursor = 0
    for stage, length in zip(STAGE_ORDER, parts):
        if length <= 0:
            continue
        schedule.append((stage, cursor, cursor + length))
        cursor += length
    if not schedule:
        schedule = [("full", 0, total_epochs)]
    return schedule


def stage_at_epoch(schedule: Sequence[Tuple[str, int, int]], epoch: int) -> str:
    for stage, start, end in schedule:
        if start <= epoch < end:
            return stage
    return schedule[-1][0]
