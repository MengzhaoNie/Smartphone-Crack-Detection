from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

def pr_auc_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
    p = probs.detach().float().cpu().numpy().reshape(-1)
    y = labels.detach().float().cpu().numpy().reshape(-1)
    y_bin = (y > 0.5).astype(np.float64)
    if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
        return 0.5
    try:
        return float(average_precision_score(y_bin, p))
    except ValueError:
        return 0.5

@torch.no_grad()
def _collect_pr_auc(model, loader, device) -> float:
    model.eval()
    probs, labels = ([], [])
    for batch in loader:
        if len(batch) == 2:
            inputs, masks = batch
        else:
            inputs, _, masks = batch
        inputs = inputs.to(device)
        masks = masks.to(device)
        logits = model(inputs)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
        probs.append(torch.sigmoid(logits).reshape(-1).cpu())
        labels.append(masks.reshape(-1).cpu())
    if not probs:
        return 0.5
    return pr_auc_from_probs(torch.cat(probs), torch.cat(labels))

def eq7_weight(s_orig: float, s_gan: float, w0: float=0.5, delta_w: float=1.0) -> float:
    w = w0 + delta_w * (1.0 / (1.0 + np.exp(-(s_orig - s_gan))) - 0.5)
    return float(np.clip(w, 0.0, 1.0))

@torch.no_grad()
def compute_dynamic_weight(model, orig_loader, gan_loader, device, w0: float=0.5, delta_w: float=1.0) -> Tuple[float, Dict[str, float]]:
    s_orig = _collect_pr_auc(model, orig_loader, device)
    s_gan = _collect_pr_auc(model, gan_loader, device)
    w_t = eq7_weight(s_orig, s_gan, w0=w0, delta_w=delta_w)
    info = {'S_orig': s_orig, 'S_gan': s_gan, 'w_orig': w_t, 'w_gan': 1.0 - w_t}
    return (w_t, info)
