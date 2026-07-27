from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm

def _mean_std(values) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    t = torch.tensor(values, dtype=torch.float64)
    return (float(t.mean()), float(t.std(unbiased=False)) if t.numel() > 1 else (float(t.mean()), 0.0))

@torch.no_grad()
def calculate_per_image_metrics(model, data_loader, device, threshold: float=0.633, predict_fn: Optional[Callable]=None) -> Dict[str, float]:
    model.eval()
    ious, dices = ([], [])
    precs, recs = ([], [])
    total_tp = total_fp = total_fn = 0.0
    for batch in tqdm(data_loader, desc='metrics', leave=False):
        if predict_fn is not None:
            outputs, masks = predict_fn(model, batch, device)
        else:
            if len(batch) == 2:
                inputs, masks = batch
            else:
                inputs, _, masks = batch
            if inputs.dim() == 5:
                inputs = inputs[:, inputs.shape[1] // 2]
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model(inputs)
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
        pred = (torch.sigmoid(outputs) > threshold).float()
        b = pred.size(0)
        pred_f = pred.view(b, -1)
        mask_f = masks.view(b, -1)
        tp = (pred_f * mask_f).sum(dim=1)
        fp = pred_f.sum(dim=1) - tp
        fn = mask_f.sum(dim=1) - tp
        eps = 1e-08
        iou = (tp + eps) / (tp + fp + fn + eps)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        prec = (tp + eps) / (tp + fp + eps)
        rec = (tp + eps) / (tp + fn + eps)
        ious.extend(iou.tolist())
        dices.extend(dice.tolist())
        precs.extend(prec.tolist())
        recs.extend(rec.tolist())
        total_tp += float(tp.sum())
        total_fp += float(fp.sum())
        total_fn += float(fn.sum())
    eps = 1e-08
    precision = (total_tp + eps) / (total_tp + total_fp + eps)
    recall = (total_tp + eps) / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    miou, miou_std = _mean_std(ious)
    mdice, mdice_std = _mean_std(dices)
    mprec, mprec_std = _mean_std(precs)
    mrec, mrec_std = _mean_std(recs)
    return {'iou': miou, 'dice': mdice, 'miou': miou, 'mdice': mdice, 'miou_std': miou_std, 'mdice_std': mdice_std, 'precision': precision, 'recall': recall, 'f1': f1, 'mprecision': mprec, 'mrecall': mrec, 'mprecision_std': mprec_std, 'mrecall_std': mrec_std, 'n_images': float(len(ious))}
