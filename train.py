from __future__ import annotations
import argparse
import os
import types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data import get_dataloader
from util.metrics import calculate_per_image_metrics as calculate_epoch_metrics

def build_model(name: str, args):
    name = name.lower().replace('-', '_')
    if name == 'unet':
        from models.baselines.Unet import create_unet
        return create_unet(num_classes=1)
    if name == 'resunet':
        from models.baselines.Resunet import create_resunet
        return create_resunet(num_classes=1)
    if name == 'skpnet':
        from models.baselines.SKPNet import create_skpnet
        return create_skpnet(num_classes=1)
    if name == 'mambacracknet':
        from models.baselines.Mambacracknet import create_mambacracknet
        return create_mambacracknet(num_classes=1, img_size=args.img_size)
    if name == 'fpn':
        from models.baselines.FPN import create_fpn
        return create_fpn(num_classes=1)
    if name == 'pspnet':
        from models.baselines.PsPnet import create_pspnet
        return create_pspnet(num_classes=1)
    if name == 'deeplabv3':
        from models.baselines.DeepLabV3 import create_deeplabv3
        return create_deeplabv3(num_classes=1)
    if name == 'deeplabv3plus':
        from models.baselines.DeepLabV3Plus import create_deeplabv3plus
        return create_deeplabv3plus(num_classes=1)
    if name == 'transunet':
        from models.baselines.TransUNet import create_transunet
        return create_transunet(img_size=args.img_size, num_classes=1)
    if name == 'crackformer':
        from models.baselines.CrackFormer import create_crackformer
        return create_crackformer(num_classes=1)
    if name == 'defnet':
        from models.baselines.DefNet import create_defnet
        return create_defnet(num_classes=1, img_size=args.img_size)
    if name == 'ukan_only':
        from models.ablation.CNN_Kan import create_ukan_only
        return create_ukan_only(num_classes=1)
    if name == 'vmamba_only':
        from models.ablation.vmamba_only import create_vmamba_only
        return create_vmamba_only(num_classes=1)
    if name == 'dual_encoder_concat':
        from models.ablation.dual_encoder_concat import create_dual_encoder_concat
        return create_dual_encoder_concat(num_classes=1)
    if name in ('dual_encoder', 'teacher'):
        from models.ablation.dual_encoder import create_dual_encoder
        return create_dual_encoder(num_classes=1)
    raise ValueError(f'Unknown model: {name}. Choose from: unet, resunet, skpnet, mambacracknet, fpn, pspnet, deeplabv3, deeplabv3plus, transunet, crackformer, defnet, ukan_only, vmamba_only, dual_encoder_concat, dual_encoder')
AVAILABLE_MODELS = ['unet', 'resunet', 'skpnet', 'mambacracknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'transunet', 'crackformer', 'defnet', 'ukan_only', 'vmamba_only', 'dual_encoder_concat', 'dual_encoder']

class SoftIoUDiceLoss(nn.Module):

    def __init__(self, smooth: float=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        pred = torch.sigmoid(logits).view(-1)
        target = target.view(-1)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        iou = (inter + self.smooth) / (union + self.smooth)
        dice = (2 * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - iou + (1 - dice)

class BCEDiceLoss(nn.Module):

    def __init__(self, smooth: float=1.0, bce_weight: float=0.7, pos_weight: float=10.0):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight.to(logits.device))
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum()
        dice = (2 * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice)

def fix_batchnorm_for_small_batches(model: nn.Module) -> nn.Module:
    original_forward = model.forward

    def new_forward(self, x):
        if x.size(0) != 1:
            return original_forward(x)
        bn_states = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_states[name] = module.training
                module.eval()
        out = original_forward(x)
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in bn_states and bn_states[name]:
                module.train()
        return out
    model.forward = types.MethodType(new_forward, model)
    return model

def load_state_dict_flexible(model, state_dict):
    clean = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean, strict=True)

def train_one_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_name = args.model.lower().replace('-', '_')
    out_dir = Path(args.output_dir or f'./output_{model_name}')
    log_dir = Path(args.log_dir or f'./runs_{model_name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    writer = SummaryWriter(str(log_dir))
    train_loader, val_loader = get_dataloader(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size, augment=True)
    model = build_model(model_name, args).to(device)
    model = fix_batchnorm_for_small_batches(model)
    use_amp = bool(args.amp) and model_name != 'mambacracknet'
    if model_name == 'mambacracknet':
        criterion = BCEDiceLoss(pos_weight=args.pos_weight).to(device)
        use_amp = False
        print('MambaCrackNet: using BCEDiceLoss, AMP disabled')
    else:
        criterion = SoftIoUDiceLoss().to(device)
    if args.pretrained and Path(args.pretrained).is_file():
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        load_state_dict_flexible(model, state)
        print(f'Loaded pretrained: {args.pretrained}')
    n_train = sum((p.numel() for p in model.parameters() if p.requires_grad))
    n_all = sum((p.numel() for p in model.parameters()))
    print(f'Model={model_name}  params={n_train}/{n_all}  device={device}  amp={use_amp}')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult)
    scaler = GradScaler(enabled=use_amp)
    best_dice = -1.0
    results = {'epoch': [], 'learning_rate': [], 'train_loss': [], 'train_iou': [], 'train_dice': [], 'train_miou_std': [], 'train_mdice_std': [], 'train_precision': [], 'train_recall': [], 'train_f1': [], 'val_iou': [], 'val_dice': [], 'val_miou_std': [], 'val_mdice_std': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}
    start_epoch = 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        load_state_dict_flexible(model, ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_dice = float(ckpt.get('best_metrics', {}).get('dice', best_dice))
        print(f'Resumed from {args.resume}, start_epoch={start_epoch}, best_dice={best_dice:.4f}')
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for inputs, masks in pbar:
            inputs = inputs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=True)
                loss = criterion(outputs, masks)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += float(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        epoch_loss = running_loss / max(len(train_loader), 1)
        train_metrics = calculate_epoch_metrics(model, train_loader, device, args.threshold)
        val_metrics = calculate_epoch_metrics(model, val_loader, device, args.threshold)
        print(f"[{model_name}] epoch {epoch + 1}: loss={epoch_loss:.4f} train_mDice={train_metrics['mdice']:.4f}±{train_metrics['mdice_std']:.4f} val_mDice={val_metrics['mdice']:.4f}±{val_metrics['mdice_std']:.4f} val_mIoU={val_metrics['miou']:.4f}±{val_metrics['miou_std']:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        for k, v in train_metrics.items():
            writer.add_scalar(f'{k}/train', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'{k}/val', v, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        payload = {'epoch': epoch, 'model': model_name, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_metrics': val_metrics if val_metrics['dice'] > best_dice else {'dice': best_dice}}
        torch.save(payload, out_dir / f'last_model_{model_name}_{args.dataset_name}.pth')
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            payload['best_metrics'] = val_metrics
            torch.save(payload, out_dir / f'best_model_{model_name}_{args.dataset_name}.pth')
            print(f'  saved best (dice={best_dice:.4f})')
        scheduler.step()
        results['epoch'].append(epoch + 1)
        results['learning_rate'].append(optimizer.param_groups[0]['lr'])
        results['train_loss'].append(epoch_loss)
        for split, metrics in (('train', train_metrics), ('val', val_metrics)):
            results[f'{split}_iou'].append(metrics['miou'])
            results[f'{split}_dice'].append(metrics['mdice'])
            results[f'{split}_miou_std'].append(metrics['miou_std'])
            results[f'{split}_mdice_std'].append(metrics['mdice_std'])
            for k in ('precision', 'recall', 'f1'):
                results[f'{split}_{k}'].append(metrics[k])
        df = pd.DataFrame(results)
        df.to_csv(out_dir / f'training_metrics_{model_name}_{args.dataset_name}.csv', index=False)
        try:
            df.to_excel(out_dir / f'training_metrics_{model_name}_{args.dataset_name}.xlsx', index=False)
        except Exception:
            pass
    writer.close()
    print(f'Done. best val dice={best_dice:.4f}  outputs -> {out_dir}')

def get_args():
    parser = argparse.ArgumentParser('Unified crack-segmentation trainer')
    parser.add_argument('--model', type=str, required=True, choices=AVAILABLE_MODELS, help='model name to train')
    parser.add_argument('--data_root', type=str, required=True, help='dataset root with train/val')
    parser.add_argument('--dataset_name', type=str, default='new_dataset')
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001, help='Adam lr=0.001')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--t0', type=int, default=15, help='CosineAnnealingWarmRestarts T_0')
    parser.add_argument('--t_mult', type=int, default=2, help='CosineAnnealingWarmRestarts T_mult')
    parser.add_argument('--threshold', type=float, default=0.633)
    parser.add_argument('--pos_weight', type=float, default=10.0, help='for mambacracknet BCE')
    parser.add_argument('--amp', action='store_true', help='enable AMP (ignored for mambacracknet)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='resume from last/best checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()
if __name__ == '__main__':
    args = get_args()
    train_one_model(args)
