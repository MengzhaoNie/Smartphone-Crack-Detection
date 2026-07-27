from __future__ import annotations
import argparse
import types
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data import get_dataloader
from train import SoftIoUDiceLoss, build_model, fix_batchnorm_for_small_batches, load_state_dict_flexible
from util.metrics import calculate_per_image_metrics
from util.progressive_unfreeze import apply_unfreeze_stage, parse_stage_epochs, stage_at_epoch

def get_args():
    p = argparse.ArgumentParser('Progressive-unfreeze transfer')
    p.add_argument('--model', default='dual_encoder')
    p.add_argument('--pretrained', required=True, help='public-dataset teacher checkpoint')
    p.add_argument('--data_root', required=True, help='smartphone / target domain root')
    p.add_argument('--dataset_name', default='smartphone')
    p.add_argument('--img_size', type=int, default=448)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--stage_epochs', default='20,20,60', help='epochs per stage: decoder,cnn_kan,full (must sum ≈ --epochs)')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--lr_decoder', type=float, default=None, help='override lr in decoder stage')
    p.add_argument('--lr_full', type=float, default=None, help='override lr in full stage')
    p.add_argument('--t0', type=int, default=15)
    p.add_argument('--t_mult', type=int, default=2)
    p.add_argument('--threshold', type=float, default=0.633)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--log_dir', default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def _lr_for_stage(args, stage: str) -> float:
    if stage == 'decoder' and args.lr_decoder is not None:
        return float(args.lr_decoder)
    if stage == 'full' and args.lr_full is not None:
        return float(args.lr_full)
    if stage == 'cnn_kan' and args.lr_full is not None:
        base = args.lr_decoder if args.lr_decoder is not None else args.lr
        return float((base * args.lr_full) ** 0.5)
    return float(args.lr)

def _build_optimizer(model, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError('No trainable parameters after freeze.')
    return optim.Adam(params, lr=lr)

def train_transfer(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    out_dir = Path(args.output_dir or f'./output_{args.model}_transfer_{args.dataset_name}')
    log_dir = Path(args.log_dir or f'./runs_{args.model}_transfer_{args.dataset_name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    model = build_model(args.model, types.SimpleNamespace(img_size=args.img_size)).to(device)
    model = fix_batchnorm_for_small_batches(model)
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    load_state_dict_flexible(model, state)
    print(f'Loaded pretrained teacher: {args.pretrained}')
    schedule = parse_stage_epochs(args.stage_epochs, args.epochs)
    print('Unfreeze schedule:')
    for stage, start, end in schedule:
        print(f'  [{start:3d}, {end:3d})  {stage}')
    train_loader, val_loader = get_dataloader(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    criterion = SoftIoUDiceLoss(smooth=1.0).to(device)
    current_stage = None
    optimizer = None
    scheduler = None
    best_mdice = -1.0
    rows = []
    for epoch in range(args.epochs):
        stage = stage_at_epoch(schedule, epoch)
        if stage != current_stage:
            info = apply_unfreeze_stage(model, stage)
            lr = _lr_for_stage(args, stage)
            optimizer = _build_optimizer(model, lr)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult)
            current_stage = stage
            print(f"==> enter stage={info['stage']} trainable={info['trainable']}/{info['total']} lr={lr:.2e} kind={info['kind']}")
        model.train()
        running = 0.0
        for inputs, masks in tqdm(train_loader, desc=f'ep{epoch + 1}/{args.epochs}[{stage}]'):
            inputs, masks = (inputs.to(device), masks.to(device))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        scheduler.step()
        train_loss = running / max(len(train_loader), 1)
        val_metrics = calculate_per_image_metrics(model, val_loader, device, args.threshold)
        print(f"epoch {epoch + 1}: stage={stage} loss={train_loss:.4f} val_mDice={val_metrics['mdice']:.4f} val_mIoU={val_metrics['miou']:.4f}")
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('mdice/val', val_metrics['mdice'], epoch)
        writer.add_scalar('miou/val', val_metrics['miou'], epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        payload = {'epoch': epoch, 'stage': stage, 'model': args.model, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_metrics': val_metrics}
        torch.save(payload, out_dir / 'last.pth')
        if val_metrics['mdice'] > best_mdice:
            best_mdice = val_metrics['mdice']
            torch.save(payload, out_dir / 'best.pth')
            print(f'  saved best mDice={best_mdice:.4f}')
        rows.append({'epoch': epoch + 1, 'stage': stage, 'train_loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], **val_metrics})
        pd.DataFrame(rows).to_csv(out_dir / 'transfer_metrics.csv', index=False)
    writer.close()
    print(f'Done. best_mDice={best_mdice:.4f} -> {out_dir}')
if __name__ == '__main__':
    train_transfer(get_args())
