from __future__ import annotations
import argparse
import types
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets.gan_dataset import GANAugmentedDataset, GAN_TYPE_ALIASES, OriginalOnlyView, holdout_split_70_15_15, resolve_gan_type, stems_identical
from train import build_model, fix_batchnorm_for_small_batches, load_state_dict_flexible
from util.gan_weight import compute_dynamic_weight
from util.losses import IoUDiceLoss
from util.metrics import calculate_per_image_metrics

def get_args():
    p = argparse.ArgumentParser('Teacher training with offline GAN + dynamic weight')
    p.add_argument('--model', default='dual_encoder')
    p.add_argument('--data_root', required=True, help='original dataset root with train/val')
    p.add_argument('--gan_root', required=True, help='GAN augmented images root')
    p.add_argument('--gan_type', default='enhance', help=f'GAN variant folder / alias. Known: {sorted(set(GAN_TYPE_ALIASES.keys()))}')
    p.add_argument('--layout', default='auto', choices=['auto', 'roadcrack', 'standard'])
    p.add_argument('--img_size', type=int, default=448)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--w0', type=float, default=0.5, help='base weight on original data')
    p.add_argument('--delta_w', type=float, default=1.0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--log_dir', default=None)
    p.add_argument('--pretrained', default=None)
    p.add_argument('--threshold', type=float, default=0.633)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force_holdout', action='store_true', help='Always re-split the train pool 70/15/15 (ignore disk val/)')
    return p.parse_args()

def _build_loaders(args, gan_type, device_hint=None):
    train_pool = GANAugmentedDataset(args.data_root, args.gan_root, 'train', gan_type, args.img_size, args.layout, args.max_samples, augment=False)
    use_holdout = bool(args.force_holdout)
    val_ds = None
    test_ds = None
    if not use_holdout:
        try:
            val_candidate = GANAugmentedDataset(args.data_root, args.gan_root, 'val', gan_type, args.img_size, args.layout, args.max_samples, augment=False)
            if stems_identical(train_pool, val_candidate):
                print('WARNING: train and val image sets are identical. Auto-splitting pool into 70/15/15.')
                use_holdout = True
            else:
                val_ds = val_candidate
                train_ds = GANAugmentedDataset(args.data_root, args.gan_root, 'train', gan_type, args.img_size, args.layout, args.max_samples, samples=train_pool.samples, augment=True)
        except Exception as e:
            print(f'WARNING: val load failed ({e}); holdout-splitting train pool 70/15/15.')
            use_holdout = True
    if use_holdout:
        train_ds, val_ds, test_ds = holdout_split_70_15_15(train_pool, seed=args.seed)
        print(f'Holdout 70/15/15: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}')
    else:
        test_ds = None
    gan_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    orig_val_loader = DataLoader(OriginalOnlyView(val_ds, use_gan=False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    gan_val_loader = DataLoader(OriginalOnlyView(val_ds, use_gan=True), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(OriginalOnlyView(test_ds, use_gan=False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return (train_ds, val_ds, test_ds, gan_train_loader, orig_val_loader, gan_val_loader, test_loader)

def train_teacher_gan(args):
    torch.manual_seed(args.seed)
    gan_type = resolve_gan_type(args.gan_type)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    out_dir = Path(args.output_dir or f'./output_{args.model}_gan_{gan_type}')
    log_dir = Path(args.log_dir or f'./runs_{args.model}_gan_{gan_type}')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    model = build_model(args.model, types.SimpleNamespace(img_size=args.img_size)).to(device)
    model = fix_batchnorm_for_small_batches(model)
    if args.pretrained and Path(args.pretrained).is_file():
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        load_state_dict_flexible(model, state)
        print(f'Loaded pretrained: {args.pretrained}')
    criterion = IoUDiceLoss(eps=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    train_ds, val_ds, test_ds, gan_train_loader, orig_val_loader, gan_val_loader, test_loader = _build_loaders(args, gan_type)
    print(f'model={args.model} gan_type={gan_type} layout={train_ds.layout} train_pairs={len(train_ds)} val_pairs={len(val_ds)} device={device}')
    print(f'GAN image dir: {train_ds.gan_dir}')
    print('Loss = Dice_Loss + IoU_Loss (ε=1); w(t) from val PR-AUC')
    w_t = float(args.w0)
    rows = []
    best_dice = -1.0
    weight_log = out_dir / f'gan_weight_log_{gan_type}.txt'
    with open(weight_log, 'w', encoding='utf-8') as f:
        f.write(f'initial_orig_weight(w0): {w_t:.4f}  delta_w: {args.delta_w}\n')
        f.write('Epoch,w_orig,w_gan,S_orig,S_gan\n')
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for orig, gan, mask in tqdm(gan_train_loader, desc=f'epoch {epoch + 1}/{args.epochs}'):
            orig, gan, mask = (orig.to(device), gan.to(device), mask.to(device))
            optimizer.zero_grad(set_to_none=True)
            out_o = model(orig)
            out_g = model(gan)
            if out_o.shape[-2:] != mask.shape[-2:]:
                out_o = F.interpolate(out_o, size=mask.shape[2:], mode='bilinear', align_corners=False)
                out_g = F.interpolate(out_g, size=mask.shape[2:], mode='bilinear', align_corners=False)
            loss = w_t * criterion(out_o, mask) + (1.0 - w_t) * criterion(out_g, mask)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        scheduler.step()
        w_t, w_info = compute_dynamic_weight(model, orig_val_loader, gan_val_loader, device, args.w0, args.delta_w)
        val_metrics = calculate_per_image_metrics(model, orig_val_loader, device, args.threshold)
        train_loss = running / max(len(gan_train_loader), 1)
        print(f"epoch {epoch + 1}: loss={train_loss:.4f} w_orig={w_t:.3f} w_gan={1 - w_t:.3f} S_orig={w_info['S_orig']:.4f} S_gan={w_info['S_gan']:.4f} val_mDice={val_metrics['mdice']:.4f} val_mIoU={val_metrics['miou']:.4f}")
        writer.add_scalar('w/orig', w_t, epoch)
        writer.add_scalar('w/gan', 1.0 - w_t, epoch)
        writer.add_scalar('S/orig', w_info['S_orig'], epoch)
        writer.add_scalar('S/gan', w_info['S_gan'], epoch)
        writer.add_scalar('dice/val_mdice', val_metrics['mdice'], epoch)
        writer.add_scalar('iou/val_miou', val_metrics['miou'], epoch)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        with open(weight_log, 'a', encoding='utf-8') as f:
            f.write(f"{epoch + 1},{w_t:.6f},{1.0 - w_t:.6f},{w_info['S_orig']:.6f},{w_info['S_gan']:.6f}\n")
        payload = {'epoch': epoch, 'model': args.model, 'gan_type': gan_type, 'w_t': w_t, 'w_info': w_info, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_metrics': val_metrics}
        torch.save(payload, out_dir / 'last.pth')
        if val_metrics['mdice'] > best_dice:
            best_dice = val_metrics['mdice']
            torch.save(payload, out_dir / 'best.pth')
            with open(weight_log, 'a', encoding='utf-8') as f:
                f.write(f"best(Epoch {epoch + 1}) w_orig={w_t:.6f} mDice={best_dice:.6f} mIoU={val_metrics['miou']:.6f}\n")
            print(f'  saved best mDice={best_dice:.4f}')
        rows.append({'epoch': epoch + 1, 'w_orig': w_t, 'w_gan': 1.0 - w_t, 'S_orig': w_info['S_orig'], 'S_gan': w_info['S_gan'], 'train_loss': train_loss, **val_metrics})
        pd.DataFrame(rows).to_csv(out_dir / f'gan_teacher_metrics_{gan_type}.csv', index=False)
    if test_loader is not None:
        test_metrics = calculate_per_image_metrics(model, test_loader, device, args.threshold)
        print(f"Final test (holdout): mIoU={test_metrics['miou']:.4f} mDice={test_metrics['mdice']:.4f} P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f}")
        pd.DataFrame([test_metrics]).to_csv(out_dir / f'gan_teacher_test_{gan_type}.csv', index=False)
    writer.close()
    print(f'Done. best_mDice={best_dice:.4f} -> {out_dir}')
if __name__ == '__main__':
    train_teacher_gan(get_args())
