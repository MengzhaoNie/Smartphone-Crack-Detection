from __future__ import annotations
import argparse
import csv
import types
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from train import AVAILABLE_MODELS, build_model, load_state_dict_flexible
from util.metrics import calculate_per_image_metrics
MODES = ('atmospheric_blur', 'box_blur', 'median_blur', 'motion_blur', 'illumination')

def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode='RGB')

def _from_pil(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert('RGB'), dtype=np.float32)

def atmospheric_blur(img: Image.Image, intensity: float) -> Image.Image:
    if intensity <= 0:
        return img
    radius = max(0.1, 12.0 * (intensity / 100.0))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def box_blur(img: Image.Image, intensity: float) -> Image.Image:
    if intensity <= 0:
        return img
    radius = max(1, int(round(15 * intensity / 100.0)))
    return img.filter(ImageFilter.BoxBlur(radius=radius))

def median_blur(img: Image.Image, intensity: float) -> Image.Image:
    if intensity <= 0:
        return img
    size = int(round(15 * intensity / 100.0))
    size = max(3, size | 1)
    return img.filter(ImageFilter.MedianFilter(size=size))

def motion_blur(img: Image.Image, intensity: float, angle_deg: float=0.0) -> Image.Image:
    if intensity <= 0:
        return img
    length = max(1, int(round(25 * intensity / 100.0)))
    if length == 1:
        return img
    arr = _from_pil(img)
    kernel = np.ones(length, dtype=np.float32) / length
    pad = length // 2
    out = np.zeros_like(arr)
    for c in range(3):
        ch = np.pad(arr[:, :, c], ((0, 0), (pad, pad)), mode='edge')
        cs = np.cumsum(ch, axis=1)
        left = np.concatenate([np.zeros((ch.shape[0], 1), dtype=np.float32), cs[:, :-length]], axis=1)
        right = cs[:, length - 1:]
        n = arr.shape[1]
        summed = right[:, :n] - left[:, :n]
        out[:, :, c] = summed / length
    blurred = _to_pil(out)
    if abs(angle_deg) > 0.001:
        blurred = blurred.rotate(angle_deg, resample=Image.BILINEAR, expand=False)
    return blurred

def illumination(img: Image.Image, intensity: float) -> Image.Image:
    if intensity <= 0:
        return img
    factor = max(0.05, 1.0 - 0.95 * (intensity / 100.0))
    return ImageEnhance.Brightness(img).enhance(factor)

def illuminate_bright(img: Image.Image, intensity: float) -> Image.Image:
    if intensity <= 0:
        return img
    factor = 1.0 + 2.0 * (intensity / 100.0)
    return ImageEnhance.Brightness(img).enhance(factor)

def apply_perturbation(img: Image.Image, mode: str, intensity: float) -> Image.Image:
    mode = mode.lower()
    if mode == 'atmospheric_blur':
        return atmospheric_blur(img, intensity)
    if mode == 'box_blur':
        return box_blur(img, intensity)
    if mode == 'median_blur':
        return median_blur(img, intensity)
    if mode == 'motion_blur':
        return motion_blur(img, intensity)
    if mode == 'illumination':
        return illumination(img, intensity)
    if mode == 'illumination_bright':
        return illuminate_bright(img, intensity)
    raise ValueError(f'Unknown mode: {mode}')

class PerturbedCrackDataset(Dataset):

    def __init__(self, data_root: str, split: str, img_size: int, mode: str, intensity: float, max_samples: int | None=None):
        self.root = Path(data_root)
        self.img_dir = self.root / split / 'images'
        self.mask_dir = self.root / split / 'masks'
        if not self.img_dir.is_dir():
            raise FileNotFoundError(self.img_dir)
        self.mode = mode
        self.intensity = intensity
        self.img_size = img_size
        stems = sorted({p.stem for p in self.img_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}})
        self.samples: List[Tuple[Path, Path]] = []
        for stem in stems:
            img = None
            mask = None
            for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                if img is None and (self.img_dir / f'{stem}{ext}').is_file():
                    img = self.img_dir / f'{stem}{ext}'
                if mask is None and (self.mask_dir / f'{stem}{ext}').is_file():
                    mask = self.mask_dir / f'{stem}{ext}'
            if img and mask:
                self.samples.append((img, mask))
        if max_samples:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f'No pairs in {self.img_dir}')
        self.tf_img = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.tf_mask = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p = self.samples[idx]
        img = Image.open(img_p).convert('RGB')
        img = apply_perturbation(img, self.mode, self.intensity)
        mask = Image.open(mask_p).convert('L')
        x = self.tf_img(img)
        y = (self.tf_mask(mask) > 0.5).float()
        return (x, y)

def load_model(args, device):
    model = build_model(args.model, types.SimpleNamespace(img_size=args.img_size)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    load_state_dict_flexible(model, state)
    model.eval()
    return model

def eval_one(model, data_root, split, img_size, mode, intensity, batch_size, device, threshold, max_samples, num_workers):
    ds = PerturbedCrackDataset(data_root, split, img_size, mode, intensity, max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return calculate_per_image_metrics(model, loader, device, threshold)

def get_args():
    p = argparse.ArgumentParser('Robustness eval')
    p.add_argument('--model', required=True, choices=AVAILABLE_MODELS)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--img_size', type=int, default=448)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.633)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--modes', default=','.join(MODES), help='comma-separated modes; add illumination_bright for overexposure curve')
    p.add_argument('--intensities', default='0,20,40,60,80,100')
    p.add_argument('--output_dir', default='./robustness_out')
    p.add_argument('--save_vis', action='store_true', help='save example perturbed images at 40%')
    return p.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    intensities = [float(x) for x in args.intensities.split(',') if x.strip()]
    model = load_model(args, device)
    rows = []
    for mode in modes:
        for inten in intensities:
            print(f'==> {mode} @ {inten:.0f}%')
            metrics = eval_one(model, args.data_root, args.split, args.img_size, mode, inten, args.batch_size, device, args.threshold, args.max_samples, args.num_workers)
            row = {'model': args.model, 'mode': mode, 'intensity': inten, 'miou': metrics['miou'], 'mdice': metrics['mdice'], 'miou_std': metrics['miou_std'], 'mdice_std': metrics['mdice_std'], 'precision': metrics['precision'], 'recall': metrics['recall'], 'f1': metrics['f1']}
            rows.append(row)
            print(f"    mIoU={metrics['miou']:.4f}±{metrics['miou_std']:.4f}  mDice={metrics['mdice']:.4f}")
    csv_path = out_dir / f'robustness_{args.model}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    snap = [r for r in rows if abs(r['intensity'] - 40.0) < 1e-06]
    if snap:
        snap_path = out_dir / f'robustness_{args.model}_intensity40.csv'
        with open(snap_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(snap[0].keys()))
            w.writeheader()
            w.writerows(snap)
        print(f'40% snapshot: {snap_path}')
    if args.save_vis:
        vis_dir = out_dir / 'vis'
        vis_dir.mkdir(exist_ok=True)
        img_dir = Path(args.data_root) / args.split / 'images'
        sample = next(img_dir.glob('*.png'), None) or next(img_dir.glob('*.jpg'), None)
        if sample:
            base = Image.open(sample).convert('RGB').resize((args.img_size, args.img_size))
            base.save(vis_dir / 'orig.png')
            for mode in modes:
                apply_perturbation(base, mode, 40.0).save(vis_dir / f'{mode}_40.png')
    try:
        import matplotlib.pyplot as plt
        for metric in ('miou', 'mdice'):
            plt.figure(figsize=(8, 5))
            for mode in modes:
                xs = [r['intensity'] for r in rows if r['mode'] == mode]
                ys = [r[metric] for r in rows if r['mode'] == mode]
                plt.plot(xs, ys, marker='o', label=mode)
            plt.xlabel('Perturbation intensity (%)')
            plt.ylabel(metric)
            plt.title(f'Robustness - {args.model} ({metric})')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = out_dir / f'robustness_{args.model}_{metric}.png'
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f'curve: {fig_path}')
    except Exception as e:
        print(f'(plot skipped: {e})')
    print(f'Done: {csv_path}')
if __name__ == '__main__':
    main()
