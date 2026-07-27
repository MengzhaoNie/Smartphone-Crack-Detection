from __future__ import annotations
import argparse
import csv
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from train import AVAILABLE_MODELS, build_model, load_state_dict_flexible
from util.metrics import calculate_per_image_metrics
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

class PairDataset(Dataset):

    def __init__(self, pairs: List[Tuple[Path, Path]], img_size: int=448):
        self.pairs = pairs
        self.tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.mtf = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        x = self.tf(Image.open(ip).convert('RGB'))
        y = (self.mtf(Image.open(mp).convert('L')) > 0.5).float()
        return (x, y)

def _pairs_from_split(root: Path, split: str) -> List[Tuple[Path, Path]]:
    img_dir = root / split / 'images'
    mask_dir = root / split / 'masks'
    if not img_dir.is_dir():
        img_dir = root / 'images'
        mask_dir = root / 'masks'
    pairs = []
    if not img_dir.is_dir():
        return pairs
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        for ext in IMG_EXTS:
            m = mask_dir / f'{p.stem}{ext}'
            if m.is_file():
                pairs.append((p, m))
                break
    return pairs

def pairs_from_manifest(combined_root: Path, split: str, source: str) -> List[Tuple[Path, Path]]:
    man = combined_root / 'manifest.csv'
    if not man.is_file():
        raise FileNotFoundError(f'Need manifest.csv under {combined_root}')
    pairs = []
    with open(man, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('split') != split:
                continue
            if row.get('source', '').lower() != source.lower():
                continue
            stem = row['stem']
            img = combined_root / split / 'images' / f'{stem}.png'
            mask = combined_root / split / 'masks' / f'{stem}.png'
            if img.is_file() and mask.is_file():
                pairs.append((img, mask))
    return pairs

def get_args():
    p = argparse.ArgumentParser('Per-dataset eval')
    p.add_argument('--model', required=True, choices=AVAILABLE_MODELS)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--combined_root', default=None, help='prepared public_combined root with manifest.csv')
    p.add_argument('--crack500', default=None)
    p.add_argument('--deepcrack', default=None)
    p.add_argument('--cfd', default=None)
    p.add_argument('--cracktree', default=None)
    p.add_argument('--split', default='test')
    p.add_argument('--img_size', type=int, default=448)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.633)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--output', default='./tab1_per_dataset.csv')
    return p.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model = build_model(args.model, types.SimpleNamespace(img_size=args.img_size)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    load_state_dict_flexible(model, state)
    model.eval()
    datasets: Dict[str, List[Tuple[Path, Path]]] = {}
    if args.combined_root:
        root = Path(args.combined_root)
        for name in ('Crack500', 'DeepCrack', 'CFD', 'CrackTree'):
            datasets[name] = pairs_from_manifest(root, args.split, name)
    else:
        mapping = {'Crack500': args.crack500, 'DeepCrack': args.deepcrack, 'CFD': args.cfd, 'CrackTree': args.cracktree}
        for name, path in mapping.items():
            if not path:
                continue
            datasets[name] = _pairs_from_split(Path(path), args.split)
    if not datasets:
        raise SystemExit('Provide --combined_root or individual dataset roots.')
    rows = []
    for name, pairs in datasets.items():
        if not pairs:
            print(f'WARNING: {name} has 0 pairs, skip')
            continue
        print(f'==> {name}: {len(pairs)} images')
        loader = DataLoader(PairDataset(pairs, args.img_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        m = calculate_per_image_metrics(model, loader, device, args.threshold)
        rows.append({'model': args.model, 'dataset': name, 'n': len(pairs), 'miou': m['miou'], 'mdice': m['mdice'], 'miou_std': m['miou_std'], 'mdice_std': m['mdice_std'], 'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1']})
        print(f"    mIoU={m['miou']:.4f}  mDice={m['mdice']:.4f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'CSV: {out}')
if __name__ == '__main__':
    main()
