from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def _find_image(directory: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        for name in (f'{stem}{ext}', f'{stem}{ext.upper()}'):
            p = directory / name
            if p.is_file():
                return p
    return None

def _list_stems(directory: Path) -> List[str]:
    if not directory.is_dir():
        return []
    return sorted({p.stem for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS})

def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    root = Path(root)
    pairs: List[Tuple[Path, Path]] = []
    seen = set()

    def add(img_dir: Path, mask_dir: Path):
        if not img_dir.is_dir() or not mask_dir.is_dir():
            return
        for stem in _list_stems(img_dir):
            img = _find_image(img_dir, stem)
            mask = _find_image(mask_dir, stem)
            if img and mask:
                key = img.resolve()
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((img, mask))
    for split in ('train', 'val', 'test'):
        add(root / split / 'images', root / split / 'masks')
    add(root / 'images', root / 'masks')
    if not pairs:
        add(root, root)
    if not pairs:
        raise RuntimeError(f'No keyframe pairs under {root}. Need images/+masks/ (flat or train/val/test).')
    return pairs

def split_70_15_15(n: int, seed: int):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(round(n * 0.7))
    n_val = int(round(n * 0.15))
    return (idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:])

def save_resized(img_p: Path, mask_p: Path, out_img: Path, out_mask: Path, size: int):
    img = Image.open(img_p).convert('RGB').resize((size, size), Image.BILINEAR)
    mask = Image.open(mask_p).convert('L').resize((size, size), Image.NEAREST)
    mask = mask.point(lambda v: 255 if v > 127 else 0)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img)
    mask.save(out_mask)

def main():
    p = argparse.ArgumentParser('Prepare smartphone keyframe splits')
    p.add_argument('--input', required=True, help='root with keyframe images/masks')
    p.add_argument('--output', required=True)
    p.add_argument('--img_size', type=int, default=448)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--write_clips_stub', action='store_true', help='also create empty prev/next dirs for later temporal KD (copies key as placeholder)')
    args = p.parse_args()
    pairs = collect_pairs(Path(args.input))
    print(f'Found {len(pairs)} keyframe pairs in {args.input}')
    train_i, val_i, test_i = split_70_15_15(len(pairs), args.seed)
    splits = {'train': train_i, 'val': val_i, 'test': test_i}
    out_root = Path(args.output)
    rows = []
    for split, indices in splits.items():
        for j, i in enumerate(indices):
            img_p, mask_p = pairs[i]
            stem = f'kf_{split}_{j:05d}_{img_p.stem}'
            out_img = out_root / split / 'images' / f'{stem}.png'
            out_mask = out_root / split / 'masks' / f'{stem}.png'
            save_resized(img_p, mask_p, out_img, out_mask, args.img_size)
            rows.append({'split': split, 'stem': stem, 'src_image': str(img_p), 'src_mask': str(mask_p)})
            if args.write_clips_stub:
                for nb in ('prev', 'next'):
                    nb_dir = out_root / split / nb
                    nb_dir.mkdir(parents=True, exist_ok=True)
                    Image.open(out_img).save(nb_dir / f'{stem}.png')
    with open(out_root / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {'total': len(pairs), 'train': len(train_i), 'val': len(val_i), 'test': len(test_i), 'seed': args.seed, 'img_size': args.img_size, 'ref_counts': {'train': 1239, 'val': 265, 'test': 265, 'total': 1769}, 'input': str(Path(args.input))}
    (out_root / 'split_info.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Done: {out_root}')
    if abs(summary['total'] - 1769) > 100:
        print('WARNING: count mismatch')
if __name__ == '__main__':
    main()
