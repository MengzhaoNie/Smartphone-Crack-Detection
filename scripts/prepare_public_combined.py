from __future__ import annotations
import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from PIL import Image, ImageFilter
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_EXTS = {'.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff'}

def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def _find_mask(stem: str, mask_dirs: Sequence[Path], mask_names: Optional[Dict[str, Path]]=None) -> Optional[Path]:
    if mask_names and stem in mask_names:
        return mask_names[stem]
    for d in mask_dirs:
        if not d.is_dir():
            continue
        for ext in MASK_EXTS:
            for name in (f'{stem}{ext}', f'{stem}_mask{ext}', f'{stem}_lab{ext}', f'{stem}mask{ext}'):
                p = d / name
                if p.is_file():
                    return p
    return None

def _index_masks(mask_dirs: Sequence[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for d in mask_dirs:
        if not d.is_dir():
            continue
        for p in d.rglob('*'):
            if p.is_file() and p.suffix.lower() in MASK_EXTS:
                stem = p.stem
                for suf in ('_mask', '_lab', '_gt', '_label', 'mask'):
                    if stem.lower().endswith(suf):
                        stem = stem[:-len(suf)]
                        break
                out.setdefault(stem, p)
                out.setdefault(p.stem, p)
    return out

def _collect_from_dirs(image_dirs: Sequence[Path], mask_dirs: Sequence[Path], source: str) -> List[Tuple[Path, Path, str]]:
    mask_index = _index_masks(mask_dirs)
    pairs = []
    for img_dir in image_dirs:
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.rglob('*')):
            if not _is_image(p):
                continue
            if any((part.lower() in {'mask', 'masks', 'gt', 'label', 'labels', 'lab'} for part in p.parts)):
                if img_dir.name.lower() in {'mask', 'masks', 'gt', 'label', 'labels', 'lab'}:
                    continue
            m = _find_mask(p.stem, mask_dirs, mask_index)
            if m is None:
                continue
            pairs.append((p, m, source))
    return pairs

def discover_pairs(root: Path, source: str) -> List[Tuple[Path, Path, str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Missing dataset root: {root}')
    candidates: List[Tuple[List[Path], List[Path]]] = []
    named = [(['train_crop', 'test_crop', 'val_crop'], ['train_crop', 'test_crop', 'val_crop']), (['train_img', 'test_img', 'val_img'], ['train_lab', 'test_lab', 'val_lab']), (['images', 'image', 'img', 'JPEGImages'], ['masks', 'mask', 'gt', 'groundTruth', 'Annotations', 'labels']), (['train/images', 'val/images', 'test/images'], ['train/masks', 'val/masks', 'test/masks'])]
    for imgs, masks in named:
        idirs = [root / x for x in imgs if (root / x).is_dir()]
        mdirs = [root / x for x in masks if (root / x).is_dir()]
        if idirs and mdirs:
            candidates.append((idirs, mdirs))
    for sub in root.iterdir() if root.is_dir() else []:
        if not sub.is_dir():
            continue
        files = [p for p in sub.iterdir() if p.is_file()]
        stems_img = {p.stem for p in files if p.suffix.lower() in {'.jpg', '.jpeg'}}
        stems_png = {p.stem for p in files if p.suffix.lower() == '.png'}
        if stems_img and stems_png:
            candidates.append(([sub], [sub]))
    for images_dir in root.rglob('images'):
        if images_dir.is_dir():
            sibling = images_dir.parent / 'masks'
            if sibling.is_dir():
                candidates.append(([images_dir], [sibling]))
    seen = set()
    pairs: List[Tuple[Path, Path, str]] = []
    for idirs, mdirs in candidates:
        for img, mask, src in _collect_from_dirs(idirs, mdirs, source):
            key = (img.resolve(), mask.resolve())
            if key in seen:
                continue
            seen.add(key)
            pairs.append((img, mask, src))
    if not pairs:
        raise RuntimeError(f'No image/mask pairs found under {root}. Expected layouts like images/+masks/, train_img/+train_lab/, or train_crop/.')
    return pairs

def dilate_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.convert('L')
    m = mask.convert('L')
    for _ in range(max(1, radius)):
        m = m.filter(ImageFilter.MaxFilter(3))
    return m.point(lambda v: 255 if v > 127 else 0)

def save_pair(img_path: Path, mask_path: Path, out_img: Path, out_mask: Path, size: int, dilate: int=0):
    img = Image.open(img_path).convert('RGB').resize((size, size), Image.BILINEAR)
    mask = Image.open(mask_path)
    if dilate > 0:
        mask = dilate_mask(mask, dilate)
    else:
        mask = mask.convert('L')
    mask = mask.resize((size, size), Image.NEAREST)
    mask = mask.point(lambda v: 255 if v > 127 else 0)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img)
    mask.save(out_mask)

def split_70_15_15(n: int, seed: int):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(round(n * 0.7))
    n_val = int(round(n * 0.15))
    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:]
    return (train, val, test)

def main():
    p = argparse.ArgumentParser('Combine public crack datasets')
    p.add_argument('--crack500', type=str, required=True)
    p.add_argument('--deepcrack', type=str, required=True)
    p.add_argument('--cfd', type=str, required=True)
    p.add_argument('--cracktree', type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--img_size', type=int, default=448)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cracktree_dilate', type=int, default=2, help='dilation iterations for CrackTree centerlines (0=off)')
    p.add_argument('--copy_mode', choices=['save', 'symlink'], default='save')
    args = p.parse_args()
    sources = {'Crack500': Path(args.crack500), 'DeepCrack': Path(args.deepcrack), 'CFD': Path(args.cfd), 'CrackTree': Path(args.cracktree)}
    all_pairs: List[Tuple[Path, Path, str]] = []
    for name, root in sources.items():
        pairs = discover_pairs(root, name)
        print(f'{name}: {len(pairs)} pairs from {root}')
        all_pairs.extend(pairs)
    uniq = {}
    for img, mask, src in all_pairs:
        uniq[str(img.resolve())] = (img, mask, src)
    all_pairs = list(uniq.values())
    print(f'Total unique pairs: {len(all_pairs)}')
    train_i, val_i, test_i = split_70_15_15(len(all_pairs), args.seed)
    splits = {'train': train_i, 'val': val_i, 'test': test_i}
    out_root = Path(args.output)
    manifest_rows = []
    for split, indices in splits.items():
        for j, i in enumerate(indices):
            img_p, mask_p, src = all_pairs[i]
            stem = f'{src}_{img_p.stem}_{j:05d}'
            out_img = out_root / split / 'images' / f'{stem}.png'
            out_mask = out_root / split / 'masks' / f'{stem}.png'
            dilate = args.cracktree_dilate if src == 'CrackTree' else 0
            if args.copy_mode == 'save':
                save_pair(img_p, mask_p, out_img, out_mask, args.img_size, dilate=dilate)
            else:
                out_img.parent.mkdir(parents=True, exist_ok=True)
                out_mask.parent.mkdir(parents=True, exist_ok=True)
                if out_img.exists():
                    out_img.unlink()
                if out_mask.exists():
                    out_mask.unlink()
                out_img.symlink_to(img_p.resolve())
                out_mask.symlink_to(mask_p.resolve())
            manifest_rows.append({'split': split, 'stem': stem, 'source': src, 'src_image': str(img_p), 'src_mask': str(mask_p), 'dilate': dilate})
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    summary = {'total': len(all_pairs), 'train': len(train_i), 'val': len(val_i), 'test': len(test_i), 'seed': args.seed, 'img_size': args.img_size, 'ref_counts': {'total': 12887, 'train': 9021, 'val': 1933, 'test': 1933}, 'sources': {k: str(v) for k, v in sources.items()}}
    (out_root / 'split_info.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Done: {out_root}')
    if abs(summary['total'] - 12887) > 500:
        print('WARNING: count mismatch')
if __name__ == '__main__':
    main()
