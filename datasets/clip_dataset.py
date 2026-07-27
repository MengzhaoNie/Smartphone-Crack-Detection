from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def _find_with_ext(directory: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = directory / f'{stem}{ext}'
        if p.is_file():
            return p
        p2 = directory / f'{stem}{ext.upper()}'
        if p2.is_file():
            return p2
    return None

class CrackClipDataset(Dataset):

    def __init__(self, root_dir: str, split: str='train', img_size: int=448, transform=None):
        self.root = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.split_dir = self.root / split
        self.images_dir = self.split_dir / 'images'
        self.masks_dir = self.split_dir / 'masks'
        self.neighbors_dir = self.split_dir / 'neighbors'
        self.transform = transform or transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.mask_transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])
        self.samples: List[dict] = []
        manifest = self.split_dir / 'clips.csv'
        if manifest.is_file():
            self._load_manifest(manifest)
        else:
            self._load_from_folders()

    def _load_manifest(self, manifest: Path):
        with open(manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({'key': self._resolve(row['keyframe']), 'prev': self._resolve(row.get('prev') or row['keyframe']), 'next': self._resolve(row.get('next') or row['keyframe']), 'mask': self._resolve(row['mask'])})

    def _resolve(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_file():
            return p
        cand = self.split_dir / path_str
        if cand.is_file():
            return cand
        cand = self.root / path_str
        if cand.is_file():
            return cand
        raise FileNotFoundError(path_str)

    def _load_from_folders(self):
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f'Missing images dir: {self.images_dir}')
        stems = []
        for p in sorted(self.images_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                stems.append(p.stem)
        stems = sorted(set(stems))
        for stem in stems:
            key = _find_with_ext(self.images_dir, stem)
            mask = _find_with_ext(self.masks_dir, stem)
            if key is None or mask is None:
                continue
            prev = None
            nxt = None
            if self.neighbors_dir.is_dir():
                prev = _find_with_ext(self.neighbors_dir, f'{stem}_prev')
                nxt = _find_with_ext(self.neighbors_dir, f'{stem}_next')
            prev_dir = self.split_dir / 'prev'
            next_dir = self.split_dir / 'next'
            if prev is None and prev_dir.is_dir():
                prev = _find_with_ext(prev_dir, stem)
            if nxt is None and next_dir.is_dir():
                nxt = _find_with_ext(next_dir, stem)
            self.samples.append({'key': key, 'prev': prev or key, 'next': nxt or key, 'mask': mask})

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        prev = self._load_rgb(s['prev'])
        key = self._load_rgb(s['key'])
        nxt = self._load_rgb(s['next'])
        frames = torch.stack([prev, key, nxt], dim=0)
        mask = self.mask_transform(Image.open(s['mask']).convert('L'))
        mask = (mask > 0.5).float()
        return (frames, mask)

def get_clip_dataloaders(data_root: str, batch_size: int=4, num_workers: int=2, img_size: int=448):
    train_ds = CrackClipDataset(data_root, split='train', img_size=img_size)
    val_ds = CrackClipDataset(data_root, split='val', img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return (train_loader, val_loader)
