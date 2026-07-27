
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


GAN_TYPE_ALIASES: Dict[str, str] = {
    "enhance": "enhance",
    "gan_enhance": "enhance",
    "cyclegan": "shade",
    "style": "shade",
    "shade": "shade",
    "slight_bright": "slight_bright",
    "medium_bright": "medium_bright",
    "slight_dark": "slight_dark",
    "medium_dark": "medium_dark",
    "super_resolution": "EDSR",
    "sr": "EDSR",
    "edsr": "EDSR",
    "EDSR": "EDSR",
}


def resolve_gan_type(gan_type: str) -> str:
    key = gan_type.strip()
    if key in GAN_TYPE_ALIASES:
        return GAN_TYPE_ALIASES[key]
    low = key.lower()
    if low in GAN_TYPE_ALIASES:
        return GAN_TYPE_ALIASES[low]
    return key


def _find_image(directory: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = directory / f"{stem}{ext}"
        if p.is_file():
            return p
        p2 = directory / f"{stem}{ext.upper()}"
        if p2.is_file():
            return p2
    return None


def _list_stems(directory: Path) -> List[str]:
    stems = []
    if not directory.is_dir():
        return stems
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            stems.append(p.stem)
    return stems


def _stem_set(samples: Sequence[Tuple[Path, Path, Path]]) -> set:
    return {a[0].name for a in samples}


class GANAugmentedDataset(Dataset):


    def __init__(
        self,
        data_root: str,
        gan_root: str,
        split: str = "train",
        gan_type: str = "enhance",
        img_size: int = 448,
        layout: str = "auto",
        max_samples: Optional[int] = None,
        samples: Optional[List[Tuple[Path, Path, Path]]] = None,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.gan_type = resolve_gan_type(gan_type)

        self.images_dir = self.data_root / split / "images"
        self.masks_dir = self.data_root / split / "masks"

        gan_root_p = Path(gan_root)
        if samples is not None:

            self.layout = layout if layout != "auto" else "holdout"
            self.gan_dir = Path(samples[0][1]).parent if samples else Path(".")
            self.samples = list(samples)
            if max_samples is not None and max_samples > 0:
                self.samples = self.samples[:max_samples]
        else:
            if not self.images_dir.is_dir():
                raise FileNotFoundError(f"Missing images: {self.images_dir}")
            if not self.masks_dir.is_dir():
                raise FileNotFoundError(f"Missing masks: {self.masks_dir}")

            self.layout = layout if layout != "auto" else self._detect_layout(gan_root_p)
            self.gan_dir = self._resolve_gan_dir(gan_root_p)
            if not self.gan_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing GAN images for type={self.gan_type} layout={self.layout}: {self.gan_dir}"
                )

            self.samples = []
            for stem in _list_stems(self.images_dir):
                orig = _find_image(self.images_dir, stem)
                mask = _find_image(self.masks_dir, stem)
                gan = _find_image(self.gan_dir, stem)
                if orig and mask and gan:
                    self.samples.append((orig, gan, mask))

            if max_samples is not None and max_samples > 0:
                self.samples = self.samples[:max_samples]

        if not self.samples:
            raise RuntimeError(
                f"No paired samples for split={split}, gan_type={self.gan_type}. "
                f"orig={self.images_dir}, gan={getattr(self, 'gan_dir', '?')}"
            )

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _detect_layout(self, gan_root: Path) -> str:
        if (gan_root / self.gan_type / "images").is_dir():
            return "roadcrack"
        if (gan_root / self.split / self.gan_type).is_dir():
            return "standard"
        if (gan_root / "images").is_dir() and gan_root.name == self.gan_type:
            return "roadcrack_direct"
        raise FileNotFoundError(
            f"Cannot detect GAN layout under {gan_root} for type={self.gan_type}. "
            f"Expected {gan_root}/{self.gan_type}/images or {gan_root}/{self.split}/{self.gan_type}"
        )

    def _resolve_gan_dir(self, gan_root: Path) -> Path:
        if self.layout == "roadcrack":
            return gan_root / self.gan_type / "images"
        if self.layout == "roadcrack_direct":
            return gan_root / "images"
        return gan_root / self.split / self.gan_type

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)

    def _load_mask(self, path: Path) -> Image.Image:
        return Image.open(path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

    def __getitem__(self, idx: int):
        orig_p, gan_p, mask_p = self.samples[idx]
        orig = self._load_rgb(orig_p)
        gan = self._load_rgb(gan_p)
        mask = self._load_mask(mask_p)


        if self.augment:
            if random.random() < 0.5:
                orig = TF.hflip(orig)
                gan = TF.hflip(gan)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                orig = TF.vflip(orig)
                gan = TF.vflip(gan)
                mask = TF.vflip(mask)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                orig = TF.rotate(orig, angle)
                gan = TF.rotate(gan, angle)
                mask = TF.rotate(mask, angle)

        orig_t = self.normalize(TF.to_tensor(orig))
        gan_t = self.normalize(TF.to_tensor(gan))
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        return orig_t, gan_t, mask_t


def stems_identical(a: GANAugmentedDataset, b: GANAugmentedDataset) -> bool:
    return len(a) == len(b) and _stem_set(a.samples) == _stem_set(b.samples)


def holdout_split_70_15_15(
    pool: GANAugmentedDataset,
    seed: int = 42,
) -> Tuple[GANAugmentedDataset, GANAugmentedDataset, GANAugmentedDataset]:

    n = len(pool.samples)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    def _subset(indices: List[int], split_name: str, augment: bool) -> GANAugmentedDataset:
        samples = [pool.samples[i] for i in indices]

        return GANAugmentedDataset(
            data_root=str(pool.data_root),
            gan_root=str(pool.data_root),
            split=split_name,
            gan_type=pool.gan_type,
            img_size=pool.img_size,
            layout=pool.layout,
            samples=samples,
            augment=augment,
        )

    train_ds = _subset(train_idx, "train", augment=True)
    val_ds = _subset(val_idx, "val", augment=False)
    test_ds = _subset(test_idx, "test", augment=False)
    return train_ds, val_ds, test_ds


class OriginalOnlyView(Dataset):


    def __init__(self, paired: GANAugmentedDataset, use_gan: bool = False):
        self.paired = paired
        self.use_gan = use_gan

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx):
        orig, gan, mask = self.paired[idx]
        return (gan if self.use_gan else orig), mask


def get_gan_dataloaders(
    data_root,
    gan_root,
    gan_type,
    batch_size=4,
    num_workers=2,
    img_size=448,
    layout="auto",
    max_samples=None,
):
    train = GANAugmentedDataset(
        data_root, gan_root, "train", gan_type, img_size, layout, max_samples, augment=True
    )
    try:
        val = GANAugmentedDataset(
            data_root, gan_root, "val", gan_type, img_size, layout, max_samples, augment=False
        )
    except Exception:
        val = train

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
