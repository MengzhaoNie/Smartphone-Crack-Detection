import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF


class CrackDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, img_size=448, augment=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.augment = augment and split == "train"

        self.img_formats = [".jpg", ".jpeg", ".png", ".bmp"]
        self.images_dir = os.path.join(root_dir, split, "images")
        self.masks_dir = os.path.join(root_dir, split, "masks")

        self.image_files = []
        for img_format in self.img_formats:
            self.image_files.extend(
                [
                    f[: -len(img_format)]
                    for f in os.listdir(self.images_dir)
                    if f.lower().endswith(img_format)
                ]
            )
        self.image_files = sorted(list(set(self.image_files)))

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_files)

    def _resolve(self, directory, stem):
        for fmt in self.img_formats:
            p = os.path.join(directory, stem + fmt)
            if os.path.exists(p):
                return p
        return None

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self._resolve(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name + ".png")
        if not os.path.exists(mask_path):
            mask_path = self._resolve(self.masks_dir, img_name)

        image = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)


        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            if random.random() < 0.3:
                image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            if random.random() < 0.3:
                image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))

        if self.transform is not None:

            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
        else:
            image = self.normalize(TF.to_tensor(image))
            mask = TF.to_tensor(mask)

        mask = (mask > 0.5).float()
        return image, mask


def get_dataloader(data_root, batch_size=4, num_workers=4, img_size=448, augment=True):
    train_dataset = CrackDataset(
        root_dir=data_root, split="train", img_size=img_size, augment=augment
    )
    val_dataset = CrackDataset(
        root_dir=data_root, split="val", img_size=img_size, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
