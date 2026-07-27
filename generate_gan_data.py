
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from data import CrackDataset
from models.gan import GAN_TYPES, create_cyclegan, create_enhance_gan, create_srnet


def load_gan(gan_type: str, ckpt: str, device):
  if gan_type == "enhance":
    model = create_enhance_gan()
  elif gan_type == "cyclegan":
    model = create_cyclegan()
  else:
    model = create_srnet(scale=2)
  state = torch.load(ckpt, map_location=device, weights_only=False)
  model.load_state_dict(state.get("model_state_dict", state), strict=False)
  return model.to(device).eval()


@torch.no_grad()
def apply_gan(model, gan_type: str, img: Image.Image, device) -> Image.Image:
  to_tensor = transforms.ToTensor()
  x = to_tensor(img).unsqueeze(0).to(device)
  if gan_type == "enhance":
    out = model.generate(x)
  elif gan_type == "cyclegan":
    out = model.to_target_style(x)
  else:
    out = model.super_resolve(x, target_size=x.shape[-2:])
  out = out.squeeze(0).clamp(0, 1).cpu()
  arr = (out.permute(1, 2, 0).numpy() * 255).astype("uint8")
  return Image.fromarray(arr)


def main():
  p = argparse.ArgumentParser("Generate GAN-augmented images")
  p.add_argument("--gan_type", choices=GAN_TYPES, required=True)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--data_root", required=True)
  p.add_argument("--output_root", default="./gan_augmented")
  p.add_argument("--splits", nargs="+", default=["train", "val"])
  p.add_argument("--device", default="cuda")
  args = p.parse_args()

  device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
  model = load_gan(args.gan_type, args.checkpoint, device)
  out_root = Path(args.output_root)

  for split in args.splits:
    ds = CrackDataset(args.data_root, split=split, transform=None)
    out_dir = out_root / split / args.gan_type
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(ds)), desc=f"gen {split}"):
      img_name = ds.image_files[i]
      img_path = None
      for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        cand = Path(ds.images_dir) / f"{img_name}{ext}"
        if cand.is_file():
          img_path = cand
          break
      if img_path is None:
        continue
      img = Image.open(img_path).convert("RGB")
      aug = apply_gan(model, args.gan_type, img, device)
      aug.save(out_dir / f"{img_name}.png")

  print(f"GAN images saved under {out_root}")


if __name__ == "__main__":
  main()
