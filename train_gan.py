
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data import CrackDataset
from models.gan import GAN_TYPES, create_cyclegan, create_enhance_gan, create_srnet


def edge_enhance_tensor(x: torch.Tensor) -> torch.Tensor:


  mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
  img = torch.clamp(x * std + mean, 0, 1)

  blur = F.avg_pool2d(img, 3, 1, 1)
  sharp = torch.clamp(img + 0.6 * (img - blur), 0, 1)
  return (sharp - mean) / std


def color_jitter_batch(x: torch.Tensor) -> torch.Tensor:
  mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
  img = torch.clamp(x * std + mean, 0, 1)

  b = 0.85 + torch.rand(1, device=x.device) * 0.3
  c = 0.85 + torch.rand(1, device=x.device) * 0.3
  out = torch.clamp((img - 0.5) * c + 0.5 + (b - 1.0) * 0.2, 0, 1)
  return (out - mean) / std


def gan_loss_d(d_out_real, d_out_fake):
  loss_real = F.binary_cross_entropy_with_logits(d_out_real, torch.ones_like(d_out_real))
  loss_fake = F.binary_cross_entropy_with_logits(d_out_fake, torch.zeros_like(d_out_fake))
  return 0.5 * (loss_real + loss_fake)


def gan_loss_g(d_out_fake):
  return F.binary_cross_entropy_with_logits(d_out_fake, torch.ones_like(d_out_fake))


def train_enhance(model, loader, device, epochs, lr, out_dir):
  opt_g = optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
  opt_d = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
  for ep in range(epochs):
    model.train()
    for imgs, _ in tqdm(loader, desc=f"enhance ep{ep+1}"):
      imgs = imgs.to(device)
      target = edge_enhance_tensor(imgs)

      mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
      std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
      x01 = torch.clamp(imgs * std + mean, 0, 1)

      fake = model.generate(x01)
      fake_n = (fake - mean) / std


      opt_d.zero_grad()
      d_real = model.discriminator(target)
      d_fake = model.discriminator(fake_n.detach())
      loss_d = gan_loss_d(d_real, d_fake)
      loss_d.backward()
      opt_d.step()


      opt_g.zero_grad()
      d_fake_g = model.discriminator(fake_n)
      loss_g = gan_loss_g(d_fake_g) + F.l1_loss(fake_n, target)
      loss_g.backward()
      opt_g.step()

    torch.save({"model_state_dict": model.state_dict(), "epoch": ep}, out_dir / "last.pth")
  torch.save({"model_state_dict": model.state_dict()}, out_dir / "best.pth")


def train_cyclegan(model, loader, device, epochs, lr, out_dir):
  opt = optim.Adam(
    list(model.G_ab.parameters()) + list(model.G_ba.parameters()) +
    list(model.D_a.parameters()) + list(model.D_b.parameters()),
    lr=lr, betas=(0.5, 0.999),
  )
  for ep in range(epochs):
    model.train()
    for imgs, _ in tqdm(loader, desc=f"cyclegan ep{ep+1}"):
      real_a = imgs.to(device)
      real_b = color_jitter_batch(real_a)
      mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
      std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
      a01 = torch.clamp(real_a * std + mean, 0, 1)
      b01 = torch.clamp(real_b * std + mean, 0, 1)

      fake_b = model.to_target_style(a01)
      fake_a = model.to_source_style(b01)
      rec_a = model.to_source_style(fake_b)
      rec_b = model.to_target_style(fake_a)

      fake_b_n = (fake_b - mean) / std
      fake_a_n = (fake_a - mean) / std


      opt.zero_grad()
      loss_d = gan_loss_d(model.D_a(real_a), model.D_a(fake_a_n.detach()))
      loss_d = loss_d + gan_loss_d(model.D_b(real_b), model.D_b(fake_b_n.detach()))
      loss_d.backward()
      opt.step()


      opt.zero_grad()
      fake_b_n = (model.to_target_style(a01) - mean) / std
      fake_a_n = (model.to_source_style(b01) - mean) / std
      loss_g = gan_loss_g(model.D_b(fake_b_n)) + gan_loss_g(model.D_a(fake_a_n))
      loss_g = loss_g + F.l1_loss(rec_a, a01) + F.l1_loss(rec_b, b01)
      loss_g.backward()
      opt.step()

    torch.save({"model_state_dict": model.state_dict(), "epoch": ep}, out_dir / "last.pth")
  torch.save({"model_state_dict": model.state_dict()}, out_dir / "best.pth")


def train_sr(model, loader, device, epochs, lr, out_dir):
  opt = optim.Adam(model.parameters(), lr=lr)
  for ep in range(epochs):
    model.train()
    for imgs, _ in tqdm(loader, desc=f"sr ep{ep+1}"):
      imgs = imgs.to(device)
      mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
      std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
      hr = torch.clamp(imgs * std + mean, 0, 1)
      lr_img = F.interpolate(hr, scale_factor=0.5, mode="bilinear", align_corners=False)
      pred = model.super_resolve(lr_img, target_size=hr.shape[-2:])
      loss = F.l1_loss(pred, hr) + 0.1 * F.mse_loss(pred, hr)
      opt.zero_grad()
      loss.backward()
      opt.step()
    torch.save({"model_state_dict": model.state_dict(), "epoch": ep}, out_dir / "last.pth")
  torch.save({"model_state_dict": model.state_dict()}, out_dir / "best.pth")


def main():
  p = argparse.ArgumentParser("Train GAN augmentation models")
  p.add_argument("--gan_type", choices=GAN_TYPES, required=True)
  p.add_argument("--data_root", required=True)
  p.add_argument("--epochs", type=int, default=20)
  p.add_argument("--batch_size", type=int, default=4)
  p.add_argument("--lr", type=float, default=2e-4)
  p.add_argument("--img_size", type=int, default=448)
  p.add_argument("--output_dir", default=None)
  p.add_argument("--device", default="cuda")
  args = p.parse_args()

  device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
  out_dir = Path(args.output_dir or f"./output_gan_{args.gan_type}")
  out_dir.mkdir(parents=True, exist_ok=True)

  tf = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
  ds = CrackDataset(args.data_root, split="train", transform=tf)
  loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

  if args.gan_type == "enhance":
    model = create_enhance_gan().to(device)
    train_enhance(model, loader, device, args.epochs, args.lr, out_dir)
  elif args.gan_type == "cyclegan":
    model = create_cyclegan().to(device)
    train_cyclegan(model, loader, device, args.epochs, args.lr, out_dir)
  else:
    model = create_srnet(scale=2).to(device)
    train_sr(model, loader, device, args.epochs, args.lr, out_dir)

  print(f"Saved GAN checkpoint -> {out_dir / 'best.pth'}")


if __name__ == "__main__":
  main()
