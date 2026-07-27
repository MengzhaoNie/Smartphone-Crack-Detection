
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


from train import AVAILABLE_MODELS, build_model

BG = (0x44, 0x00, 0x46)
FG = (0xFD, 0xE1, 0x00)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class _Args:
    def __init__(self, img_size: int):
        self.img_size = img_size


def load_checkpoint(model, ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(clean, strict=True)
    model.to(device).eval()
    return model


def find_mask(mask_dir: Path | None, stem: str, src_name: str | None = None) -> Path | None:
    if mask_dir is None:
        return None
    candidates = []
    if src_name:
        candidates.append(mask_dir / src_name)
        candidates.append(mask_dir / Path(src_name).name)
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        candidates.append(mask_dir / f"{stem}{ext}")
    for p in candidates:
        if p.exists():
            return p
    return None


def metrics(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool).ravel()
    gt = gt.astype(bool).ravel()
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
    dice = (2 * tp + 1e-8) / (pred.sum() + gt.sum() + 1e-8)
    return float(iou), float(dice)


def colorize(pred: np.ndarray) -> Image.Image:
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[pred == 0] = BG
    out[pred == 1] = FG
    return Image.fromarray(out, mode="RGB")


@torch.no_grad()
def predict_one(model, image: Image.Image, device, img_size: int, threshold: float) -> np.ndarray:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    if logits.shape[-2:] != (img_size, img_size):
        logits = F.interpolate(logits, size=(img_size, img_size), mode="bilinear", align_corners=True)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob > threshold).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser("Unified crack-segmentation predictor")
    parser.add_argument("--model", type=str, required=True, choices=AVAILABLE_MODELS)
    parser.add_argument("--checkpoint", type=str, required=True, help="path to .pth")
    parser.add_argument("--image_dir", type=str, required=True, help="folder of input images")
    parser.add_argument("--output_dir", type=str, required=True, help="folder to save predictions")
    parser.add_argument("--mask_dir", type=str, default=None, help="optional GT masks for IoU/Dice")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--threshold", type=float, default=0.633)
    parser.add_argument("--colorize", action="store_true", help="save #440046/#FDE100 colored mask")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(args.model, _Args(args.img_size))
    model = load_checkpoint(model, Path(args.checkpoint), device)
    print(f"Loaded {args.model} from {args.checkpoint}")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTS and p.is_file())
    if not images:
        raise SystemExit(f"No images in {image_dir}")

    rows = [("stem", "iou", "dice", "file")]
    for img_path in tqdm(images, desc="predict"):
        image = Image.open(img_path).convert("RGB")
        image_r = image.resize((args.img_size, args.img_size), Image.BILINEAR)
        pred = predict_one(model, image_r, device, args.img_size, args.threshold)

        mask_path = find_mask(mask_dir, img_path.stem)
        if mask_path is not None:
            gt = np.array(
                Image.open(mask_path).convert("L").resize((args.img_size, args.img_size), Image.NEAREST)
            )
            gt = (gt > 127).astype(np.uint8)
            iou, dice = metrics(pred, gt)
        else:
            iou = dice = float("nan")

        if args.colorize:
            vis = colorize(pred)
            out_name = f"{img_path.stem}__{args.model}.png"
            vis.save(out_dir / out_name)
        else:
            out_name = f"{img_path.stem}__{args.model}.png"
            Image.fromarray((pred * 255).astype(np.uint8), mode="L").save(out_dir / out_name)

        rows.append((img_path.stem, f"{iou:.6f}", f"{dice:.6f}", out_name))
        print(f"{out_name}  IoU={iou:.4f} Dice={dice:.4f}")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
