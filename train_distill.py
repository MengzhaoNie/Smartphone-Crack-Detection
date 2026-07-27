
from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.clip_dataset import get_clip_dataloaders
from models.distill.losses import DistillLossBundle
from models.distill.student_kd import build_student_kd
from models.distill.teacher_adapter import build_teacher_adapter


def load_teacher(args, device):

    arch = (args.teacher_arch or "unet").lower()
    if arch in ("dual_encoder", "teacher"):
        from models.ablation.dual_encoder import create_dual_encoder
        teacher = create_dual_encoder(num_classes=1)
    elif arch == "unet":
        from models.baselines.Unet import create_unet
        teacher = create_unet(num_classes=1)
    else:

        from train import build_model
        teacher = build_model(arch, types.SimpleNamespace(img_size=args.img_size))

    if args.teacher_ckpt and Path(args.teacher_ckpt).is_file():
        ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = teacher.load_state_dict(clean, strict=False)
        print(f"Loaded teacher ckpt: {args.teacher_ckpt}  missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("WARNING: no teacher_ckpt, using random teacher weights")

    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return build_teacher_adapter(teacher, feat_dim=args.feat_dim).to(device)


from util.metrics import calculate_per_image_metrics


@torch.no_grad()
def evaluate_student(student, loader, device, threshold: float):
    return calculate_per_image_metrics(student, loader, device, threshold)


def train_distill(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mode = args.mode.lower()
    use_convlstm = mode == "full"
    out_dir = Path(args.output_dir or f"./output_distill_{args.student}_{mode}")
    log_dir = Path(args.log_dir or f"./runs_distill_{args.student}_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    writer = SummaryWriter(str(log_dir))

    train_loader, val_loader = get_clip_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size
    )
    print(f"clips: train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    teacher = load_teacher(args, device)
    student = build_student_kd(
        backbone=args.student,
        use_convlstm=use_convlstm,
        feat_dim=args.feat_dim,
        pretrained=args.pretrained_backbone,
    ).to(device)

    if getattr(args, "resume_student", None) and Path(args.resume_student).is_file():
        ckpt = torch.load(args.resume_student, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = student.load_state_dict(state, strict=False)
        print(f"Resumed student from {args.resume_student} (missing={len(missing)} unexpected={len(unexpected)})")

    criterion = DistillLossBundle(
        mode=mode, w_task=args.w_task, w_feat=args.w_feat, w_resp=args.w_resp, w_temp=args.w_temp
    ).to(device)


    optimizer = optim.Adam(
        list(student.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"student={args.student} mode={mode} convlstm={use_convlstm} params={n_params} device={device}")

    best_dice = -1.0
    rows = []

    for epoch in range(args.epochs):
        student.train()
        teacher.eval()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for frames, masks in pbar:
            frames = frames.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_out = teacher.forward_clip(frames)

            with autocast(enabled=use_amp):
                student_out = student.forward_clip(frames)
                loss, parts = criterion(student_out, teacher_out, masks)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", task=f"{parts['task'].item():.3f}")

        scheduler.step()
        train_loss = running / max(len(train_loader), 1)
        val_metrics = evaluate_student(student, val_loader, device, args.threshold)

        print(
            f"[{args.student}/{mode}] epoch {epoch+1}: loss={train_loss:.4f} "
            f"val_mDice={val_metrics['mdice']:.4f}±{val_metrics['mdice_std']:.4f} "
            f"val_mIoU={val_metrics['miou']:.4f}±{val_metrics['miou_std']:.4f}"
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"{k}/val", v, epoch)

        payload = {
            "epoch": epoch,
            "student": args.student,
            "mode": mode,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metrics": val_metrics,
        }
        torch.save(payload, out_dir / f"last_student_{args.student}_{mode}.pth")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(payload, out_dir / f"best_student_{args.student}_{mode}.pth")
            print(f"  saved best dice={best_dice:.4f}")

        rows.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics, "lr": optimizer.param_groups[0]["lr"]})
        pd.DataFrame(rows).to_csv(out_dir / f"metrics_{args.student}_{mode}.csv", index=False)

    writer.close()
    print(f"Done. best val dice={best_dice:.4f} -> {out_dir}")


def get_args():
    p = argparse.ArgumentParser("ConvLSTM knowledge distillation")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, default=None)
    p.add_argument("--teacher_arch", type=str, default="unet", help="unet | dual_encoder | other train.py model")
    p.add_argument(
        "--student",
        type=str,
        default="mobilevit",
        choices=["efficientnet", "mobilenet", "shufflenet", "fastscnn", "mobilevit"],
        help="efficientnet | mobilenet | shufflenet | fastscnn | mobilevit",
    )
    p.add_argument("--mode", type=str, default="full", choices=["task", "kd", "full"])
    p.add_argument(
        "--curriculum",
        action="store_true",
        help="sequential task, kd, full stages",
    )
    p.add_argument(
        "--curriculum_epochs",
        default="30,30,40",
        help="epochs for task,kd,full when --curriculum is set",
    )
    p.add_argument("--resume_student", type=str, default=None, help="warm-start student weights")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--feat_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--t0", type=int, default=15, help="CosineAnnealingWarmRestarts T_0")
    p.add_argument("--t_mult", type=int, default=2)
    p.add_argument("--eta_min", type=float, default=1e-6)
    p.add_argument("--w_task", type=float, default=1.0)
    p.add_argument("--w_feat", type=float, default=0.5)
    p.add_argument("--w_resp", type=float, default=0.5)
    p.add_argument("--w_temp", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.633)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--pretrained_backbone", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.curriculum:
        parts = [int(x.strip()) for x in args.curriculum_epochs.split(",") if x.strip()]
        while len(parts) < 3:
            parts.append(0)
        stages = [("task", parts[0]), ("kd", parts[1]), ("full", parts[2])]
        base = Path(args.output_dir or f"./output_distill_{args.student}_curriculum")
        prev = args.resume_student
        for stage, nep in stages:
            if nep <= 0:
                continue
            args.mode = stage
            args.epochs = nep
            args.output_dir = str(base / stage)
            args.log_dir = str(base / f"runs_{stage}")
            args.resume_student = prev
            print(f"\n======== CURRICULUM STAGE: {stage} ({nep} epochs) ========")
            train_distill(args)
            prev = str(Path(args.output_dir) / f"best_student_{args.student}_{stage}.pth")
        print(f"Curriculum finished. Final best: {prev}")
    else:
        train_distill(args)
