from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.distill.student_kd import build_student_kd


class StudentDeployWrapper(torch.nn.Module):


  def __init__(self, student_kd):
    super().__init__()
    self.student = student_kd.student

  def forward(self, x):
    return self.student(x)


def load_student(ckpt_path: str, backbone: str, use_convlstm: bool, feat_dim: int, device):
  model = build_student_kd(backbone=backbone, use_convlstm=use_convlstm, feat_dim=feat_dim)
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  state = ckpt.get("model_state_dict", ckpt)
  clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
  model.load_state_dict(clean, strict=False)
  return StudentDeployWrapper(model).to(device).eval()


def main():
  p = argparse.ArgumentParser("Export student ONNX")
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--student", default="mobilevit", choices=["efficientnet", "mobilenet", "shufflenet", "fastscnn", "mobilevit"])
  p.add_argument("--img_size", type=int, default=448)
  p.add_argument("--output", default="student.onnx")
  p.add_argument("--opset", type=int, default=13)
  p.add_argument("--feat_dim", type=int, default=64)
  p.add_argument("--use_convlstm", action="store_true")
  p.add_argument("--device", default="cpu")
  args = p.parse_args()

  device = torch.device(args.device)
  model = load_student(args.checkpoint, args.student, args.use_convlstm, args.feat_dim, device)
  dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)

  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  torch.onnx.export(
    model,
    dummy,
    str(out_path),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=args.opset,
    do_constant_folding=True,
  )
  print(f"Exported ONNX -> {out_path}")

  try:
    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))
    print("ONNX check passed")
  except ImportError:
    print("onnx not installed, skip checker")


if __name__ == "__main__":
  main()
