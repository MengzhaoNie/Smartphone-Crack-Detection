from __future__ import annotations
import argparse
import csv
import time
import types
from pathlib import Path
import torch
from train import AVAILABLE_MODELS, build_model

def count_params(model) -> int:
    return sum((p.numel() for p in model.parameters()))

def size_mb(model) -> float:
    n = sum((p.numel() * p.element_size() for p in model.parameters()))
    n += sum((b.numel() * b.element_size() for b in model.buffers()))
    return n / 1024 ** 2

def try_flops(model, img_size: int, device) -> float | None:
    try:
        from thop import profile
        x = torch.randn(1, 3, img_size, img_size, device=device)
        model_ = model.to(device).eval()
        flops, _ = profile(model_, inputs=(x,), verbose=False)
        return flops / 1000000000.0
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            x = torch.randn(1, 3, img_size, img_size, device=device)
            flops = FlopCountAnalysis(model.to(device).eval(), x).total()
            return flops / 1000000000.0
        except Exception:
            return None

@torch.no_grad()
def latency_ms(model, img_size: int, device, warmup: int=10, repeats: int=50) -> float:
    model = model.to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeats

def build_student(name: str, feat_dim: int):
    from models.distill.student_kd import build_student_kd
    return build_student_kd(backbone=name, use_convlstm=False, feat_dim=feat_dim)

def measure(name: str, model, img_size: int, device, warmup: int, repeats: int) -> dict:
    n = count_params(model)
    row = {'name': name, 'params_M': round(n / 1000000.0, 4), 'size_MB': round(size_mb(model), 4), 'flops_G': None, 'latency_ms': None}
    flops = try_flops(model, img_size, device)
    if flops is not None:
        row['flops_G'] = round(flops, 4)
    try:
        row['latency_ms'] = round(latency_ms(model, img_size, device, warmup, repeats), 4)
    except Exception as e:
        row['latency_ms'] = f'err:{e}'
    return row

def get_args():
    p = argparse.ArgumentParser('Efficiency report')
    p.add_argument('--model', default=None, choices=AVAILABLE_MODELS)
    p.add_argument('--student', default=None, choices=['efficientnet', 'mobilenet', 'shufflenet', 'fastscnn', 'mobilevit'])
    p.add_argument('--all_baselines', action='store_true')
    p.add_argument('--all_students', action='store_true')
    p.add_argument('--feat_dim', type=int, default=64)
    p.add_argument('--img_size', type=int, default=448)
    p.add_argument('--device', default='cuda')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--output', default='./efficiency.csv')
    return p.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    rows = []
    if args.all_baselines:
        for name in AVAILABLE_MODELS:
            print(f'measure {name} ...')
            try:
                m = build_model(name, types.SimpleNamespace(img_size=args.img_size))
                rows.append(measure(name, m, args.img_size, device, args.warmup, args.repeats))
            except Exception as e:
                rows.append({'name': name, 'params_M': None, 'size_MB': None, 'flops_G': None, 'latency_ms': f'err:{e}'})
            print(rows[-1])
    if args.all_students:
        for name in ('efficientnet', 'mobilenet', 'shufflenet', 'fastscnn', 'mobilevit'):
            print(f'measure student {name} ...')
            m = build_student(name, args.feat_dim)
            rows.append(measure(f'student_{name}', m, args.img_size, device, args.warmup, args.repeats))
            print(rows[-1])
    if args.model:
        m = build_model(args.model, types.SimpleNamespace(img_size=args.img_size))
        rows.append(measure(args.model, m, args.img_size, device, args.warmup, args.repeats))
        print(rows[-1])
    if args.student:
        m = build_student(args.student, args.feat_dim)
        rows.append(measure(f'student_{args.student}', m, args.img_size, device, args.warmup, args.repeats))
        print(rows[-1])
    if not rows:
        raise SystemExit('Specify --model / --student / --all_baselines / --all_students')
    out = Path(args.output)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Saved: {out}')
    if any((r.get('flops_G') is None for r in rows)):
        print('Note: install `thop` or `fvcore` for FLOPs (pip install thop).')
if __name__ == '__main__':
    main()
