# Smartphone-Crack-Detection

Real-time road crack detection on smartphones through ConvLSTM-based temporal knowledge distillation from a CNN-KAN and VMamba dual-path network.

## Setup

```bash
conda create -n CV python=3.11
conda activate CV
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install timm==1.0.3 einops==0.8.2 fvcore numpy pandas pillow scikit-learn opencv-python matplotlib tqdm tensorboard openpyxl onnx onnxruntime
```

Weights and datasets are **not** in this repo.  
Self-built dataset: [https://doi.org/10.6084/m9.figshare.33090545](https://doi.org/10.6084/m9.figshare.33090545)

## Layout

```text
train.py / predict.py / train_transfer.py / train_distill.py / train_teacher_gan.py
train_gan.py / generate_gan_data.py
eval_robustness.py / eval_per_dataset.py / estimate_efficiency.py
export_student_onnx.py
scripts/prepare_*.py
models/{modules,baselines,ablation,distill,gan}/
datasets/  util/
```

## Data

Prepare via [`data/README.md`](data/README.md):

```bash
python scripts/prepare_public_combined.py --crack500 ... --deepcrack ... --cfd ... --cracktree ... --output ./public_combined
python scripts/prepare_smartphone_keyframes.py --input ... --output ./smartphone_crack
python scripts/extract_gps_neighbors.py --data_root ./smartphone_crack --gps_csv P_gps.dat --video capture.mp4 --distance_m 1.0
```

Expected: `<root>/{train,val,test}/{images,masks}/` and `neighbors/` + `clips.csv` for ConvLSTM.

## Train

```bash
python train.py --model dual_encoder --data_root /path/to/public_combined
python train_transfer.py --model dual_encoder --pretrained public_best.pth --data_root /path/to/smartphone_crack
python train_teacher_gan.py --model dual_encoder --data_root /path/to/data --gan_root /path/to/gan --gan_type enhance
python train_distill.py --data_root /path/to/clips --teacher_ckpt teacher.pth --teacher_arch dual_encoder --student mobilevit --mode full
```

`--model`: `unet` `resunet` `skpnet` `mambacracknet` `fpn` `pspnet` `deeplabv3` `deeplabv3plus` `transunet` `crackformer` `defnet` `ukan_only` `vmamba_only` `dual_encoder_concat` `dual_encoder`

`--student`: `efficientnet` `mobilenet` `shufflenet` `fastscnn` `mobilevit`

Defaults: Adam lr=`1e-3`, CosineAnnealingWarmRestarts(`T_0=15`, `T_mult=2`), batch=`8`, threshold=`0.633`, input=`448`.

## Eval

```bash
python eval_per_dataset.py --model dual_encoder --checkpoint best.pth --combined_root ./public_combined
python eval_robustness.py --model dual_encoder --checkpoint best.pth --data_root ./smartphone_crack
python estimate_efficiency.py --all_baselines --all_students
python predict.py --model dual_encoder --checkpoint best.pth --image_dir ./images --output_dir ./out
```

## ONNX export

```bash
python export_student_onnx.py --checkpoint student.pth --student mobilevit --output student.onnx
```

## License

[`LICENSE`](LICENSE)
