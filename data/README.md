# Dataset

Raw images are **not** stored in this repository.

Self-built smartphone keyframes: [https://doi.org/10.6084/m9.figshare.33090545](https://doi.org/10.6084/m9.figshare.33090545)

Public benchmarks (Crack500 / DeepCrack / CFD / CrackTree) must be downloaded from their original releases.

## Prepare keyframe splits (70 / 15 / 15)

```bash
python scripts/prepare_smartphone_keyframes.py \
  --input  /path/to/keyframes \
  --output /path/to/smartphone_crack --seed 42
```

## GPS ~1 m neighbors for ConvLSTM

Supported GPS inputs:

- Standard CSV header: `frame_idx,timestamp,lat,lon,path`
- Native acquisition `.dat` (first line metadata + `Image_Name,latitude,longitude,...`)

`path` / `Image_Name` is optional if you pass `--video` (uses `frame_idx`) or a sequential `--frames_dir`.

```bash
python scripts/extract_gps_neighbors.py \
  --data_root /path/to/smartphone_crack \
  --gps_csv /path/to/P_gps.dat \
  --video /path/to/capture.mp4 \
  --distance_m 1.0 \
  --img_size 448
```

Or from already extracted frames:

```bash
python scripts/extract_gps_neighbors.py \
  --data_root /path/to/smartphone_crack \
  --gps_csv P_gps.dat \
  --frames_dir /path/to/video_frames \
  --distance_m 1.0
```

Writes:

```text
<data_root>/<split>/neighbors/<stem>_prev.png
<data_root>/<split>/neighbors/<stem>_next.png
<data_root>/<split>/clips.csv
```

`train_distill.py` reads these clips automatically.

## Public combined set

```bash
python scripts/prepare_public_combined.py \
  --crack500  /path/to/Crack500 \
  --deepcrack /path/to/DeepCrack \
  --cfd       /path/to/CFD \
  --cracktree /path/to/CrackTree \
  --output    /path/to/public_combined --seed 42
```

## Layout

```text
<data_root>/
  train|val|test/
    images/
    masks/
    neighbors/          # from extract_gps_neighbors.py
    clips.csv           # optional manifest
```
