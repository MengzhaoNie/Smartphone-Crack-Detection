from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _gps_text_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("samplingnumber"):
        lines = lines[1:]
    return lines


def _frame_idx_from_name(name: str, fallback: int) -> int:
    stem = Path(name.strip()).stem
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            n = int(tail)
            return n - 1 if n > 0 else 0
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            pass
    return fallback


def load_gps_csv(path: Path) -> List[dict]:
    rows: List[dict] = []
    lines = _gps_text_lines(path)
    if len(lines) < 2:
        raise RuntimeError(f"GPS file too short: {path}")

    import io

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    fields = {c.lower().strip(): c for c in (reader.fieldnames or [])}

    def col(*names: str) -> Optional[str]:
        for n in names:
            if n in fields:
                return fields[n]
        return None

    c_frame = col("frame", "frame_idx", "frame_id", "index", "idx", "id")
    c_time = col("timestamp", "time", "t", "sec", "seconds", "gpstimestamp")
    c_lat = col("lat", "latitude", "y")
    c_lon = col("lon", "lng", "longitude", "x")
    c_path = col("path", "image", "file", "filename", "name", "image_name")
    if c_lat is None or c_lon is None:
        raise ValueError(f"GPS csv needs lat/lon columns: {path}")

    for i, row in enumerate(reader):
        lat = float(str(row[c_lat]).strip())
        lon = float(str(row[c_lon]).strip())
        img_name = row.get(c_path, "").strip() if c_path else ""
        if c_frame and row.get(c_frame, "").strip() != "":
            frame_idx = int(float(str(row[c_frame]).strip()))
        elif img_name:
            frame_idx = _frame_idx_from_name(img_name, i)
        else:
            frame_idx = i
        if c_time and row.get(c_time, "").strip() != "":
            raw_t = str(row[c_time]).strip()
            try:
                ts = float(raw_t)
            except ValueError:
                ts = float(i)
        else:
            ts = float(frame_idx)
        rows.append(
            {
                "i": i,
                "frame_idx": frame_idx,
                "t": ts,
                "lat": lat,
                "lon": lon,
                "path": img_name,
            }
        )
    if len(rows) < 3:
        raise RuntimeError(f"GPS track too short: {path}")
    return rows


def cumulative_distance_m(track: Sequence[dict]) -> np.ndarray:
    d = np.zeros(len(track), dtype=np.float64)
    for i in range(1, len(track)):
        d[i] = d[i - 1] + haversine_m(
            track[i - 1]["lat"], track[i - 1]["lon"], track[i]["lat"], track[i]["lon"]
        )
    return d


def find_neighbor_index(cum: np.ndarray, key_i: int, target_m: float, direction: int) -> int:
    key_d = cum[key_i]
    if direction < 0:
        target = key_d - target_m
        if target <= cum[0]:
            return 0
        j = key_i
        while j > 0 and cum[j] > target:
            j -= 1
        if j + 1 <= key_i and abs(cum[j] - target) > abs(cum[j + 1] - target):
            j = j + 1
        return max(0, min(j, key_i))
    target = key_d + target_m
    if target >= cum[-1]:
        return len(cum) - 1
    j = key_i
    while j < len(cum) - 1 and cum[j] < target:
        j += 1
    if j - 1 >= key_i and abs(cum[j] - target) > abs(cum[j - 1] - target):
        j = j - 1
    return max(key_i, min(j, len(cum) - 1))


def match_keyframe_to_track(
    key_stem: str,
    track: List[dict],
    stem_to_idx: Dict[str, int],
) -> Optional[int]:
    if key_stem in stem_to_idx:
        return stem_to_idx[key_stem]
    for k, idx in stem_to_idx.items():
        if key_stem.endswith(k) or k.endswith(key_stem) or key_stem in k or k in key_stem:
            return idx
    digits = "".join(ch for ch in key_stem if ch.isdigit())
    if digits:
        try:
            n = int(digits[-6:])
            for row in track:
                if int(row["frame_idx"]) == n:
                    return int(row["i"])
        except ValueError:
            pass
    return None


def build_stem_index(track: List[dict], frames_dir: Optional[Path]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in track:
        if row["path"]:
            stem = Path(row["path"]).stem
            out[stem] = int(row["i"])
            out[Path(row["path"]).name] = int(row["i"])
        out[str(row["frame_idx"])] = int(row["i"])
        out[f"{int(row['frame_idx']):06d}"] = int(row["i"])
        out[f"{int(row['frame_idx']):08d}"] = int(row["i"])
    if frames_dir and frames_dir.is_dir():
        files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        if len(files) == len(track):
            for i, p in enumerate(files):
                out[p.stem] = i
                out[p.name] = i
    return out


def open_video(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def read_video_frame(cap, frame_idx: int):
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_frame_image(frames_dir: Path, track_row: dict, fallback_idx: int):
    from PIL import Image

    if track_row.get("path"):
        p = Path(track_row["path"])
        if not p.is_file():
            p = frames_dir / Path(track_row["path"]).name
        if p.is_file():
            return Image.open(p).convert("RGB")
    files = sorted([x for x in frames_dir.iterdir() if x.suffix.lower() in IMG_EXTS])
    if 0 <= fallback_idx < len(files):
        return Image.open(files[fallback_idx]).convert("RGB")
    stem = f"{int(track_row['frame_idx']):06d}"
    for ext in IMG_EXTS:
        p = frames_dir / f"{stem}{ext}"
        if p.is_file():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"No frame for idx={fallback_idx} under {frames_dir}")


def list_key_stems(images_dir: Path) -> List[str]:
    return sorted({p.stem for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS})


def process_split(
    split_dir: Path,
    track: List[dict],
    cum: np.ndarray,
    stem_to_idx: Dict[str, int],
    distance_m: float,
    img_size: int,
    video_path: Optional[Path],
    frames_dir: Optional[Path],
) -> Tuple[int, int]:
    from PIL import Image

    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    nb_dir = split_dir / "neighbors"
    nb_dir.mkdir(parents=True, exist_ok=True)
    if not images_dir.is_dir():
        return 0, 0

    cap = open_video(video_path) if video_path else None
    clips = []
    ok_n = 0
    miss_n = 0
    try:
        for stem in list_key_stems(images_dir):
            key_i = match_keyframe_to_track(stem, track, stem_to_idx)
            if key_i is None:
                miss_n += 1
                continue
            prev_i = find_neighbor_index(cum, key_i, distance_m, -1)
            next_i = find_neighbor_index(cum, key_i, distance_m, +1)
            if prev_i == key_i and key_i > 0:
                prev_i = key_i - 1
            if next_i == key_i and key_i < len(track) - 1:
                next_i = key_i + 1

            def grab(idx: int) -> Image.Image:
                row = track[idx]
                if cap is not None:
                    arr = read_video_frame(cap, int(row["frame_idx"]))
                    return Image.fromarray(arr)
                if frames_dir is None:
                    raise RuntimeError("Need --video or --frames_dir")
                return load_frame_image(frames_dir, row, idx)

            prev_img = grab(prev_i).resize((img_size, img_size), Image.BILINEAR)
            next_img = grab(next_i).resize((img_size, img_size), Image.BILINEAR)
            prev_path = nb_dir / f"{stem}_prev.png"
            next_path = nb_dir / f"{stem}_next.png"
            prev_img.save(prev_path)
            next_img.save(next_path)

            key_rel = f"images/{stem}.png"
            key_file = None
            for ext in IMG_EXTS:
                p = images_dir / f"{stem}{ext}"
                if p.is_file():
                    key_file = p.name
                    key_rel = f"images/{p.name}"
                    break
            mask_rel = f"masks/{stem}.png"
            for ext in IMG_EXTS:
                p = masks_dir / f"{stem}{ext}"
                if p.is_file():
                    mask_rel = f"masks/{p.name}"
                    break
            clips.append(
                {
                    "keyframe": key_rel,
                    "prev": f"neighbors/{prev_path.name}",
                    "next": f"neighbors/{next_path.name}",
                    "mask": mask_rel,
                    "key_gps_i": key_i,
                    "prev_gps_i": prev_i,
                    "next_gps_i": next_i,
                    "dist_prev_m": float(abs(cum[key_i] - cum[prev_i])),
                    "dist_next_m": float(abs(cum[next_i] - cum[key_i])),
                }
            )
            ok_n += 1
    finally:
        if cap is not None:
            cap.release()

    if clips:
        with open(split_dir / "clips.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(clips[0].keys()))
            w.writeheader()
            w.writerows(clips)
    return ok_n, miss_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--gps_csv", required=True)
    p.add_argument("--video", default=None)
    p.add_argument("--frames_dir", default=None)
    p.add_argument("--distance_m", type=float, default=1.0)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--splits", default="train,val,test")
    args = p.parse_args()

    if not args.video and not args.frames_dir:
        raise SystemExit("Provide --video and/or --frames_dir")

    root = Path(args.data_root)
    track = load_gps_csv(Path(args.gps_csv))
    cum = cumulative_distance_m(track)
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    video_path = Path(args.video) if args.video else None
    stem_to_idx = build_stem_index(track, frames_dir)

    summary = {
        "gps_points": len(track),
        "track_length_m": float(cum[-1]),
        "distance_m": args.distance_m,
        "splits": {},
    }
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        ok_n, miss_n = process_split(
            split_dir,
            track,
            cum,
            stem_to_idx,
            args.distance_m,
            args.img_size,
            video_path,
            frames_dir,
        )
        summary["splits"][split] = {"ok": ok_n, "miss": miss_n}
        print(f"{split}: ok={ok_n} miss={miss_n}")

    (root / "gps_neighbors_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
