# tools/offline_augment_patients.py
import argparse, shutil, random, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import albumentations as A
from typing import Optional, Dict, List, Tuple

# ---------- Augmentation (Replayable) ----------
def build_aug():
    # ReplayCompose lets us sample once, then replay the exact params
    return A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10,
                           border_mode=0, p=0.7),
        A.ElasticTransform(alpha=50, sigma=7, alpha_affine=10, border_mode=0, p=0.2),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        A.GaussNoise(var_limit=(5, 15), p=0.2),
    ], additional_targets={'mask': 'mask'})

# ---------- I/O helpers ----------
def save_image(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)

def copy_subset(src_sub: Path, dst_sub: Path):
    if dst_sub.exists():
        print(f"[INFO] removing existing {dst_sub}")
        shutil.rmtree(dst_sub)
    shutil.copytree(src_sub, dst_sub)

# ---------- Patient parsing & renaming ----------
# Try to extract an integer patient ID from filename or parent folder
_PATIENT_PATTERNS = [
    re.compile(r'(?:^|[_-])patient[_-]?(\d+)(?=[_-]|$)', re.IGNORECASE),
    re.compile(r'(?:^|[_-])(\d{1,4})(?=[_-]|$)')  # fallback: bare number chunk in name
]

def extract_patient_id(path: Path) -> Optional[int]:
    name = path.stem
    for pat in _PATIENT_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    # try parent folder name e.g. .../patient_12/img.png
    parent = path.parent.name
    for pat in _PATIENT_PATTERNS:
        m = pat.search(parent)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None

def rename_with_new_patient_id(path: Path, new_pid: int) -> Path:
    """Return a NEW Path with the patient number in the filename replaced by new_pid,
    preserving any zero-padding width (e.g., 01 -> 41)."""
    path = Path(path)
    name, suf = path.stem, path.suffix
    for pat in _PATIENT_PATTERNS:
        m = pat.search(name)
        if m:
            start, end = m.span(1)                  # span of the digits
            width = end - start                     # keep zero padding
            new_digits = f"{new_pid:0{width}d}"
            new_name = name[:start] + new_digits + name[end:]
            return path.with_name(new_name + suf)
    # fallback: prefix new id
    return path.with_name(f"patient_{new_pid}__{name}{suf}")

# ---------- Pairing ----------
def pair_paths(img_dir: Path, msk_dir: Path):
    pairs = []
    for img_p in sorted(img_dir.glob("*")):
        if not img_p.is_file():
            continue
        stem = img_p.stem
        msk_candidates = list(msk_dir.glob(stem + ".*"))
        if not msk_candidates:
            print(f"[WARN] No mask for {img_p.name}, skipping")
            continue
        msk_p = msk_candidates[0]
        pairs.append((img_p, msk_p))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="e.g., data/SEGTHOR_CLEAN")
    ap.add_argument("--dst", required=True, help="e.g., data/SEGTHOR_AUGMENTED")
    ap.add_argument("--images-dirname", default="img")
    ap.add_argument("--masks-dirname",  default="gt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    print(vars(args))

    src = Path(args.src)
    dst = Path(args.dst)
    train_src = src / "train"
    val_src   = src / "val"

    # 1) Copy val exactly (no aug)
    if val_src.exists():
        copy_subset(val_src, dst / "val")

    # 2) Copy train originals
    copy_subset(train_src, dst / "train")

    img_dir = train_src / args.images_dirname
    msk_dir = train_src / args.masks_dirname
    pairs = pair_paths(img_dir, msk_dir)

    # Group by patient
    groups: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    missing_pid = 0
    for img_p, msk_p in pairs:
        pid = extract_patient_id(img_p)
        if pid is None:
            missing_pid += 1
            print(f"[WARN] Could not parse patient ID from {img_p.name}; skipping")
            continue
        groups[pid].append((img_p, msk_p))

    patient_ids = sorted(groups.keys())
    N_patients = 40
    print(f"[INFO] Found {N_patients} patients, {len(pairs)-missing_pid} pairs total.")

    aug = build_aug()
    out_img_dir = dst / "train" / args.images_dirname
    out_msk_dir = dst / "train" / args.masks_dirname

    # 3) For each patient: sample ONE transform, replay for all slices; save under new patient ID = pid + N
    for pid in patient_ids:
        slices = groups[pid]
        # Use first slice to sample transform
        img0 = np.array(Image.open(slices[0][0]))
        msk0 = np.array(Image.open(slices[0][1])).astype(np.uint8)
        first = aug(image=img0, mask=msk0)
        replay = first["replay"]

        new_pid = pid + N_patients  # continue numbering: e.g., 1 -> 41 (if N=40)
        for img_p, msk_p in slices:
            img_np = np.array(Image.open(img_p))
            msk_np = np.array(Image.open(msk_p)).astype(np.uint8)

            out = A.ReplayCompose.replay(replay, image=img_np, mask=msk_np)
            ai, am = out["image"], out["mask"].astype(np.uint8)

            img_aug_name = rename_with_new_patient_id(img_p, new_pid).name
            msk_aug_name = rename_with_new_patient_id(msk_p, new_pid).name
            img_aug_p = out_img_dir / img_aug_name
            msk_aug_p = out_msk_dir / msk_aug_name

            save_image(img_aug_p, ai)
            save_image(msk_aug_p, am)

    print(f"[DONE] Originals copied to {dst/'train'}, augmented copies written with patient IDs shifted by +{N_patients}.")
    if val_src.exists():
        print(f"[DONE] Validation copied unchanged to {dst/'val'}")

if __name__ == "__main__":
    main()
