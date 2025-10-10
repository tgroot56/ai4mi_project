#!/usr/bin/env python3
"""Post-process already-stitched NIfTI prediction volumes.

Loads every `*.nii.gz` file from --src_folder, applies
`improvements.postprocessing.postprocess_volume`, and writes the result
to --dst_folder preserving filenames.

This helper is used by the postprocessing job when predictions are
already present as full volumes under `stitched_data/baseline/.../pred`.
"""
from pathlib import Path
import argparse
import nibabel as nib

from improvements.postprocessing import postprocess_volume


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src_folder', required=True)
    p.add_argument('--dst_folder', required=True)
    p.add_argument('--num_classes', type=int, default=5)
    p.add_argument('--pp_min_size_voxels', type=int, default=1000)
    p.add_argument('--pp_keep_largest', action='store_true')
    return p.parse_args()


def main():
    args = get_args()
    src = Path(args.src_folder)
    dst = Path(args.dst_folder)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob('*.nii.gz'))
    if not files:
        print(f"No NIfTI files found in {src}")
        return

    for f in files:
        print(f"Processing {f}")
        img = nib.load(str(f))
        data = img.get_fdata().astype(int)

        try:
            data_pp = postprocess_volume(data,
                                         args.num_classes,
                                         min_size_voxels=args.pp_min_size_voxels,
                                         keep_largest=args.pp_keep_largest)
        except Exception as e:
            print(f"postprocess_volume failed for {f}: {e}; saving original volume")
            data_pp = data

        out = dst / f.name
        nib.save(nib.Nifti1Image(data_pp, img.affine, img.header), str(out))
        print(f"Saved post-processed volume: {out}")


if __name__ == '__main__':
    main()
