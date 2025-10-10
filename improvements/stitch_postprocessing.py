#!/usr/bin/env python3

import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from PIL import Image

# Import post-processing function
from postprocess import postprocess_volume


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch 2D slices back to 3D volumes")
    
    parser.add_argument('--data_folder', type=str, required=True,
                        help="Path to folder containing sliced 2D images")
    parser.add_argument('--dest_folder', type=str, required=True,
                        help="Destination folder for stitched 3D volumes")
    parser.add_argument('--num_classes', type=int, required=True,
                        help="Number of classes (e.g., 255 for grayscale, 5 for SEGTHOR)")
    parser.add_argument('--grp_regex', type=str, required=True,
                        help="Regex pattern to extract patient ID from filename")
    parser.add_argument('--source_scan_pattern', type=str, required=True,
                        help="Pattern for original scan files with {id_} placeholder")
    
    # # Post-processing options
    # parser.add_argument('--pp_min_size_voxels', type=int, default=1000,
    #                 help='Min component size (voxels) for 3D post-processing when keep_largest is False')
    # parser.add_argument('--pp_keep_largest', action='store_true',
    #                 help='If set, keep only the largest 3D component per class when post-processing')

    
    args = parser.parse_args()
    print(f"Args: {args}")
    return args


def group_slices_by_patient(data_folder: Path, grp_regex: str) -> Dict[str, List[Path]]:
    slice_files = list(data_folder.glob("*.png"))
    patient_slices = defaultdict(list)
    
    pattern = re.compile(grp_regex)
    
    for slice_file in slice_files:
        match = pattern.match(slice_file.stem)
        if match:
            patient_id = match.group(1)
            patient_slices[patient_id].append(slice_file)
        else:
            print(f"Warning: Could not extract patient ID from {slice_file}")
    
    for patient_id in patient_slices:
        patient_slices[patient_id].sort(key=lambda x: x.stem)
    
    return dict(patient_slices)


def load_original_scan_info(source_scan_pattern: str, patient_id: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    scan_path = Path(source_scan_pattern.format(id_=patient_id))
    
    if not scan_path.exists():
        raise FileNotFoundError(f"Original scan not found: {scan_path}")
    
    img = nib.load(scan_path)
    original_data = img.get_fdata()
    
    return original_data, img


def stitch_patient_slices(slice_files: List[Path], target_shape: Tuple[int, int, int], 
                         num_classes: int) -> np.ndarray:
    target_x, target_y, target_z = target_shape
    
    volume_3d = np.zeros((target_x, target_y, target_z), dtype=np.uint8)
    num_slices = min(len(slice_files), target_z)
    
    for i in range(num_slices):
        slice_file = slice_files[i]

        slice_img = Image.open(slice_file)
        slice_array = np.array(slice_img, dtype=np.uint8)
        
        if len(slice_array.shape) == 3:
            slice_array = slice_array[:, :, 0]
        
        if slice_array.shape != (target_x, target_y):
            from skimage.transform import resize
            slice_array = resize(slice_array, (target_x, target_y), 
                               anti_aliasing=False, preserve_range=True, order=0)
            slice_array = slice_array.astype(np.uint8)
        
        if num_classes == 5:  # SEGTHOR case
            slice_array = slice_array // 63  # This gives the correct class indices
            slice_array = np.clip(slice_array, 0, 4)  # Ensure values stay in 0-4 range
        
        volume_3d[:, :, i] = slice_array
    
    if len(slice_files) < target_z:
        print(f"Warning: Only {len(slice_files)} slices available for target depth {target_z}")
    elif len(slice_files) > target_z:
        print(f"Warning: {len(slice_files)} slices available but only using first {target_z}")
    
    return volume_3d


def save_nifti_volume(volume: np.ndarray, original_img: nib.Nifti1Image, 
                     dest_path: Path, patient_id: str) -> None:
    dest_path.mkdir(parents=True, exist_ok=True)
    
    nifti_img = nib.Nifti1Image(volume, original_img.affine, header=original_img.header)
    
    output_file = dest_path / f"{patient_id}.nii.gz"
    nib.save(nifti_img, output_file)
    print(f"Saved stitched volume: {output_file} with shape {volume.shape}")


def main():
    args = get_args()
    
    data_folder = Path(args.data_folder)
    dest_folder = Path(args.dest_folder)
    
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")
    
    print(f"Processing slices from: {data_folder}")
    print(f"Output directory: {dest_folder}")
    
    # Group slices by patient ID
    patient_slices = group_slices_by_patient(data_folder, args.grp_regex)
    print(f"Found {len(patient_slices)} patients")
    
    for patient_id, slice_files in patient_slices.items():
        print(f"\nProcessing patient {patient_id} with {len(slice_files)} slices")
        
        try:
            original_data, original_img = load_original_scan_info(
                args.source_scan_pattern, patient_id
            )
            target_shape = original_data.shape
            print(f"Target shape for {patient_id}: {target_shape}")
            
            volume_3d = stitch_patient_slices(slice_files, target_shape, args.num_classes)

            save_nifti_volume(volume_3d, original_img, dest_folder, patient_id)

            # # Post-process the stitched 3D volume (remove small components or keep largest)
            # try:
            #     volume_3d_pp = postprocess_volume(
            #         volume_3d,
            #         args.num_classes,
            #         min_size_voxels=args.pp_min_size_voxels,
            #         keep_largest=args.pp_keep_largest,
            #     )
            #     print(f"Applied 3D post-processing (keep_largest={args.pp_keep_largest}, min_size_voxels={args.pp_min_size_voxels}) for patient {patient_id}")
            # except Exception as e:
            #     print(f"Warning: post-processing failed for {patient_id}, saving unprocessed volume. Error: {e}")
            #     volume_3d_pp = volume_3d
            
            # save_nifti_volume(volume_3d_pp, original_img, dest_folder, patient_id)
            
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            continue
    
    print(f"\nStitching completed! Results saved in: {dest_folder}")


if __name__ == "__main__":
    main()