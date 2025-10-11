
import numpy as np
from scipy import ndimage as ndi


# 3D volume post-processing
def postprocess_volume(volume: np.ndarray,
                       num_classes: int,
                       min_size_voxels: int = 1000,
                       keep_largest: bool = True,
                       spacing: tuple | None = None) -> np.ndarray:
    """
    Post-process a 3D integer segmentation volume.
    - volume: 3D array with integer class labels, shape (Z, H, W) or (H, W, Z) depending on your pipeline.
              This function preserves the input shape and dtype.
    - num_classes: number of classes (including background 0).
    - min_size_voxels: minimum connected component size in voxels to keep (used when keep_largest is False).
    - keep_largest: if True, keep only the largest connected component per class (3D).
    - spacing: optional voxel spacing (dz, dy, dx). If provided and min_size_voxels is 0,
               you can convert a min_size_mm threshold externally to voxels before calling.
    Returns a new 3D array of same shape and dtype.
    """
    if volume.dtype.kind not in ('u', 'i'):
        volume = volume.astype(np.int32)

    orig_dtype = volume.dtype
    vol = volume.copy()
    out = vol.copy()

    for cls in range(1, num_classes):
        mask = (vol == cls)
        if not mask.any():
            continue

        labeled, ncomp = ndi.label(mask)
        if ncomp == 0:
            continue

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background

        if keep_largest:
            largest_label = int(np.argmax(sizes))
            if sizes[largest_label] > 0:
                keep_mask = (labeled == largest_label)
            else:
                keep_mask = np.zeros_like(mask, dtype=bool)
        else:
            keep_labels = np.nonzero(sizes >= min_size_voxels)[0]
            if keep_labels.size > 0:
                keep_mask = np.isin(labeled, keep_labels)
            else:
                keep_mask = np.zeros_like(mask, dtype=bool)

        out[vol == cls] = 0
        out[keep_mask] = cls

    return out.astype(orig_dtype)

