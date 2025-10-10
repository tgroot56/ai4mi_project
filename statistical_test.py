import argparse
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
import os

def load_metric(dir_path: Path, seed_dir: Path, name: str) -> np.ndarray:
    npy = seed_dir / f"{name}.npy"
    npz = seed_dir / f"{name}.npz"
    if npy.exists():
        return np.load(npy)
    if npz.exists():
        data = np.load(npz)
        if "arr_0" in data.files:
            return data["arr_0"]
        return data[data.files[0]]
    raise FileNotFoundError(f"Could not find {name}.npy or {name}.npz in {seed_dir}")

def collect_seed_dirs(model_dir: Path):
    seed_dirs = sorted([p for p in model_dir.glob("metrics_seed*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No 'metrics_seed*/' folders found in {model_dir}")
    return seed_dirs

def stack_across_seeds(model_dir: Path, metric_name: str) -> np.ndarray:
    seeds = collect_seed_dirs(model_dir)
    arrays = []
    for sd in seeds:
        arr = load_metric(model_dir, sd, metric_name)
        arr = np.asarray(arr).ravel()
        arrays.append(arr)
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Patient counts differ across seeds for {model_dir} / {metric_name}: {lengths}")
    return np.vstack(arrays)

def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 1234):
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.median(values[idx], axis=1)
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return lo, hi

def rank_biserial_from_wilcoxon(stat_w: float, n_nonzero: int) -> float:
    T = n_nonzero * (n_nonzero + 1) / 2
    return (2 * stat_w - T) / T

def analyze_pair(baseline_dir: Path, improved_dir: Path, metric_name: str, higher_is_better: bool):
    base = stack_across_seeds(baseline_dir, metric_name)
    impv = stack_across_seeds(improved_dir, metric_name)
    if base.shape != impv.shape:
        raise ValueError(f"Shape mismatch for {metric_name}: {base.shape} vs {impv.shape}")
    base_patient = base.mean(axis=0)
    impv_patient = impv.mean(axis=0)
    if higher_is_better:
        d = impv_patient - base_patient
        direction_note = "(improved − baseline)"
    else:
        d = base_patient - impv_patient
        direction_note = "(baseline − improved)"
    nonzero_mask = d != 0
    d_nz = d[nonzero_mask]
    n_patients = d.size
    n_nonzero = d_nz.size
    if n_nonzero == 0:
        raise ValueError(f"All paired differences are zero for {metric_name}; Wilcoxon not defined.")
    w_stat, p_val = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    median_impr = float(np.median(d))
    ci_lo, ci_hi = bootstrap_ci(d)
    r_rb = rank_biserial_from_wilcoxon(w_stat, n_nonzero)
    return {
        "metric": metric_name,
        "direction": direction_note,
        "n_patients": int(n_patients),
        "median_improvement": median_impr,
        "ci95": (float(ci_lo), float(ci_hi)),
        "wilcoxon_W": float(w_stat),
        "wilcoxon_p": float(p_val),
        "rank_biserial_r": float(r_rb),
    }

def main():
    parser = argparse.ArgumentParser(description="Statistical comparison of segmentation metrics.")
    parser.add_argument('--baseline_dir', type=Path, required=True, help="Path to baseline model metrics directory")
    parser.add_argument('--improved_dir', type=Path, required=True, help="Path to improved model metrics directory")
    parser.add_argument('--dest', type=Path, required=True, help="Path to save the results as a text file")
    args = parser.parse_args()

    results = []
    results.append(analyze_pair(args.baseline_dir, args.improved_dir, "3d_dice", higher_is_better=True))
    results.append(analyze_pair(args.baseline_dir, args.improved_dir, "3d_hd95", higher_is_better=False))

    output_lines = []
    output_lines.append("\n=== Paired comparison (averaged across seeds per patient) ===")
    for res in results:
        output_lines.append(f"\nMetric: {res['metric']}  {res['direction']}")
        output_lines.append(f"N patients: {res['n_patients']}")
        output_lines.append(f"Median improvement: {res['median_improvement']:.6f}")
        lo, hi = res["ci95"]
        output_lines.append(f"95% bootstrap CI (median): [{lo:.6f}, {hi:.6f}]")
        output_lines.append(f"Wilcoxon W: {res['wilcoxon_W']:.2f}, p={res['wilcoxon_p']:.3g}")
        output_lines.append(f"Rank-biserial r: {res['rank_biserial_r']:.3f}")

    dest_path = args.dest
    if dest_path.is_dir():
        dest_path = dest_path / "statistical_test.txt"
    dest_path.write_text('\n'.join(output_lines))
    print(f"Results saved to {dest_path}")

if __name__ == "__main__":
    main()