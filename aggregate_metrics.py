#!/usr/bin/env python3
# filepath: /home/scur1050/ai4mi_final_project/ai4mi_project/aggregate_metrics.py

"""
Aggregate metrics across multiple seeds and calculate mean ± standard deviation.
This script processes metrics from multiple seed experiments and creates summary tables.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

def load_metrics_from_seed(metrics_folder: Path, metric_name: str) -> Optional[np.ndarray]:
    """Load metrics for a specific seed"""
    metric_file = metrics_folder / f"{metric_name}.npy"
    
    if metric_file.exists():
        return np.load(metric_file)
    else:
        print(f"Warning: {metric_file} not found")
        return None

def aggregate_seed_metrics(base_path: Path, experiment_name: str, seeds: List[int], 
                          metrics: List[str] = ['3d_dice', '3d_hd95']) -> Dict:
    """Aggregate metrics across seeds for a specific experiment"""
    
    aggregated_results = {}
    
    for metric_name in metrics:
        seed_data = []
        valid_seeds = []
        
        for seed in seeds:
            metrics_path = base_path / f"{experiment_name}" / f"metrics_seed{seed}"
            
            if not metrics_path.exists():
                print(f"Warning: Metrics folder not found: {metrics_path}")
                continue
                
            metric_data = load_metrics_from_seed(metrics_path, metric_name)
            
            if metric_data is not None:
                seed_data.append(metric_data)
                valid_seeds.append(seed)
        
        if seed_data:
            # Stack data from all seeds: shape (n_seeds, n_patients, n_classes)
            stacked_data = np.stack(seed_data, axis=0)
            
            # Calculate statistics across seeds (axis=0)
            mean_across_seeds = np.mean(stacked_data, axis=0)  # (n_patients, n_classes)
            std_across_seeds = np.std(stacked_data, axis=0)    # (n_patients, n_classes)
            
            # Calculate overall statistics (across both patients and seeds)
            overall_mean = np.mean(stacked_data)
            overall_std = np.std(stacked_data)
            
            # Calculate per-class statistics (across patients and seeds)
            per_class_mean = np.mean(stacked_data, axis=(0, 1))  # (n_classes,)
            per_class_std = np.std(stacked_data, axis=(0, 1))    # (n_classes,)
            
            # Calculate foreground-only statistics (excluding background class 0)
            fg_data = stacked_data[:, :, 1:]  # Remove background class
            fg_overall_mean = np.mean(fg_data)
            fg_overall_std = np.std(fg_data)
            
            aggregated_results[metric_name] = {
                'raw_data': stacked_data,
                'valid_seeds': valid_seeds,
                'mean_per_patient': mean_across_seeds,
                'std_per_patient': std_across_seeds,
                'overall_mean': overall_mean,
                'overall_std': overall_std,
                'per_class_mean': per_class_mean,
                'per_class_std': per_class_std,
                'fg_overall_mean': fg_overall_mean,
                'fg_overall_std': fg_overall_std,
                'n_seeds': len(valid_seeds),
                'n_patients': stacked_data.shape[1],
                'n_classes': stacked_data.shape[2]
            }
        else:
            print(f"No valid data found for metric {metric_name} in experiment {experiment_name}")
    
    return aggregated_results

def print_and_save_summary(results_dict: Dict, output_file: Path, class_names: Optional[List[str]] = None):
    """Print and save summary statistics to console and text file"""
    
    if class_names is None:
        class_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
    
    # Prepare output content
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("CROSS-SEED METRICS SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    for experiment_name, metrics_data in results_dict.items():
        output_lines.append(f"EXPERIMENT: {experiment_name.upper()}")
        output_lines.append("-" * 50)
        
        for metric_name, metric_stats in metrics_data.items():
            if not metric_stats:
                continue
                
            output_lines.append(f"\n{metric_name.upper()}:")
            output_lines.append(f"  Seeds used: {metric_stats['valid_seeds']} ({metric_stats['n_seeds']} total)")
            output_lines.append(f"  Patients: {metric_stats['n_patients']}")
            output_lines.append(f"  Classes: {metric_stats['n_classes']}")
            
            # Overall statistics (all classes including background)
            output_lines.append(f"  Overall (all classes): {metric_stats['overall_mean']:.4f} ± {metric_stats['overall_std']:.4f}")
            
            # Foreground-only statistics (excluding background)
            output_lines.append(f"  Foreground only: {metric_stats['fg_overall_mean']:.4f} ± {metric_stats['fg_overall_std']:.4f}")
            
            # Per-class statistics
            output_lines.append("  Per-class statistics:")
            for i, (mean_val, std_val) in enumerate(zip(metric_stats['per_class_mean'], metric_stats['per_class_std'])):
                class_name = class_names[i] if i < len(class_names) else f'Class_{i}'
                output_lines.append(f"    {class_name:<12}: {mean_val:.4f} ± {std_val:.4f}")
        
        output_lines.append("")
        output_lines.append("=" * 50)
        output_lines.append("")
    
    # Add notes
    output_lines.append("NOTES:")
    output_lines.append("- Values shown as mean ± standard deviation across seeds")
    output_lines.append("- For Dice: Higher values indicate better performance")
    output_lines.append("- For HD95: Lower values indicate better performance")
    output_lines.append("- Foreground metrics exclude the background class")
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✓ Summary saved to: {output_file}")

def save_json_summary(results_dict: Dict, output_file: Path):
    """Save summary statistics as JSON for programmatic access"""
    
    json_data = {}
    
    for experiment_name, metrics_data in results_dict.items():
        json_data[experiment_name] = {}
        
        for metric_name, metric_stats in metrics_data.items():
            if not metric_stats:
                continue
                
            json_data[experiment_name][metric_name] = {
                'overall_mean': float(metric_stats['overall_mean']),
                'overall_std': float(metric_stats['overall_std']),
                'foreground_mean': float(metric_stats['fg_overall_mean']),
                'foreground_std': float(metric_stats['fg_overall_std']),
                'per_class_mean': [float(x) for x in metric_stats['per_class_mean']],
                'per_class_std': [float(x) for x in metric_stats['per_class_std']],
                'n_seeds': metric_stats['n_seeds'],
                'n_patients': metric_stats['n_patients'],
                'n_classes': metric_stats['n_classes'],
                'valid_seeds': metric_stats['valid_seeds']
            }
    
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ JSON summary saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Aggregate metrics across multiple seeds')
    parser.add_argument('--base_path', type=Path, required=True,
                        help='Base path containing experiment results (e.g., improvements/results/)')
    parser.add_argument('--experiments', nargs='+', required=True,
                        help='List of experiment names (e.g., baseline preprocessing)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='List of seeds used in experiments')
    parser.add_argument('--metrics', nargs='+', default=['3d_dice', '3d_hd95'],
                        choices=['3d_dice', '3d_hd95', '3d_hd', '3d_jaccard', '3d_assd'],
                        help='Metrics to aggregate')
    parser.add_argument('--output_dir', type=Path, default=Path('improvements/results/aggregated'),
                        help='Directory to save aggregated results')
    parser.add_argument('--class_names', nargs='+', 
                        default=['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta'],
                        help='Names of the classes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("METRICS AGGREGATION ACROSS SEEDS")
    print("=" * 60)
    print(f"Base path: {args.base_path}")
    print(f"Experiments: {args.experiments}")
    print(f"Seeds: {args.seeds}")
    print(f"Metrics: {args.metrics}")
    print()
    
    # Check if base path exists
    if not args.base_path.exists():
        print(f"Error: Base path {args.base_path} does not exist")
        return
    
    # Aggregate results for all experiments
    all_results = {}
    
    for experiment in args.experiments:
        print(f"Processing experiment: {experiment}")
        results = aggregate_seed_metrics(args.base_path, experiment, args.seeds, args.metrics)
        
        if results:
            all_results[experiment] = results
            print(f"✓ Successfully processed {experiment}")
            
            # Print quick summary
            for metric_name, metric_stats in results.items():
                if metric_stats:
                    print(f"  {metric_name}: {metric_stats['fg_overall_mean']:.3f} ± {metric_stats['fg_overall_std']:.3f} "
                          f"(foreground, {metric_stats['n_seeds']} seeds)")
        else:
            print(f"✗ No data found for {experiment}")
        print()
    
    if not all_results:
        print("No valid results found. Exiting.")
        return
    
    # Print and save summary
    output_txt = args.output_dir / "metrics_summary.txt"
    output_json = args.output_dir / "metrics_summary.json"
    
    print_and_save_summary(all_results, output_txt, args.class_names)
    save_json_summary(all_results, output_json)
    
    # Print recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Find best experiment for each metric (using foreground metrics)
    for metric in args.metrics:
        best_exp = None
        best_value = None
        
        for exp_name, exp_data in all_results.items():
            if metric in exp_data and exp_data[metric]:
                value = exp_data[metric]['fg_overall_mean']
                
                if best_exp is None:
                    best_exp = exp_name
                    best_value = value
                else:
                    if metric == '3d_dice' and value > best_value:  # Higher is better
                        best_exp = exp_name
                        best_value = value
                    elif 'hd' in metric and value < best_value:  # Lower is better
                        best_exp = exp_name
                        best_value = value
        
        if best_exp:
            best_std = all_results[best_exp][metric]['fg_overall_std']
            print(f"🏆 Best {metric} (foreground): {best_exp} ({best_value:.3f} ± {best_std:.3f})")

if __name__ == "__main__":
    main()