#!/usr/bin/env python3
# filepath: /home/scur1050/ai4mi_final_project/ai4mi_project/aggregate_metrics.py

"""
Aggregate metrics across multiple seeds and calculate mean ± standard deviation.
This script processes metrics from multiple seed experiments and creates summary tables.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from collections import defaultdict

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
            metrics_path = base_path / f"{experiment_name}_seed{seed}" / "metrics"
            
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
            
            aggregated_results[metric_name] = {
                'raw_data': stacked_data,
                'valid_seeds': valid_seeds,
                'mean_per_patient': mean_across_seeds,
                'std_per_patient': std_across_seeds,
                'overall_mean': overall_mean,
                'overall_std': overall_std,
                'per_class_mean': per_class_mean,
                'per_class_std': per_class_std,
                'n_seeds': len(valid_seeds),
                'n_patients': stacked_data.shape[1],
                'n_classes': stacked_data.shape[2]
            }
        else:
            print(f"No valid data found for metric {metric_name} in experiment {experiment_name}")
    
    return aggregated_results

def create_summary_table(results_dict: Dict, class_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Create a summary table with mean ± std for each experiment and metric"""
    
    if class_names is None:
        # Default SEGTHOR class names
        class_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
    
    summary_data = []
    
    for experiment_name, metrics_data in results_dict.items():
        row_data = {'Experiment': experiment_name}
        
        for metric_name, metric_stats in metrics_data.items():
            if metric_stats:  # Check if data exists
                # Overall statistics
                row_data[f'{metric_name}_overall'] = f"{metric_stats['overall_mean']:.3f} ± {metric_stats['overall_std']:.3f}"
                row_data[f'{metric_name}_n_seeds'] = metric_stats['n_seeds']
                
                # Per-class statistics
                for i, class_name in enumerate(class_names[:metric_stats['n_classes']]):
                    if i < len(metric_stats['per_class_mean']):
                        mean_val = metric_stats['per_class_mean'][i]
                        std_val = metric_stats['per_class_std'][i]
                        row_data[f'{metric_name}_{class_name}'] = f"{mean_val:.3f} ± {std_val:.3f}"
        
        summary_data.append(row_data)
    
    return pd.DataFrame(summary_data)

def save_detailed_results(results_dict: Dict, output_dir: Path, class_names: Optional[List[str]] = None):
    """Save detailed results including per-patient statistics"""
    
    if class_names is None:
        class_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for experiment_name, metrics_data in results_dict.items():
        exp_output_dir = output_dir / experiment_name
        exp_output_dir.mkdir(exist_ok=True)
        
        for metric_name, metric_stats in metrics_data.items():
            if not metric_stats:
                continue
                
            # Save raw aggregated data
            np.save(exp_output_dir / f"{metric_name}_aggregated.npy", metric_stats['raw_data'])
            
            # Save per-patient means and stds
            np.save(exp_output_dir / f"{metric_name}_patient_means.npy", metric_stats['mean_per_patient'])
            np.save(exp_output_dir / f"{metric_name}_patient_stds.npy", metric_stats['std_per_patient'])
            
            # Create per-patient CSV
            n_patients, n_classes = metric_stats['mean_per_patient'].shape
            patient_data = []
            
            for patient_idx in range(n_patients):
                row = {'Patient': f'Patient_{patient_idx:02d}'}
                for class_idx in range(n_classes):
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f'Class_{class_idx}'
                    mean_val = metric_stats['mean_per_patient'][patient_idx, class_idx]
                    std_val = metric_stats['std_per_patient'][patient_idx, class_idx]
                    row[f'{class_name}_mean'] = mean_val
                    row[f'{class_name}_std'] = std_val
                patient_data.append(row)
            
            patient_df = pd.DataFrame(patient_data)
            patient_df.to_csv(exp_output_dir / f"{metric_name}_per_patient.csv", index=False)
            
            # Save summary statistics as JSON
            summary_stats = {
                'overall_mean': float(metric_stats['overall_mean']),
                'overall_std': float(metric_stats['overall_std']),
                'per_class_mean': metric_stats['per_class_mean'].tolist(),
                'per_class_std': metric_stats['per_class_std'].tolist(),
                'n_seeds': metric_stats['n_seeds'],
                'n_patients': metric_stats['n_patients'],
                'n_classes': metric_stats['n_classes'],
                'valid_seeds': metric_stats['valid_seeds']
            }
            
            with open(exp_output_dir / f"{metric_name}_summary.json", 'w') as f:
                json.dump(summary_stats, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Aggregate metrics across multiple seeds')
    parser.add_argument('--base_path', type=Path, required=True,
                        help='Base path containing experiment results (e.g., results/)')
    parser.add_argument('--experiments', nargs='+', required=True,
                        help='List of experiment names (e.g., baseline preprocessing data_augmentation)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='List of seeds used in experiments')
    parser.add_argument('--metrics', nargs='+', default=['3d_dice', '3d_hd95'],
                        choices=['3d_dice', '3d_hd95', '3d_hd', '3d_jaccard', '3d_assd'],
                        help='Metrics to aggregate')
    parser.add_argument('--output_dir', type=Path, default=Path('results/aggregated_metrics'),
                        help='Directory to save aggregated results')
    parser.add_argument('--class_names', nargs='+', 
                        default=['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta'],
                        help='Names of the classes')
    parser.add_argument('--summary_only', action='store_true',
                        help='Only create summary table, skip detailed per-patient files')
    
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
                    print(f"  {metric_name}: {metric_stats['overall_mean']:.3f} ± {metric_stats['overall_std']:.3f} "
                          f"({metric_stats['n_seeds']} seeds)")
        else:
            print(f"✗ No data found for {experiment}")
        print()
    
    if not all_results:
        print("No valid results found. Exiting.")
        return
    
    # Create and save summary table
    summary_df = create_summary_table(all_results, args.class_names)
    
    # Save summary table
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = args.output_dir / "summary_table.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Also save as formatted text table
    summary_txt_path = args.output_dir / "summary_table.txt"
    with open(summary_txt_path, 'w') as f:
        f.write("CROSS-SEED METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nNotes:\n")
        f.write("- Values shown as mean ± standard deviation across seeds\n")
        f.write("- Lower HD95 values indicate better performance\n")
        f.write("- Higher Dice values indicate better performance\n")
    
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    print(f"✓ Summary table saved to: {summary_csv_path}")
    print(f"✓ Formatted summary saved to: {summary_txt_path}")
    
    # Save detailed results unless summary_only is specified
    if not args.summary_only:
        save_detailed_results(all_results, args.output_dir, args.class_names)
        print(f"✓ Detailed results saved to: {args.output_dir}")
    
    # Print recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Find best experiment for each metric
    for metric in args.metrics:
        metric_col = f"{metric}_overall"
        if metric_col in summary_df.columns:
            # Extract numeric values (mean) from "mean ± std" format
            summary_df[f'{metric}_numeric'] = summary_df[metric_col].str.extract(r'(\d+\.\d+)').astype(float)
            
            if metric == '3d_dice':
                best_idx = summary_df[f'{metric}_numeric'].idxmax()
                print(f"🏆 Best {metric}: {summary_df.iloc[best_idx]['Experiment']} "
                      f"({summary_df.iloc[best_idx][metric_col]})")
            elif 'hd' in metric:
                best_idx = summary_df[f'{metric}_numeric'].idxmin()  # Lower is better for HD metrics
                print(f"🏆 Best {metric}: {summary_df.iloc[best_idx]['Experiment']} "
                      f"({summary_df.iloc[best_idx][metric_col]})")

if __name__ == "__main__":
    main()