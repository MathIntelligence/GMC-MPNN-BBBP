#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
import pandas as pd
from pathlib import Path


def run_training_script(script_path, args_dict, seed):
    """Run training script for a specific seed."""
    cmd = [
        sys.executable,
        script_path,
        '--input_path', args_dict['input_path'],
        '--features_folder', args_dict['features_folder'],
        '--results_path', args_dict['results_path'],
        '--seed', str(seed),
    ]
    
    # Add optional arguments if provided
    if 'smiles_column' in args_dict:
        cmd.extend(['--smiles_column', args_dict['smiles_column']])
    if 'target_columns' in args_dict:
        if isinstance(args_dict['target_columns'], list):
            cmd.extend(['--target_columns'] + args_dict['target_columns'])
        else:
            cmd.extend(['--target_columns', args_dict['target_columns']])
    if 'batch_size' in args_dict:
        cmd.extend(['--batch_size', str(args_dict['batch_size'])])
    if 'max_epochs' in args_dict:
        cmd.extend(['--max_epochs', str(args_dict['max_epochs'])])
    if 'split_type' in args_dict:
        cmd.extend(['--split_type', args_dict['split_type']])
    if 'num_workers' in args_dict:
        cmd.extend(['--num_workers', str(args_dict['num_workers'])])
    
    print(f"\n{'='*60}")
    print(f"Running seed {seed}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Error: Training failed for seed {seed}")
        return False
    
    return True


def collect_and_average_results(results_path, seeds, output_csv=None):
    """Collect results from all seeds and calculate averages."""
    results_path = Path(results_path)
    
    all_seed_results = []
    
    # Load results from each seed
    for seed in seeds:
        csv_path = results_path / f"all_results_seed_{seed}.csv"
        if not csv_path.exists():
            print(f"Warning: Results file not found for seed {seed}: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        df['seed'] = seed
        all_seed_results.append(df)
    
    if not all_seed_results:
        print("Error: No results found for any seed!")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_seed_results, ignore_index=True)
    
    # Group by kernel and calculate mean test metrics across seeds
    group_cols = ['kernel', 'feature_file'] if 'kernel' in combined_df.columns else []
    numeric_cols = combined_df.select_dtypes(include=[float, int]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['seed', 'kernel']]
    
    # Calculate per-kernel averages across seeds
    if group_cols:
        test_cols = [col for col in numeric_cols if col.startswith('test_')]
        avg_results = combined_df.groupby(group_cols)[test_cols].mean().reset_index()
        
        overall_avg = combined_df[test_cols].mean()
    else:
        test_cols = [col for col in numeric_cols if col.startswith('test_')]
        avg_results = combined_df[test_cols].mean().to_frame(name='mean').T
        overall_avg = avg_results.iloc[0]
    
    # Save per-kernel averages
    if output_csv is None:
        output_csv = results_path / "averaged_test_results.csv"
    else:
        output_csv = Path(output_csv)
    
    avg_results.to_csv(output_csv, index=False)
    print(f"\nPer-kernel averaged test results saved to: {output_csv}")
    
    # Save overall average
    overall_csv = results_path / "overall_averaged_test_results.csv"
    overall_avg.to_frame(name="mean").to_csv(overall_csv)
    print(f"Overall averaged test results saved to: {overall_csv}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Overall Average Test Metrics (across all kernels and seeds):")
    print(f"{'='*60}")
    for metric_name, metric_value in overall_avg.items():
        print(f"  {metric_name}: {metric_value:.6f}")
    
    combined_csv = results_path / "all_seeds_combined_results.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nCombined results (all seeds) saved to: {combined_csv}")
    
    return avg_results, overall_avg


def main():
    parser = argparse.ArgumentParser(
        description="Train model with multiple seeds and average test results"
    )
    parser.add_argument(
        '--training_script',
        type=str,
        required=True,
        help='Path to training script (e.g., b3db_cls_training_1seed.py)'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--features_folder',
        type=str,
        required=True,
        help='Path to folder containing feature files'
    )
    parser.add_argument(
        '--results_path',
        type=str,
        required=True,
        help='Path to store results'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3, 4],
        help='Seeds to run (default: 0 1 2 3 4)'
    )
    parser.add_argument(
        '--smiles_column',
        type=str,
        default='smiles',
        help='Column name for SMILES strings'
    )
    parser.add_argument(
        '--target_columns',
        nargs='+',
        default=['labels'],
        help='Target column(s)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (uses script default if not specified)'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=None,
        help='Max epochs (uses script default if not specified)'
    )
    parser.add_argument(
        '--split_type',
        type=str,
        default=None,
        help='Split type (uses script default if not specified)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of workers (uses script default if not specified)'
    )
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip training, only average existing results'
    )
    
    args = parser.parse_args()
    
    # Prepare arguments dictionary
    args_dict = {
        'input_path': args.input_path,
        'features_folder': args.features_folder,
        'results_path': args.results_path,
        'smiles_column': args.smiles_column,
        'target_columns': args.target_columns,
    }
    
    if args.batch_size is not None:
        args_dict['batch_size'] = args.batch_size
    if args.max_epochs is not None:
        args_dict['max_epochs'] = args.max_epochs
    if args.split_type is not None:
        args_dict['split_type'] = args.split_type
    if args.num_workers is not None:
        args_dict['num_workers'] = args.num_workers
    
    # Run training for each seed
    if not args.skip_training:
        print(f"Training with seeds: {args.seeds}")
        t0 = time.time()
        
        for seed in args.seeds:
            success = run_training_script(args.training_script, args_dict, seed)
            if not success:
                print(f"Warning: Seed {seed} failed, but continuing...")
        
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed:.2f} seconds")
        print(f"{'='*60}")
    
    # Collect and average results
    print(f"\n{'='*60}")
    print("Collecting and averaging results...")
    print(f"{'='*60}")
    
    collect_and_average_results(args.results_path, args.seeds)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

