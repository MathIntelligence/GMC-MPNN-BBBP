#!/usr/bin/env python3
"""
Test script to reproduce results by training only the best kernel for each seed.
This trains a single kernel per seed instead of all kernels.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path


# Best kernels for each dataset and seed
BEST_KERNELS = {
    'BBBP': {
        0: 266,
        1: 152,
        2: 953,
        3: 404,
        4: 451,
    },
    'B3DB_cls': {
        0: 64,
        1: 589,
        2: 538,
        3: 1133,
        4: 417,
    },
    'B3DB_reg': {
        0: 34,
        1: 227,
        2: 66,
        3: 894,
        4: 77,
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    'BBBP': {
        'training_script': 'train_bbbp.py',
        'input_path': '<path_to_datasets>/BBBP.csv',
        'features_folder': '<path_to_features>/BBBP',
        'results_path': '<path_to_results>/BBBP/test',
        'target_columns': ['labels'],
        'split_type': 'SCAFFOLD_BALANCED',
    },
    'B3DB_cls': {
        'training_script': 'train_b3db_cls.py',
        'input_path': '<path_to_datasets>/B3DB_cls.csv',
        'features_folder': '<path_to_features>/B3DB_cls',
        'results_path': '<path_to_results>/B3DB_cls/test',
        'target_columns': ['labels'],
        'split_type': 'SCAFFOLD_BALANCED',
    },
    'B3DB_reg': {
        'training_script': 'train_b3db_regression.py',
        'input_path': '<path_to_datasets>/bbbp_regression.csv',
        'features_folder': '<path_to_features>/B3DB_reg',
        'results_path': '<path_to_results>/B3DB_reg/test',
        'target_columns': ['logBB'],
        'split_type': 'SCAFFOLD_BALANCED',
    },
}


def find_kernel_file(features_folder, kernel_num):
    """Find the kernel file for a given kernel number."""
    features_path = Path(features_folder)
    
    # Try different possible filename patterns
    patterns = [
        f"kernel_{kernel_num}.npz",
        f"kernel_{kernel_num:04d}.npz",
        f"B3DB_cls_ker{kernel_num}_cutoff12.0.npz",
        f"B3DB_reg_ker{kernel_num}_cutoff12.0.npz",
        f"BBBP_ker{kernel_num}_cutoff12.0.npz",
    ]
    
    for pattern in patterns:
        kernel_file = features_path / pattern
        if kernel_file.exists():
            return kernel_file
    
    # If exact match not found, search for files containing the kernel number
    for feature_file in features_path.glob("*.npz"):
        stem = feature_file.stem
        # Extract kernel number from filename
        if stem.startswith("kernel_"):
            file_kernel_num = int(stem.split("_")[1])
            if file_kernel_num == kernel_num:
                return feature_file
        else:
            # Try to extract from other patterns
            match = re.search(r'ker(\d+)', stem, re.IGNORECASE)
            if match:
                file_kernel_num = int(match.group(1))
                if file_kernel_num == kernel_num:
                    return feature_file
    
    raise FileNotFoundError(
        f"Kernel {kernel_num} not found in {features_folder}. "
        f"Tried patterns: {patterns}"
    )


def create_temp_features_folder(features_folder, kernel_num):
    """Create a temporary folder with only the specified kernel file."""
    kernel_file = find_kernel_file(features_folder, kernel_num)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix=f"kernel_{kernel_num}_")
    
    # Copy or symlink the kernel file (use copy on Windows, symlink on Unix)
    temp_kernel_file = Path(temp_dir) / kernel_file.name
    if sys.platform == 'win32':
        # Windows: copy the file (symlinks require admin privileges)
        shutil.copy2(kernel_file, temp_kernel_file)
    else:
        # Unix/Linux: create symlink (more efficient)
        temp_kernel_file.symlink_to(kernel_file.resolve())
    
    return temp_dir, kernel_file.name


def run_training(dataset, seed, kernel_num, config, temp_features_folder=None):
    """Run training for a specific dataset, seed, and kernel."""
    if temp_features_folder is None:
        # Create temporary folder with only the specified kernel
        temp_features_folder, kernel_filename = create_temp_features_folder(
            config['features_folder'], kernel_num
        )
        cleanup_temp = True
    else:
        cleanup_temp = False
    
    try:
        cmd = [
            sys.executable,
            config['training_script'],
            '--input_path', config['input_path'],
            '--features_folder', temp_features_folder,
            '--results_path', config['results_path'],
            '--seed', str(seed),
        ]
        
        # Add optional arguments
        if 'target_columns' in config:
            if isinstance(config['target_columns'], list):
                cmd.extend(['--target_columns'] + config['target_columns'])
            else:
                cmd.extend(['--target_columns', config['target_columns']])
        
        if 'batch_size' in config:
            cmd.extend(['--batch_size', str(config['batch_size'])])
        if 'max_epochs' in config:
            cmd.extend(['--max_epochs', str(config['max_epochs'])])
        if 'split_type' in config:
            cmd.extend(['--split_type', config['split_type']])
        if 'num_workers' in config:
            cmd.extend(['--num_workers', str(config['num_workers'])])
        
        print(f"\n{'='*60}")
        print(f"Training {dataset} - Seed {seed} - Kernel {kernel_num}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            print(f"Error: Training failed for {dataset}, seed {seed}, kernel {kernel_num}")
            return False
        
        print(f"Successfully completed training for {dataset}, seed {seed}, kernel {kernel_num}")
        return True
        
    finally:
        # Clean up temporary folder
        if cleanup_temp and os.path.exists(temp_features_folder):
            shutil.rmtree(temp_features_folder)
            print(f"Cleaned up temporary folder: {temp_features_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Test script to reproduce results by training best kernels"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['BBBP', 'B3DB_cls', 'B3DB_reg'],
        help='Dataset name'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Specific seed to run (0-4). If not specified, runs all seeds.'
    )
    parser.add_argument(
        '--kernel',
        type=int,
        default=None,
        help='Specific kernel number. If not specified, uses best kernel for the seed.'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        default=None,
        help='Override input CSV path'
    )
    parser.add_argument(
        '--features_folder',
        type=str,
        default=None,
        help='Override features folder path'
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default=None,
        help='Override results path'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (uses default if not specified)'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=None,
        help='Max epochs (uses default if not specified)'
    )
    
    args = parser.parse_args()
    
    # Get dataset configuration
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    config = DATASET_CONFIGS[args.dataset].copy()
    
    # Override paths if provided
    if args.input_path:
        config['input_path'] = args.input_path
    if args.features_folder:
        config['features_folder'] = args.features_folder
    if args.results_path:
        config['results_path'] = args.results_path
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs
    
    # Determine which seeds to run
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(BEST_KERNELS[args.dataset].keys())
    
    # Run training for each seed
    results = []
    for seed in seeds:
        # Get kernel number
        if args.kernel is not None:
            kernel_num = args.kernel
        else:
            kernel_num = BEST_KERNELS[args.dataset][seed]
        
        print(f"\n{'='*70}")
        print(f"Dataset: {args.dataset} | Seed: {seed} | Kernel: {kernel_num}")
        print(f"{'='*70}")
        
        success = run_training(args.dataset, seed, kernel_num, config)
        results.append({
            'dataset': args.dataset,
            'seed': seed,
            'kernel': kernel_num,
            'success': success
        })
    
    # Print summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['dataset']} - Seed {result['seed']} - Kernel {result['kernel']}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nCompleted: {successful}/{len(results)}")
    
    if successful == len(results):
        print("All training jobs completed successfully!")
    else:
        print("Some training jobs failed. Check logs for details.")


if __name__ == "__main__":
    main()

