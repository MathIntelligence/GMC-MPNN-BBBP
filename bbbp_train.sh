#!/bin/bash
#SBATCH -J model_training                   # Job name
#SBATCH -A <your_account_name>             # Replace with your project allocation/account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-20                       # Adjust based on total kernels and batch size
#SBATCH -o ./logs/train_%A_%a.out
#SBATCH -e ./logs/train_%A_%a.err

echo "Job ID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID"
date

# === User-defined paths ===
FEATURE_FOLDER="/path/to/feature_folder"
DATA_PATH="/path/to/dataset.csv"
OUTPUT_DIR="/path/to/save_outputs"
TRAINING_SCRIPT="your_training_script.py"   # e.g., b3db_cls_training.py
PREFIX="your_prefix"                        # e.g., B3DB_cls

# === Kernel indexing parameters ===
TOTAL_KERNELS=1600
KERNELS_PER_TASK=80
START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * KERNELS_PER_TASK + 1 ))
END_INDEX=$(( SLURM_ARRAY_TASK_ID * KERNELS_PER_TASK ))

# === Training Loop ===
for (( KERNEL_INDEX=START_INDEX; KERNEL_INDEX<=END_INDEX && KERNEL_INDEX<=TOTAL_KERNELS; KERNEL_INDEX++ ))
do
    ATOM_FEATURE_FILE="${FEATURE_FOLDER}/${PREFIX}_ker${KERNEL_INDEX}_cutoff12.0.npz"
    
    echo "[$(date)] Training with kernel index ${KERNEL_INDEX}..."
    
    python "$TRAINING_SCRIPT" \
        --input_path "$DATA_PATH" \
        --feature_file "$ATOM_FEATURE_FILE" \
        --results_path "$OUTPUT_DIR"
done

echo "Completed SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"
date
