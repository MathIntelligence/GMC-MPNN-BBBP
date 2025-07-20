#!/bin/bash
#SBATCH -J ggl_feature_extraction              # Job name
#SBATCH -A <your_account_name>                # Replace with your allocation/account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --time=01:00:00
#SBATCH --array=1-54                          # Adjust range based on total kernel count and batch size
#SBATCH -o ./logs/ggl_extract_%A_%a.out
#SBATCH -e ./logs/ggl_extract_%A_%a.err

echo "Job ID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID"
date

# === User-defined paths ===
SCRIPT_PATH="path/to/get_ggl_ligand_features.py"
DATA_FOLDER="path/to/mol2_files"
FEATURE_FOLDER="path/to/save_features"
CSV_FILE="path/to/dataset.csv"
CUTOFF=12.0

# === Kernel indexing per job ===
KERNELS_PER_TASK=30
START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * KERNELS_PER_TASK + 1 ))
END_INDEX=$(( SLURM_ARRAY_TASK_ID * KERNELS_PER_TASK ))

echo "Processing kernels from $START_INDEX to $END_INDEX"

for (( KERNEL_INDEX=START_INDEX; KERNEL_INDEX<=END_INDEX; KERNEL_INDEX++ ))
do
    echo "Running kernel $KERNEL_INDEX"
    python3 "$SCRIPT_PATH" -k "$KERNEL_INDEX" -c "$CUTOFF" -f "$CSV_FILE" -dd "$DATA_FOLDER" -fd "$FEATURE_FOLDER" &
done

wait
echo "Finished array task $SLURM_ARRAY_TASK_ID"
date
