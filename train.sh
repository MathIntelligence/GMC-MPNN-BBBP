#!/bin/bash
#SBATCH -J multi_seed_training          # Job name
#SBATCH -A <project_allocation>                # Project allocation account name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=20             # Number of tasks (cores) per node
#SBATCH --time=24:00:00                  # Maximum run time (24 hours)
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --partition=<partition>             # Partition to use
#SBATCH --qos=<qos>                   # Quality of Service
#SBATCH --array=0-4                      # Array job: 5 tasks (seeds 0-4) - runs in parallel
#SBATCH -o <path_to_results>/multi_seed_%A_%a.out
#SBATCH -e <path_to_results>/multi_seed_%A_%a.err

# Print job information
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
pwd; hostname; date

# Configuration - MODIFY THESE PATHS AS NEEDED
TRAINING_SCRIPT="train_b3db_cls.py"  # Change to: train_bbbp.py, train_b3db_regression.py, etc.
INPUT_PATH="<path_to_datasets>/BBBP.csv"
FEATURES_FOLDER="<path_to_features>/BBBP"
RESULTS_PATH="<path_to_results>/BBBP/multi_seed"
TARGET_COLUMNS="labels"  # Change to: logBB for regression
BATCH_SIZE=32
MAX_EPOCHS=100
SPLIT_TYPE="SCAFFOLD_BALANCED"

# Each array task runs one seed (0-4)
SEED=$SLURM_ARRAY_TASK_ID

echo "Running training for seed ${SEED}"

# Run training for this specific seed
python ${TRAINING_SCRIPT} \
    --input_path ${INPUT_PATH} \
    --features_folder ${FEATURES_FOLDER} \
    --results_path ${RESULTS_PATH} \
    --seed ${SEED} \
    --target_columns ${TARGET_COLUMNS} \
    --batch_size ${BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --split_type ${SPLIT_TYPE}

echo "Array task ${SLURM_ARRAY_TASK_ID} (seed ${SEED}) completed!"

# Note: After all 5 array tasks complete, run averaging separately:
# python train.py --training_script ${TRAINING_SCRIPT} --input_path ${INPUT_PATH} \
#     --features_folder ${FEATURES_FOLDER} --results_path ${RESULTS_PATH} \
#     --seeds 0 1 2 3 4 --target_columns ${TARGET_COLUMNS} --skip_training

