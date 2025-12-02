#!/bin/bash
#SBATCH -J ligand_only_features_multi_job        # Job name
#SBATCH -A ISAAC-UTK0323                         # Project allocation account name
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks-per-node=30                     # Number of tasks (cores) per node
#SBATCH --time=01:00:00                          # Maximum run time
#SBATCH --partition=campus                       # Partition to use
#SBATCH --array=1-54                             # Array range to cover 1600 kernels (1600 / 30 = ~54)
#SBATCH --qos=campus
#SBATCH -o /lustre/isaac24/proj/UTK0323/GMC-BBBP/features/output/ligand_o_features_multi-%A_%a.out  # Output file
#SBATCH -e /lustre/isaac24/proj/UTK0323/GMC-BBBP/features/output/ligand_o_features_multi-%A_%a.err  # Error file

# Print job information
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
pwd; hostname; date

# Path to the Python script and necessary folders
SCRIPT_PATH="/lustre/isaac24/proj/UTK0323/GMC-BBBP/ggl_bbbp/get_ggl_ligand_features.py"
DATA_FOLDER="/lustre/isaac24/proj/UTK0323/GMC-BBBP/data/B3DB_cls"
FEATURE_FOLDER="/lustre/isaac24/proj/UTK0323/GMC-BBBP/features/B3DB_cls_all"
CSV_FILE="/lustre/isaac24/proj/UTK0323/GMC-BBBP/data/B3DB_cls.csv"

# Fixed cutoff value
CUTOFF=12.0

# Calculate start and end indices for the current array task
START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * 30 + 1 ))
END_INDEX=$(( SLURM_ARRAY_TASK_ID * 30 ))

echo "Processing kernels $START_INDEX to $END_INDEX"

# Loop over each kernel index in this subset
for (( KERNEL_INDEX=START_INDEX; KERNEL_INDEX<=END_INDEX && KERNEL_INDEX<=1600; KERNEL_INDEX++ ))
do
    # Run the Python script in parallel for each kernel index in this batch
    echo "Starting task for kernel index $KERNEL_INDEX"
    python3 $SCRIPT_PATH -k $KERNEL_INDEX -c $CUTOFF -f $CSV_FILE -dd $DATA_FOLDER -fd $FEATURE_FOLDER &
done

# Wait for all background jobs to complete
wait

echo "All tasks for array job $SLURM_ARRAY_TASK_ID are done!"
date
