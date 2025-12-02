# GMC-MPNN for BBBP Datasets

This repository provides code for hyperparameter optimization, feature extraction, training, and evaluation of the GMC-MPNN model, specifically designed for predicting blood-brain barrier permeability (BBBP).

---

## Installation

Install Chemprop following the official guide: 
🔗 https://chemprop.readthedocs.io/en/latest/installation.html

📚 Additional Dependencies

```bash
pip install pandas numpy scipy scikit-learn biopandas rdkit
```

Depending on your system, `rdkit` may require Conda:

```bash
conda install -c rdkit rdkit
```

We recommend setting up a clean environment using `conda` or `virtualenv`, and ensuring all dependencies are satisfied as specified in the Chemprop documentation.

---

## 1. Hyperparameter Optimization

To search for optimal hyperparameters using Chemprop's CLI:

```bash
chemprop hpopt \
    --data-path <path_to_dataset.csv> \
    --task-type <classification|regression> \
    --search-parameter-keywords all \
    --split-type SCAFFOLD_BALANCED \
    --hpopt-save-dir <path_to_output_dir> \
    --raytune-num-gpus 1
```

---

## 2. GGL Feature Extraction

To compute GGL-based ligand features:

```bash
python <script_path> -k <kernel_index> -c <cutoff> -f <csv_file> -dd <data_folder> -fd <feature_folder>
```

**Example:**

```bash
python get_ggl_ligand_features.py -k 1551 -c 12.0 -f dataset.csv -dd ./mol2_files -fd ./features
```

### On SLURM:

Use the provided SLURM job script:

```bash
sbatch extract_ggl_features.sh
```

---

## 3. Model Training

To train models with multiple seeds (0-4) and automatically average test results:

```bash
python train.py \
    --training_script <training_script> \
    --input_path <path_to_dataset.csv> \
    --features_folder <path_to_features> \
    --results_path <path_to_results> \
    --seeds 0 1 2 3 4 \
    --target_columns <target_column_name>
```

**Example for B3DB_cls:**

```bash
python train.py \
    --training_script train_b3db_cls.py \
    --input_path /path/to/B3DB_cls.csv \
    --features_folder /path/to/features/B3DB_cls \
    --results_path /path/to/results/B3DB_cls/multi_seed \
    --seeds 0 1 2 3 4 \
    --target_columns labels \
    --batch_size 32 \
    --max_epochs 100 \
    --split_type SCAFFOLD_BALANCED
```

The script will train models for each seed and automatically calculate averaged test results across seeds.

### Parallel Training with SLURM

To submit parallel training jobs:

```bash
sbatch train.sh
```

Update `train.sh` with your dataset-specific paths and configuration before submitting.

---

## 4. Reproducing Results

To reproduce results by training only the best kernel for each seed:

```bash
python test.py \
    --dataset <dataset_name> \
    --input_path <path_to_dataset.csv> \
    --features_folder <path_to_features> \
    --results_path <path_to_results>
```

**Example for B3DB_cls:**

```bash
python test.py \
    --dataset B3DB_cls \
    --input_path /path/to/B3DB_cls.csv \
    --features_folder /path/to/features/B3DB_cls \
    --results_path /path/to/results/B3DB_cls/test
```

The script automatically uses the best kernel for each seed

---

## 5. Data Access

We provide the following for reproducibility and testing:

* ✅ All datasets
* ✅ GGL feature files (`.npz`)

📥 **Access via OneDrive**  
🔗 http://bit.ly/4558Ovg

---

## Contact

For questions or support, please contact:
📧 [ducnguyen@utk.edu](mailto:ducnguyen@utk.edu)

