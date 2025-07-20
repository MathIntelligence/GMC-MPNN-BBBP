# GMC-MPNN for BBBP Dataset

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
    --data-path /path/to/dataset.csv \
    --task-type classification \
    --search-parameter-keywords all \
    --split-type SCAFFOLD_BALANCED \
    --hpopt-save-dir /path/to/output_dir/ \
    --raytune-num-gpus 1
````

### Key Arguments:

* `--data-path`: Path to the input CSV dataset
* `--task-type`: Use `classification` or `regression`
* `--search-parameter-keywords`: Choose from `basic`, `learning_rate`, or `all`
* `--split-type`: Data split strategy (e.g., `SCAFFOLD_BALANCED`)
* `--hpopt-save-dir`: Output directory for tuning results
* `--raytune-num-gpus`: GPUs allocated per trial (default = 1)

---

## 2. GGL Feature Extraction

To compute GGL-based ligand features:

```bash
python ${SCRIPT_PATH} -k ${KERNEL_INDEX} -c ${CUTOFF} -f ${CSV_FILE} -dd ${DATA_FOLDER} -fd ${FEATURE_FOLDER}
```

**Arguments:**

* `SCRIPT_PATH`: Path to `get_ggl_ligand_features.py`
* `KERNEL_INDEX`: Kernel index to use (e.g., 1551, 1556)
* `CUTOFF`: Distance cutoff (recommended range: 5–20 Å)
* `CSV_FILE`: Dataset with molecule identifiers
* `DATA_FOLDER`: Folder containing `.mol2` files
* `FEATURE_FOLDER`: Output folder for storing `.npz` feature files

### On SLURM:

Use the provided SLURM job script:

```bash
sbatch extract_ggl_features.sh
```

---

## 3. Model Training

To train the model with extracted features:

```bash
python bbbp_${DATASET_NAME}_training.py \
       --input_path ${DATA_PATH} \
       --feature_file ${ATOM_FEATURE_FILE} \
       --results_path ${OUTPUT_DIR}
```

* `DATA_PATH`: Path to training dataset
* `ATOM_FEATURE_FILE`: Path to GGL `.npz` file
* `OUTPUT_DIR`: Output folder (results saved under `test_${FILE_INDEX}`)

Submit as a SLURM job:

```bash
sbatch bbbp_train.sh
```

To analyze the results of all kernel training, run:

```bash
python process_result.py
```

---

## 4. Model Testing

To evaluate a trained model:

```bash
python bbbp_testing.py \
       --input_path ${DATA_PATH} \
       --feature_file ${ATOM_FEATURE_FILE} \
       --model_path ${MODEL_PATH} \
       --batch_size ${BATCH_SIZE} \
       --output_dir ${OUTPUT_DIR}
```

---

## 5. Pretrained Models & Data Access

We provide the following for reproducibility and testing:

* ✅ All BBBP classification datasets
* ✅ Pretrained model checkpoints
* ✅ Sample GGL feature files (`.npz`)

📥 **Access via OneDrive**  
🔗 http://bit.ly/46S3uxz

---

## Contact

For questions or support, please contact:
📧 [ducnguyen@utk.edu](mailto:ducnguyen@utk.edu)

