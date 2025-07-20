````markdown
# GMC-MPNN For BBBP Dataset

This repository contains scripts for hyperparameter optimization, feature extraction, model training, and evaluation for D-MPNN with GGL features.

## Installation

Before running any of the scripts in this repository, Chemprop must be installed.

Please follow the official installation guide here:  
🔗 **https://chemprop.readthedocs.io/en/latest/installation.html**

We recommend creating a clean Python environment (e.g., via `conda` or `virtualenv`) and installing all dependencies as specified.

---

## 1. Hyperparameter Optimization

To perform hyperparameter optimization using Chemprop's CLI interface, use the following command format:

```bash
chemprop hpopt \
    --data-path /path/to/dataset.csv \
    --task-type classification \
    --search-parameter-keywords all \
    --split-type SCAFFOLD_BALANCED \
    --hpopt-save-dir /path/to/output_dir/ \
    --raytune-num-gpus 1
````

### Description of Arguments:

* `--data-path`: Path to the input dataset (CSV format).
* `--task-type`: Set to `classification` or `regression` depending on your task.
* `--search-parameter-keywords`: Specify which parameters to optimize. Options include:

  * `basic`: Includes `depth`, `ffn_num_layers`, `dropout`, `message_hidden_dim`, `ffn_hidden_dim`
  * `learning_rate`: Includes `max_lr`, `init_lr`, `final_lr`, `warmup_epochs`
  * `all`: Includes all of the above plus `activation`, `aggregation`, `aggregation_norm`, and `batch_size`
* `--split-type`: The data splitting strategy (e.g., `SCAFFOLD_BALANCED` for scaffold-based split with label balancing).
* `--hpopt-save-dir`: Directory where the optimization results will be saved.
* `--raytune-num-gpus`: Number of GPUs to allocate per trial (default is 1).

---

## 2. Feature Extraction

To extract GGL ligand features, run:

```bash
python ${SCRIPT_PATH} -k ${KERNEL_INDEX} -c ${CUTOFF} -f ${CSV_FILE} -dd ${DATA_FOLDER} -fd ${FEATURE_FOLDER}
```

**Arguments:**

* `SCRIPT_PATH`: Path to `get_ggl_ligand_features.py`
* `KERNEL_INDEX`: Kernel index to use (integer)
* `CUTOFF`: Cutoff distance (integer between 5 and 20)
* `CSV_FILE`: Path to the CSV dataset
* `DATA_FOLDER`: Directory containing `.mol2` files
* `FEATURE_FOLDER`: Output directory for storing generated features

To run on a SLURM cluster:

```bash
sbatch extract_ggl_features.sh
```

---

## 3. Model Training

To train the model using Chemprop:

```bash
python bbbp_{DATASET_NAME}_training.py \
       --input_path ${DATA_PATH} \
       --feature_file ${ATOM_FEATURE_FILE} \
       --results_path ${OUTPUT_DIR}
```

**Arguments:**

* `DATA_PATH`: Path to the training dataset
* `ATOM_FEATURE_FILE`: Path to the atom-level features
* `OUTPUT_DIR`: Directory to store outputs; results will be saved under `test_${FILE_INDEX}`

Submit jobs via SLURM:

```bash
sbatch bbbp_train.sh          
```

To analyze validation results:

```bash
python process_result.py
```

---

## 4. Model Testing

After identifying the best kernel setting, evaluate test performance by running:

```bash
python bbbp_testing.py \
       --input_path ${DATA_PATH} \
       --feature_file ${ATOM_FEATURE_FILE} \
       --model_path ${MODEL_PATH} \
       --batch_size ${BATCH_SIZE} \
       --output_dir ${OUTPUT_DIR}
```

---

## 5. Pretrained Models and Dataset Access

To facilitate reproducibility and evaluation, we provide:

- ✅ All BBBP datasets
- ✅ Pretrained model checkpoints for each setting
- ✅ Sample GGL feature files

📥 **Download them from OneDrive**:  
🔗 [https://liveutk-my.sharepoint.com/...](https://liveutk-my.sharepoint.com/:f:/g/personal/tnguy122_vols_utk_edu/EnRFV9Zt7LhBp9pxIn_x7tIBWj3sGA0P3kRhm8QehSK4Tw?e=wMUqHh)


---

## Contact

For questions, suggestions, or issues, please contact:
📧 [ducnguyen@utk.edu](mailto:ducnguyen@utk.edu)


