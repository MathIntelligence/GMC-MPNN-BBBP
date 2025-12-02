import argparse
import os
import re
import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn, utils
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class BBBPTrainer:
    def __init__(self, args):
        self.input_path = args.input_path
        self.features_folder = Path(args.features_folder)
        if not self.features_folder.is_dir():
            raise ValueError(f"Features folder does not exist: {self.features_folder}")
        self.smiles_column = args.smiles_column
        self.target_columns = args.target_columns
        self.num_workers = args.num_workers
        self.results_path = args.results_path
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.split_type = args.split_type
        self.split_sizes = tuple(args.split_sizes)

    def load_data(self):
        df_input = pd.read_csv(self.input_path)
        smis = df_input[self.smiles_column].values
        ys = df_input[self.target_columns].values
        mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
        return mols, ys
    
    def load_extra_atom_features(self, feature_file):
        extra_atom_featuress = np.load(feature_file)
        return [extra_atom_featuress[f"arr_{i}"] for i in range(len(extra_atom_featuress))]

    def prepare_data(self, mols, ys, extra_atom_featuress, seed):
        all_data = [
            data.MoleculeDatapoint(mol, y, V_f=V_f)
            for mol, y, V_f in zip(mols, ys, extra_atom_featuress)
        ]
        train_indices, val_indices, test_indices = data.make_split_indices(
            mols, self.split_type, self.split_sizes, seed=seed
        )
        return data.split_data_by_indices(all_data, train_indices, val_indices, test_indices)

    def create_datasets(self, train_data, val_data, test_data, extra_atom_featuress):
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra_atom_featuress[0].shape[1])
        train_dset = data.MoleculeDataset(train_data[0], featurizer)
        val_dset = data.MoleculeDataset(val_data[0], featurizer)
        test_dset = data.MoleculeDataset(test_data[0], featurizer)
        
        extra_atom_features_scaler = train_dset.normalize_inputs("V_f")
        val_dset.normalize_inputs("V_f", extra_atom_features_scaler)
        test_dset.normalize_inputs("V_f", extra_atom_features_scaler)
        
        return train_dset, val_dset, test_dset, featurizer

    def create_dataloaders(self, train_dset, val_dset, test_dset):
        train_loader = data.build_dataloader(train_dset, batch_size=self.batch_size, num_workers=self.num_workers)
        val_loader = data.build_dataloader(val_dset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        test_loader = data.build_dataloader(test_dset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return train_loader, val_loader, test_loader

    def create_model(self, featurizer):
        mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, depth=3)  
        agg = nn.MeanAggregation(norm=4.0)  
        ffn = nn.BinaryClassificationFFN(
            n_tasks=len(self.target_columns), 
            n_layers=2,  
            hidden_dim=700, 
            dropout=0, 
            activation="RELU"
        )
        return models.MPNN(mp, agg, ffn, batch_norm=False)

    def train_and_test(self, mpnn, train_loader, val_loader, test_loader, output_dir):
        logger = pl.loggers.CSVLogger(save_dir=output_dir, name="training_logs")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename="best_model_epoch_{epoch:02d}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=10,  # Number of epochs with no improvement to wait
            verbose=True,
            mode="min"
        )

        trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=True,
            enable_progress_bar=True,
            accelerator="auto",  
            devices="auto", 
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=5
        )
        trainer.fit(mpnn, train_loader, val_loader)

        best_model_path = checkpoint_callback.best_model_path
        best_val_loss = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None
        if "epoch=" in best_model_path:
            best_epoch = int(best_model_path.split("epoch=")[-1].split(".")[0])
        else:
            best_epoch = "Unknown"

        # Evaluate best checkpoint on validation set
        if best_model_path:
            val_results = trainer.validate(mpnn, val_loader, ckpt_path=best_model_path)
        else:
            val_results = trainer.validate(mpnn, val_loader)
        val_metrics = val_results[0] if val_results else {}

        # Explicitly load the best checkpoint for testing
        if best_model_path:
            test_results = trainer.test(mpnn, test_loader, ckpt_path=best_model_path)
        else:
            # Fallback if no checkpoint was saved
            test_results = trainer.test(mpnn, test_loader)

        test_metrics = test_results[0] if test_results else {}

        with open(os.path.join(output_dir, "log.txt"), "w") as log_file:
            log_file.write(f"Best model checkpoint saved at: {best_model_path}\n")
            log_file.write(f"Best validation loss: {best_val_loss}\n")
            log_file.write(f"Best epoch: {best_epoch}\n")
            log_file.write(f"Validation results: {val_metrics}\n")
            log_file.write(f"Test results: {test_metrics}\n")

        print(f"Training complete. Best model saved at {best_model_path}.")
        print(f"Best validation loss: {best_val_loss}. Best epoch: {best_epoch}.")
        val_roc = val_metrics.get("val/roc") or val_metrics.get("test/roc")
        print(f"Validation ROC: {val_roc if val_roc is not None else 'N/A'}")
        print(f"Test ROC: {test_metrics.get('test/roc', 'N/A')}")

        # Include validation metrics alongside test metrics for downstream saving
        combined_metrics = dict(test_metrics)
        for key, value in val_metrics.items():
            # Avoid overwriting test metrics; prefix validation keys
            combined_metrics[f"val_{key.replace('/', '_')}"] = value

        return combined_metrics, best_model_path, best_epoch

    def run(self, seed):
        pl.seed_everything(seed)
        mols, ys = self.load_data()

        feature_files = sorted(self.features_folder.glob("*.npz"))
        if not feature_files:
            raise ValueError(f"No .npz feature files found in {self.features_folder}")

        aggregated_results = []

        for feature_file in feature_files:
            feature_file = Path(feature_file)
            print(f"\n=== Processing feature file: {feature_file.name} ===")
            extra_atom_featuress = self.load_extra_atom_features(feature_file)

            # Extract kernel number from filename
            stem = feature_file.stem
            if stem.startswith("kernel_"):
                kernel_num = int(stem.split("_")[1])
            else:
                parts = stem.split("_")
                kernel_num = None
                for part in parts:
                    if part.startswith("ker") and len(part) > 3:
                        try:
                            kernel_num = int(part[3:])
                            break
                        except ValueError:
                            continue
                if kernel_num is None:
                    match = re.search(r'(\d+)', stem)
                    if match:
                        kernel_num = int(match.group(1))
                    else:
                        raise ValueError(f"Could not extract kernel number from filename: {feature_file.name}")

            train_data, val_data, test_data = self.prepare_data(mols, ys, extra_atom_featuress, seed)
            train_dset, val_dset, test_dset, featurizer = self.create_datasets(
                train_data, val_data, test_data, extra_atom_featuress
            )
            train_loader, val_loader, test_loader = self.create_dataloaders(
                train_dset, val_dset, test_dset
            )

            print(f"Training dataset size: {len(train_dset)}")
            print(f"Validation dataset size: {len(val_dset)}")
            print(f"Test dataset size: {len(test_dset)}")

            kernel_output_dir = os.path.join(self.results_path, f"kernel_{kernel_num}", f"seed_{seed}")
            os.makedirs(kernel_output_dir, exist_ok=True)

            mpnn = self.create_model(featurizer)
            combined_metrics, best_model_ckpt_path, best_epoch = self.train_and_test(
                mpnn, train_loader, val_loader, test_loader, kernel_output_dir
            )

            print(
                f"Training complete for kernel {kernel_num}, seed {seed}. "
                f"Results saved in {kernel_output_dir} (best epoch {best_epoch})."
            )

            combined_metrics.update(
                {
                    "kernel": kernel_num,
                    "feature_file": str(feature_file),
                    "seed": seed,
                }
            )
            aggregated_results.append(combined_metrics)

        return aggregated_results

def get_args():
    parser = argparse.ArgumentParser(description="BBBP Trainer")
    parser.add_argument('--input_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--features_folder', type=str, required=True, help='Path to folder containing feature files')
    parser.add_argument('--smiles_column', type=str, default='smiles', help='Column name for SMILES strings')
    parser.add_argument('--target_columns', nargs='+', default=['labels'], help='Target column(s) for classification')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--results_path', type=str, required=True, help='Path to store results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')  
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs for training') 
    parser.add_argument('--split_type', type=str, choices=['RANDOM', 'SCAFFOLD_BALANCED'], default='SCAFFOLD_BALANCED', help='Dataset split type')
    parser.add_argument('--split_sizes', nargs=3, type=float, default=(0.8, 0.1, 0.1), help='Proportions for train/val/test split')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    return parser.parse_args()

def main():
    args = get_args()
    trainer = BBBPTrainer(args)

    print(f"Running with seed {args.seed}")
    metrics_list = trainer.run(args.seed)
    
    if not metrics_list:
        print("No results generated.")
        return
    
    results_df = pd.DataFrame(metrics_list)
    results_csv_path = os.path.join(trainer.results_path, f"all_results_seed_{args.seed}.csv")
    results_df.to_csv(results_csv_path, index=False)

    drop_cols = [col for col in ["kernel", "feature_file", "seed"] if col in results_df.columns]
    average_metrics = results_df.drop(columns=drop_cols).mean(numeric_only=True)
    average_csv_path = os.path.join(trainer.results_path, f"average_results_seed_{args.seed}.csv")
    average_metrics.to_frame(name="mean").to_csv(average_csv_path)

    print(f"Results saved to {results_csv_path}")
    print(f"Average metrics saved to {average_csv_path}")
    print("Average metrics:")
    for metric_name, metric_value in average_metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print('Done!')
    print('Elapsed time: ', time.time()-t0)
