import os
import argparse
import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn, utils
import csv

def load_data(input_path, smiles_column, target_columns):
    """Loads SMILES and target data from a CSV file."""
    df_input = pd.read_csv(input_path)
    smis = df_input[smiles_column].values
    ys = df_input[target_columns].values
    mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
    return mols, ys, smis

def load_extra_atom_features(feature_file):
    """Loads pre-computed atom features."""
    extra_atom_featuress = np.load(feature_file)
    return [extra_atom_featuress[f"arr_{i}"] for i in range(len(extra_atom_featuress))]

def create_dataloader(dataset, batch_size, num_workers):
    """Creates a DataLoader for the dataset."""
    return data.build_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

def load_and_test_model(model_path, test_loader):
    """Loads a trained model and evaluates it on the test data."""
    model = models.MPNN.load_from_checkpoint(model_path)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices="auto",
    )
    
    # Get predictions (will be class probabilities)
    predictions = trainer.predict(model, test_loader)
    predictions = torch.cat(predictions).numpy()
    
    # Calculate metrics (will include 'test/roc')
    test_results = trainer.test(model, test_loader)
    return test_results[0], predictions

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Load and test BBBP classification model with multiple seeds")
    parser.add_argument('--input_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--feature_file', type=str, required=True, help='Path to feature file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save test results')
    parser.add_argument('--smiles_column', type=str, default='smiles', help='Column name for SMILES strings')
    parser.add_argument('--target_columns', nargs='+', default=['labels'], help='Target column(s) for classification')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--split_type', type=str, choices=['RANDOM', 'SCAFFOLD_BALANCED'], default='SCAFFOLD_BALANCED', help='Dataset split type')
    parser.add_argument('--split_sizes', nargs=3, type=float, default=(0.8, 0.1, 0.1), help='Proportions for train/val/test split')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all data once
    mols, ys, smis = load_data(args.input_path, args.smiles_column, args.target_columns)
    extra_atom_featuress = load_extra_atom_features(args.feature_file)
    
    # Combine into MoleculeDatapoint objects
    all_datapoints = [
        data.MoleculeDatapoint(mol=mol, y=y, V_f=V_f)
        for mol, y, V_f in zip(mols, ys, extra_atom_featuress)
    ]

    # Prepare results file
    csv_path = os.path.join(args.output_dir, 'test_results_scaffold_classification.csv')
    
    # Initialize featurizer here
    temp_featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra_atom_featuress[0].shape[1])

    # **NEW: Initialize a list to store results for statistics**
    all_roc_auc_scores = []

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['seed', 'test_roc_auc'] 
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        predictions_dir = os.path.join(args.output_dir, 'predictions_scaffold_classification')
        os.makedirs(predictions_dir, exist_ok=True)

        for seed in range(21):
            print(f"--- Testing with seed {seed} using {args.split_type} split ---")
            pl.seed_everything(seed)
            
            # Get split indices
            train_indices, val_indices, test_indices = data.make_split_indices(
                mols,
                args.split_type,
                args.split_sizes,
                seed=seed,
            )
            
            # Use chemprop's helper function to split the data
            train_data_nested, val_data_nested, test_data_nested = data.split_data_by_indices(
                all_datapoints,
                train_indices,
                val_indices,
                test_indices
            )
            
            # Unpack the single fold from the nested list structure
            train_data = train_data_nested[0]
            val_data = val_data_nested[0]
            test_data = test_data_nested[0]

            # Create datasets for each split
            train_dset = data.MoleculeDataset(train_data, temp_featurizer)
            val_dset = data.MoleculeDataset(val_data, temp_featurizer)
            test_dset = data.MoleculeDataset(test_data, temp_featurizer)
            
            # Normalize features based on the TRAINING set
            scaler = train_dset.normalize_inputs("V_f")
            val_dset.normalize_inputs("V_f", scaler)
            test_dset.normalize_inputs("V_f", scaler)
            
            test_ys = np.array([d.y for d in test_dset])

            test_loader = create_dataloader(test_dset, args.batch_size, args.num_workers)

            # Load model and get results
            test_results, predictions = load_and_test_model(args.model_path, test_loader)
            
            roc_auc = test_results.get('test/roc')

            # **NEW: Add the score to our list for later calculation**
            if roc_auc is not None:
                all_roc_auc_scores.append(roc_auc)
            
            # Create the dictionary row with a matching key
            results_to_write = {
                'seed': seed,
                'test_roc_auc': roc_auc
            }
            writer.writerow(results_to_write)

            # Save predictions to a separate CSV for this seed
            predictions_df = pd.DataFrame({
                'true_label': test_ys.flatten(),
                'predicted_probability': predictions.flatten()
            })
            predictions_csv_path = os.path.join(predictions_dir, f"predictions_seed_{seed}.csv")
            predictions_df.to_csv(predictions_csv_path, index=False)

            # Print the ROC-AUC score safely
            print(f"Test ROC-AUC: {roc_auc:.4f}" if roc_auc is not None else "Test ROC-AUC: N/A")
            print(f"Predictions for seed {seed} saved to: {predictions_csv_path}")

    # Print overall statistics
    print("\n" + "="*50)
    print("Overall Test Results (21 Seeds)")
    print("="*50)

    if all_roc_auc_scores:  # Ensure the list is not empty
        mean_roc_auc = np.mean(all_roc_auc_scores)
        std_roc_auc = np.std(all_roc_auc_scores)
        max_roc_auc = np.max(all_roc_auc_scores)

        print(f"Mean ROC-AUC:           {mean_roc_auc:.4f}")
        print(f"Standard Deviation:     {std_roc_auc:.4f}")
        print(f"Maximum ROC-AUC:        {max_roc_auc:.4f}")
    else:
        print("No valid ROC-AUC scores were recorded to calculate statistics.")
    
    print("="*50)
    print(f"\nAll test results saved to {csv_path}")


if __name__ == "__main__":
    main()