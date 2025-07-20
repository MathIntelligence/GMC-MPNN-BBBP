import sys
import os
import pandas as pd
from itertools import product
import ntpath
import argparse
import time
import numpy as np
from features_extraction.ggl_ligand import *


class GeometricGraphLearningFeatures:
    df_kernels = pd.read_csv('kernels.csv')

    def __init__(self, args):
        self.kernel_index = args.kernel_index
        self.cutoff = args.cutoff
        self.path_to_csv = args.path_to_csv
        self.data_folder = args.data_folder
        self.feature_folder = args.feature_folder


    def get_ggl_atom_features(self, parameters):
        df_pdbids = pd.read_csv(self.path_to_csv)
        pdbids = df_pdbids['id'].tolist()
        smiles = df_pdbids['smiles'].tolist()

        Kernel = KernelFunction(kernel_type=parameters['type'], kappa=parameters['power'], tau=parameters['tau'])
        GGL = GGL_LIGAND(Kernel=Kernel, cutoff=parameters['cutoff'])

        extra_atom_features = []

        for _pdbid in pdbids:
            lig_file = f'{self.data_folder}/{_pdbid}.mol2'

            # Get atom features for the current .mol2 file
            ggl_atom_features = GGL.get_atom_features(lig_file)

            # Append features to the list
            extra_atom_features.append(ggl_atom_features)

        return extra_atom_features

    def main(self):
        adjusted_index = self.kernel_index - 1

        parameters = {
            'type': self.df_kernels.loc[adjusted_index, 'type'],
            'power': self.df_kernels.loc[adjusted_index, 'power'],
            'tau': self.df_kernels.loc[adjusted_index, 'tau'],
            'cutoff': self.cutoff
        }

        atom_features_list = self.get_ggl_atom_features(parameters)
        output_file_name = f'ker{self.kernel_index}_cutoff{self.cutoff}.npz'
        output_path = os.path.join(self.feature_folder, output_file_name)

        # Save the list of atom features as an NPZ file
        np.savez(output_path, *atom_features_list)

        print(f"Processed features saved to {output_path}")



def get_args(args):

    parser = argparse.ArgumentParser(description="Get GGL Features")
    parser.add_argument('-k', '--kernel-index', help='Kernel Index (see kernels/kernels.csv)',
                        type=int)
    parser.add_argument('-c', '--cutoff', help='distance cutoff to define binding site',
                        type=float, default=12.0)
    parser.add_argument('-f', '--path_to_csv',
                        help='path to CSV file containing PDBIDs and pK values')
    parser.add_argument('-dd', '--data_folder', type=str,
    					help='path to data folder directory')
    parser.add_argument('-fd', '--feature_folder', type=str,
    					help='path to the directory where features will be saved')

    args = parser.parse_args()
    return args


def cli_main():
    args = get_args(sys.argv[1:])
    GGL_Features = GeometricGraphLearningFeatures(args)
    GGL_Features.main()


if __name__ == "__main__":
    t0 = time.time()
    cli_main()
    print('Done!')
    print('Elapsed time: ', time.time()-t0)
