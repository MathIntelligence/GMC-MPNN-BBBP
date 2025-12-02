import numpy as np
import pandas as pd
import os
from os import listdir
from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
from biopandas.mol2 import PandasMol2


class KernelFunction:

    def __init__(self, kernel_type='exponential_kernel', kappa=2.0, tau=1.0):
        self.kernel_type = kernel_type
        self.kappa = kappa
        self.tau = tau
        self.kernel_function = self.build_kernel_function(kernel_type)

    def build_kernel_function(self, kernel_type):
        if kernel_type[0] in ['E', 'e']:
            return self.exponential_kernel
        elif kernel_type[0] in ['L', 'l']:
            return self.lorentz_kernel

    def exponential_kernel(self, d, vdw_radii):
        eta = self.tau*vdw_radii
        return np.exp(-(d/eta)**self.kappa)

    def lorentz_kernel(self, d, vdw_radii):
        eta = self.tau*vdw_radii
        return 1/(1+(d/eta)**self.kappa)


class GGL_LIGAND:

    ligand_atom_types_df = pd.read_csv('utils/ligand_SYBYL_atom_types.csv')
    ligand_atom_types = ligand_atom_types_df['AtomType'].tolist()
    ligand_atom_radii = ligand_atom_types_df['Radius'].tolist()

    def __init__(self, Kernel, cutoff=12.0):
        self.Kernel = Kernel
        self.cutoff = cutoff

        self.pairwise_atom_type_radii = self.get_pairwise_atom_type_radii()

    def get_pairwise_atom_type_radii(self):
        ligand_atom_radii_dict = {a: r for (a, r) in zip(self.ligand_atom_types, self.ligand_atom_radii)}

        pairwise_atom_type_radii = {i[0]+"-"+i[1]: ligand_atom_radii_dict[i[0]] +
                                    ligand_atom_radii_dict[i[1]] for i in product(self.ligand_atom_types, self.ligand_atom_types)}

        return pairwise_atom_type_radii

    def mol2_to_df(self, mol2_file):
        df_mol2_all = PandasMol2().read_mol2(mol2_file).df
        df_mol2 = df_mol2_all[df_mol2_all['atom_type']!='H']
        df = pd.DataFrame(data={'ATOM_INDEX': df_mol2['atom_id'],
                                'ATOM_ELEMENT': df_mol2['atom_type'],
                                'X': df_mol2['x'],
                                'Y': df_mol2['y'],
                                'Z': df_mol2['z']})

        if len(set(df["ATOM_ELEMENT"]) - set(self.ligand_atom_types)) > 0:
            print(
                "WARNING: Ligand contains unsupported atom types. Only supported atom-type pairs are counted.")
        return(df)


    def get_atom_features(self, ligand_file):
        ligand = self.mol2_to_df(ligand_file)

        atom_pairs = list(product(ligand["ATOM_ELEMENT"], ligand["ATOM_ELEMENT"]))
        atom_pairs = [x[0] + "-" + x[1] for x in atom_pairs]

        # Calculate pairwise radii
        pairwise_radii = [self.pairwise_atom_type_radii.get(x, np.nan) for x in atom_pairs]
        pairwise_radii = np.asarray(pairwise_radii)

        # Calculate distances
        distances = cdist(ligand[["X", "Y", "Z"]], ligand[["X", "Y", "Z"]], metric="euclidean")
        pairwise_radii = pairwise_radii.reshape(distances.shape[0], distances.shape[1])

        # Calculate MWC distances
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)

        # Exclude diagonal entries as they represent distance to itself
        np.fill_diagonal(mwcg_distances, np.nan)

        covalent_bond_mask = distances >= (pairwise_radii.sum(axis=1).reshape(-1, 1))  
        cutoff_mask = distances > self.cutoff     
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)
        mwcg_distances[covalent_bond_mask | cutoff_mask] = np.nan
        np.fill_diagonal(mwcg_distances, np.nan)


        row_mean = np.nanmean(mwcg_distances, axis=1).reshape(-1, 1)
        row_min = np.nanmin(mwcg_distances, axis=1).reshape(-1, 1)
        row_max = np.nanmax(mwcg_distances, axis=1).reshape(-1, 1)
        row_sum = np.nansum(mwcg_distances, axis=1).reshape(-1, 1)
        row_std = np.nanstd(mwcg_distances, axis=1).reshape(-1, 1)
        row_median = np.nanmedian(mwcg_distances, axis=1).reshape(-1, 1)

        features_array = np.hstack((row_min, row_max, row_sum, row_mean, row_median, row_std))

        return features_array
