import deepchem as dc
from mat_data_utils import load_data_from_smiles
import numpy as np
from torch.utils.data import Dataset
import os
import pickle

class MoleculeDataset(Dataset):

    def __init__(self, dc_dataset_name: str, split: str, featurizer: str, prepare_data_for_mat: bool,
                 download_dataset: bool = False, root_datasets_dir: str="") -> None:
       
        self.dc_dataset_name = dc_dataset_name
        self.split = split
        self.featurizer = featurizer
        self.root_datasets_dir = root_datasets_dir
        self.dataset_path, self.dataset_split_path = self._prepare_directories(download_dataset)
        self.smiles, self.vectorized_molecules, self.labels, self.w = self._prepare_dc_datasets(download_dataset)
        self.node_features, self.adjacency_matrix, \
            self.distance_matrices = self._prepare_dataset_for_mat(prepare_data_for_mat, download_dataset)


    def __getitem__(self, index):

        # extra features generated
        if self.prepare_data_for_mat:
            return self.smiles[index], self.vectorized_molecules[index], self.labels[index], self.w[index], \
                self.node_features[index], self.adjacency_matrix[index], self.distance_matrices[index]

        else:
            return self.smiles[index], self.vectorized_molecules[index], self.labels[index], self.w[index]


    def _prepare_directories(self, download_dataset):
        
        if download_dataset:
            dataset_path = f"{self.root_datasets_dir}/{self.dc_dataset_name}"
            dataset_dir_created = os.path.isdir(dataset_path)

            dataset_split_path = f"{dataset_path}/{self.split}"
            dataset_split_dir_created = os.path.isdir(dataset_split_path)

            if not dataset_dir_created:
                os.mkdir(dataset_path)
            
            if not dataset_split_dir_created:
                os.mkdir(dataset_split_path)

            return dataset_path, dataset_split_path
        
        else:
            return None, None


    def _prepare_dataset_for_mat(self, prepare_data_for_mat, download_dataset):
        '''
        Converts smiles molecules into (node features, adjacency matrices, distance matrices)
        which is acceptable by Molecule Attention Transformer
        '''

        if not prepare_data_for_mat:
            return None, None, None

        node_features_path = f"{self.dataset_split_path}/node_features.npy"
        adjacency_matrices_path = f"{self.dataset_split_path}/adjacency_matrices.npy"
        distance_matrices_path = f"{self.dataset_split_path}/distance_matrices.npy"

        extra_features_already_prepared = os.path.isfile(node_features_path) and os.path.isfile(adjacency_matrices_path) \
            and os.path.isfile(distance_matrices_path)

        if extra_features_already_prepared and download_dataset:

            with open(node_features_path, "rb") as fp:
                pickle.dump(node_features, fp)

            with open(adjacency_matrices_path, "rb") as fp:
                pickle.dump(adjacency_matrices, fp)

            with open(distance_matrices_path, "rb") as fp:
                pickle.dump(distance_matrices, fp)

        else:
            molecules_extra_features, _ = load_data_from_smiles(self.smiles, self.labels)
            
            node_features = []
            adjacency_matrices = []
            distance_matrices = []
            print(f"type(extra_features): {type(molecules_extra_features)}")
            print(f"len(extra_features): {len(molecules_extra_features)}")
            print(f"node features: {type(molecules_extra_features[0][0])}")
            print(f"adjacency matrices: {type(molecules_extra_features[0][1])}")
            print(f"distance matrices: {type(molecules_extra_features[0][2])}")

            with open(node_features_path, "rb") as fp:
                node_features = pickle.load(fp)

            with open(adjacency_matrices_path, "rb") as fp:
                adjacency_matrices = pickle.load(fp)

            with open(distance_matrices_path, "rb") as fp:
                distance_matrices = pickle.load(fp)

            # collect all extra features into lists
            for extra_features in molecules_extra_features:
                node_features.append(extra_features[0])
                adjacency_matrices.append(extra_features[1])
                distance_matrices.append(extra_features[2])
            
        return node_features, adjacency_matrices, distance_matrices


    def _prepare_dc_datasets(self, download_dataset: bool):
        '''
        Downloads dataset from Deepchem MoleculeNet
        
        Parameters
        ----------
        dataset_name: str
            name of the dataset to download from repository
        featurizer: str="ECFP"
            type of molecule featurizer 
        prepare_data_for_mat: bool=False
            converts smiles representation into (node features, adjacency matrices, distance matrices)
        download_dataset: bool=False
            Download dataset from MoleculeNet
        root_datasets_dir: str=""
            Path where dataset should be downloaded or where is it already stored
        '''

        smiles_path = f"{self.dataset_split_path}/smiles_{self.split}.npy"
        X_path = f"{self.dataset_split_path}/X_{self.split}.npy"
        y_path = f"{self.dataset_split_path}/y_{self.split}.npy"
        w_path = f"{self.dataset_split_path}/w_{self.split}.npy"
        split_dataset_already_downloaded = os.path.isfile(smiles_path) and os.path.isfile(X_path) \
            and os.path.isfile(y_path) and os.path.isfile(w_path)

        if split_dataset_already_downloaded:
            smiles = np.load(smiles_path, allow_pickle=True)
            X = np.load(X_path, allow_pickle=True) 
            y = np.load(y_path, allow_pickle=True)
            w = np.load(w_path, allow_pickle=True)

        elif not split_dataset_already_downloaded and download_dataset:
            
            # download datasets from deepchem
            if self.dc_dataset_name == "HIV":
                tasks, datasets, transformers = dc.molnet.load_hiv(featurizer=self.featurizer)
            elif self.dc_dataset_name == "TOX21":
                tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=self.featurizer)
            elif self.dc_dataset_name == "Delaney":
                tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=self.featurizer)
            else:
                raise Exception(f"Dataset {self.dataset_name} not implemented.")

            split_id = 0 if self.split == "train" else 2
            smiles, X, y, w = datasets[split_id].ids, datasets[split_id].X, datasets[split_id].y, datasets[split_id].w
            print(f"Smiles type: {type(smiles)}")

            np.save(smiles_path, smiles)
            np.save(X_path, X)
            np.save(y_path, y)
            np.save(w_path, w)

        return smiles, X, y, w