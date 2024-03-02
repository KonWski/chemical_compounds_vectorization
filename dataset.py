import deepchem as dc
from mat_data_utils import load_data_from_smiles
import numpy as np
from torch.utils.data import Dataset
import os

class MoleculeDataset(Dataset):

    def __init__(self, dc_dataset_name: str, split: str, featurizer: str, prepare_data_for_mat: bool,
                 download_dataset: bool = False, root_datasets_dir: str="") -> None:
        self.dc_dataset_name = dc_dataset_name
        self.featurizer = featurizer
        self.prepare_data_for_mat = prepare_data_for_mat

        self.smiles, self.vectorized_molecules, self.labels, self.w = self._prepare_dc_datasets(dc_dataset_name, split, featurizer, 
                                                                                        download_dataset, root_datasets_dir)

        self.node_features = None
        self.adjacency_matrix = None
        self.distance_matrices = None

        if prepare_data_for_mat:
            self.node_features, self.adjacency_matrix, self.distance_matrices = self._prepare_dataset_for_mat(self.smiles[0], self.labels[0])


    def __getitem__(self, index):

        # extra features generated
        if self.prepare_data_for_mat:
            return self.smiles[index], self.vectorized_molecules[index], self.labels[index], self.w[index], \
                self.node_features[index], self.adjacency_matrix[index], self.distance_matrices[index]

        else:
            return self.smiles[index], self.vectorized_molecules[index], self.labels[index], self.w[index]


    def _prepare_dataset_for_mat(self, smiles, labels, split, root_datasets_dir):
        '''
        Converts smiles molecules into (node features, adjacency matrices, distance matrices)
        which is acceptable by Molecule Attention Transformer
        '''

        dataset_split_path = f"{root_datasets_dir}/{self.dataset_name}/{split}"
        node_features_path = ""
        adjacency_matrices_path = ""
        distance_matrices_path = ""

        extra_features_already_prepared = os.path.isfile(node_features_path) and os.path.isfile(adjacency_matrices_path) \
            and os.path.isfile(distance_matrices_path)

        molecules_extra_features, _ = load_data_from_smiles(smiles, labels)
        
        node_features = []
        adjacency_matrices = []
        distance_matrices = []
        print(f"type(extra_features): {type(molecules_extra_features)}")
        print(f"len(extra_features): {len(molecules_extra_features)}")
        print(f"node features: {type(molecules_extra_features[0][0])}")
        print(f"adjacency matrices: {type(molecules_extra_features[0][1])}")
        print(f"distance matrices: {type(molecules_extra_features[0][2])}")

        # collect all extra features into lists
        for extra_features in molecules_extra_features:
            node_features.append(extra_features[0])
            adjacency_matrices.append(extra_features[1])
            distance_matrices.append(extra_features[2])

        return node_features, adjacency_matrices, distance_matrices


    def _prepare_dc_datasets(self, dataset_name: str, split: str, featurizer: str, download_dataset: bool, root_datasets_dir: str):
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

        dataset_path = f"{root_datasets_dir}/{dataset_name}"
        dataset_split_path = f"{dataset_path}/{split}"
        main_dir_created = os.path.isdir(dataset_path)
        split_dataset_already_downloaded = os.path.isdir(dataset_split_path)

        if not main_dir_created:
            os.mkdir(dataset_path)
        
        smiles_path = f"{dataset_split_path}/smiles_{split}.npy"
        X_path = f"{dataset_split_path}/X_{split}.npy"
        y_path = f"{dataset_split_path}/y_{split}.npy"
        w_path = f"{dataset_split_path}/w_{split}.npy"

        if split_dataset_already_downloaded:
            smiles = np.load(smiles_path, allow_pickle=True)
            X = np.load(X_path, allow_pickle=True) 
            y = np.load(y_path, allow_pickle=True)
            w = np.load(w_path, allow_pickle=True)

        elif not split_dataset_already_downloaded and download_dataset:
            
            # download datasets from deepchem
            if dataset_name == "HIV":
                tasks, datasets, transformers = dc.molnet.load_hiv(featurizer=featurizer)
            elif dataset_name == "TOX21":
                tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer)
            elif dataset_name == "Delaney":
                tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer)
            else:
                raise Exception(f"Dataset {dataset_name} not implemented.")

            os.mkdir(dataset_split_path)

            split_id = 0 if split == "train" else 2
            smiles, X, y, w = datasets[split_id].ids, datasets[split_id].X, datasets[split_id].y, datasets[split_id].w
            print(f"Smiles type: {type(smiles)}")

            np.save(smiles_path, smiles)
            np.save(X_path, X)
            np.save(y_path, y)
            np.save(w_path, w)

        return smiles, X, y, w