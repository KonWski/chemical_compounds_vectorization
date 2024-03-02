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
            extra_features = self._prepare_dataset_for_mat(self.smiles[0], self.labels[0])


    def __getitem__(self, index):
        raise NotImplementedError


    def _prepare_dataset_for_mat(self, smiles, labels):
        '''
        Converts smiles molecules into (node features, adjacency matrices, distance matrices)
        which is acceptable by Molecule Attention Transformer
        '''

        extra_features, _ = load_data_from_smiles(smiles, labels)
        print(f"node features: {type(extra_features[0])}")
        print(f"adjacency matrices: {type(extra_features[1])}")
        print(f"distance matrices: {type(extra_features[2])}")

        return extra_features


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
        dataset_already_downloaded = os.path.isdir(dataset_path)

        print(f"dataset_path: {dataset_path}")
        print(f"dataset_already_downloaded: {dataset_already_downloaded}")
        
        smiles_path = f"{dataset_split_path}/smiles_{split}.npy"
        X_path = f"{dataset_split_path}/X_{split}.npy"
        y_path = f"{dataset_split_path}/y_{split}.npy"
        w_path = f"{dataset_split_path}/w_{split}.npy"

        if dataset_already_downloaded:
            smiles = np.load(smiles_path)
            X = np.load(X_path) 
            y = np.load(y_path)
            w = np.load(w_path)

        elif not dataset_already_downloaded and download_dataset:
            
            # download datasets from deepchem
            if dataset_name == "HIV":
                tasks, datasets, transformers = dc.molnet.load_hiv(featurizer=featurizer)
            elif dataset_name == "TOX21":
                tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer)
            elif dataset_name == "Delaney":
                tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer)
            else:
                raise Exception(f"Dataset {dataset_name} not implemented.")

            os.mkdir(dataset_path)
            os.mkdir(dataset_split_path)

            split_id = 0 if split == "train" else 2
            smiles, X, y, w = datasets[split_id].ids, datasets[split_id].X, datasets[split_id].y, datasets[split_id].w
            print(f"Smiles type: {type(smiles)}")

            np.save(smiles_path, smiles)
            np.save(X_path, X)
            np.save(y_path, y)
            np.save(w_path, w)

        return smiles, X, y, w