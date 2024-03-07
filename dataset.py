import deepchem as dc
from mat_data_utils import load_data_from_smiles
import numpy as np
from torch.utils.data import Dataset, Sampler
import os
import pickle
from torch.utils.data import DataLoader
from typing import Any, Callable, Iterable, TypeVar, List, Optional, Union
import torch

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]

class MoleculeDataset(Dataset):

    def __init__(self, dc_dataset_name: str, split: str, featurizer: str, prepare_data_for_mat: bool,
                 download_dataset: bool = False, root_datasets_dir: str="") -> None:
       
        self.dc_dataset_name = dc_dataset_name
        self.split = split
        self.featurizer = featurizer
        self.prepare_data_for_mat = prepare_data_for_mat
        self.root_datasets_dir = root_datasets_dir
        self.dataset_path, self.dataset_split_path = self._prepare_directories(download_dataset)
        self.smiles, self.vectorized_molecules, self.labels, self.w = self._prepare_dc_datasets(download_dataset)
        self.node_features, self.adjacency_matrix, \
            self.distance_matrices = self._prepare_dataset_for_mat(download_dataset)


    def __getitem__(self, index):

        # extra features generated
        if self.prepare_data_for_mat:
            return self.smiles[index], self.vectorized_molecules[index], self.labels[index], self.w[index], \
                self.node_features[index], self.adjacency_matrix[index], self.distance_matrices[index]

        else:
            return self.smiles[index], self.vectorized_molecules[index], self.labels[index], self.w[index]


    def __len__(self) -> int:
        return len(self.smiles)


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


    def _prepare_dataset_for_mat(self, download_dataset):
        '''
        Converts smiles molecules into (node features, adjacency matrices, distance matrices)
        which is acceptable by Molecule Attention Transformer
        '''

        if not self.prepare_data_for_mat:
            return None, None, None

        node_features_path = f"{self.dataset_split_path}/node_features.npy"
        adjacency_matrices_path = f"{self.dataset_split_path}/adjacency_matrices.npy"
        distance_matrices_path = f"{self.dataset_split_path}/distance_matrices.npy"

        extra_features_already_prepared = os.path.isfile(node_features_path) and os.path.isfile(adjacency_matrices_path) \
            and os.path.isfile(distance_matrices_path)

        if extra_features_already_prepared:

            with open(node_features_path, "rb") as fp:
                node_features = pickle.load(fp)

            with open(adjacency_matrices_path, "rb") as fp:
                adjacency_matrices = pickle.load(fp)

            with open(distance_matrices_path, "rb") as fp:
                distance_matrices = pickle.load(fp)

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

            # collect all extra features into lists
            for extra_features in molecules_extra_features:
                node_features.append(extra_features[0])
                adjacency_matrices.append(extra_features[1])
                distance_matrices.append(extra_features[2])

            if download_dataset:

                with open(node_features_path, "wb") as fp:
                    pickle.dump(node_features, fp)

                with open(adjacency_matrices_path, "wb") as fp:
                    pickle.dump(adjacency_matrices, fp)

                with open(distance_matrices_path, "wb") as fp:
                    pickle.dump(distance_matrices, fp)

        # convert outputs to tensors
        print(f"np.array(node_features).shape: {np.array(node_features).shape}")
        print(f"np.array(adjacency_matrices).shape: {np.array(adjacency_matrices).shape}")
        print(f"np.array(distance_matrices).shape: {np.array(distance_matrices).shape}")

        node_features = torch.Tensor(np.array(node_features)) 
        adjacency_matrices = torch.Tensor(np.array(adjacency_matrices)) 
        distance_matrices = torch.Tensor(np.array(distance_matrices)) 

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

        else:
            
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

            if download_dataset:
                np.save(smiles_path, smiles)
                np.save(X_path, X)
                np.save(y_path, y)
                np.save(w_path, w)

        return smiles, X, y, w


class MoleculeDataLoader(DataLoader):

    def __init__(self,
                 dataset: Dataset[T_co], 
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, 
                 collate_fn: Optional[_collate_fn_t] = None):

        super().__init__(dataset, batch_size, shuffle)

        if dataset.prepare_data_for_mat and collate_fn is None:
            self.collate_fn = self._collate_extra_features
        else:
            self.collate_fn = self.collate_fn

    

    def _collate_extra_features(self, batch):

        # print(f"type(batch): {type(batch)}")
        # print(f"len(batch): {len(batch)}")
        # print(f"type(batch[0]): {type(batch[0])}")
        # print(f"len(batch[0]): {len(batch[0])}")
        # print(f"type(batch[0][0]): {type(batch[0][0])}")
        # print(f"batch[0][0]: {batch[0][0]}")
        # print(f"type(batch[0][1]): {type(batch[0][1])}")


        if self.dataset.prepare_data_for_mat:

            smiles_list, vectorized_molecules_list, labels_list, w_list, \
                node_features_list, adjacency_matrices_list, distance_matrices_list = [], [], [], [], [], [], []

            for molecule in batch:

                smiles, vectorized_molecule, label, w, node_features, adjacency_matrix, distance_matrix = molecule
                max_size = 0
                    
                smiles_list.append(smiles)
                vectorized_molecules_list.append(vectorized_molecule)
                labels_list.append(label)
                w_list.append(w)

                node_features_list.append(node_features)
                adjacency_matrices_list.append(adjacency_matrix)

                if adjacency_matrix.shape[0] > max_size:
                    max_size = adjacency_matrix.shape[0]

            for molecule in batch:
                adjacency_matrices_list.append(self._pad_array(adjacency_matrix, (max_size, max_size)))
                distance_matrices_list.append(self._pad_array(distance_matrix, (max_size, max_size)))
                node_features_list.append(self._pad_array(node_features, (max_size, node_features.shape[1])))

            return [smiles_list, vectorized_molecules_list, labels_list, w_list, \
                node_features_list, adjacency_matrices_list, distance_matrices_list]
            
        else:
            return batch


    def _pad_array(self, array, shape, dtype=np.float32):
        """Pad a 2-dimensional array with zeros.

        Args:
            array (ndarray): A 2-dimensional array to be padded.
            shape (tuple[int]): The desired shape of the padded array.
            dtype (data-type): The desired data-type for the array.

        Returns:
            A 2-dimensional array of the given shape padded with zeros.
        """
        padded_array = np.zeros(shape, dtype=dtype)
        padded_array[:array.shape[0], :array.shape[1]] = array
        return padded_array



def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """
    adjacency_list, distance_list, features_list = [], [], []
    labels = []

    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        distance_list.append(pad_array(molecule.distance_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))

    return [FloatTensor(features) for features in (adjacency_list, features_list, distance_list, labels)]