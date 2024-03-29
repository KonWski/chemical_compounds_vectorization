import deepchem as dc
from data_utils import load_data_from_smiles
import numpy as np
from torch.utils.data import Dataset
import os
import pickle
from torch.utils.data import DataLoader
from typing import Any, Callable, TypeVar, List, Optional
import torch
from torch.nn import BCELoss, MSELoss
from sklearn.metrics import mean_squared_error, log_loss
import logging

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]

class MoleculeDataset(Dataset):

    def __init__(self, dc_dataset_name: str, split: str, featurizer: str, splitter: str, prepare_data_for_mat: bool,
                 download_dataset: bool = False, root_datasets_dir: str = "", dataset_task_name: str = None, 
                 model_type: str = None) -> None:
       
        self.dc_dataset_name = dc_dataset_name
        self.split = split
        self.featurizer = featurizer
        self.splitter = splitter
        self.prepare_data_for_mat = prepare_data_for_mat
        self.root_datasets_dir = root_datasets_dir
        self.dataset_path, self.dataset_split_path = self._prepare_directories(download_dataset)
        self.smiles, self.vectorized_molecules, self.labels, \
            self.w, self.dataset_task_name = self._prepare_dc_datasets(download_dataset, splitter, dataset_task_name)
        self.node_features, self.adjacency_matrix, \
            self.distance_matrices = self._prepare_dataset_for_mat(download_dataset)
        self.criterion = self._get_criterion(model_type)
        self.prediction_task = self._get_prediction_task()


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

                logging.info(f"Extra features saved to {self.dataset_split_path}")

        return node_features, adjacency_matrices, distance_matrices


    def _prepare_dc_datasets(self, download_dataset: bool, splitter: str, dataset_task_name: str):
        '''
        Downloads dataset from Deepchem MoleculeNet
        
        Parameters
        ----------
        download_dataset: bool=False
            Download dataset from MoleculeNet
        '''

        smiles_path = f"{self.dataset_split_path}/smiles_{self.split}.npy"
        X_path = f"{self.dataset_split_path}/X_{self.split}.npy"
        w_path = f"{self.dataset_split_path}/w_{self.split}.npy"
        if dataset_task_name:
            y_path = f"{self.dataset_split_path}/y_{dataset_task_name.lower().replace(' ', '_')}_{self.split}.npy"
        else:
            y_path = f"{self.dataset_split_path}/y_{self.split}.npy"       
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
                dataset_tasks, datasets, _ = dc.molnet.load_hiv(featurizer=self.featurizer, splitter=splitter)
            elif self.dc_dataset_name == "TOX21":
                dataset_tasks, datasets, _ = dc.molnet.load_tox21(featurizer=self.featurizer, splitter=splitter)
            elif self.dc_dataset_name == "Delaney":
                dataset_tasks, datasets, _ = dc.molnet.load_delaney(featurizer=self.featurizer, splitter=splitter)
            elif self.dc_dataset_name == "Sider":
                dataset_tasks, datasets, _ = dc.molnet.load_sider(featurizer=self.featurizer, splitter=splitter)
            elif self.dc_dataset_name == "BACE":
                dataset_tasks, datasets, _ = dc.molnet.load_bace_classification(featurizer=self.featurizer, splitter=splitter)
            else:
                raise Exception(f"Dataset {self.dataset_name} not implemented.")

            split_id = 0 if self.split == "train" else 2
            smiles, X, y, w = datasets[split_id].ids, datasets[split_id].X, datasets[split_id].y, datasets[split_id].w

            # if download_dataset:
            #     np.save(smiles_path, smiles)
            #     np.save(X_path, X)
            #     np.save(y_path, y)
            #     np.save(w_path, w)

            # fiter out task
            if dataset_task_name:
                task_id = dataset_tasks.index(dataset_task_name)
                y = y[:,task_id]
            elif len(dataset_tasks) == 1:
                dataset_task_name = dataset_tasks[0]
            else:
                raise Exception("Please specify dataset task name - dataset consists of more than 1 task.")

        return smiles, X, y, w, dataset_task_name


    def _get_criterion(self, model_type: str):

        # download datasets from deepchem
        if self.dc_dataset_name in ["HIV", "TOX21", "Sider", "BACE"]:
            if model_type == "mat":
                return BCELoss()
            elif model_type == "svm":
                return log_loss

        elif self.dc_dataset_name == "Delaney":
            if model_type == "mat":
                return MSELoss()
            elif model_type == "svm":
                return mean_squared_error

        else:
            raise Exception(f"Dataset {self.dataset_name} not implemented.")


    def _get_prediction_task(self):

        # download datasets from deepchem
        if self.dc_dataset_name in ["HIV", "TOX21", "Sider", "BACE"]:
            prediction_task = "classification"
        elif self.dc_dataset_name == "Delaney":
            prediction_task = "regression"

        return prediction_task


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

        if self.dataset.prepare_data_for_mat:

            smiles_list, vectorized_molecules_list, labels_list, w_list, \
                node_features_list, adjacency_matrices_list, distance_matrices_list = [], [], [], [], [], [], []

            max_size = 0

            for molecule in batch:
                
                smiles, vectorized_molecule, label, w, node_features, adjacency_matrix, distance_matrix = molecule

                smiles_list.append(smiles)
                vectorized_molecules_list.append(vectorized_molecule)
                w_list.append(w)                
                labels_list.append(label)

                if adjacency_matrix.shape[0] > max_size:
                    max_size = adjacency_matrix.shape[0]

            for molecule in batch:

                smiles, vectorized_molecule, label, w, node_features, adjacency_matrix, distance_matrix = molecule

                adjacency_matrices_list.append(self._pad_array(adjacency_matrix, (max_size, max_size)))
                distance_matrices_list.append(self._pad_array(distance_matrix, (max_size, max_size)))
                node_features_list.append(self._pad_array(node_features, (max_size, node_features.shape[1])))

            # convert list to tensors
            node_features_list = torch.Tensor(np.array(node_features_list)) 
            adjacency_matrices_list = torch.Tensor(np.array(adjacency_matrices_list)) 
            distance_matrices_list = torch.Tensor(np.array(distance_matrices_list)) 

            # convert deepchem labels to torch like labels
            if self.dataset.prediction_task == "classification":
                labels_list = [int(arr[0]) for arr in labels_list]
                labels = np.array(labels_list)
                labels_tensor = np.zeros((labels.size, 2)) # only binary classification
                labels_tensor[np.arange(labels.size), labels] = 1
                labels_tensor = torch.Tensor(labels_tensor)               
            else:
                labels_tensor = torch.Tensor(np.array(labels_list)) 

            return [smiles_list, vectorized_molecules_list, labels_tensor, w_list, \
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