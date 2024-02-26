import deepchem as dc
from mat_data_utils import load_data_from_smiles
import numpy as np
from torch.utils.data import Dataset

class MoleculeDataset(Dataset):

    def __init__(self, dc_dataset_name: str, featurizer: str, prepare_data_for_mat: bool, dc_dataset) -> None:
        self.dc_dataset_name = dc_dataset_name
        self.featurizer = featurizer
        self.smiles = dc_dataset.ids
        self.vectorized_molecules = dc_dataset.X
        self.labels = dc_dataset.y
        self.node_features = None
        self.adjacency_matrix = None
        self.distance_matrices = None

        if prepare_data_for_mat:
            extra_features = self._prepare_dataset_for_mat(self.smiles, self.labels)


    def __getitem__(self, index):
        raise NotImplementedError


    def _prepare_dataset_for_mat(self, smiles, labels):
        '''
        Converts smiles molecules into (node features, adjacency matrices, distance matrices)
        which is acceptable by Molecule Attention Transformer
        '''

        extra_features, _ = load_data_from_smiles(smiles, labels)
                
        return extra_features



def download_dataset(dataset_name: str, featurizer: str="ECFP", prepare_data_for_mat: bool=False):
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
    '''

    if dataset_name == "HIV":
        tasks, datasets, transformers = dc.molnet.load_hiv(featurizer=featurizer)
    elif dataset_name == "TOX21":
        tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer)
    elif dataset_name == "Delaney":
        tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer)
    else:
        raise Exception(f"Dataset {dataset_name} not implemented.")

    train_dataset = MoleculeDataset(dataset_name, featurizer, prepare_data_for_mat, datasets[0])
    test_dataset = MoleculeDataset(dataset_name, featurizer, prepare_data_for_mat, datasets[2])

    # if prepare_data_for_mat:
    #     train_X, train_y, valid_X, valid_y, test_X, test_y = prepare_dataset_for_mat(datasets)

    return train_dataset, test_dataset


def prepare_dataset_for_mat(datasets, add_dummy_node=True, one_hot_formal_charge=False, use_data_saving=True):
    '''
    Converts smiles molecules into (node features, adjacency matrices, distance matrices)
    which is acceptable by Molecule Attention Transformer

    '''

    train_dataset, valid_dataset, test_dataset = datasets

    train_smiles, train_labels = train_dataset.ids, train_dataset.y
    valid_smiles, valid_labels = valid_dataset.ids, valid_dataset.y
    test_smiles, test_labels = test_dataset.ids, test_dataset.y

    train_X, train_y = load_data_from_smiles(train_smiles, train_labels, add_dummy_node=add_dummy_node,
                                         one_hot_formal_charge=one_hot_formal_charge)
    
    valid_X, valid_y = load_data_from_smiles(valid_smiles, valid_labels, add_dummy_node=add_dummy_node,
                                        one_hot_formal_charge=one_hot_formal_charge)

    test_X, test_y = load_data_from_smiles(test_smiles, test_labels, add_dummy_node=add_dummy_node,
                                        one_hot_formal_charge=one_hot_formal_charge)
    
    return train_X, train_y, valid_X, valid_y, test_X, test_y


if __name__ == "main":
    md = download_dataset("HIV", "ECFP", True)