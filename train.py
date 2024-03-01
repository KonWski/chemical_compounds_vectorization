from dataset import MoleculeDataset
import torch
from torch.optim import Adam
from torch.nn import BCELoss
import logging


def train_model(
        device, 
        n_epochs: int,
        dataset_name: str,
        download_dataset: bool,
        root_datasets_dir: str,
        batch_size: int,
        model_type: str
    ):
    '''

    Parameters
    ----------
    device 
    n_epochs: int
        number of training epochs
    dataset_name: str
        Name of dataset available through MoleculeNet
    download_dataset: bool
        Download dataset from MoleculeNet
    batch_size: int
        number of images inside single batch
    model_type: str
        type of model which will be trained
    '''

    if dataset_name == "":
        prepare_data_for_mat = False
    else:
        prepare_data_for_mat = False

    # datasets and dataloaders
    train_dataset = MoleculeDataset(dataset_name, "train", "ECFP", True, download_dataset, root_datasets_dir)
    test_dataset = MoleculeDataset(dataset_name, "test", "ECFP", True, download_dataset, root_datasets_dir)

    