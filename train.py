from dataset import download_dataset
import torch
from torch.optim import Adam
from torch.nn import BCELoss
import logging


def train_model(
        device, 
        n_epochs: int,
        dataset_name: str,
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
    batch_size: int
        number of images inside single batch
    model_type: str
        type of model which will be trained
    '''

    # datasets and dataloaders
    train_dataset, test_dataset = download_dataset("HIV", "ECFP", True)