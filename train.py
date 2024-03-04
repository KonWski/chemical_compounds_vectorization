from dataset import MoleculeDataset, collate_extra_features
import torch
from torch.optim import Adam
from torch.nn import BCELoss, CrossEntropyLoss
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


    # datasets and dataloaders
    trainset = MoleculeDataset(dataset_name, "train", "ECFP", True, download_dataset, root_datasets_dir)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, collate_fn=collate_extra_features, shuffle=True)

    testset = MoleculeDataset(dataset_name, "test", "ECFP", True, download_dataset, root_datasets_dir)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, collate_fn=collate_extra_features, shuffle=True)

    # number of observations
    len_train_dataset = len(trainset)
    len_test_dataset = len(testset)

    for epoch in range(n_epochs):

        for state, loader, len_dataset in zip(["train", "test"], [train_loader, test_loader], [len_train_dataset, len_test_dataset]):

            # calculated parameters
            running_loss = 0.0
            running_corrects = 0

            criterion = CrossEntropyLoss()

            # if state == "train":
            #     model.train()
            # else:
            #     model.eval()

            for id, batch in enumerate(loader, 0):

                with torch.set_grad_enabled(state == 'train'):
                    
                    adjacency_matrix, node_features, distance_matrix, y = batch

