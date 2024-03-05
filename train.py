from dataset import MoleculeDataset, MoleculeDataLoader
import torch
from torch.optim import Adam
from torch.nn import BCELoss, CrossEntropyLoss
import logging
import yaml

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
    train_loader = MoleculeDataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = MoleculeDataset(dataset_name, "test", "ECFP", True, download_dataset, root_datasets_dir)
    test_loader = MoleculeDataLoader(testset, batch_size=batch_size, shuffle=True)

    # number of observations
    len_train_dataset = len(trainset)
    len_test_dataset = len(testset)

    # model loading
    with open(f'model_params/{model_type}.yaml', 'r') as yaml_config:
        model_params = yaml.safe_load(yaml_config)

    print(model_params)

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
                    
                    smiles, vectorized_molecules, labels, w, node_features, adjacency_matrices, distance_matrices = batch
                break
            break
        
        break