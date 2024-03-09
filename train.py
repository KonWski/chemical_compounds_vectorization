from dataset import MoleculeDataset, MoleculeDataLoader
import torch
from torch.optim import Adam
import logging
import yaml
from pathlib import Path
from mat_model import make_model, load_checkpoint, save_checkpoint
from datetime import datetime

def train_model(
        device, 
        n_epochs: int,
        dataset_name: str,
        download_dataset: bool,
        root_datasets_dir: str,
        checkpoint_path: str,
        batch_size: int,
        model_type: str,
        load_model: bool
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
    root_datasets_dir: str
        Path where dataset should be downloaded or where is it already stored
    checkpoint_path: str
        Path to the loaded checkpoint
    batch_size: int
        number of images inside single batch
    model_type: str
        type of model which will be trained
    load_model: bool
        continue learning using existing model and optimizer
    '''

    # datasets and dataloaders
    trainset = MoleculeDataset(dataset_name, "train", "ECFP", True, download_dataset, root_datasets_dir)
    train_loader = MoleculeDataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = MoleculeDataset(dataset_name, "test", "ECFP", True, download_dataset, root_datasets_dir)
    test_loader = MoleculeDataLoader(testset, batch_size=batch_size, shuffle=True)

    # number of observations
    len_train_dataset = len(trainset)
    len_test_dataset = len(testset)

    if load_model:
        model, optimizer, loaded_checkpoint = load_checkpoint(checkpoint_path)
        best_test_loss = loaded_checkpoint["test_loss"]
        start_epoch = loaded_checkpoint["epoch"] + 1
        yaml_config_path = loaded_checkpoint["yaml_config_path"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    else:

        yaml_config_path = f'{Path(__file__).parent}/models_params/{model_type}.yaml'

        # model params loading
        with open(yaml_config_path, 'r') as yaml_config:
            model_params = yaml.safe_load(yaml_config)
            model_params["d_atom"] = trainset.node_features[0].shape[1]

        model = make_model(**model_params)
        optimizer = Adam(model.parameters(), lr=1e-5)
        best_test_loss = float("inf")
        start_epoch = 0

    for epoch in range(start_epoch, n_epochs):

        checkpoint = {}

        for state, loader, len_dataset in zip(["train", "test"], [train_loader, test_loader], [len_train_dataset, len_test_dataset]):

            # calculated parameters
            running_loss = 0.0
            criterion = trainset.criterion

            if state == "train":
                model.train()
            else:
                model.eval()

            for id, batch in enumerate(loader, 0):

                with torch.set_grad_enabled(state == 'train'):
                    
                    smiles, vectorized_molecules, labels, w, node_features, adjacency_matrices, distance_matrices = batch
                    batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

                    # print(8*"-")
                    # print("----INPUTS-----")
                    # print(8*"-")
                    # print("node_features".upper())
                    # print(node_features)
                    # print("batch_mask".upper())
                    # print(batch_mask)
                    # print("adjacency_matrices".upper())
                    # print(adjacency_matrices)
                    # print("distance_matrices".upper())
                    # print(distance_matrices)

                    outputs = model(node_features, batch_mask, adjacency_matrices, distance_matrices, None)
                    # print(f"output: {outputs}")
                    # print(f"labels: {labels}")
                    
                    loss = criterion(outputs, labels)

                    if state == "train":
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item()

            # save and log epoch statistics
            checkpoint["test_loss"] = round(running_loss / len_dataset, 2)
            logging.info(f"Epoch: {epoch}, state: {state}, loss: {checkpoint['test_loss']}")

        if checkpoint["test_loss"] < best_test_loss:
            
            # update best test loss for further training
            best_test_loss = checkpoint["test_loss"]
            
            # save model to checkpoint
            checkpoint["epoch"] = epoch
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            checkpoint["save_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            checkpoint['dc_dataset_name'] = dataset_name
            checkpoint['yaml_config_path'] = yaml_config_path

            save_checkpoint(checkpoint, checkpoint_path)

        else:
            logging.info(8*"-")