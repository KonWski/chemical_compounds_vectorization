from dataset import MoleculeDataset, MoleculeDataLoader
import torch
from torch.optim import Adam
import logging
from mat.mat_model import make_model, load_checkpoint, save_checkpoint
from datetime import datetime
from utils import load_yaml_config
from sklearn.metrics import roc_auc_score
from torch import softmax

def train_mat(
        device, 
        n_epochs: int,
        dataset_name: str,
        splitter: str,
        download_dataset: bool,
        root_datasets_dir: str,
        checkpoint_path: str,
        batch_size: int,
        model_type: str,
        load_model: bool,
        featurizer_type: str,
        config_name: str,
        dataset_task_name: str
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
    featurizer_type: str
        featurizer used for initial compounds vectorization
    config_name: str
        configuration name selected from yaml describing model
    dataset_task_name: str
        task used for filtering down tox21 dataset
    '''

    # datasets and dataloaders
    trainset = MoleculeDataset(dataset_name, "train", featurizer_type, splitter, True, download_dataset, 
                               root_datasets_dir, dataset_task_name, model_type)
    train_loader = MoleculeDataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = MoleculeDataset(dataset_name, "test", featurizer_type, splitter, True, download_dataset, 
                              root_datasets_dir, dataset_task_name, model_type)
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
        model_params, yaml_config_path = load_yaml_config(model_type, config_name, trainset)
        model = make_model(**model_params)
        optimizer = Adam(model.parameters(), lr=1e-5)
        best_test_loss = float("inf")
        start_epoch = 0

    model = model.to(device)

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
                    
                    _, _, labels, _, node_features, adjacency_matrices, distance_matrices = batch
                    batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

                    labels = labels.to(device)
                    node_features = node_features.to(device)
                    adjacency_matrices = adjacency_matrices.to(device)
                    distance_matrices = distance_matrices.to(device)
                    batch_mask = batch_mask.to(device)

                    outputs = model(node_features, batch_mask, adjacency_matrices, distance_matrices, None)
                    if trainset.prediction_task == "classification":
                        outputs = torch.sigmoid(outputs)

                    loss = criterion(outputs, labels.cpu().detach().numpy())

                    if state == "train":
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item()

            # save and log epoch statistics
            checkpoint["test_loss"] = round(running_loss / len_dataset, 2)
            
            if trainset.prediction_task == "classification":
                proba = softmax(outputs, 1)
                auc = roc_auc_score(labels, proba)

                logging.info(f"Epoch: {epoch}, state: {state}, loss: {checkpoint['test_loss']}, auc: {auc}")

            else:
                logging.info(f"Epoch: {epoch}, state: {state}, loss: {checkpoint['test_loss']}, auc: {auc}")

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
            checkpoint['dataset_task_name'] = trainset.dataset_task_name

            save_checkpoint(checkpoint, checkpoint_path, f"{model_type}_epoch_{epoch}_{dataset_name}")

        else:
            logging.info(8*"-")