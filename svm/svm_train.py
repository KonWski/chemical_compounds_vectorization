from dataset import MoleculeDataset
import logging
from sklearn import svm
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
from utils import load_yaml_config
from torch.nn import AdaptiveAvgPool2d, Flatten, Sequential
from torch import stack, Tensor, unsqueeze, squeeze

def train_svm(
        featurizer_type: str,
        dataset_name: str,
        download_dataset: bool,
        root_datasets_dir: str,
        checkpoint_path: str,
        config_name: str,
        dataset_task_name: str,
        model_type: str
    ):
    '''

    Parameters
    ----------
    device 
    dataset_name: str
        Name of dataset available through MoleculeNet
    download_dataset: bool
        Download dataset from MoleculeNet
    root_datasets_dir: str
        Path where dataset should be downloaded or where is it already stored
    checkpoint_path: str
        Path to the loaded checkpoint
    config_name: str
        configuration name selected from yaml describing model
    dataset_task_name: str
        task used for filtering down tox21 dataset
    model_type: str
        type of model which will be trained
    '''

    # datasets
    trainset = MoleculeDataset(dataset_name, "train", featurizer_type, True, download_dataset, root_datasets_dir, dataset_task_name, model_type)
    testset = MoleculeDataset(dataset_name, "test", featurizer_type, True, download_dataset, root_datasets_dir, dataset_task_name, model_type)

    # train and test data
    X_train, y_train = np.array(trainset.vectorized_molecules), np.ravel(np.array(trainset.labels))
    X_test, y_test = np.array(testset.vectorized_molecules), np.ravel(np.array(testset.labels))

    # extra layers necessary to make use of irregular (in shape) matrices
    level_off_shape_transforms = Sequential(AdaptiveAvgPool2d(32), Flatten())
    
    for phase, dataset in zip(["train", "test"], [trainset, testset]):

        leveled_off_node_features = []
        leveled_off_adjacency_matrices = []
        leveled_off_distance_matrices = []

        for nf, am, dm in zip(dataset.node_features, dataset.adjacency_matrix, dataset.distance_matrices):

            leveled_off_node_features.append(level_off_shape_transforms(unsqueeze(Tensor(nf), 0)))
            leveled_off_adjacency_matrices.append(level_off_shape_transforms(unsqueeze(Tensor(am), 0)))
            leveled_off_distance_matrices.append(level_off_shape_transforms(unsqueeze(Tensor(dm), 0)))

        leveled_off_node_features = np.array(squeeze(stack(leveled_off_node_features), 1))
        leveled_off_adjacency_matrices = np.array(squeeze(stack(leveled_off_adjacency_matrices), 1))
        leveled_off_distance_matrices = np.array(squeeze(stack(leveled_off_distance_matrices), 1))

        if phase == "train":
            X_train = np.concatenate((X_train, leveled_off_node_features), axis=1)    
            X_train = np.concatenate((X_train, leveled_off_adjacency_matrices), axis=1)
            X_train = np.concatenate((X_train, leveled_off_distance_matrices), axis=1)
        else:
            X_test = np.concatenate((X_test, leveled_off_node_features), axis=1)    
            X_test = np.concatenate((X_test, leveled_off_adjacency_matrices), axis=1)
            X_test = np.concatenate((X_test, leveled_off_distance_matrices), axis=1)

    # load params for model from yaml
    model_params, _ = load_yaml_config("svm", config_name)

    # criterion and model type
    criterion = trainset.criterion
    model = svm.SVR(**model_params) if trainset.prediction_task == "regression" else svm.SVC(**model_params)

    # train model
    pipeline = Pipeline([('scaler', StandardScaler()), ('svm', model)])
    pipeline.fit(X_train, y_train)

    for state, X, y in zip(["train", "test"], [X_train, X_test], [y_train, y_test]):
        y_predicted = pipeline.predict(X)
        loss = round(criterion(y_true=y, y_pred=y_predicted), 2)
        logging.info(f"state: {state}, loss: {loss}")
    
    # save model to checkpoint path
    saved_model_name = f"svm_{config_name}.pkl"
    pickle_path = f"{checkpoint_path}/{saved_model_name}"

    with open(pickle_path, "wb") as f:
        pickle.dump(pipeline, f)
        logging.info(f"Model {saved_model_name} saved to {checkpoint_path}")