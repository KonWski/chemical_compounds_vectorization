from dataset import MoleculeDataset
import logging
from sklearn import svm
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
from utils import load_yaml_config

def train_svm(
        featurizer_type: str,
        dataset_name: str,
        download_dataset: bool,
        root_datasets_dir: str,
        checkpoint_path: str,
        config_name: str
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
    '''

    # datasets
    trainset = MoleculeDataset(dataset_name, "train", featurizer_type, True, download_dataset, root_datasets_dir)
    testset = MoleculeDataset(dataset_name, "test", featurizer_type, True, download_dataset, root_datasets_dir)

    # train and test data
    X_train, y_train = np.array(trainset.vectorized_molecules), np.array(trainset.labels)
    X_test, y_test = np.array(testset.vectorized_molecules), np.array(testset.labels)

    # load params for model from yaml
    model_params, _ = load_yaml_config("svm", config_name)

    # train model
    pipeline = Pipeline([('scaler', StandardScaler()), ('svr', svm.SVR(**model_params))])
    pipeline.fit(X_train, y_train)

    for state, X, y in zip(["train", "test"], [X_train, X_test], [y_train, y_test]):
        y_predicted = pipeline.predict(X)
        loss = round(mean_squared_error(y_true=y, y_pred=y_predicted), 2)
        logging.info(f"state: {state}, loss: {loss}")
    
    # save model to checkpoint path
    with open(f"{checkpoint_path}/svm_{config_name}.pkl", "wb") as f:
        pickle.dump(pipeline, f)