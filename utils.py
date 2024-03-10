import os
import yaml
from pathlib import Path
from dataset import MoleculeDataset

def prepare_checkpoint_directory(checkpoint_path):
    
    checkpoint_dir_created = os.path.isdir(checkpoint_path)

    if not checkpoint_dir_created:
        os.mkdir(checkpoint_path)


def load_yaml_config(model_type: str, config_name: str = "default", dataset: MoleculeDataset = None):
    '''
    loads yaml with model settings and selects the subset of configuration
    '''

    yaml_config_path = f'{Path(__file__).parent}/models_params/{model_type}.yaml'

    with open(yaml_config_path, 'r') as yaml_config:
        yaml_config = yaml.safe_load(yaml_config)
        model_params = yaml_config[config_name]

        if model_type == "mat":
            model_params["d_atom"] = dataset.node_features[0].shape[1]

    return model_params