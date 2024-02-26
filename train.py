from dataset import CIFAR10GAN
import torch
from models import DiscriminatorCIFAR10, GeneratorCIFAR10
from torch.optim import Adam
from torch.nn import BCELoss
import logging
from torchvision import transforms
from torchvision.utils import save_image
from mat_data_utils import construct_loader

def train_model(
        device, 
        n_epochs: int,
        dataset_name: str,
        batch_size: int,
        download_datasets: bool,
        root_datasets_dir: str,
        model_type: str
    ):
    '''

    Parameters
    ----------
    device 
    n_epochs: int
        number of training epochs
    batch_size: int
        number of images inside single batch
    ref_images_dir: str
        path to directory where ref images will be stored
    download_datasets: bool
        True -> download dataset from torchvision repo
    root_datasets_dir: str
        path to directory where dataset should be downloaded (download_datasets = True)
        or where dataset is already stored
    class_name: str
        one of ten classes in CIFAR10 dataset
    latent_vector_length: int
        length of random vector which will be transformed into an image by generator
    init_generator_weights: bool
        Init generator's weights using normal distribiution
    init_discriminator_weights: bool
        Init discriminator's weights using normal distribiution
    '''


