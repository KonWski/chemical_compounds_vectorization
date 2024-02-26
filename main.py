import argparse
import logging
import torch
from train import train_model

def get_args():

    parser = argparse.ArgumentParser(description='Paramaters for model training')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Number of images in batch')
    parser.add_argument('--root_datasets_dir', type=str, help='Path where dataset should be downloaded or where is it already stored')

    parser.add_argument('--dataset_name', type=str, help='Name of dataset available through MoleculeNet', 
                        choices=["HIV", "TOX21", "Delaney"])
    parser.add_argument('--download_dataset', type=str, help='Download dataset from MoleculeNet')

    parser.add_argument('--model_type', type=str, help="type of model which will be trained", 
                        choices=["message_passing_neural_network", "mol2vec"])    

    args = vars(parser.parse_args())
    
    # parse str to boolean
    str_true = ["Y", "y", "Yes", "yes", "true", "True"]
    bool_params = ["download_datasets", "init_generator_weights", "init_discriminator_weights"]
    for param in bool_params:
        if args[param] in str_true:
            args[param] = True
        else:
            args[param] = False

    # log input parameters
    logging.info(8*"-")
    logging.info("PARAMETERS")
    logging.info(8*"-")

    for parameter in args.keys():
        logging.info(f"{parameter}: {args[parameter]}")
    logging.info(8*"-")

    return args

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    model = train_model(device, args["n_epochs"], args["batch_size"], args["ref_images_dir"], 
                        args["download_datasets"], args["root_datasets_dir"], args["class_name"],
                        args["latent_vector_length"], args["init_generator_weights"],
                        args["init_discriminator_weights"])