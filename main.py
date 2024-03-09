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

    parser.add_argument('--model_type', type=str, help="Type of model which will be trained", 
                        choices=["mat", "mol2vec"])    
    parser.add_argument('--checkpoint_path', type=str, help="Path to the loaded checkpoint")
    parser.add_argument('--load_model', type=str, help="Continue learning using existing model and optimizer")   


    args = vars(parser.parse_args())
    
    # parse str to boolean
    str_true = ["Y", "y", "Yes", "yes", "true", "True"]
    bool_params = ["download_dataset", "load_model"]
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

    model = train_model(device, args["n_epochs"], args["dataset_name"], args["download_dataset"], 
                        args["root_datasets_dir"], args["checkpoint_path"], args["batch_size"], 
                        args["model_type"], args["load_model"])