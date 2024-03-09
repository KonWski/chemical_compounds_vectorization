import os

def prepare_checkpoint_directory(checkpoint_path):
    
    checkpoint_dir_created = os.path.isdir(checkpoint_path)

    if not checkpoint_dir_created:
        os.mkdir(checkpoint_path)