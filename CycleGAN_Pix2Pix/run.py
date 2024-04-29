import os
import shutil
from sklearn.model_selection import train_test_split

# Function to create necessary folders
def create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to split data and move files
def prepare_dataset(source_folder, dest_folder, split_ratio=(0.7, 0.15, 0.15)):
    files = os.listdir(source_folder)
    train_files, val_test_files = train_test_split(files, test_size=(split_ratio[1] + split_ratio[2]), random_state=42)
    val_files, test_files = train_test_split(val_test_files, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=42)

    # Function to copy files to the target directory
    def copy_files(files, target_folder):
        create_folders(target_folder)
        for file in files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(target_folder, file))

    copy_files(train_files, os.path.join(dest_folder, 'train'))
    copy_files(val_files, os.path.join(dest_folder, 'val'))
    copy_files(test_files, os.path.join(dest_folder, 'test'))


path_to_repo = '/storage/ice1/3/3/mjain335/DL/pytorch-CycleGAN-and-pix2pix/PRAM/Dataset'
path_to_data = '/storage/ice1/3/3/mjain335/DL/pytorch-CycleGAN-and-pix2pix/datasets'

# Create required directories
create_folders(path_to_data)
create_folders(f"{path_to_data}/A")
create_folders(f"{path_to_data}/B")

# Path to your datasets
sketches_folder = os.path.join(path_to_repo, 'sketches')
originals_folder = os.path.join(path_to_repo, 'original_images')

# Prepare datasets
prepare_dataset(sketches_folder, f"{path_to_data}/A")
prepare_dataset(originals_folder, f"{path_to_data}/B")

