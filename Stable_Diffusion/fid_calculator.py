import torch
from pytorch_fid.fid_score import calculate_fid_given_paths

# Paths to the datasets
# These should be paths to directories containing images. Each directory should contain images of one class.
path_real = './Results/original'
path_fake = './Results/reproduced'

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Calculate the FID score
fid_value = calculate_fid_given_paths([path_real, path_fake], batch_size=50, device=device, dims=2048)
print(f'FID score: {fid_value}')
