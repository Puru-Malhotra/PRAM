from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from fashion_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint


# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 16
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
torch.manual_seed(0)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders to handle batching
train_loader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=batch_size, shuffle=False)

# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

# Checkpoint callback to save the latest model weights
checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints/',
    filename='latest-model',
    save_top_k=1,  # Save only the most recent version
    every_n_epochs=1,  # Save every epoch
    save_on_train_epoch_end=True  # Ensure it saves at the end of the epoch
)


trainer = pl.Trainer(gpus=1, precision=32, strategy='ddp', callbacks=[logger, checkpoint_callback], max_epochs=5)


# Train!
trainer.fit(model, train_loader, test_loader)
