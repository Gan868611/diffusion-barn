import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings('ignore')
from  dataset.KULBarnDataset import KULBarnDataset
import numpy as np
from torch.utils.data import DataLoader
from model.cnn_model_behavior_cloning import CNNModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from omegaconf import OmegaConf
from datetime import datetime
import pytz
import shutil

# Load hyperparameters from config.yaml
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = OmegaConf.load(config_path)

df = pd.read_csv(config.inspection_data)
df.head()



# # remove rows with success = 0
df = df[df['success'] == True]

# take random 90% of the world ids for training
ids = df['world_idx'].unique()

test_ids = list(range(0, 300, 5))

non_test_ids = np.setdiff1d(ids, test_ids)

test_df = df[df['world_idx'].isin(test_ids)]
test_dataset = KULBarnDataset(test_df, mode="test")



print("Test Dataset Length:", len(test_dataset))

# dataloader

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Initialize the model
num_lidar_features = len(test_dataset.lidar_cols)
num_non_lidar_features = len(test_dataset.non_lidar_cols)
num_actions = len(test_dataset.actions_cols)
model = CNNModel(num_lidar_features, num_non_lidar_features, num_actions)
model_path = '/jackal_ws/src/mlda-barn-2024/outputs/behavior_cloning_cnn/v1/cnn_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Move the model and loss function to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_fn = loss_fn.to(device)
print(device)


from tqdm import tqdm

def test_model(model, test_loader, loss_fn):
    model.eval()

    losses = []
    for lidar, non_lidar, actions in tqdm(test_loader):
        # Move the data to the device that is used
        lidar = lidar.to(device).unsqueeze(1)
        non_lidar = non_lidar.to(device).unsqueeze(1)
        actions = actions.to(device)

        # Forward pass
        actions_pred = model(lidar.float(), non_lidar.float())
        loss = loss_fn(actions_pred, actions.float())

        # Save the loss
        losses.append(loss.item())

    # return the average loss for this epoch
    return sum(losses)/len(losses)


import sys



sys.stdout.flush()

test_loss = test_model(model, test_loader=test_loader, loss_fn=loss_fn)
# cnn_test_losses.append(test_loss)

print(f"Test Loss: {test_loss}")
sys.stdout.flush()

