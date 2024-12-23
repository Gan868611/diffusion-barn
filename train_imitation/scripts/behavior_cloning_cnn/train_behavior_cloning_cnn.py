import sys
import os
sys.path.append('/jackal_ws/src/mlda-barn-2024/train_imitation')
import warnings
warnings.filterwarnings('ignore')
from dataset.KULBarnDataset import KULBarnDataset
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

train_ids = np.random.choice(non_test_ids, int(0.8 * len(non_test_ids)), replace=False)
train_df = df[df['world_idx'].isin(train_ids)]
train_dataset = KULBarnDataset(train_df, mode="train")

# take the remaining of the world ids for validation
val_ids = np.setdiff1d(non_test_ids, train_ids)
val_df = df[df['world_idx'].isin(val_ids)]
val_dataset = KULBarnDataset(val_df, mode="val")


print(len(train_ids), len(val_ids))


print("Train Dataset Length:", len(train_dataset))
print("Val Dataset Length:", len(val_dataset))


# dataloader


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
# test dataloader
lidar, non_lidar, actions = next(iter(train_loader))
print(f"Non lidar shape: {non_lidar.shape}")
print(f"Lidar shape: {lidar.shape}")
# print size dataloader
print(f"Train loader size: {len(train_loader)}")
print(f"Val loader size: {len(val_loader)}")
print(lidar, non_lidar, actions)



# # make a CustomLoss prioritizing the angular velocity
# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()

#     def forward(self, pred, target):
#         # increase the loss of the second element of the prediction
#         # this is the angular velocity
#         loss = (pred - target) ** 2
#         loss[:, 1] *= 2
#         return loss.mean()


# Initialize the model
num_lidar_features = len(train_dataset.lidar_cols)
num_non_lidar_features = len(train_dataset.non_lidar_cols)
num_actions = len(train_dataset.actions_cols)
model = CNNModel(num_lidar_features, num_non_lidar_features, num_actions)

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Move the model and loss function to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_fn = loss_fn.to(device)
print(device)


from tqdm import tqdm

def train_model(model, train_loader, loss_fn, optimizer):
    model.train()

    losses = []
    for lidar, non_lidar, actions in tqdm(train_loader):
        # Move the data to the device that is used
        lidar = lidar.to(device).unsqueeze(1)
        non_lidar = non_lidar.to(device).unsqueeze(1)
        actions = actions.to(device)

        # Forward pass
        actions_pred = model(lidar.float(), non_lidar.float())
        loss = loss_fn(actions_pred, actions.float())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the loss
        losses.append(loss.item())

    # return the average loss for this epoch
    return sum(losses)/len(losses)


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
NUM_EPOCHS = config.num_epochs

random_val_loss = test_model(model, val_loader, loss_fn)
print("Random val loss:", random_val_loss)
sys.stdout.flush()

cnn_train_losses = []
cnn_val_losses = []
best_val_loss = float('inf')
patience = config.patience
no_improve_epochs = 0

for epoch in range(NUM_EPOCHS):
    train_loss = train_model(model, train_loader, loss_fn, optimizer)
    val_loss = test_model(model, val_loader, loss_fn)
    cnn_train_losses.append(train_loss)
    cnn_val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}")
    sys.stdout.flush()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping due to no improvement after {} epochs.".format(patience))
            break


# plot the loss
import matplotlib.pyplot as plt

plt.plot(cnn_train_losses, label='Train Loss')
plt.plot(cnn_val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# plt.show()

from datetime import datetime, timedelta
import pytz

singapore_tz = pytz.timezone('Asia/Singapore')
current_time = datetime.now(singapore_tz)
adjusted_time = current_time - timedelta(minutes=9)
timestamp = adjusted_time.strftime('%y%m%d_%H%M%S')

base_path = '/jackal_ws/src/mlda-barn-2024/outputs/behavior_cloning_cnn/'
dir_path = os.path.join(base_path, timestamp)
os.makedirs(dir_path, exist_ok=True)

plot_filepath = os.path.join(dir_path, 'loss_plot.png')
plt.savefig(plot_filepath)

# save the model
torch.save(model.state_dict(), dir_path + '/cnn_model.pth')

dataset_script_src = config.dataset_script_src
dataset_script_dst = os.path.join(dir_path, 'KULBarnDataset.py')
shutil.copyfile(dataset_script_src, dataset_script_dst)


config_dst = os.path.join(dir_path, 'config.yaml')
shutil.copyfile(config_path, config_dst)







# %%
